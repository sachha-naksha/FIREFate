# downloaded: ipykernel, jupyterlab, notebook, scipy, requests, pytz, seaborn, plotly, h5py
# Core libraries
import os, time, gc, itertools, pickle, psutil
from collections import OrderedDict, defaultdict
from tqdm import tqdm
# Scientific and Data libraries
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, ttest_1samp, norm
# Plotting libraries
import seaborn as sns, matplotlib.pyplot as plt
import plotly.express as px, plotly.graph_objs as go, plotly.offline as py
# Bioinformatics libraries
from Bio.Seq import Seq
from Bio import SeqIO
from skbio.alignment import StripedSmithWaterman
import pybedtools, logomaker
pybedtools.set_bedtools_path("/opt/packages/bedtools/2.30.0") #### Required so that it can find the bedtools executable
# PyTorch libraries
import torch
import torch.nn.functional as F
# Multiprocessing
from multiprocessing import Pool, cpu_count
# h5py for handling HDF5 files
import h5py

def seq2tensor(seq):
    """
    Converts a DNA sequence string to a tensor containing only 01234 (ACGTN).

    Args:
    - seq (str): DNA sequence

    Returns:
    - tensor: containing only 01234 (ACGTN)
    """
    # Convert sequence to uppercase
    seq_out = seq.upper()

    # Define allowed DNA characters
    allowed = set("ACTGN")

    # Replace invalid characters with 'N'
    if not set(seq_out).issubset(allowed):
        invalid = set(seq_out) - allowed
        seq_out = seq_out.translate(str.maketrans("".join(list(invalid)), 'N' * len(invalid)))

    # Map DNA characters to numerical values
    seq_out = seq_out.translate(str.maketrans('ACGTN', '01234'))

    # Convert sequence to tensor
    return torch.from_numpy(np.asarray(list(seq_out), dtype='int'))

def get_reverse_complement(seq):
    """
    Get the reverse complement of a DNA sequence.

    Args:
    seq (str): The DNA sequence.

    Returns:
    str: The reverse complement of the input DNA sequence.
    """
    seq = seq.upper()
    # Use translate to get the reverse complement and then reverse the result
    return seq.translate(str.maketrans('ATCG', 'TAGC'))[::-1]

def get_random_seq(size):
    """
    Generate a random DNA sequence of a given size.

    Args:
    size (int): The length of the DNA sequence to generate.

    Returns:
    str: The randomly generated DNA sequence.
    """
    bases = list('ATCG')  # List of DNA bases
    random_seq = ''.join(np.random.choice(bases, size))  # Generate random sequence
    return random_seq

def one_hot4seqs(list_seq):
    """
    Encode a list of DNA sequences into a one-hot matrix.

    Args:
    list_seq (list): List of DNA sequences to encode

    Returns:
    torch.Tensor: One-hot matrix of shape (N, 4, seq_length)
    """
    # Convert each sequence to tensor and uppercase
    list_seq_out = []
    for i in range(len(list_seq)):
        list_seq_out.append(seq2tensor(list_seq[i].upper()))

    # Stack the tensors, apply one-hot encoding, and permute the dimensions
    return F.one_hot(torch.stack(list_seq_out),
                     num_classes=5).permute((0, 2, 1))[:, :-1, :]
#### Function to get the sequences from fasta or bedfile
#### Function for reading FASTA
def read_fasta(fasta_file):
    dic_fasta = OrderedDict()
    with open(fasta_file, 'r') as infile:
        for line in infile:
            if line.startswith('>'):
                seq_name = line.strip().split(' ')[0][1:]
                dic_fasta[seq_name] = ''
            else:
                dic_fasta[seq_name] += line.strip()
    return dic_fasta

#### Function for getting bed sequences
def get_bed_peak_sequence(bed_file, genome_sequence_fa, return_seq_name=False):
    pybed = pybedtools.BedTool(bed_file)
    pybed = pybed.sequence(fi=genome_sequence_fa, s=True)
    fa_file = pybed.seqfn
    if return_seq_name:
        return list(read_fasta(fa_file).values()),list(read_fasta(fa_file).keys())
    else:
        return list(read_fasta(fa_file).values())

#### Function for shuffling DNA sequences
def dinuc_shuffle(seq, num_shufs=1, random_seed=6):
    """
    Creates shuffles of the given DNA sequence, preserving dinucleotide frequencies.
    
    Parameters:
        seq (str): A DNA sequence string.
        num_shufs (int): The number of shuffles to create.
        rng (np.random.RandomState or None): Random state object or None. If None, uses default RNG.
    
    Returns:
        List of str: List of shuffled DNA sequence strings.
    """
    rng = np.random.RandomState(random_seed)

    # Convert the DNA sequence to integer tensor (A=0, C=1, G=2, T=3)
    seq_upper = seq.upper().replace('N', 'G')
    seq_tensor = seq2tensor(seq_upper).numpy()
    seq_len = len(seq_upper)
    num_chars = 4  # Number of different nucleotides

    # Build transition matrix for dinucleotide frequencies
    shuf_next_inds = [np.where(seq_tensor[:-1] == t)[0] + 1 for t in range(num_chars)]

    shuffled_sequences = []
    
    try:
        for _ in range(num_shufs):
            # Shuffle the next indices while keeping the last index fixed
            for t in range(num_chars):
                inds = np.arange(len(shuf_next_inds[t]))
                if len(inds) > 1:
                    rng.shuffle(inds[:-1])  # Shuffle indices, keep the last index same
                shuf_next_inds[t] = shuf_next_inds[t][inds]

            counters = [0] * num_chars
            result_tokens = np.empty(seq_len, dtype=int)

            # Build the shuffled sequence
            ind = 0
            result_tokens[0] = seq_tensor[ind]
            for j in range(1, seq_len):
                t = seq_tensor[ind]
                ind = shuf_next_inds[t][counters[t]]
                counters[t] += 1
                result_tokens[j] = seq_tensor[ind]

            # Convert the shuffled sequence back to string
            int_to_char = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
            shuffled_seq = ''.join(int_to_char[token] for token in result_tokens)
            shuffled_sequences.append(shuffled_seq)
    except:
        print("sequence failed to dinucleotide shuffle, ignoring...:",seq)
        
    
    return shuffled_sequences
#### Function for CESeek Tests
def z_test_one_tailed(x, A, population_variance=None):
    """
    Perform a one-tailed Z-test to evaluate whether x is significantly higher than the distribution of values in A.

    Args:
    - x (float): The value to test.
    - A (list or array): The list of values to compare against.
    - population_variance (float, optional): Known population variance.
                                                If not provided, the sample variance will be used.

    Returns:
    - z_score (float): The Z-score for the test.
    - p_value (float): The p-value for the test (one-tailed, checking if x is greater than A).
    """

    # Convert A to a NumPy array for calculation
    A = np.array(A)

    # Calculate mean and standard deviation of the list A
    mean_A = np.mean(A)

    # Use population variance if provided, otherwise use sample variance
    if population_variance is not None:
        std_A = np.sqrt(population_variance)
    else:
        std_A = np.std(A, ddof=1)  # Sample standard deviation with Bessel's correction

    # Calculate the Z-score
    n = len(A)
    z_score = (x - mean_A) / (std_A / np.sqrt(n))

    # Calculate the one-tailed p-value (testing if x is greater than the mean of A)
    p_value = norm.sf(z_score)  # Use the survival function (1 - CDF) to get the upper tail probability

    return z_score, p_value

def fisher_exact_test_for_motif_hits(a, A, b, B):
    # Construct the contingency table
    table = [[a, A - a], [b, B - b]]

    # Perform Fisher's exact test (one-sided)
    _, p_value = fisher_exact(table, alternative='greater')

    return p_value

def strand_conversion(strand1, strand2):
    if strand1 != strand2:
        return strand1, strand2
    else:
        if strand1 == "+":
            return "-", "-"
        else:
            return "+", "+"
    
def set_device_and_threads(device=None, num_threads=None):
    device_string = device
    if device:
        assert device in ['cpu', 'cuda'], "Invalid device"
    else:
        device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_string == 'cpu':
        if (num_threads is not None) and (num_threads>=1):
            torch.set_num_threads(num_threads)
        else:
            torch.set_num_threads(torch.get_num_threads())
    device = torch.device(device_string)
    return device