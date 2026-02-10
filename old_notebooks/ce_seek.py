
#### Functions for manipulating DNA sequences
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