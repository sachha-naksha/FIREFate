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

from .motif_analysis_utility import *
from gimmemotifs.motif import default_motifs

class MotifProcessor:
    ########################################################################################################
    #### Codes Related to Motif Processing ####
    ########################################################################################################
    def __init__(self):
        self.motif_pwm_path = None
        self.dict_motif = None
        self.dict_motif_cutoff = None
        self.sequences = None
        self.sequence_names = None
        self.sequence_file_fa = None
        self.sequence_file_bed = None
        self.sequences_control = None
        # self.sequence_control_file_fa = None
        # self.sequence_control_file_bed = None
        self.sequence_length = None
        self.motif_hits = None
        self.dict_motif_hits = None
        self.dict_motif_hits_control = None
        self.species = "Human"

        # if motif_pwm_path is not None:
        #     self.motif_pwm_path = motif_pwm_path
        #     self.dict_motif, self.dict_motif_cutoff = self.read_motif_pwm()
        #     motif_filters = []
        #     for pwm in self.dict_motif.values():
        #         pwm = np.array(pwm) + 0.001
        #         pwm = pwm / pwm.sum(axis=1, keepdims=True)
        #         pwm = np.log2(pwm / 0.25)
        #         motif_filters.append(pwm)
        #     motif_length_max = max([pwm.shape[0] for pwm in motif_filters])
        #     for ind, pwm in enumerate(motif_filters):
        #         zero_padding_length = motif_length_max - pwm.shape[0]
        #         motif_filters[ind] = np.vstack((
        #             np.zeros((int(zero_padding_length / 2), 4)),
        #             pwm,
        #             np.zeros((zero_padding_length - int(zero_padding_length / 2), 4))
        #         ))
        #     self.motif_filters = np.array(motif_filters, dtype=float)
        #     self.motif_cutoffs = np.array(list(self.dict_motif_cutoff.values()))

    def process_custom_motif_pwm(self):
        dict_motif = OrderedDict()
        dict_motif_cutoff = OrderedDict()
        motif_name = None
        if os.path.isfile(self.motif_pwm_path):
            with open(self.motif_pwm_path, 'r') as infile:
                for line in infile:
                    if line.startswith('>'):
                        if motif_name is not None:
                            dict_motif[motif_name] = np.array(motif_pwm)
                        motif_name = line.strip().split('\t')[1]
                        motif_pwm = []
                        try:
                            dict_motif_cutoff[motif_name] = float(line.strip().split('\t')[2])
                        except:
                            dict_motif_cutoff[motif_name] = -np.inf
                    else:
                        motif_pwm.append(list(map(float, line.strip().split('\t'))))
                dict_motif[motif_name] = np.array(motif_pwm)
        else:
            for motif_file in os.listdir(self.motif_pwm_path):
                with open("%s/%s" % (self.motif_pwm_path, motif_file), 'r') as infile:
                    for line in infile:
                        if line.startswith('>'):
                            if motif_name is not None:
                                dict_motif[motif_name] = np.array(motif_pwm)
                            motif_name = line.strip().split('\t')[1]
                            motif_pwm = []
                            try:
                                dict_motif_cutoff[motif_name] = float(line.strip().split('\t')[2])
                            except:
                                dict_motif_cutoff[motif_name] = -np.inf
                        else:
                            motif_pwm.append(list(map(float, line.strip().split('\t'))))
                    dict_motif[motif_name] = np.array(motif_pwm)
        
        for motif in dict_motif.keys():
            dict_motif[motif] = np.array(dict_motif[motif], dtype=float)
        return dict_motif, dict_motif_cutoff

    def process_default_motif_pwm(self, motifs):
        dict_motif = OrderedDict()
        for motif in motifs:
            dict_motif[motif.id] = np.array(motif.pwm, dtype=float)
        return dict_motif

    def create_motif_filters_and_cutoffs(self, dict_motif, dict_motif_cutoff):
        motif_filters = []
        for pwm in dict_motif.values():
            pwm = np.array(pwm) + 0.001
            pwm = pwm / pwm.sum(axis=1, keepdims=True)
            pwm = np.log2(pwm / 0.25)
            motif_filters.append(pwm)
        motif_length_max = max([pwm.shape[0] for pwm in motif_filters])
        for ind, pwm in enumerate(motif_filters):
            zero_padding_length = motif_length_max - pwm.shape[0]
            motif_filters[ind] = np.vstack((
                np.zeros((int(zero_padding_length / 2), 4)),
                pwm,
                np.zeros((zero_padding_length - int(zero_padding_length / 2), 4))
            ))
        return np.array(motif_filters, dtype=float), np.array(list(dict_motif_cutoff.values()))
    
    def set_motifs(self, motif_pwm_path = None, TF_formatting="auto", verbose=True):
        if motif_pwm_path is None:
            if verbose:
                print("No motif pwm path entered. Loading default motifs for your species ...")

            if self.species in ["Mouse", "Human", "Rat"]: # If species is vertebrate, we use gimmemotif default motifs as a default.
                motifs = default_motifs()
                self.motif_pwm_path = None
                dict_motif, dict_motif_cutoff = self.process_default_motif_pwm(motifs)
                self.motif_db_name = "gimme.vertebrate.v5.0"
                self.TF_formatting = True
                if verbose:
                    print(" Default motif for vertebrate: gimme.vertebrate.v5.0. \n For more information, please see https://gimmemotifs.readthedocs.io/en/master/overview.html \n")
            else:
                raise ValueError(f"We don't have default motifs for your species, Please provide motif PWM path")

        else:
            if verbose:
                print("Custom motif data entered. Processing ...")
            self.motif_pwm_path = motif_pwm_path
            dict_motif, dict_motif_cutoff = self.process_custom_motif_pwm()
            self.motif_db_name = "custom_motifs"
            if TF_formatting == "auto":
                self.TF_formatting = False
            else:
                self.TF_formatting = TF_formatting
        self.dict_motif, self.dict_motif_cutoff = dict_motif, dict_motif_cutoff
        # self.motif_filters, self.motif_cutoffs = self.create_motif_filters()
        return None
    
    def get_motif_hits(self, list_seq, dict_motif, motif_filters, motif_cutoffs, strand="+", start_index=0, percentile_cutoff=.90, device=None, num_threads=None):
        device = set_device_and_threads(device,num_threads)

        motif_cutoffs = torch.from_numpy(motif_cutoffs).to(device)
        motif_cutoffs = motif_cutoffs.view(1, -1, 1)
        motif_filters = torch.from_numpy(motif_filters).unsqueeze(1).float().to(device)

        if strand == "+":
            seq_onehot = one_hot4seqs(list_seq)
        else:
            seq_onehot = one_hot4seqs([get_reverse_complement(seq) for seq in list_seq])
        
        seq_onehot = seq_onehot.permute((0, 2, 1)).unsqueeze(1).float().to(device)
        seq_onehot.share_memory_()
        # scores dim: [N_seq, N_motif, N_length, 1]
        # print(seq_onehot.shape, motif_filters.shape)
        # print(seq_onehot)
        scores = torch.nn.functional.conv2d(seq_onehot, motif_filters, padding=0, stride=1, groups=1)
        scores = scores.squeeze(3)

        scores_max = scores.max(dim=2).values  # Shape: (N_seq, N_motif)
        percentile = torch.quantile(scores_max, percentile_cutoff, dim=0, interpolation='linear')  # Shape: (N_motif,)
        self.percentile = percentile

        binary_matrix = scores >= motif_cutoffs
        # N_seq, N_motif, N_length = binary_matrix.shape
        hit_indices = torch.nonzero(binary_matrix)
        #         self.hit_indices = hit_indices
        #
    #     hit_indices[:, 0]: The sequence index (N_seq).

    # hit_indices[:, 1]: The motif index (N_motif).

    # hit_indices[:, 2]: The position in the sequence (N_length).
        # start_index: counting the batch basically
        # hit_indices[:, 0]: counting the sequence in the batch

        if strand == "+":
            pseudo_index = (start_index + hit_indices[:, 0]) * (2 * self.sequence_length) + hit_indices[:, 2]
        else:
            pseudo_index = (start_index + hit_indices[:, 0]) * (2 * self.sequence_length) + binary_matrix.shape[2] - hit_indices[:, 2]

        list_pseudo_indices, list_motif_cutoffs = [], []
        # Convert the indices to pseudo-indices
        for motif_ind, motif_name in enumerate(dict_motif):
            mask = (hit_indices[:, 1] == motif_ind)
            zero_padding_length = motif_filters.shape[1] - dict_motif[motif_name].shape[0]
            if strand == "+":
                list_pseudo_indices.append((pseudo_index[mask].cpu().numpy() + int(zero_padding_length / 2)).tolist())
            else:
                list_pseudo_indices.append((pseudo_index[mask].cpu().numpy() + zero_padding_length - int(zero_padding_length / 2) - 1).tolist())
            list_motif_cutoffs.append(percentile[motif_ind].cpu().item())  # Store the cutoff for this motif
        if device == 'cuda':
            torch.cuda.empty_cache()

        return list_pseudo_indices, list_motif_cutoffs
       
    def load_sequences(self, input_file, sequence_set=None, genome_sequence_fa=None, fasta_input=False,
                       out_seq_len=300):
        assert sequence_set in (None, "control", 'target')

        self.sequence_length = out_seq_len
        if fasta_input:
            list_seq = list(read_fasta(input_file).values())
            list_seq_names = list(read_fasta(input_file).keys())
        else:
            assert genome_sequence_fa is not None, "Please provide genome_sequence_fa"
            list_seq,list_seq_names = get_bed_peak_sequence(input_file, genome_sequence_fa, return_seq_name=True)
        # extend or subset from the center of the sequence to the same length
        for ind, seq in enumerate(list_seq):
            if len(seq) < out_seq_len:
                list_seq[ind] = seq + 'N' * (out_seq_len - len(seq))
            elif len(seq) > out_seq_len:
                list_seq[ind] = seq[(len(seq) - out_seq_len) // 2:(len(seq) + out_seq_len) // 2]
        if sequence_set == 'control':
            self.sequences_control = list_seq
            # if fasta_input:
            #     self.sequence_control_file_fa = input_file
            # else:
            #     self.sequence_control_file_bed = input_file
        else:
            self.sequences = list_seq
            self.sequence_names = list_seq_names
            if fasta_input:
                self.sequence_file_fa = input_file
            else:
                self.sequence_file_bed = input_file
        return None

    def scan_motif_on_batch_sequences(self, list_seq, num_threads=1, batch_max_number=1e7):

        batch_seq_size = int(batch_max_number / self.motif_filters.shape[0] / 2)

        dict_motif_hits = OrderedDict()

        for i in range(0, len(list_seq), batch_seq_size):
            list_seq_batch = list_seq[i:(i + batch_seq_size)]
            for strand in ["+", "-"]:
                list_pseudo_indices, _ = self.get_motif_hits(list_seq_batch, self.motif_filters, self.motif_cutoffs, strand=strand,
                                                          start_index=i, num_threads=num_threads)
                for motif_ind, motif_name in enumerate(self.dict_motif):
                    if motif_name not in dict_motif_hits:
                        dict_motif_hits[motif_name] = {}
                        dict_motif_hits[motif_name]['+'] = []
                        dict_motif_hits[motif_name]['-'] = []
                    dict_motif_hits[motif_name][strand] += list_pseudo_indices[motif_ind]

        # self.dict_motif_hits = dict_motif_hits
        return dict_motif_hits
    
    def scan_motif_on_sequences(self, list_seq, batch_max_number=1e7, **kwargs):
        dict_motif = self.dict_motif
        dict_motif_cutoff = self.dict_motif_cutoff
        motif_filters, motif_cutoffs = self.create_motif_filters_and_cutoffs(dict_motif, dict_motif_cutoff)
        batch_seq_size = int(batch_max_number / motif_filters.shape[0] / 2)
        dict_motif_hits = OrderedDict()

        for i in range(0, len(list_seq), batch_seq_size):
            list_seq_batch = list_seq[i:(i + batch_seq_size)]
            for strand in ["+", "-"]:
                list_pseudo_indices, _ = self.get_motif_hits(list_seq_batch, dict_motif, motif_filters, motif_cutoffs, 
                                                             strand=strand, start_index=i, **kwargs)
                for motif_ind, motif_name in enumerate(dict_motif.keys()):
                    if motif_name not in dict_motif_hits:
                        dict_motif_hits[motif_name] = {}
                        dict_motif_hits[motif_name]['+'] = []
                        dict_motif_hits[motif_name]['-'] = []
                    dict_motif_hits[motif_name][strand] += list_pseudo_indices[motif_ind]

        # self.dict_motif_hits = dict_motif_hits
        return dict_motif_hits
    
    def calculate_homer_score_for_motifs(self, list_seq, batch_max_number=1e7, **kwargs):
        mask = np.array(list(self.dict_motif_cutoff.values())) == -np.inf
        keys = np.array(list(self.dict_motif.keys()))[mask]
        
        dict_motif = OrderedDict((k, self.dict_motif[k]) for k in keys)
        dict_motif_cutoff = OrderedDict((k, -np.inf) for k in keys)
        motif_filters, motif_cutoffs = self.create_motif_filters_and_cutoffs(dict_motif, dict_motif_cutoff)
        self.motifs_with_no_scores=list(dict_motif.keys())

        batch_seq_size = int(batch_max_number / motif_filters.shape[0] / 2)

        dict_motif_cutoffs = OrderedDict()

        for i in range(0, len(list_seq), batch_seq_size):
            list_seq_batch = list_seq[i:(i + batch_seq_size)]
            for strand in ["+", "-"]:
                _, list_motif_cutoffs = self.get_motif_hits(list_seq_batch, dict_motif, motif_filters, motif_cutoffs,
                                                            strand=strand, start_index=i, **kwargs)
                for motif_ind, motif_name in enumerate(dict_motif.keys()):
                    if motif_name not in dict_motif_cutoffs:
                        dict_motif_cutoffs[motif_name] = {}
                        dict_motif_cutoffs[motif_name]['+'] = []
                        dict_motif_cutoffs[motif_name]['-'] = []
                    dict_motif_cutoffs[motif_name][strand].append(list_motif_cutoffs[motif_ind])
        # Calculate the final cutoff for each motif based on the scores from all sequences
        for motif_name in dict_motif_cutoffs.keys():
            # Combine the cutoffs from both strands
            combined_cutoff = []
            if '+' in dict_motif_cutoffs[motif_name]:
                combined_cutoff += dict_motif_cutoffs[motif_name]['+']
            if '-' in dict_motif_cutoffs[motif_name]:
                combined_cutoff += dict_motif_cutoffs[motif_name]['-']
            
            if len(combined_cutoff) > 0:
                # Set the cutoff to the 99th percentile of the scores for this motif
                final_cutoff = np.mean(combined_cutoff)
                self.dict_motif_cutoff[motif_name] = final_cutoff
            else:
                self.dict_motif_cutoff[motif_name] = np.nan
        return None

    def scan_motifs(self, batch_max_number=1e7, **kwargs):
        # assert self.sequences is not None, "Please provide sequences by calling load_sequences() first"
        assert self.dict_motif is not None, "Please provide motif PWMs by calling set_motifs() first"
        assert not (-np.inf in self.dict_motif_cutoff.values()), "Scores not found for some for motifs"
        # if -np.inf in self.dict_motif_cutoff.values():
        #     assert self.sequences_control is not None, "Please provide control sequences by calling load_sequences() first"
        #     self.calculate_homer_score_for_motifs(self.sequences_control, batch_max_number=batch_max_number, **kwargs)

        self.dict_motif_hits = self.scan_motif_on_sequences(self.sequences, batch_max_number=batch_max_number, **kwargs)
        if self.sequences_control is not None:
            self.dict_motif_hits_control = self.scan_motif_on_sequences(self.sequences_control, batch_max_number=batch_max_number, **kwargs)
    
    def single_motif_enrichment(self, motif_name1):
        assert motif_name1 in self.dict_motif, "Invalid motif name"
        assert self.sequences is not None, "Please provide target sequences by calling load_sequences() first"
        assert self.sequences_control is not None, "Please provide control sequences by calling load_sequences() first"
        assert self.dict_motif_hits is not None, "Please scan motifs on target sequencess by calling scan_motifs() first"
        assert self.dict_motif_hits_control is not None, "Please scan motifs on target sequencess by calling scan_motifs() first"

        dict_motif_hits_temp = {'target': [], 'control': []}
        for strand in ["+", "-"]:
            dict_motif_hits_temp['target'] += [i // (self.sequence_length * 2) for i in
                                                self.dict_motif_hits[motif_name1][strand]]
            dict_motif_hits_temp['control'] += [i // (self.sequence_length * 2) for i in
                                                self.dict_motif_hits_control[motif_name1][strand]]
        for key in ["target", "control"]:
            dict_motif_hits_temp[key] = set(dict_motif_hits_temp[key])
        pvalue = fisher_exact_test_for_motif_hits(len(dict_motif_hits_temp['target']), len(self.sequences),
                                                  len(dict_motif_hits_temp['control']), len(self.sequences_control))
        return pvalue