import gc
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from dictys.utils.numpy import ArrayLike, NDArray
from joblib import Memory
from scipy import stats
from scipy.stats import hypergeom
from tqdm import tqdm

##################################### Data retrieval ############################################


def get_tf_indices(dictys_dynamic_object, tf_list):
    """
    Get the indices of transcription factors from a list, if present in ndict and nids[0].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    tf_mappings_to_gene_hashmap = dictys_dynamic_object.nids[0]
    tf_indices = []
    tf_gene_indices = []
    missing_tfs = []
    for gene in tf_list:
        # Check if the gene is in the gene_hashmap
        if gene in gene_hashmap:
            gene_index = gene_hashmap[gene]  # Get the index in gene_hashmap
            # Check if the gene index is present in tf_mappings_to_gene_hashmap
            match = np.where(tf_mappings_to_gene_hashmap == gene_index)[0]
            if match.size > 0:  # If a match is found
                tf_indices.append(int(match[0]))  # Append the position of the match
                tf_gene_indices.append(int(gene_index))  # Also append the gene index
            else:
                missing_tfs.append(gene)  # Gene exists but not as a TF
        else:
            missing_tfs.append(gene)  # Gene not found at all
    return tf_indices, tf_gene_indices, missing_tfs


def get_gene_indices(dictys_dynamic_object, gene_list):
    """
    Get the indices of target genes from a list, if present in ndict and nids[1].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    gene_indices = []
    for gene in gene_list:
        if gene in gene_hashmap:
            gene_indices.append(gene_hashmap[gene])
    return gene_indices


def check_if_gene_in_ndict(dictys_dynamic_object, gene_name, return_index=False):
    """
    Check if a gene is in the ndict of the dynamic object.
    """
    # Input validation
    if not hasattr(dictys_dynamic_object, "ndict"):
        raise AttributeError("Dynamic object does not have ndict attribute")
    # Handle single gene case
    if isinstance(gene_name, str):
        is_present = gene_name in dictys_dynamic_object.ndict
        if return_index:
            return dictys_dynamic_object.ndict.get(gene_name, None)
        return is_present
    # Handle list of genes case
    elif isinstance(gene_name, (list, tuple, set)):
        results = {
            "present": [],
            "missing": [],
            "indices": {} if return_index else None,
        }
        for gene in gene_name:
            if gene in dictys_dynamic_object.ndict:
                results["present"].append(gene)
                if return_index:
                    results["indices"][gene] = dictys_dynamic_object.ndict[gene]
            else:
                results["missing"].append(gene)
        # Add summary statistics
        results["stats"] = {
            "total_genes": len(gene_name),
            "found": len(results["present"]),
            "missing": len(results["missing"]),
            "percent_found": (len(results["present"]) / len(gene_name) * 100),
        }
        return results
    else:
        raise TypeError("gene_name must be a string or a list-like object of strings")


def curvature_of_expression(dcurve: pd.DataFrame, dtime: pd.Series):
    """
    Calculate the curvature of expression curves.
    """
    # First derivative (dx/dt)
    dx_dt = pd.DataFrame(
        np.gradient(dcurve, dtime, axis=1), index=dcurve.index, columns=dcurve.columns
    )
    # Second derivative (d2x/dt2)
    d2x_dt2 = pd.DataFrame(
        np.gradient(dx_dt, dtime, axis=1), index=dcurve.index, columns=dcurve.columns
    )
    return d2x_dt2

    
##################################### Window labels ############################################


def get_state_labels_in_window(dictys_dynamic_object, cell_labels):
    """
    Creates a mapping of window indices to their constituent cells' labels
    """
    # get cell assignment matrix from dictys_dynamic_object.prop['sc']['w']
    cell_assignment_matrix = dictys_dynamic_object.prop["sc"]["w"]
    state_labels_in_window = {}
    for window_idx in range(cell_assignment_matrix.shape[0]):
        indices_of_cells_present_in_window = np.where(
            cell_assignment_matrix[window_idx] == 1
        )[0] #these indices start from 0
        state_labels_in_window[window_idx] = [
            cell_labels[idx] for idx in indices_of_cells_present_in_window
        ]
    return state_labels_in_window


def get_state_total_counts(cell_labels):
    """
    Get total number of cells for each state in the dataset
    """
    state_counts = {}
    for label in cell_labels:
        state_counts[label] = state_counts.get(label, 0) + 1
    return state_counts


def get_top_k_fraction_labels(dictys_dynamic_object, window_idx, cell_labels, k=2):
    """
    Returns the k labels with both fractions for a given window
    """
    # Get state labels for all windows
    state_labels_dict = get_state_labels_in_window(dictys_dynamic_object, cell_labels)
    # Get window labels for specified window
    window_labels = state_labels_dict[window_idx]
    # Get total counts across all states
    state_total_counts = get_state_total_counts(cell_labels)
    # Count cells per state in this window
    window_counts = {}
    for label in window_labels:
        window_counts[label] = window_counts.get(label, 0) + 1
    total_cells_in_window = len(window_labels)
    # Calculate both fractions for each state
    state_metrics = {}
    for state in window_counts:
        window_composition = window_counts[state] / total_cells_in_window
        state_distribution = window_counts[state] / state_total_counts[state]
        state_metrics[state] = (window_composition, state_distribution)
    # Sort primarily by state_distribution, then by window_composition
    sorted_states = sorted(
        state_metrics.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True
    )
    return sorted_states[:k]


def window_labels_to_count_df(window_labels_dict):
    """
    Converts a dictionary of window indices to cell labels into a DataFrame
    with counts of each label per window.
    """
    from collections import Counter

    import pandas as pd

    # Get all unique labels
    all_labels = set()
    for labels in window_labels_dict.values():
        all_labels.update(labels)

    # Sort labels for consistency
    all_labels = sorted(all_labels)

    # Get all window indices
    window_indices = sorted(window_labels_dict.keys())

    # Initialize DataFrame with zeros
    count_df = pd.DataFrame(0, index=all_labels, columns=window_indices)

    # Fill in the counts for each window
    for window_idx, labels in window_labels_dict.items():
        # Count occurrences of each label in this window
        label_counts = Counter(labels)

        # Update the DataFrame
        for label, count in label_counts.items():
            count_df.loc[label, window_idx] = count

    return count_df


##################################### Plotting ############################################

def create_enriched_links_per_state(enriched_links, state_LF):
    # Create dictionaries to map genes to their states
    gene_to_state = dict(zip(state_LF['gene'], state_LF['color']))

    # Initialize lists to store links for each state
    state1_links = []  # For Red (PB)
    state2_links = []  # For Blue (GC)
    TFs = set[Any]()
    targets_in_lf = set[Any]()

    # Iterate over each row in the enriched_links DataFrame
    for _, row in enriched_links.iterrows():
        tf_str = row['TF']
        # Extract the TF name from the string representation of a tuple
        tf = tf_str.strip("(,)' ").replace("'", "")
        TFs.add(tf)
        
        # Handle the targets as a string representation of a list
        if isinstance(row['common'], str):
            # If it's a string representation of a list, convert it to a list
            targets_str = row['common'].strip("[]").replace("'", "")
            targets = [t.strip() for t in targets_str.split(",")]
        else:
            # If it's already a list
            targets = row['common']
    
        # Assign each TF-target pair to the appropriate state
        for target in targets:
            if target and target in gene_to_state:
                targets_in_lf.add(target)
                state = gene_to_state[target]
                link = (tf, target)
                if state == 'Red':
                    state1_links.append(link)
                elif state == 'Blue':
                    state2_links.append(link)
    TFs = list(TFs)
    targets_in_lf = list(targets_in_lf)

    return state1_links, state2_links, TFs, targets_in_lf

def extract_tf_gene_info(enriched_links_df):
    """
    Extract TF-Gene links and unique TF/Gene names from enriched links dataframe.
    """
    
    # Validate input
    if enriched_links_df is None or enriched_links_df.empty:
        return [], [], []
    
    required_cols = ['TF', 'Gene']
    for col in required_cols:
        if col not in enriched_links_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Remove any rows with NaN values in TF or Gene columns
    clean_df = enriched_links_df.dropna(subset=['TF', 'Gene'])
    
    if clean_df.empty:
        return [], [], []
    
    # Extract links as list of tuples
    links_list = list(zip(clean_df['TF'], clean_df['Gene']))
    
    # Get unique TF names (sorted for consistency)
    unique_tfs = sorted(clean_df['TF'].unique().tolist())
    
    # Get unique gene names (sorted for consistency)
    unique_genes = sorted(clean_df['Gene'].unique().tolist())
    
    return links_list, unique_tfs, unique_genes


##################################### Plotting (moved) ############################################
# Every figure now lives in ``firefate.utils.plots``; re-exported here so existing
# ``from firefate.utils.custom import *`` call sites (notebooks, ``utils_custom`` shim)
# keep working.

from firefate.utils.plots import (  # noqa: E402,F401
    cluster_heatmap,
    create_pathway_color_scheme,
    fig_expression_gradient_heatmap,
    fig_expression_linear_heatmap,
    fig_regulation_heatmap,
    plot_expression_for_multiple_genes,
    plot_force_heatmap,
    plot_force_heatmap_with_clustering,
    plot_gene_expression_subplots,
    plot_tf_gene_coregulation_heatmap,
)
