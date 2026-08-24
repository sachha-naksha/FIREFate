"""Cell-state labels per trajectory window, and their counts."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


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
