####
# 1. Loads smoothened networks and builds episodic GRNs.
# 2. Enriches regulons in an episodic manner to specify temporal action of TFs responsible for regulon expression dynamics.
####

import gc
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Optional, Tuple, Union

import dictys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dictys.net import stat
from dictys.utils.numpy import ArrayLike, NDArray
from joblib import Memory
from scipy import stats
from scipy.stats import hypergeom
from tqdm import tqdm

from utils_custom import *
from pseudotime_curves import *

class AlignTimeScales:
    """
    standardize the window and sampled points time-scales to S0 or S1 pseudotime values depending on the trajectory range being queried.
    """
    def __init__(self, dictys_dynamic_object=None,
        trajectory_range=(1, 3),
        num_points=40,
        dist=0.001,
        sparsity=0.01,
        total_episodes=8,
        ):
        """
        initialize the aligner with an optional dictys dynamic object.
        """
        self.dictys_dynamic_object = dictys_dynamic_object
        self.trajectory_range = trajectory_range
        self.num_points = num_points
        self.dist = dist
        self.sparsity = sparsity
        self.total_episodes = total_episodes

    def pseudotime_of_windows(self):
        """
        returns - numpy array (of window centroids' pseudotime values)
        S0 or S1 pseudotime values depending on the trajectory range.
        """
        # get 2d array of all nodes' pseudotimes per window
        node_specific_pseudotimes_per_window = self.dictys_dynamic_object.point['s'].dist
        # if traj range starts from 0 return the 0th pseudotime out of 0,1,2,3 and if it starts from 1 return the 1st pseudotime out of 0,1,2,3
        idx_for_querying_pseudotimes = self.trajectory_range[0]
        # return the pseudotime values
        return node_specific_pseudotimes_per_window[:, idx_for_querying_pseudotimes]

    def pseudotime_of_sampled_points(self):
        """
        returns - numpy array of sampled points' pseudotime values. 
        these points are not necessarily the window centroids, they are just equi-spaced points along the trajectory.
        S0 or S1 pseudotime values depending on the trajectory range.
        [should be directly corresponding to window centroid time points]
        """
        # sample equi-spaced points and instantiate smoothing function    
        pts, fsmooth = self.dictys_dynamic_object.linspace(self.trajectory_range[0], self.trajectory_range[1], self.num_points, self.dist)
        # pseudo time values (x axis)
        stat1_x = stat.pseudotime(self.dictys_dynamic_object, pts)
        tmp_x = stat1_x.compute(pts)
        dx = pd.Series(tmp_x[0])
        # return numpy array of pseudotime values
        return dx.to_numpy()


class EpisodeDynamics:
    """
    workflow for episodic grn extraction, filtering, force calculation, and enrichment.
    """

    def __init__(self, dictys_dynamic_object, output_folder, mode="expression",
                 trajectory_range=(1, 3), num_points=40, dist=0.001, sparsity=0.01, n_processes=16):
        
        # Core parameters
        self.dictys_dynamic_object = dictys_dynamic_object
        self.output_folder = output_folder
        self.mode = mode
        self.trajectory_range = trajectory_range
        self.num_points = num_points
        self.dist = dist
        self.sparsity = sparsity
        self.n_processes = n_processes
        
        # Initialize composed objects with same parameters
        self.curves = SmoothedCurves(
            dictys_dynamic_object=dictys_dynamic_object,
            trajectory_range=trajectory_range,
            num_points=num_points,
            dist=dist,
            sparsity=sparsity,
            mode=mode
        )
        
        self.time_aligner = AlignTimeScales(
            dictys_dynamic_object=dictys_dynamic_object,
            trajectory_range=trajectory_range,
            num_points=num_points,
            dist=dist,
            sparsity=sparsity
        )
        
        # State variables
        self.lcpm_dcurve = None
        self.dtime = None
        self.episode_beta_dcurve = None
        self.filtered_edges = None
        self.filtered_edges_p001 = None
        self.tf_lcpm_episode = None
        self.force_curves = None
        self.avg_force_df = None
        self.episodic_grn_edges = None
        self.lf_genes = None
        self.lf_in_object = None
        self.episodic_enrichment_df = None

    def compute_expression_curves(self, mode="expression"):
        """
        expression curves for all genes using the curves component.
        """
        # Update the mode in the curves object
        self.curves.mode = mode
        lcpm_dcurve, dtime = self.curves.get_smoothed_curves()
        self.lcpm_dcurve = lcpm_dcurve
        self.dtime = dtime
        return lcpm_dcurve, dtime

    def build_episode_grn(self, time_slice=slice(0, 5)):
        """
        build the episodic grn (weighted and binarized) for the specified episode/time window.
        """
        pts, fsmooth = self.dictys_dynamic_object.linspace(
            self.trajectory_range[0],
            self.trajectory_range[1],
            self.num_points,
            self.dist,
        )
        stat1_net = fsmooth(stat.net(self.dictys_dynamic_object))
        stat1_netbin = stat.fbinarize(stat1_net, sparsity=self.sparsity)
        dnet = stat1_net.compute(pts)
        dnetbin = stat1_netbin.compute(pts)
        dnet_episode = dnet[:, :, time_slice]
        dnetbin_episode = dnetbin[:, :, time_slice]

        # Map indices to gene names
        ndict = self.dictys_dynamic_object.ndict
        index_to_gene = {idx: name for name, idx in ndict.items()}
        target_names = [index_to_gene[idx] for idx in range(dnetbin_episode.shape[1])]
        tf_gene_indices = [
            self.dictys_dynamic_object.nids[0][tf_idx]
            for tf_idx in range(dnetbin_episode.shape[0])
        ]
        tf_names = [index_to_gene[idx] for idx in tf_gene_indices]

        # reshape to dataframe
        index_tuples = [(tf, target) for tf in tf_names for target in target_names]
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=["TF", "Target"])
        n_tfs, n_targets, n_times = dnet_episode.shape
        reshaped_data = dnet_episode.reshape(-1, n_times)
        episode_beta_dcurve = pd.DataFrame(
            reshaped_data,
            index=multi_index,
            columns=[f"time_{i}" for i in range(n_times)],
        )
        episode_beta_dcurve = episode_beta_dcurve[episode_beta_dcurve.sum(axis=1) != 0]

        # remove tfs with names starting with ZNF and ZBTB
        episode_beta_dcurve = episode_beta_dcurve[
            ~episode_beta_dcurve.index.get_level_values(0).str.startswith("ZNF")
            & ~episode_beta_dcurve.index.get_level_values(0).str.startswith("ZBTB")
        ]
        self.episode_beta_dcurve = episode_beta_dcurve
        return episode_beta_dcurve

    def filter_edges(
        self,
        min_nonzero_timepoints=3,
        alpha=0.05,
        min_observations=3,
        check_direction_invariance=True,
        n_processes=16,
        chunk_size=8000,
        pval_threshold=0.001,
    ):
        """
        filter episodic grn edges for significance and direction invariance.
        """
        filtered_edges = filter_edges_by_significance_and_direction(
            self.episode_beta_dcurve,
            min_nonzero_timepoints=min_nonzero_timepoints,
            alpha=alpha,
            min_observations=min_observations,
            check_direction_invariance=check_direction_invariance,
            n_processes=n_processes,
            chunk_size=chunk_size,
            save_intermediate=False,
            intermediate_path=self.output_folder,
        )
        self.filtered_edges = filtered_edges
        filtered_edges_p001 = filtered_edges[filtered_edges["p_value"] < pval_threshold]
        self.filtered_edges_p001 = filtered_edges_p001
        return filtered_edges_p001

    def compute_tf_expression(self):
        """
        compute tf expression for the episode (matching time window).
        """
        tf_names = self.filtered_edges_p001.index.get_level_values(0).unique()
        tf_lcpm_values = self.lcpm_dcurve.loc[tf_names]
        n_time_cols = len(
            [col for col in self.filtered_edges_p001.columns if col.startswith("time_")]
        )
        tf_lcpm_episode = tf_lcpm_values.iloc[:, 0:n_time_cols]
        tf_lcpm_episode.columns = [
            col for col in self.filtered_edges_p001.columns if col.startswith("time_")
        ][:n_time_cols]
        self.tf_lcpm_episode = tf_lcpm_episode
        return tf_lcpm_episode

    def calculate_forces(self, n_processes=20, chunk_size=30000, epsilon=1e-10):
        """
        calculate force curves for the filtered episodic grn.
        """
        beta_curves_for_force = self.filtered_edges_p001.drop("p_value", axis=1)
        force_curves = calculate_force_curves_parallel(
            beta_curves=beta_curves_for_force,
            tf_expression=self.tf_lcpm_episode,
            n_processes=n_processes,
            chunk_size=chunk_size,
            epsilon=epsilon,
            save_intermediate=False,
        )
        self.force_curves = force_curves
        avg_force = force_curves.mean(axis=1)
        avg_force_df = avg_force.to_frame(name="avg_force")
        self.avg_force_df = avg_force_df
        return avg_force_df

    def select_top_edges(self, percentile=98):
        """
        select the top k% of edges by absolute average force to build the episodic grn.
        """
        threshold = np.percentile(np.abs(self.avg_force_df["avg_force"]), percentile)
        top_percent_mask = np.abs(self.avg_force_df["avg_force"]) >= threshold
        episodic_grn_edges = self.avg_force_df[top_percent_mask].copy()
        self.episodic_grn_edges = episodic_grn_edges
        return episodic_grn_edges

    def select_top_activating_and_repressing_edges(
        self, percentile_positive=98.5, percentile_negative=0.5
    ):
        """
        select the top k% of edges by average force to build the episodic grn.
        selects top k% positive and top k% negative edges separately.
        returns the selected edges as a dataframe.
        """
        avg_force = self.avg_force_df["avg_force"]
        # Separate positive and negative selection
        positive_forces = avg_force[avg_force > 0]
        negative_forces = avg_force[avg_force < 0]
        # Top k% positive
        if len(positive_forces) > 0:
            pos_threshold = np.percentile(positive_forces, percentile_positive)
            top_pos_edges = positive_forces[positive_forces >= pos_threshold]
        else:
            top_pos_edges = pd.Series(dtype=avg_force.dtype)
        # Top k% negative (most negative)
        if len(negative_forces) > 0:
            neg_threshold = np.percentile(negative_forces, percentile_negative)
            top_neg_edges = negative_forces[negative_forces <= neg_threshold]
        else:
            top_neg_edges = pd.Series(dtype=avg_force.dtype)
        episodic_grn_edges = pd.concat([top_pos_edges, top_neg_edges]).to_frame(
            name="avg_force"
        )
        episodic_grn_edges = episodic_grn_edges.sort_values(
            by="avg_force", ascending=False
        )
        self.episodic_grn_edges = episodic_grn_edges
        return episodic_grn_edges

    def set_lf_genes(self, lf_genes):
        """
        check if the lf genes are in the dictys dynamic object.
        """
        self.lf_genes = lf_genes
        self.lf_in_object = check_if_gene_in_ndict(
            self.dictys_dynamic_object, lf_genes, return_index=True
        )
        return self.lf_in_object

    def annotate_lf_in_grn(self):
        """
        annotate which targets in the episodic grn are lf genes.
        """
        if self.lf_genes is None:
            raise ValueError("LF genes not set. Use set_lf_genes() first.")
        self.episodic_grn_edges["is_in_lf"] = (
            self.episodic_grn_edges.index.get_level_values(1).isin(self.lf_genes)
        )
        return self.episodic_grn_edges

    def calculate_enrichment(self):
        """
        calculate tf enrichment for lf genes in the episodic grn.
        """
        lf_in_episodic_grn = self.episodic_grn_edges[
            self.episodic_grn_edges["is_in_lf"]
        ]
        lf_genes_active_in_episode = lf_in_episodic_grn.index.get_level_values(
            1
        ).unique()
        target_genes_in_episodic_grn = self.episodic_grn_edges.index.get_level_values(
            1
        ).unique()
        enrichment_df = calculate_tf_episodic_enrichment(
            self.episodic_grn_edges,
            total_lf_genes=len(lf_genes_active_in_episode),
            total_genes_in_grn=len(target_genes_in_episodic_grn),
        )
        # sort the enrichment_score in descending order
        episodic_enrichment_df_sorted = enrichment_df.sort_values(
            by="enrichment_score", ascending=False
        )
        # drop rows with 0 enrichment score
        episodic_enrichment_df_sorted = episodic_enrichment_df_sorted[
            episodic_enrichment_df_sorted["enrichment_score"] != 0
        ]
        self.episodic_enrichment_df = episodic_enrichment_df_sorted
        return episodic_enrichment_df_sorted

    def episodic_composition(self):
        """
        Cellular composition of the episode is the union of all 
        window compositions present in the episode.
        """
        return None


def calculate_tf_episodic_enrichment(df, total_lf_genes, total_genes_in_grn):
    """
    Calculate TF enrichment scores using hypergeometric test

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with MultiIndex (TF, Target) and columns 'avg_force', 'is_in_lf'
    total_lf_genes : int
        Total number of active LF genes in the episode
    total_genes_in_grn : int
        Total number of genes in the episodic GRN

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: TF, p_value, enrichment_score, genes_in_lf, genes_dwnstrm, weights
    """

    results = []

    # Group by TF (level 0 of the MultiIndex)
    for tf in df.index.get_level_values(0).unique():
        tf_data = df.loc[tf]

        # Calculate metrics for this TF
        tf_lf_targets = tf_data[
            "is_in_lf"
        ].sum()  # Number of LF targets for this TF (k)
        tf_total_targets = len(tf_data)  # Total targets for this TF (n)

        # Get LF gene names and their weights for this TF
        lf_mask = tf_data["is_in_lf"]
        genes_in_lf = tuple(str(gene) for gene in tf_data[lf_mask].index.tolist())
        weights = tuple(tf_data[lf_mask]["avg_force"].tolist())

        # Get downstream genes that are NOT in LF (False for is_in_lf)
        non_lf_mask = ~tf_data["is_in_lf"]
        genes_dwnstrm = tuple(str(gene) for gene in tf_data[non_lf_mask].index.tolist())

        # Hypergeometric test parameters:
        # N = total_genes_in_grn (population size)
        # K = total_lf_genes (number of success states in population)
        # n = tf_total_targets (sample size)
        # k = tf_lf_targets (number of observed successes)

        # Calculate p-value using hypergeometric distribution
        # P(X >= k) = 1 - P(X <= k-1)
        if tf_total_targets > 0 and total_lf_genes > 0:
            p_value = hypergeom.sf(
                tf_lf_targets - 1, total_genes_in_grn, total_lf_genes, tf_total_targets
            )

            # Calculate enrichment score (fold enrichment)
            expected = (tf_total_targets * total_lf_genes) / total_genes_in_grn
            enrichment_score = tf_lf_targets / expected if expected > 0 else 0
        else:
            p_value = 1.0
            enrichment_score = 0

        results.append(
            {
                "TF": str(tf),
                "p_value": p_value,
                "enrichment_score": enrichment_score,
                "genes_in_lf": genes_in_lf,
                "genes_dwnstrm": genes_dwnstrm,
                "weights": weights,
            }
        )

    return pd.DataFrame(results)


#################################### LF + DyGRN ############################################

def get_unique_regs_by_target(max_force_df):
    """
    Create dictionary of unique TF-target pairs for each target
    """
    # Get unique targets
    targets = max_force_df.index.get_level_values(1).unique()
    # Initialize dictionary
    tf_target_pairs_per_gene = {}
    # Process each target
    for target in targets:
        # Get rows for this target
        target_mask = max_force_df.index.get_level_values(1) == target
        target_data = max_force_df[target_mask]
        # Get unique TFs for this target
        unique_tfs = target_data.index.get_level_values(0).unique()
        # Create list of tuples
        tf_target_pairs = [(str(tf), str(target)) for tf in unique_tfs]
        # Store in dictionary
        tf_target_pairs_per_gene[target] = tf_target_pairs
    return tf_target_pairs_per_gene


def filter_edges_by_significance_and_direction(
    df,
    min_nonzero_timepoints=3,
    alpha=0.05,
    min_observations=3,
    check_direction_invariance=True,
    n_processes=None,
    chunk_size=10000,
    save_intermediate=False,
    intermediate_path=None,
):
    """
    Filter edges for significance and direction invariance using chunked multiprocessing.

    Parameters:
        df (pd.DataFrame): DataFrame with TF-Target as index and time points as columns
        min_nonzero_timepoints (int): Minimum number of non-zero time points required
        alpha (float): Significance level for t-test
        min_observations (int): Minimum number of observations needed for t-test
        check_direction_invariance (bool): Whether to filter for direction invariance
        n_processes (int): Number of processes to use
        chunk_size (int): Number of rows to process per chunk
        save_intermediate (bool): Whether to save intermediate results
        intermediate_path (str): Path to save intermediate results

    Returns:
        pd.DataFrame: Filtered DataFrame with p-values added
    """

    if n_processes is None:
        n_processes = min(mp.cpu_count(), 16)  # Cap at 16 to avoid memory issues

    print(f"Processing {len(df):,} rows using {n_processes} processes...")
    print(f"Chunk size: {chunk_size:,} rows")
    print(
        f"Direction invariance check: {'Enabled' if check_direction_invariance else 'Disabled'}"
    )

    start_time = time.time()

    # Identify time columns (exclude p_value if it exists)
    time_cols = [col for col in df.columns if col != "p_value"]
    print(f"Time columns: {time_cols}")

    # Create chunks of indices directly (much more memory efficient)
    print("Creating index chunks...")
    total_rows = len(df)
    index_chunks = []

    for i in range(0, total_rows, chunk_size):
        end_idx = min(i + chunk_size, total_rows)
        chunk_indices = df.index[i:end_idx]
        index_chunks.append(chunk_indices)

    total_chunks = len(index_chunks)
    print(f"Created {total_chunks} chunks of indices")

    # Create partial function that takes DataFrame and indices
    process_func = partial(
        filter_chunk_of_edges,
        df=df,
        time_cols=time_cols,
        min_nonzero_timepoints=min_nonzero_timepoints,
        alpha=alpha,
        min_observations=min_observations,
        check_direction_invariance=check_direction_invariance,
    )

    # Process chunks with progress tracking
    all_results = []

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Submit all chunks
        future_to_chunk = {
            executor.submit(process_func, chunk_indices): i
            for i, chunk_indices in enumerate(index_chunks)
        }

        # Process results with progress bar
        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)

                    # Optional: save intermediate results
                    if save_intermediate and intermediate_path:
                        chunk_df = pd.DataFrame(
                            chunk_results, columns=["index", "keep", "p_value"]
                        )
                        chunk_df.to_parquet(
                            f"{intermediate_path}_chunk_{chunk_idx}.parquet"
                        )

                except Exception as exc:
                    print(f"Chunk {chunk_idx} generated an exception: {exc}")
                    # Add dummy results for failed chunk
                    chunk_size_actual = len(index_chunks[chunk_idx])
                    dummy_results = [
                        (index_chunks[chunk_idx][i], False, np.nan)
                        for i in range(chunk_size_actual)
                    ]
                    all_results.extend(dummy_results)

                pbar.update(1)

                # Periodic garbage collection
                if len(all_results) % (chunk_size * 10) == 0:
                    gc.collect()

    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    # Sort results to maintain original order
    print("Sorting results...")
    index_to_position = {idx: pos for pos, idx in enumerate(df.index)}
    all_results.sort(key=lambda x: index_to_position[x[0]])
    # Extract results
    indices, keep_rows, p_values = zip(*all_results)
    # Create result DataFrame efficiently
    print("Creating result DataFrame...")
    # Only keep the time columns
    result_df = df[time_cols].copy()
    # Add p-values
    result_df["p_value"] = p_values
    # Filter significant rows
    significant_df = result_df[list(keep_rows)].copy()
    # Clean up memory
    del all_results, indices, keep_rows, p_values
    gc.collect()

    return significant_df


def filter_chunk_of_edges(
    chunk_indices,
    df,
    time_cols,
    min_nonzero_timepoints=3,
    alpha=0.05,
    min_observations=3,
    check_direction_invariance=True,
):
    """
    Process a chunk of edges efficiently by working directly with DataFrame indices.

    Parameters:
        chunk_indices: Index slice to process
        df: Full DataFrame
        time_cols: List of time column names
        min_nonzero_timepoints: Minimum number of non-zero time points required
        alpha: Significance level for t-test
        min_observations: Minimum number of observations needed for t-test
        check_direction_invariance: Whether to filter for direction invariance

    Returns:
        List of tuples: (index, keep_flag, p_value)
    """
    results = []

    # Extract the chunk data efficiently using loc
    chunk_data = df.loc[chunk_indices, time_cols]

    for idx in chunk_indices:
        row = chunk_data.loc[idx]

        # Filter for minimum non-zero time points
        nonzero_mask = row != 0
        nonzero_count = nonzero_mask.sum()

        if nonzero_count < min_nonzero_timepoints:
            results.append((idx, False, np.nan))
            continue

        # Get non-zero values for statistical testing
        nonzero_values = row[nonzero_mask].values

        # Check if we have enough observations for t-test
        if len(nonzero_values) < min_observations:
            results.append((idx, False, np.nan))
            continue

        # Perform one-sample t-test against zero
        try:
            t_stat, p_value = stats.ttest_1samp(nonzero_values, 0)

            # Check significance
            is_significant = p_value < alpha

            if not is_significant:
                results.append((idx, False, p_value))
                continue

            # Check direction invariance if requested
            if check_direction_invariance:
                # All non-zero values should have the same sign
                positive_count = (nonzero_values > 0).sum()
                negative_count = (nonzero_values < 0).sum()

                # Direction is invariant if all values are positive OR all are negative
                direction_invariant = (positive_count == 0) or (negative_count == 0)

                if not direction_invariant:
                    results.append((idx, False, p_value))
                    continue

            # Edge passes all filters
            results.append((idx, True, p_value))

        except Exception as e:
            # Handle any statistical test errors
            results.append((idx, False, np.nan))

    return results


def calculate_force_curves_chunk(
    beta_chunk: pd.DataFrame, tf_expression: pd.DataFrame, epsilon: float = 1e-10
) -> pd.DataFrame:
    """
    Calculate force curves for a chunk of beta values using log transformation

    Parameters:
        beta_chunk: DataFrame chunk with multi-index (TF, Target) and time columns
        tf_expression: DataFrame with TF expression values (TF as index, time as columns)
        epsilon: Small value to avoid log(0)

    Returns:
        DataFrame with force curves for the chunk
    """
    # Get unique TFs in this chunk
    tfs_in_chunk = beta_chunk.index.get_level_values(0).unique()

    # Count number of targets per TF in this chunk
    targets_per_tf = beta_chunk.index.get_level_values(0).value_counts()

    # Get TF expression data for TFs in this chunk, in the same order as targets_per_tf
    tf_expr_subset = tf_expression.loc[targets_per_tf.index]

    # Create expanded TF expression DataFrame to match beta_chunk structure
    expanded_tf_expr = pd.DataFrame(
        np.repeat(tf_expr_subset.values, targets_per_tf.values, axis=0),
        index=beta_chunk.index,
        columns=beta_chunk.columns,
    )

    # Convert to numpy arrays for calculations
    beta_array = beta_chunk.to_numpy()
    tf_array = expanded_tf_expr.to_numpy()

    # Log transformations
    log_beta = np.log10(np.abs(beta_array) + epsilon)
    log_tf = np.log10(tf_array + epsilon)

    # Preserve signs from original beta values
    signs = np.sign(beta_array)

    # Calculate forces: force = sign(beta) * exp(log10(|beta|) + log10(tf_expr))
    force_array = signs * np.exp(log_beta + log_tf)

    # Convert back to DataFrame
    force_chunk = pd.DataFrame(
        force_array, index=beta_chunk.index, columns=beta_chunk.columns
    )

    return force_chunk


def create_balanced_chunks(df: pd.DataFrame, n_chunks: int):
    """
    Create balanced chunks by splitting DataFrame into roughly equal parts
    """
    chunk_size = len(df) // n_chunks
    remainder = len(df) % n_chunks

    chunks = []
    start_idx = 0

    for i in range(n_chunks):
        # Add one extra row to first 'remainder' chunks
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size

        chunk = df.iloc[start_idx:end_idx]
        if len(chunk) > 0:  # Only add non-empty chunks
            chunks.append(chunk)

        start_idx = end_idx

    return chunks


def calculate_force_curves_parallel(
    beta_curves: pd.DataFrame,
    tf_expression: pd.DataFrame,
    n_processes: int = None,
    chunk_size: int = 50000,
    epsilon: float = 1e-10,
    save_intermediate: bool = False,
    intermediate_path: str = None,
) -> pd.DataFrame:
    """
    Calculate force curves in parallel for large datasets

    Parameters:
        beta_curves: DataFrame with multi-index (TF, Target) and time columns
        tf_expression: DataFrame with TF expression (TF as index, time as columns)
        n_processes: Number of processes (default: CPU count)
        chunk_size: Number of rows per chunk
        epsilon: Small value to avoid log(0)
        save_intermediate: Whether to save intermediate results
        intermediate_path: Path for intermediate files

    Returns:
        DataFrame with force curves
    """

    if n_processes is None:
        n_processes = min(mp.cpu_count(), 16)  # Cap at 16 to avoid memory issues

    print(f"Processing {len(beta_curves):,} edges using {n_processes} processes...")
    print(f"Chunk size: {chunk_size:,} rows")

    start_time = time.time()

    # Remove p_value column if it exists (keep only time columns)
    time_cols = [col for col in beta_curves.columns if col.startswith("time_")]
    beta_time_only = beta_curves[time_cols].copy()

    # Ensure tf_expression has matching time columns
    tf_expr_subset = tf_expression[time_cols].copy()

    print(f"Time columns: {time_cols}")
    print(f"Beta curves shape: {beta_time_only.shape}")
    print(f"TF expression shape: {tf_expr_subset.shape}")

    # Create chunks
    n_chunks = max(1, len(beta_time_only) // chunk_size)
    chunks = create_balanced_chunks(beta_time_only, n_chunks)

    print(f"Created {len(chunks)} chunks")
    print(f"Chunk sizes: {[len(chunk) for chunk in chunks[:5]]}...")  # Show first 5

    # Create partial function for processing
    process_func = partial(
        calculate_force_curves_chunk, tf_expression=tf_expr_subset, epsilon=epsilon
    )

    # Process chunks in parallel
    force_chunks = []

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Submit all chunks
        future_to_chunk = {
            executor.submit(process_func, chunk): i for i, chunk in enumerate(chunks)
        }

        # Process results with progress bar
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    force_chunk = future.result()
                    force_chunks.append(force_chunk)

                    # Optional: save intermediate results
                    if save_intermediate and intermediate_path:
                        force_chunk.to_parquet(
                            f"{intermediate_path}_force_chunk_{chunk_idx}.parquet"
                        )

                except Exception as exc:
                    print(f"Chunk {chunk_idx} generated an exception: {exc}")
                    raise exc

                pbar.update(1)

                # Periodic garbage collection
                if len(force_chunks) % 10 == 0:
                    gc.collect()

    print(f"Processing completed in {time.time() - start_time:.2f} seconds")

    # Combine all chunks
    print("Combining results...")
    force_curves_result = pd.concat(force_chunks, axis=0)

    # Ensure the result maintains the original order
    force_curves_result = force_curves_result.loc[beta_time_only.index]

    print(f"Final shape: {force_curves_result.shape}")

    # Clean up memory
    del force_chunks, chunks
    gc.collect()

    return force_curves_result

def run_episode(
    episode_idx,
    dictys_dynamic_object_path,
    output_folder,
    trajectory_range,
    num_points,
    time_slice_start,
    time_slice_end,
    lf_genes,
    dist=0.001,
    sparsity=0.01,
    percentile=98
):
    # Load dictys object inside the process
    dictys_dynamic_object = dictys.net.dynamic_network.from_file(dictys_dynamic_object_path)
    epi = EpisodeDynamics(
        dictys_dynamic_object=dictys_dynamic_object,
        output_folder=output_folder,
        mode="expression",
        trajectory_range=trajectory_range,
        num_points=num_points,
        dist=dist,
        sparsity=sparsity
    )
    epi.compute_expression_curves()
    epi.set_lf_genes(lf_genes)
    epi.build_episode_grn(time_slice=slice(time_slice_start, time_slice_end))
    epi.filter_edges()
    epi.compute_tf_expression()
    epi.calculate_forces()
    epi.select_top_edges(percentile)
    epi.annotate_lf_in_grn()
    enrichment_df = epi.calculate_enrichment()
    # save to CSV
    out_path = os.path.join(output_folder, f'enrichment_episode_{episode_idx}.csv')
    enrichment_df.to_csv(out_path, index=False)
    return out_path
