"""Regulatory-force kernels: temporal-invariance filtering and parallel force curves."""
from __future__ import annotations

import gc
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from firefate.utils.parallel import create_balanced_chunks


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
    # Align TF expression to the beta rows BY NAME.
    #
    # The previous implementation took `value_counts()` (which orders TFs by
    # DESCENDING TARGET COUNT) and attached the repeated expression blocks to the
    # frame in ROW order. Those two orders agree only when the rows happen to be
    # grouped by descending target count; rows out of `build_episode_grn` are grouped
    # in dictys `nids[0]` (alphabetical) order and `filter_edges` makes the counts
    # unequal, so in practice most edges were scaled by another TF's expression.
    # Reindexing on the row-level TF labels makes the pairing structural instead of
    # positional.
    row_tfs = beta_chunk.index.get_level_values(0)
    missing = row_tfs.unique().difference(tf_expression.index)
    if len(missing) > 0:
        raise KeyError(
            f"TF expression missing for {len(missing)} regulator(s) in this chunk: "
            f"{sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}"
        )
    expanded_tf_expr = tf_expression.reindex(row_tfs)
    expanded_tf_expr.index = beta_chunk.index

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
