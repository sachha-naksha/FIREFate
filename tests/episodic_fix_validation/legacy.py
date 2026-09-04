"""Verbatim pre-fix implementations, kept so the fix can be A/B-tested on real data.

Two defects were fixed in ``src/firefate/temporal/`` (both `CONFIRMED` in
``tests/temporal/ISSUES.md``):

* **#1** ``calculate_force_curves_chunk`` paired TF expression ordered by
  ``value_counts()`` (descending target count) with rows grouped in network order, so
  most edges were scaled by *another* TF's expression.
* **#3** ``EpisodeDynamics.compute_tf_expression`` always took the **first** ``q``
  pseudotime columns, so every episode after the first used episode-1 regulator
  expression.

The functions below reproduce the old behaviour **exactly**, including the chunking,
so ``run_validation.py`` can compute both versions from identical inputs and attribute
any difference in the enrichment tables to the fix rather than to a rerun.

Nothing here is importable from ``firefate``; it is frozen dead code by design.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from firefate.utils.parallel import create_balanced_chunks


def legacy_calculate_force_curves_chunk(
    beta_chunk: pd.DataFrame, tf_expression: pd.DataFrame, epsilon: float = 1e-10
) -> pd.DataFrame:
    """Pre-fix ``firefate.temporal._forces.calculate_force_curves_chunk`` (ISSUES #1).

    The two lines that matter are ``value_counts()`` (count order) followed by
    ``np.repeat(..., index=beta_chunk.index)`` (row order).
    """
    targets_per_tf = beta_chunk.index.get_level_values(0).value_counts()
    tf_expr_subset = tf_expression.loc[targets_per_tf.index]

    expanded_tf_expr = pd.DataFrame(
        np.repeat(tf_expr_subset.values, targets_per_tf.values, axis=0),
        index=beta_chunk.index,
        columns=beta_chunk.columns,
    )

    beta_array = beta_chunk.to_numpy()
    tf_array = expanded_tf_expr.to_numpy()

    log_beta = np.log10(np.abs(beta_array) + epsilon)
    log_tf = np.log10(tf_array + epsilon)
    signs = np.sign(beta_array)
    force_array = signs * np.exp(log_beta + log_tf)

    return pd.DataFrame(force_array, index=beta_chunk.index, columns=beta_chunk.columns)


def legacy_calculate_force_curves(
    beta_curves: pd.DataFrame,
    tf_expression: pd.DataFrame,
    chunk_size: int = 30000,
    epsilon: float = 1e-10,
) -> pd.DataFrame:
    """Pre-fix force curves over the same balanced chunks as the production path.

    ``calculate_force_curves_parallel`` splits into ``max(1, len(beta) // chunk_size)``
    balanced chunks and calls the chunk function on each. The chunk boundaries change
    which TFs co-occur, and therefore how badly the count-vs-row ordering skews, so the
    chunking is reproduced rather than short-circuited. Runs in-process: the force
    computation is a vectorised NumPy expression and is not the bottleneck.
    """
    time_cols = [c for c in beta_curves.columns if c.startswith("time_")]
    beta_time_only = beta_curves[time_cols]
    tf_expr_subset = tf_expression[time_cols]

    n_chunks = max(1, len(beta_time_only) // chunk_size)
    chunks = create_balanced_chunks(beta_time_only, n_chunks)

    out = pd.concat(
        [legacy_calculate_force_curves_chunk(c, tf_expr_subset, epsilon) for c in chunks],
        axis=0,
    )
    return out.loc[beta_time_only.index]


def legacy_compute_tf_expression(
    lcpm_dcurve: pd.DataFrame, filtered_edges: pd.DataFrame
) -> pd.DataFrame:
    """Pre-fix ``EpisodeDynamics.compute_tf_expression`` (ISSUES #3).

    Takes the first ``n_time_cols`` columns of the branch's expression curve regardless
    of which episode is being built, then relabels them ``time_0..`` so the mismatch is
    invisible downstream.
    """
    tf_names = filtered_edges.index.get_level_values(0).unique()
    tf_lcpm_values = lcpm_dcurve.loc[tf_names]
    time_cols = [c for c in filtered_edges.columns if c.startswith("time_")]
    n_time_cols = len(time_cols)

    tf_lcpm_episode = tf_lcpm_values.iloc[:, 0:n_time_cols].copy()
    tf_lcpm_episode.columns = time_cols[:n_time_cols]
    return tf_lcpm_episode
