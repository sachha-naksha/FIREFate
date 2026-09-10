"""Time reductions of TF-force curves: one named function per context (ISSUES.md #23).

A force curve is one row per ``(TF, Target)`` link and one column per sampled
pseudotime point (see :func:`firefate.temporal._forces.calculate_force_curves`).
A sampled point is an equidistant position along the trajectory at which the
beta network is a Gaussian-smoothed blend of the surrounding pseudobulk windows;
it is not itself a dictys window.  Each analysis context collapses that row to a
single number in its own way, and the difference is deliberate:

* :func:`mean_force` -- **episodic construction.**  Plain arithmetic mean of the
  signed force over the episode's run of consecutive sampled points (five by
  default; zero points count in the denominator).  Only edges that survived the
  beta-level significance and direction-invariance filter reach this step, so
  the sign never flips across those points and the mean is a typical magnitude
  with a sign.
* :func:`softmax_peak` -- **phase assignment.**  No mean: a softmax over ``|force|``
  picks the ``top_k`` time points and their weighted pseudotime is the link's
  peak.  Arbitrary links are scored here with no sign filter, so a mean could
  cancel; the peak cannot.
* :func:`abs_max_force` -- **link validation.**  ``max_t |force(t)|``, the
  strongest regulation anywhere on the trajectory.

Keeping the three side by side is what stops one being mistaken for another.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mean_force(force_curves: pd.DataFrame) -> pd.Series:
    """Episodic reduction: mean of the signed force over the episode's sampled points.

    Every column is one sampled pseudotime point of the episode; zeros are
    included in the denominator.  Returns a Series indexed like ``force_curves``.
    """
    return force_curves.mean(axis=1)


def abs_max_force(force_curves: pd.DataFrame, links=None) -> dict:
    """Validation reduction: ``max_t |force(t)|`` per link as ``{(TF, Target): float}``.

    ``links`` restricts (and orders) the output; links absent from
    ``force_curves`` are skipped.
    """
    idx = force_curves.index if links is None else links
    return {l: float(force_curves.loc[l].abs().max())
            for l in idx if l in force_curves.index}


def softmax_peak(force_curves: pd.DataFrame, dtime, top_k: int = 5,
                 temperature: float = 1.0, method: str = 'weighted_mean') -> dict:
    """Phase reduction: softmax-weighted peak pseudotime (and force) per link.

    ``get_max_points`` scores every time point of a link by a softmax over
    ``|force|`` and keeps the ``top_k``; ``aggregate_max_points`` collapses those
    to one pseudotime per link by ``method`` (``'weighted_mean'``, ``'top1'``,
    ``'mean'`` or ``'median'``).  Returns
    ``{(TF, Target): {'pseudotime', 'window_idx', 'force', 'abs_force', 'n_points'}}``.
    Phase assignment reads only ``'pseudotime'``.
    """
    return aggregate_max_points(
        get_max_points(force_curves, dtime, top_k=top_k, temperature=temperature),
        method=method,
    )


def get_max_points(force_curves, dtime, top_k=5, temperature=1.0):
    """
    Find pseudotime points with the highest absolute force values using softmax weighting.

    Parameters:
    -----------
    force_curves : DataFrame
        Multi-indexed DataFrame with (TF, Target) as rows and time points as columns
    dtime : array-like
        Actual pseudotime values corresponding to each column/window
    top_k : int
        Number of top points to return per TF-Target pair
    temperature : float
        Temperature parameter for softmax (lower = more peaked distribution)

    Returns:
    --------
    list of dicts: Each containing TF, Target, pseudotime, window_idx, force value, and softmax_weight
    """
    max_points = []

    # Convert dtime to array if needed
    dtime = np.array(dtime)

    # Get unique TF-Target pairs
    for tf_target in force_curves.index:
        # Get the time series for this TF-Target pair
        time_series = force_curves.loc[tf_target]

        # Get absolute force values
        abs_forces = time_series.abs()

        # Apply softmax to absolute forces
        softmax_weights = np.exp(abs_forces / temperature) / np.sum(np.exp(abs_forces / temperature))

        # Get top_k indices with highest softmax weights
        top_indices = softmax_weights.nlargest(top_k).index

        for time_col in top_indices:
            # Get the window index (column position)
            window_idx = force_curves.columns.get_loc(time_col) if time_col in force_curves.columns else time_col

            max_points.append({
                'TF': tf_target[0],
                'Target': tf_target[1],
                'window_idx': window_idx,
                'pseudotime': dtime[window_idx],
                'force': time_series[time_col],
                'abs_force': abs_forces[time_col],
                'softmax_weight': softmax_weights[time_col]
            })

    return max_points


def aggregate_max_points(max_points, method='weighted_mean'):
    """
    Aggregate max points to get a single pseudotime per TF-Target regulation.

    Parameters:
    -----------
    max_points : list of dicts
        Output from get_max_points function
    method : str
        Aggregation method: 'weighted_mean', 'top1', 'mean', 'median'

    Returns:
    --------
    dict : {(TF, Target): {'pseudotime': float, 'force': float, ...}}
    """
    df = pd.DataFrame(max_points)

    aggregated = {}

    for (tf, target), group in df.groupby(['TF', 'Target']):
        if method == 'weighted_mean':
            # Normalize softmax weights within the group
            weights = group['softmax_weight'] / group['softmax_weight'].sum()
            agg_pseudotime = np.average(group['pseudotime'], weights=weights)
            agg_force = np.average(group['force'], weights=weights)
            agg_window_idx = int(np.round(np.average(group['window_idx'], weights=weights)))

        elif method == 'top1':
            # Take the one with highest softmax weight
            top_row = group.loc[group['softmax_weight'].idxmax()]
            agg_pseudotime = top_row['pseudotime']
            agg_force = top_row['force']
            agg_window_idx = top_row['window_idx']

        elif method == 'mean':
            agg_pseudotime = group['pseudotime'].mean()
            agg_force = group['force'].mean()
            agg_window_idx = int(np.round(group['window_idx'].mean()))

        elif method == 'median':
            agg_pseudotime = group['pseudotime'].median()
            agg_force = group['force'].median()
            agg_window_idx = int(np.round(group['window_idx'].median()))

        aggregated[(tf, target)] = {
            'pseudotime': float(agg_pseudotime),
            'window_idx': int(agg_window_idx),
            'force': float(agg_force),
            'abs_force': float(abs(agg_force)),
            'n_points': len(group)
        }

    return aggregated
