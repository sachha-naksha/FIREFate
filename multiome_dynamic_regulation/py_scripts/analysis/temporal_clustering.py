"""Temporal clustering of regulatory links into phases of regulation.

Nomenclature (kept consistent with the biology this module describes):

- **wave** -- a single TF-Target force linkage that rises and falls (a "force
  wave") along the lineage trajectory. One wave = one link's force curve.
- **state** -- the cell-state label a cell carries.
- **phase** -- a temporal cluster of regulatory links whose force-wave peaks
  fall in the same interval of pseudotime, the intervals being delimited by
  cell-state termination pseudotimes (3 phases for PB, 2 for GC).

Refactor of ``waves_quantification.ipynb`` into:

- :class:`StateFrequency` -- cell-state composition over pseudotime. Produces
  fate-frequency trajectories (stacked-bar and composition-curve plots), finds
  per-state extrema, and exposes the pseudotime at which a state "terminates"
  (``termination_pseudotime``). Those termination pseudotimes are the phase
  boundaries consumed by :class:`RegulatoryPhases`.

- :class:`TFForceWaves` -- TF force waves over pseudotime. Plots TF expression /
  regulation curves, computes regulatory force curves for a set of links,
  renders the 3D force landscape of a single link, and draws the clustered force
  heatmap. Its nested :class:`TFForceWaves.ForceSelector` picks per-link forces
  either lineage-specific (one branch) or combined across lineages.

- :class:`RegulatoryPhases` -- classification of input links into phases given
  the cell-state termination pseudotimes (softmax peak-pseudotime per force wave,
  binned against the boundaries).

- :class:`TFForceValidation` / :class:`PhaseValidation` -- compare the FireFate
  links against size-matched random links by abs-max TF force, pooled across the
  trajectory or split by phase.

The module-level softmax helpers (:func:`get_max_points`,
:func:`aggregate_max_points`, :func:`order_links_by_phase`, :func:`order_links`)
are the canonical implementation; :class:`RegulatoryPhases` reuses them.
"""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from ensure_firefate_path import ensure

ensure()
from firefate.core.episodic_dynamics import AlignTimeScales
from firefate.core.pseudotime_curves import SmoothedCurvesGRN
from firefate.utils.custom import (
    window_labels_to_count_df,
    plot_force_heatmap_with_clustering,
)
from firefate.utils.plots import plot_single_link_landscape


# ---------------------------------------------------------------------------
# Softmax peak-finding over force curves (canonical module-level helpers)
# ---------------------------------------------------------------------------

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


def order_links_by_phase(regulation_pseudotimes):
    """
    Order TF-Target links into phases by ascending peak pseudotime.

    Parameters:
    -----------
    regulation_pseudotimes : dict
        Output from aggregate_max_points: {(TF, Target): {'pseudotime': float, ...}}

    Returns:
    --------
    list of (TF, Target) tuples sorted by their peak pseudotime (earliest first)
    """
    return sorted(
        regulation_pseudotimes,
        key=lambda link: regulation_pseudotimes[link]['pseudotime']
    )


def order_links(force_curves, dtime, top_k=5, temperature=1.0, method='weighted_mean'):
    """
    Convenience wrapper: compute the phase ordering of links directly from force curves.

    Parameters:
    -----------
    force_curves : DataFrame
        Multi-indexed (TF, Target) force curves over pseudotime
    dtime : array-like
        Pseudotime values per column/window
    top_k, temperature : softmax parameters passed to get_max_points
    method : aggregation method passed to aggregate_max_points

    Returns:
    --------
    (ordered_links, regulation_pseudotimes)
        ordered_links : list of (TF, Target) tuples sorted by peak pseudotime
        regulation_pseudotimes : dict from aggregate_max_points
    """
    max_points = get_max_points(force_curves, dtime, top_k=top_k, temperature=temperature)
    regulation_pseudotimes = aggregate_max_points(max_points, method=method)
    ordered_links = order_links_by_phase(regulation_pseudotimes)
    return ordered_links, regulation_pseudotimes


# ---------------------------------------------------------------------------
# Class 1: cell-state frequencies and extrema / termination pseudotimes
# ---------------------------------------------------------------------------

class StateFrequency:
    """Cell-state composition over pseudotime for one or more fate branches.

    Builds the per-window cell-state count table, maps windows to pseudotime,
    and provides fate-frequency plots plus extrema / "termination" pseudotimes that
    define the phase boundaries used by :class:`RegulatoryPhases`.
    """

    def __init__(
        self,
        dictys_dynamic_object,
        cell_labels,
        colors=None,
        trajectory_range=(0, 2),
        cluster_column="Cluster",
        drop_states=("ActB-1", "earlyActB"),
    ):
        """
        Parameters
        ----------
        dictys_dynamic_object : dictys dynamic network
            Source of the soft cell-window assignment matrix and pseudotimes.
        cell_labels : pandas.DataFrame or str
            Per-cell cluster table (or a path to a CSV). Row order must match
            the columns of the cell-window assignment matrix.
        colors : dict, optional
            Mapping of state name -> colour, used by the plotting methods.
        trajectory_range : tuple
            Trajectory passed to :class:`AlignTimeScales` to resolve the
            pseudotime of each window.
        cluster_column : str
            Column in ``cell_labels`` holding the state name.
        drop_states : iterable
            States removed from the count table (matching the notebook, which
            drops the progenitor states that span all branches).
        """
        if isinstance(cell_labels, str):
            cell_labels = pd.read_csv(cell_labels, header=0)

        self.dictys_dynamic_object = dictys_dynamic_object
        self.cell_labels = cell_labels
        self.colors = colors or {}
        self.trajectory_range = trajectory_range
        self.cluster_column = cluster_column
        self.drop_states = list(drop_states)

        self._pseudotime_values_of_windows = None
        self._state_count_per_window = None

    # ------------------------------------------------------------------
    # Core tables
    # ------------------------------------------------------------------

    @property
    def pseudotime_values_of_windows(self):
        """Pseudotime value of each window centroid (cached)."""
        if self._pseudotime_values_of_windows is None:
            self._pseudotime_values_of_windows = AlignTimeScales(
                self.dictys_dynamic_object,
                trajectory_range=list(self.trajectory_range),
            ).pseudotime_of_windows()
        return self._pseudotime_values_of_windows

    @property
    def state_count_per_window(self):
        """States x windows count table (cached), with ``drop_states`` removed."""
        if self._state_count_per_window is None:
            cell_assignment_matrix = self.dictys_dynamic_object.prop["sc"]["w"]
            state_labels_in_window = {}
            for window_idx in range(cell_assignment_matrix.shape[0]):
                present = np.where(cell_assignment_matrix[window_idx] == 1)[0]
                state_labels_in_window[window_idx] = [
                    self.cell_labels.iloc[int(idx)][self.cluster_column]
                    for idx in present
                ]
            counts = window_labels_to_count_df(state_labels_in_window)
            drop = [s for s in self.drop_states if s in counts.index]
            if drop:
                counts = counts.drop(index=drop)
            self._state_count_per_window = counts
        return self._state_count_per_window

    def fate_trajectory(self, window_indices):
        """Return ``(df_plot, x)`` for one fate branch.

        ``df_plot`` is the count table restricted and re-ordered to
        ``window_indices``; ``x`` is the matching pseudotime per window.
        """
        df_plot = self.state_count_per_window[window_indices]
        x = np.array([self.pseudotime_values_of_windows[i] for i in window_indices])
        return df_plot, x

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_stacked_bars(self, window_indices, n_bins=8, figsize=(6, 4)):
        """Stacked bar plot of average cell-state composition over binned windows."""
        df_plot, x = self.fate_trajectory(window_indices)
        x_min, x_max = x.min(), x.max()

        bin_edges = np.linspace(x_min, x_max, n_bins + 1)

        binned_data = {state: [0.0] * n_bins for state in df_plot.index}
        bin_counts = [0] * n_bins

        for i, time_point in enumerate(x):
            bin_idx = int(np.digitize(time_point, bin_edges) - 1)
            bin_idx = max(0, min(bin_idx, n_bins - 1))
            for state in df_plot.index:
                binned_data[state][bin_idx] += df_plot.loc[state].values[i]
            bin_counts[bin_idx] += 1

        for state in df_plot.index:
            for bin_idx in range(n_bins):
                if bin_counts[bin_idx] > 0:
                    binned_data[state][bin_idx] /= bin_counts[bin_idx]

        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        bottom = [0.0] * n_bins
        for state in df_plot.index:
            y = binned_data[state]
            ax.bar(
                range(n_bins), y,
                label=state,
                color=self.colors.get(state),
                bottom=bottom,
                alpha=0.8,
            )
            bottom = [bottom[i] + y[i] for i in range(n_bins)]

        ax.set_xlabel('Binned windows', fontsize=14, fontweight='bold', labelpad=15)
        ax.set_ylabel('Average Cell Count', fontsize=14, fontweight='bold')
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels([f"Bin {i + 1}" for i in range(n_bins)], fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        return fig, ax

    def plot_composition_curves(self, window_indices, figsize=(15, 8), xlabel='branch'):
        """Line plot (shaded) of each cell-state count over pseudotime."""
        df_plot, x = self.fate_trajectory(window_indices)

        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        for state in df_plot.index:
            y = df_plot.loc[state]
            ax.plot(x, y, label=state, color=self.colors.get(state), linewidth=2)
            ax.fill_between(x, y, color=self.colors.get(state), alpha=0.25)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cell Count')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Extrema and termination pseudotimes
    # ------------------------------------------------------------------

    def find_extrema(self, window_indices, prominence=10, distance=3):
        """Locate per-state maxima and minima of the count trajectory.

        Returns
        -------
        (extrema_info, extrema_pseudotimes)
            ``extrema_info`` keeps window indices, pseudotimes and counts of the
            maxima / minima per state; ``extrema_pseudotimes`` keeps only the
            pseudotime tuples (``maxima``, ``minima``, ``all``).
        """
        df_plot, x = self.fate_trajectory(window_indices)

        extrema_info = {}
        extrema_pseudotimes = {}

        for state in df_plot.index:
            y = df_plot.loc[state].values
            maxima_idx, _ = find_peaks(y, prominence=prominence, distance=distance)
            minima_idx, _ = find_peaks(-y, prominence=prominence, distance=distance)

            extrema_info[state] = {
                'maxima_window_idx': maxima_idx,
                'maxima_pseudotime': x[maxima_idx],
                'maxima_count': y[maxima_idx],
                'minima_window_idx': minima_idx,
                'minima_pseudotime': x[minima_idx],
                'minima_count': y[minima_idx],
            }
            extrema_pseudotimes[state] = {
                'maxima': tuple(x[maxima_idx]),
                'minima': tuple(x[minima_idx]),
                'all': tuple(sorted(np.concatenate([x[maxima_idx], x[minima_idx]]))),
            }

        return extrema_info, extrema_pseudotimes

    def plot_extrema(self, window_indices, prominence=10, distance=3, figsize=(15, 8)):
        """Composition curves annotated with maxima (^) and minima (v) markers."""
        df_plot, x = self.fate_trajectory(window_indices)
        extrema_info, _ = self.find_extrema(window_indices, prominence, distance)

        fig, ax = plt.subplots(figsize=figsize)
        for state in df_plot.index:
            y = df_plot.loc[state].values
            color = self.colors.get(state)
            ax.plot(x, y, label=state, color=color, linewidth=2)
            ax.fill_between(x, y, color=color, alpha=0.2)
            if len(extrema_info[state]['maxima_window_idx']) > 0:
                ax.scatter(extrema_info[state]['maxima_pseudotime'],
                           extrema_info[state]['maxima_count'],
                           color=color, s=150, marker='^',
                           edgecolor='black', linewidth=2, zorder=5)
            if len(extrema_info[state]['minima_window_idx']) > 0:
                ax.scatter(extrema_info[state]['minima_pseudotime'],
                           extrema_info[state]['minima_count'],
                           color=color, s=150, marker='v',
                           edgecolor='black', linewidth=2, zorder=5)

        ax.set_xlabel('Branch (pseudotime)')
        ax.set_ylabel('Cell Count')
        ax.set_title('Cell Counts with Extrema (^=maxima, v=minima)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        return fig, ax

    def termination_pseudotime(
        self,
        state,
        window_indices,
        method='threshold',
        threshold_frac=0.1,
        prominence=10,
        distance=3,
    ):
        """Pseudotime at which ``state`` terminates along ``window_indices``.

        After the state's peak (max count), this is the pseudotime where the
        population collapses. Two definitions are supported:

        - ``method='threshold'`` (default): first window after the peak whose
          count falls to ``threshold_frac`` of the peak count.
        - ``method='extrema'``: pseudotime of the first ``find_peaks`` minimum
          after the peak; falls back to the threshold rule if none is found.

        Returns the last pseudotime of the branch if the state never collapses
        within range (so it can still be used as a boundary).
        """
        df_plot, x = self.fate_trajectory(window_indices)
        if state not in df_plot.index:
            raise ValueError(f"State {state!r} not found in count table.")
        y = df_plot.loc[state].values
        peak_idx = int(np.argmax(y))

        if method == 'extrema':
            minima_idx, _ = find_peaks(-y, prominence=prominence, distance=distance)
            after = minima_idx[minima_idx > peak_idx]
            if len(after) > 0:
                return float(x[after[0]])
            # fall through to threshold rule

        peak_count = y[peak_idx]
        cutoff = threshold_frac * peak_count
        for i in range(peak_idx + 1, len(y)):
            if y[i] <= cutoff:
                return float(x[i])
        return float(x[-1])


# ---------------------------------------------------------------------------
# Class 2: TF force waves over pseudotime (inspection + force selection)
# ---------------------------------------------------------------------------

class TFForceWaves:
    """TF regulatory force waves over pseudotime.

    Wraps :class:`SmoothedCurvesGRN` for one trajectory branch and exposes:
    expression / regulation trajectory plots, force-wave computation for a set of
    links, the 3D single-link force landscape, and the clustered force heatmap.

    Phase classification of links lives in :class:`RegulatoryPhases`; the nested
    :class:`ForceSelector` picks per-link forces either lineage-specific or
    combined across lineages.
    """

    def __init__(
        self,
        dictys_dynamic_object,
        trajectory_range=(0, 2),
        num_points=100,
        dist=0.0005,
        sparsity=0.01,
    ):
        self.dictys_dynamic_object = dictys_dynamic_object
        self.trajectory_range = trajectory_range
        self.curves = SmoothedCurvesGRN(
            dictys_dynamic_object,
            trajectory_range=trajectory_range,
            num_points=num_points,
            dist=dist,
            sparsity=sparsity,
        )

        # Cached smoothed curves
        self._exp_curves = None      # (dy, dx)
        self._reg_curves = None      # (dy, dx)

        # Populated by compute_forces()
        self.beta_curves = None
        self.regulon_tf_expression = None
        self.force_curves = None
        self.dtime = None

    # ------------------------------------------------------------------
    # Expression / regulation trajectories
    # ------------------------------------------------------------------

    def expression_curves(self):
        """``(dy, dx)`` smoothed log-CPM expression curves (cached)."""
        if self._exp_curves is None:
            self._exp_curves = self.curves.get_smoothed_curves(mode="expression")
        return self._exp_curves

    def regulation_curves(self):
        """``(dy, dx)`` smoothed regulation (log target-count) curves (cached)."""
        if self._reg_curves is None:
            self._reg_curves = self.curves.get_smoothed_curves(mode="regulation")
        return self._reg_curves

    def _plot_gene_trajectories(self, dy, dx, genes, colors, ylabel, figsize):
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for gene, color in zip(genes, colors):
            if gene in dy.index:
                ax.plot(dx, dy.loc[gene], linewidth=2, color=color)
                ax.text(dx.iloc[-1], dy.loc[gene].iloc[-1], f' {gene}',
                        color=color, verticalalignment='center')
        ax.set_xlabel('Pseudotime')
        ax.set_ylabel(ylabel)
        return fig, ax

    def plot_expression(self, genes, colors, ylabel='Log (CPM)', figsize=(8, 6)):
        """Expression trajectories for ``genes`` (coloured, labelled at the end)."""
        dy, dx = self.expression_curves()
        return self._plot_gene_trajectories(dy, dx, genes, colors, ylabel, figsize)

    def plot_regulation(self, genes, colors, ylabel='Log (target counts)', figsize=(8, 6)):
        """Regulation trajectories for ``genes`` (coloured, labelled at the end)."""
        dy, dx = self.regulation_curves()
        return self._plot_gene_trajectories(dy, dx, genes, colors, ylabel, figsize)

    # ------------------------------------------------------------------
    # Force waves
    # ------------------------------------------------------------------

    def compute_forces(self, links, varname='w_in'):
        """Compute and cache beta / TF-expression / force curves for ``links``.

        Returns the force-curve DataFrame and stores ``beta_curves``,
        ``regulon_tf_expression``, ``force_curves`` and ``dtime`` on ``self``.
        """
        # beta-network curves and TF-expression curves are two independent dictys
        # `.compute` passes over the trajectory; overlap them (both release the GIL
        # in their NumPy/BLAS work).
        with ThreadPoolExecutor(max_workers=2) as ex:
            beta_future = ex.submit(self.curves.get_beta_curves, links, varname=varname)
            tf_future = ex.submit(self.curves.get_smoothed_curves, mode='tf_expression')
            beta_curves, dtime = beta_future.result()
            tf_expression, _ = tf_future.result()
        regulon_tf_expression = tf_expression.loc[
            beta_curves.index.get_level_values(0).unique()
        ]
        force_curves = SmoothedCurvesGRN.calculate_force_curves(
            beta_curves, regulon_tf_expression
        )

        self.beta_curves = beta_curves
        self.regulon_tf_expression = regulon_tf_expression
        self.force_curves = force_curves
        self.dtime = dtime
        return force_curves

    def _require_forces(self):
        if self.force_curves is None:
            raise RuntimeError("Call compute_forces(links) first.")

    def plot_landscape(self, tf_name, target_name, **kwargs):
        """3D force landscape of a single ``tf_name -> target_name`` link."""
        self._require_forces()
        return plot_single_link_landscape(
            tf_name, target_name,
            self.beta_curves, self.regulon_tf_expression,
            self.force_curves, self.dtime,
            **kwargs,
        )

    def plot_force_heatmap(self, links=None, perform_clustering=False, **kwargs):
        """Clustered (or plain) force heatmap for ``links`` (default: all)."""
        self._require_forces()
        if links is None:
            links = list(self.force_curves.index)
        df_plot = self.force_curves.loc[links]
        return plot_force_heatmap_with_clustering(
            force_df=df_plot,
            dtime=self.dtime,
            regulations=links,
            perform_clustering=perform_clustering,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Nested: pick lineage-specific or combined forces
    # ------------------------------------------------------------------

    class ForceSelector:
        """Pick per-link TF forces, lineage-specific or combined across lineages.

        A single TF-Target *force wave* exists on each lineage branch. Given one
        or more fitted :class:`TFForceWaves` branches, this returns, per link,
        either the force on one lineage (``mode='lineage'``) or the combined
        force across lineages (``mode='combined'`` -- the stronger lineage, i.e.
        ``max`` of ``max_t |force|`` taken over branches). Links already present
        in a branch's cached ``force_curves`` reuse them; others are scored on
        demand via the same GRN.
        """

        def __init__(self, waves_by_branch, varname='w_in'):
            """
            Parameters
            ----------
            waves_by_branch : TFForceWaves or dict {name: TFForceWaves}
                A single fitted branch, or a mapping of lineage name -> branch
                (e.g. ``{'PB': waves_pb, 'GC': waves_gc}``). The branches are
                assumed to share the same dictys universe.
            varname : str
                Network variable used for on-demand force curves. Default ``'w_in'``.
            """
            if isinstance(waves_by_branch, TFForceWaves):
                waves_by_branch = {'lineage': waves_by_branch}
            if not waves_by_branch:
                raise ValueError("ForceSelector needs at least one branch.")
            self.waves_by_branch = dict(waves_by_branch)
            self.varname = varname

        @property
        def branches(self):
            return list(self.waves_by_branch)

        def any_waves(self):
            """The first branch (its dictys universe is shared across branches)."""
            return next(iter(self.waves_by_branch.values()))

        def _branch_force_curves(self, waves, links):
            """``(force_curves, dtime)`` for ``links`` on one branch.

            Reuses ``waves.force_curves`` when it already holds all ``links``;
            otherwise scores them on demand via the GRN (without mutating
            ``waves``).
            """
            links = [tuple(l) for l in links]
            cached = waves.force_curves
            if cached is not None and all(l in cached.index for l in links):
                return cached.loc[links], waves.dtime
            beta_curves, dtime = waves.curves.get_beta_curves(links, varname=self.varname)
            tf_expression, _ = waves.curves.get_smoothed_curves(mode='tf_expression')
            regulon_tf_expression = tf_expression.loc[
                beta_curves.index.get_level_values(0).unique()
            ]
            force_curves = SmoothedCurvesGRN.calculate_force_curves(
                beta_curves, regulon_tf_expression
            )
            return force_curves, dtime

        @staticmethod
        def abs_max(force_curves, links=None):
            """``max_t |force(t)|`` per link as a dict {(TF, Target): float}."""
            idx = force_curves.index if links is None else links
            return {l: float(force_curves.loc[l].abs().max())
                    for l in idx if l in force_curves.index}

        def force_curves(self, links, branch=None):
            """``(force_curves, dtime)`` for ``links`` on one lineage (default: first)."""
            name = branch or self.branches[0]
            return self._branch_force_curves(self.waves_by_branch[name], links)

        def branch_abs_max_force(self, links):
            """Per-branch abs-max force: ``{branch: {link: force}}``."""
            return {name: self.abs_max(self._branch_force_curves(w, links)[0], links)
                    for name, w in self.waves_by_branch.items()}

        def abs_max_force(self, links, mode='lineage', branch=None):
            """Per-link abs-max force.

            ``mode='lineage'`` -> force on ``branch`` (default: the first branch).
            ``mode='combined'`` -> ``max`` across branches per link (the lineage
            where the link is strongest).
            """
            if mode == 'lineage':
                fc, _ = self.force_curves(links, branch=branch)
                return self.abs_max(fc, [tuple(l) for l in links])
            if mode == 'combined':
                return {l: d['abs_max_force']
                        for l, d in self.combined_abs_max_force(links).items()}
            raise ValueError("mode must be 'lineage' or 'combined'.")

        def combined_abs_max_force(self, links):
            """Per-link combined force together with the winning lineage.

            Returns ``{link: {'abs_max_force': float, 'branch': name}}`` where
            ``branch`` is the lineage on which the link's ``max_t |force|`` is
            largest (ties go to the first branch).
            """
            out = {}
            for name, am in self.branch_abs_max_force(links).items():
                for l, v in am.items():
                    if l not in out or v > out[l]['abs_max_force']:
                        out[l] = {'abs_max_force': v, 'branch': name}
            return out


# ---------------------------------------------------------------------------
# Class 3: classification of links into regulatory phases
# ---------------------------------------------------------------------------

class RegulatoryPhases:
    """Classify regulatory links into temporal *phases*.

    A phase is a cluster of links whose force-wave peaks fall in the same
    interval of pseudotime, the intervals being delimited by cell-state
    termination pseudotimes (from :class:`StateFrequency`). Each link's peak
    pseudotime is the softmax-weighted peak of its force wave; binning those
    peaks against the termination pseudotimes assigns a phase (3 phases for PB,
    2 for GC).
    """

    def __init__(self, waves, top_k=5, temperature=1.0, method='weighted_mean'):
        """
        Parameters
        ----------
        waves : TFForceWaves
            Fitted branch whose ``compute_forces(links)`` has been called so
            ``force_curves`` / ``dtime`` hold the force waves to classify.
        top_k, temperature, method :
            Softmax peak-pseudotime parameters (passed to the module helpers).
        """
        self.waves = waves
        self.top_k = top_k
        self.temperature = temperature
        self.method = method

    @staticmethod
    def assign_phases(force_curves, dtime, switch_pseudotimes,
                      top_k=5, temperature=1.0, method='weighted_mean'):
        """Bin every link in ``force_curves`` to a phase by its softmax peak.

        ``switch_pseudotimes`` are the cell-state termination pseudotimes that
        separate consecutive phases; ``N`` switches -> ``N+1`` phases (1-indexed).
        Returns ``{(TF, Target): phase}``.
        """
        boundaries = np.sort(np.asarray(switch_pseudotimes, dtype=float))
        reg_pt = aggregate_max_points(
            get_max_points(force_curves, dtime, top_k=top_k, temperature=temperature),
            method=method,
        )
        return {
            link: int(np.digitize(info['pseudotime'], boundaries, right=True)) + 1
            for link, info in reg_pt.items()
        }

    def link_peak_pseudotimes(self, links=None):
        """Softmax peak pseudotime per link.

        Returns the ``aggregate_max_points`` dict
        ``{(TF, Target): {'pseudotime': float, ...}}`` for ``links``
        (default: every link in ``waves.force_curves``).
        """
        if self.waves.force_curves is None:
            raise RuntimeError("Call waves.compute_forces(links) first.")
        df = (self.waves.force_curves if links is None
              else self.waves.force_curves.loc[links])
        max_points = get_max_points(df, self.waves.dtime,
                                    top_k=self.top_k, temperature=self.temperature)
        return aggregate_max_points(max_points, method=self.method)

    def classify_phases(self, switch_pseudotimes, links=None):
        """Assign each link to a phase by its softmax peak pseudotime.

        ``switch_pseudotimes`` is the ordered list of cell-state termination
        pseudotimes that separate consecutive phases. ``N`` switches produce
        ``N + 1`` phases (1-indexed): a link with peak pseudotime ``p`` lands in
        phase ``k`` where ``switch[k-2] < p <= switch[k-1]``. So PB uses
        ``[ActB-4_termination, earlyPB_termination]`` (3 phases) and GC uses
        ``[ActB-3_termination]`` (2 phases).

        Returns
        -------
        pandas.DataFrame with columns ``TF, Target, peak_pseudotime, phase``,
        sorted by phase then peak pseudotime.
        """
        boundaries = np.sort(np.asarray(switch_pseudotimes, dtype=float))
        reg_pt = self.link_peak_pseudotimes(links=links)

        rows = []
        for (tf, target), info in reg_pt.items():
            peak = info['pseudotime']
            phase = int(np.digitize(peak, boundaries, right=True)) + 1
            rows.append({'TF': tf, 'Target': target,
                         'peak_pseudotime': peak, 'phase': phase})

        return (
            pd.DataFrame(rows)
            .sort_values(['phase', 'peak_pseudotime'])
            .reset_index(drop=True)
        )

    def classify_phases_from_states(self, state_frequency, window_indices,
                                    boundary_states, links=None,
                                    termination_method='threshold', threshold_frac=0.1,
                                    prominence=10, distance=3):
        """Phase classification driven directly by cell-state termination pseudotimes.

        Computes the termination pseudotime of each state in ``boundary_states``
        (in order) from a :class:`StateFrequency` instance, then bins links via
        :meth:`classify_phases`.

        Example
        -------
        PB (3 phases)::

            phases.classify_phases_from_states(
                sf, PB_post_bifurcation_window_indices,
                boundary_states=['ActB-4', 'earlyPB'])

        GC (2 phases)::

            phases.classify_phases_from_states(
                sf, GC_post_bifurcation_window_indices,
                boundary_states=['ActB-3'])
        """
        switch_pseudotimes = [
            state_frequency.termination_pseudotime(
                state, window_indices,
                method=termination_method, threshold_frac=threshold_frac,
                prominence=prominence, distance=distance,
            )
            for state in boundary_states
        ]
        return self.classify_phases(switch_pseudotimes, links=links)


# ---------------------------------------------------------------------------
# Class 4: validation of enriched links vs random links
# ---------------------------------------------------------------------------

class TFForceValidation:
    """Validate enriched (FireFate) links against size-matched random links by
    abs-max TF force, pooled across the whole trajectory (no phase structure).

    The comparison metric is the absolute maximum TF force along the trajectory
    (``max_t |force(t)|``), one value per link. Forces are picked through a
    :class:`TFForceWaves.ForceSelector`, so the comparison can be run on a single
    lineage (``mode='lineage'``) or combined across lineages (``mode='combined'``):

    * ``'lineage'`` -- enriched and random links are both scored on one branch.
    * ``'combined'`` -- each enriched link is scored on its strongest lineage
      (``max`` across branches), while the random null is the *union* of the
      per-branch random forces (every random link contributes its single-branch
      force on each branch, not a cross-branch max). In this mode the result
      carries a ``branch`` column naming the winning lineage per enriched link.

    The nested :class:`PhaseSplitted` (via :meth:`split_by_phase`) reuses this
    class to split the comparison by branch-qualified phase.
    """

    def __init__(self, waves, enriched_links, varname='w_in', mode='lineage'):
        """
        Parameters
        ----------
        waves : TFForceWaves, dict {name: TFForceWaves}, or TFForceWaves.ForceSelector
            The fitted branch(es) supplying forces. A single branch (or dict with
            one entry) supports ``mode='lineage'``; pass multiple branches for
            ``mode='combined'``. ``compute_forces(enriched_links)`` should have
            been called on each branch so the enriched forces are cached (random
            links are scored on demand).
        enriched_links : list of (TF, Target)
            The enriched / prioritized links (e.g. ``PB_links_plotting``).
        varname : str
            Network variable used for random-link forces (matches the enriched
            ``compute_forces`` call). Default ``'w_in'``.
        mode : {'lineage', 'combined'}
            See the class docstring.
        """
        self.selector = (waves if isinstance(waves, TFForceWaves.ForceSelector)
                         else TFForceWaves.ForceSelector(waves, varname=varname))
        self.enriched_links = [tuple(l) for l in enriched_links]
        self.varname = varname
        self.mode = mode
        self.result_ = None      # tidy DataFrame after run()

    # ------------------------------------------------------------------
    # random-pool helpers (shared by all validators)
    # ------------------------------------------------------------------

    def _build_random_pool(self, n_tf, n_target, exclude, rng):
        """Sample a pool of non-enriched random links (``n_tf`` x ``n_target``).

        ``n_tf`` / ``n_target`` default (when ``None``) to the number of unique
        enriched TFs / targets, so the pool mirrors the shape of the enriched set.
        ``exclude`` selects the null (see :meth:`run`). Returns the list of
        ``(TF, Target)`` pool links.
        """
        d = self.selector.any_waves().dictys_dynamic_object
        nname = np.asarray(d.nname)
        tf_universe = nname[np.asarray(d.nids[0])]
        target_universe = nname[np.asarray(d.nids[1])]
        enriched_set = set(self.enriched_links)
        enr_tfs = {tf for tf, _ in self.enriched_links}
        enr_targets = {tg for _, tg in self.enriched_links}

        # default pool size = shape of the enriched set it is compared against
        if n_tf is None:
            n_tf = len(enr_tfs)
        if n_target is None:
            n_target = len(enr_targets)

        if exclude == 'tf_and_target':
            cand_tfs = np.array([t for t in tf_universe if t not in enr_tfs])
            cand_targets = np.array([g for g in target_universe if g not in enr_targets])
        elif exclude == 'links':
            cand_tfs = np.asarray(tf_universe)
            cand_targets = np.asarray(target_universe)
        else:
            raise ValueError("exclude must be 'tf_and_target' or 'links'.")

        pick_tfs = rng.choice(cand_tfs, size=min(n_tf, len(cand_tfs)), replace=False)
        pick_targets = rng.choice(cand_targets, size=min(n_target, len(cand_targets)),
                                  replace=False)
        # drop exact enriched pairs (a no-op under 'tf_and_target', essential under 'links')
        return [(tf, tg) for tf in pick_tfs for tg in pick_targets
                if (tf, tg) not in enriched_set]

    def _enriched_forces(self):
        """``(present_links, {link: abs_max_force}, {link: branch})`` for the
        enriched links.

        Forces are picked via the selector under ``self.mode`` (cached enriched
        forces reused per branch). ``branch`` is the winning lineage under
        ``mode='combined'`` and the single scored lineage under ``'lineage'``.
        """
        if self.mode == 'combined':
            combined = self.selector.combined_abs_max_force(self.enriched_links)
            enr_force = {l: d['abs_max_force'] for l, d in combined.items()}
            enr_branch = {l: d['branch'] for l, d in combined.items()}
        else:
            branch = self.selector.branches[0]
            enr_force = self.selector.abs_max_force(
                self.enriched_links, mode='lineage', branch=branch)
            enr_branch = {l: branch for l in enr_force}

        present = [l for l in self.enriched_links if l in enr_force]
        missing = [l for l in self.enriched_links if l not in enr_force]
        if missing:
            print(f"{type(self).__name__}: {len(missing)} enriched links absent from "
                  f"force curves, skipped: {missing}")
        return present, enr_force, enr_branch

    def _random_null_rows(self, pool_links):
        """Random-null rows for ``pool_links``.

        Under ``mode='combined'`` every random link contributes its single-branch
        force on each branch (the per-branch union), each tagged with the
        ``branch`` it was scored on; under ``'lineage'`` it contributes one force
        on the chosen branch (no ``branch`` column).
        """
        if self.mode == 'combined':
            return [{'link': l, 'group': 'random', 'branch': name, 'abs_max_force': f}
                    for name, am in self.selector.branch_abs_max_force(pool_links).items()
                    for l, f in am.items()]
        return [{'link': l, 'group': 'random', 'abs_max_force': f}
                for l, f in self.selector.abs_max_force(pool_links, mode='lineage').items()]

    # ------------------------------------------------------------------
    # public API (pooled, no phases)
    # ------------------------------------------------------------------

    def run(self, n_tf=None, n_target=None, exclude='tf_and_target', random_state=0):
        """Pooled enriched-vs-random table across the whole trajectory.

        A pool of ``n_tf`` x ``n_target`` non-enriched random links is scored; a
        size-matched subset (one random draw per enriched link) is kept as the
        null.

        Parameters
        ----------
        n_tf, n_target : int, optional
            Size of the random candidate pool. ``None`` (default) -> number of
            unique enriched TFs / targets, mirroring the enriched set's shape.
        exclude : {'tf_and_target', 'links'}
            How the random pool is kept "non-enriched":

            * ``'tf_and_target'`` (default): drop *every* enriched TF and *every*
              enriched target from the sampling universe.
            * ``'links'``: keep the full universe and only remove the exact
              enriched ``(TF, Target)`` pairs.
        random_state : int
            Seed for reproducible sampling.

        Returns
        -------
        pandas.DataFrame with columns ``link, group, abs_max_force`` (``group`` is
        ``'enriched'`` or ``'random'``). Under ``mode='combined'`` an extra
        ``branch`` column records the winning lineage for enriched rows and the
        lineage each random link was scored on. Also stored on ``self.result_``.
        """
        rng = np.random.default_rng(random_state)
        present, enr_force, enr_branch = self._enriched_forces()

        pool_links = self._build_random_pool(n_tf, n_target, exclude, rng)
        null_rows = self._random_null_rows(pool_links)

        n = len(present)
        if len(null_rows) < n:
            print(f"{type(self).__name__}: random null has only {len(null_rows)} "
                  f"links (< {n} enriched); using all. Increase n_tf/n_target.")
        idx = (rng.choice(len(null_rows), size=min(n, len(null_rows)), replace=False)
               if null_rows else [])

        if self.mode == 'combined':
            rows = [{'link': l, 'group': 'enriched', 'branch': enr_branch[l],
                     'abs_max_force': enr_force[l]} for l in present]
        else:
            rows = [{'link': l, 'group': 'enriched', 'abs_max_force': enr_force[l]}
                    for l in present]
        rows += [null_rows[i] for i in idx]

        self.result_ = (
            pd.DataFrame(rows).sort_values('group').reset_index(drop=True))
        return self.result_

    def plot(self, figsize=(5, 6), ylabel='Abs max TF force',
             colors=('#d1495b', '#9aa0a6')):
        """Pooled box plot of abs-max TF force, enriched vs random."""
        if self.result_ is None:
            self.run()
        return self._plot_pooled(self.result_, figsize, ylabel, colors)

    @staticmethod
    def _plot_pooled(df, figsize, ylabel, colors):
        """Two-box (enriched vs random) plot of ``abs_max_force`` from a tidy table."""
        groups = [('enriched', 'Enriched (FireFate)'),
                  ('random', 'Random (non-enriched)')]

        fig, ax = plt.subplots(figsize=figsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        data = [df[df['group'] == g]['abs_max_force'].values for g, _ in groups]
        bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True,
                        showfliers=False, medianprops=dict(color='black'))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        for pos, (vals, c) in enumerate(zip(data, colors)):
            if len(vals) == 0:
                continue
            x = pos + (np.random.rand(len(vals)) - 0.5) * 0.25
            ax.scatter(x, vals, color=c, edgecolor='black', linewidth=0.4, s=22, zorder=3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([label for _, label in groups])
        ax.set_ylabel(ylabel)
        return fig, ax

    # ------------------------------------------------------------------
    # per-phase view: factory + nested validator
    # ------------------------------------------------------------------

    def split_by_phase(self, switch_pseudotimes,
                       top_k=5, temperature=1.0, method='weighted_mean'):
        """Per-phase view of this validation (see :class:`PhaseSplitted`).

        Reuses this instance's selector, enriched links and random-pool machinery;
        only adds the phase binning. ``switch_pseudotimes`` is either one ordered
        sequence (lineage mode -- applied to the single branch) or a
        ``{branch: sequence}`` mapping (combined mode -- one set of phase
        boundaries per lineage). ``top_k, temperature, method`` are the softmax
        peak-pseudotime parameters, kept identical to the enriched assignment.
        """
        return TFForceValidation.PhaseSplitted(
            self, switch_pseudotimes,
            top_k=top_k, temperature=temperature, method=method)

    class PhaseSplitted:
        """Per-phase enriched-vs-random comparison, nested inside (and built from)
        a :class:`TFForceValidation` instance.

        TF force is now an established validation metric, so this only *adds* the
        phase split on top of the outer validator -- it reuses that instance's
        force selector, enriched links and random-pool sampling by composition
        (``self.v``) rather than duplicating them.

        Each enriched link is scored on its stronger lineage (``mode='combined'``)
        or the single branch (``mode='lineage'``) -- exactly the force the outer
        validator already picks -- and then binned into a *phase of that lineage*
        by the softmax peak-pseudotime rule (:meth:`RegulatoryPhases.assign_phases`)
        against that branch's cell-state termination pseudotimes. Phases are
        therefore branch-qualified: a link appears only under its winning branch's
        phase (e.g. PB 1-3, GC 1-2). Within each ``(branch, phase)`` cell a
        size-matched set of random non-enriched links -- drawn through the outer
        validator's :meth:`_build_random_pool`, scored and phase-binned on the same
        branch -- forms the null.
        """

        def __init__(self, validation, switch_pseudotimes,
                     top_k=5, temperature=1.0, method='weighted_mean'):
            """
            Parameters
            ----------
            validation : TFForceValidation
                The outer (already-built) validator supplying the selector,
                enriched links, mode and random-pool machinery.
            switch_pseudotimes : sequence or dict {branch: sequence}
                Phase boundaries. A single sequence is applied to the (single)
                lineage branch; a mapping gives one set of boundaries per lineage
                (combined mode). ``N`` boundaries -> ``N + 1`` phases.
            top_k, temperature, method :
                Softmax peak-pseudotime parameters, kept identical to the enriched
                phase assignment for a fair comparison.
            """
            self.v = validation
            self.switch_by_branch = self._normalize_switches(switch_pseudotimes)
            self.softmax_kwargs = dict(top_k=top_k, temperature=temperature, method=method)
            self.result_ = None

        def _normalize_switches(self, switch_pseudotimes):
            """``{branch: sorted boundaries}``: accepts a per-branch mapping or a
            single sequence (applied to the lineage's single branch)."""
            if isinstance(switch_pseudotimes, dict):
                return {b: np.sort(np.asarray(v, dtype=float))
                        for b, v in switch_pseudotimes.items()}
            branch = self.v.selector.branches[0]
            return {branch: np.sort(np.asarray(switch_pseudotimes, dtype=float))}

        def _assign_phases(self, force_curves, dtime, branch):
            """Bin every link in ``force_curves`` to a phase on ``branch`` (same
            softmax rule and boundaries used for the enriched links)."""
            return RegulatoryPhases.assign_phases(
                force_curves, dtime, self.switch_by_branch[branch], **self.softmax_kwargs)

        # ------------------------------------------------------------------
        # public API (per branch-qualified phase)
        # ------------------------------------------------------------------

        def run(self, n_tf=None, n_target=None, exclude='tf_and_target', random_state=0):
            """Build the per-(branch, phase) enriched-vs-random comparison table.

            Each enriched link is placed under its winning branch's phase; a pool
            of ``n_tf`` x ``n_target`` random non-enriched links (see
            :meth:`TFForceValidation._build_random_pool`) is scored and
            phase-binned on each branch, and within every ``(branch, phase)`` cell
            a random subset is drawn to match the enriched count there.

            Parameters
            ----------
            n_tf, n_target, exclude, random_state :
                See :meth:`TFForceValidation.run` (the same random-pool controls).

            Returns
            -------
            pandas.DataFrame with columns ``link, branch, phase, group,
            abs_max_force`` (``group`` is ``'enriched'`` or ``'random'``). Also
            stored on ``self.result_``.
            """
            rng = np.random.default_rng(random_state)
            present, enr_force, enr_branch = self.v._enriched_forces()

            # enriched links grouped by their (winning) lineage
            enr_by_branch = defaultdict(list)
            for link in present:
                enr_by_branch[enr_branch[link]].append(link)

            # one random pool, scored on each branch that carries enriched links
            # (union null -- a random link keeps its own single-branch force)
            pool_links = self.v._build_random_pool(n_tf, n_target, exclude, rng)

            rows = []
            n_per_cell = defaultdict(int)     # (branch, phase) -> # enriched links
            for branch, links in enr_by_branch.items():
                fc, dtime = self.v.selector.force_curves(links, branch=branch)
                phases = self._assign_phases(fc, dtime, branch)
                for link in links:
                    p = phases[link]
                    n_per_cell[(branch, p)] += 1
                    rows.append({'link': link, 'branch': branch, 'phase': p,
                                 'group': 'enriched', 'abs_max_force': enr_force[link]})

            for branch in enr_by_branch:
                fc, dtime = self.v.selector.force_curves(pool_links, branch=branch)
                rnd_phases = self._assign_phases(fc, dtime, branch)
                rnd_force = self.v.selector.abs_max(fc)
                pool_by_phase = defaultdict(list)
                for link, p in rnd_phases.items():
                    pool_by_phase[p].append(link)
                for (b, p), n in n_per_cell.items():
                    if b != branch:
                        continue
                    cands = pool_by_phase.get(p, [])
                    if len(cands) < n:
                        print(f"PhaseSplitted: {branch} phase {p} has only "
                              f"{len(cands)} random links in the pool (< {n} "
                              f"enriched); using all. Increase n_tf/n_target.")
                    chosen_idx = (rng.choice(len(cands), size=min(n, len(cands)),
                                             replace=False) if cands else [])
                    for i in chosen_idx:
                        link = cands[i]
                        rows.append({'link': link, 'branch': branch, 'phase': p,
                                     'group': 'random', 'abs_max_force': rnd_force[link]})

            self.result_ = (
                pd.DataFrame(rows)
                .sort_values(['branch', 'phase', 'group'])
                .reset_index(drop=True)
            )
            return self.result_

        def plot(self, figsize=(9, 6), ylabel='Abs max TF force',
                 colors=('#d1495b', '#9aa0a6')):
            """Box plot of abs-max TF force, enriched vs random, one group of boxes
            per branch-qualified phase (e.g. ``PB-1 PB-2 PB-3 GC-1 GC-2``)."""
            if self.result_ is None:
                self.run()
            df = self.result_
            cells = sorted(set(zip(df['branch'], df['phase'])))
            groups = [('enriched', 'Enriched (FireFate)'),
                      ('random', 'Random (non-enriched)')]
            width = 0.35

            fig, ax = plt.subplots(figsize=figsize)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            for gi, (g, _label) in enumerate(groups):
                offset = (gi - 0.5) * width
                positions, data = [], []
                for ci, (branch, p) in enumerate(cells):
                    vals = df[(df['branch'] == branch) & (df['phase'] == p)
                              & (df['group'] == g)]['abs_max_force'].values
                    positions.append(ci + offset)
                    data.append(vals)
                bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color='black'))
                for patch in bp['boxes']:
                    patch.set_facecolor(colors[gi])
                    patch.set_alpha(0.6)
                # one jittered point per link
                for pos, vals in zip(positions, data):
                    if len(vals) == 0:
                        continue
                    x = pos + (np.random.rand(len(vals)) - 0.5) * width * 0.5
                    ax.scatter(x, vals, color=colors[gi], edgecolor='black',
                               linewidth=0.4, s=22, zorder=3)

            ax.set_xticks(range(len(cells)))
            ax.set_xticklabels([f'{b}-{p}' for b, p in cells])
            ax.set_ylabel(ylabel)
            handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[gi], alpha=0.6)
                       for gi in range(len(groups))]
            ax.legend(handles, [label for _, label in groups], frameon=False)
            return fig, ax
