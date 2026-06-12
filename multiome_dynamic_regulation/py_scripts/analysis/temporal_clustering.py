"""Temporal clustering of regulatory links into waves.

Refactor of ``waves_quantification.ipynb`` into three classes:

- :class:`FateFrequency` -- cell-state composition over pseudotime. Produces
  fate-frequency trajectories (stacked-bar and composition-curve plots), finds
  per-state extrema, and exposes the pseudotime at which a state "terminates"
  (``termination_pseudotime``). Those termination pseudotimes are the wave-switch boundaries
  consumed by :class:`TFForceWaves`.

- :class:`TFForceWaves` -- TF force trajectories over pseudotime. Plots TF
  expression / regulation curves, computes regulatory forces for a set of links,
  renders the 3D force landscape of a single link, runs the softmax peak-finding
  per force curve, draws the clustered force heatmap, and classifies links into
  waves by binning each link's softmax peak pseudotime against the cell-state
  termination pseudotimes (3 waves for PB, 2 for GC).

- :class:`WaveValidation` -- (skeleton) compares the FireFate links against
  random links in the data frame, split by wave. Implementation pending.

The module-level softmax helpers (:func:`get_max_points`,
:func:`aggregate_max_points`, :func:`order_links_by_wave`, :func:`order_links`)
are the canonical implementation; :class:`TFForceWaves` reuses them.
"""

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


def order_links_by_wave(regulation_pseudotimes):
    """
    Order TF-Target links into waves by ascending peak pseudotime.

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
    Convenience wrapper: compute the wave ordering of links directly from force curves.

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
    ordered_links = order_links_by_wave(regulation_pseudotimes)
    return ordered_links, regulation_pseudotimes


# ---------------------------------------------------------------------------
# Class 1: cell-state fate frequencies and extrema / termination pseudotimes
# ---------------------------------------------------------------------------

class FateFrequency:
    """Cell-state composition over pseudotime for one or more fate branches.

    Builds the per-window cell-state count table, maps windows to pseudotime,
    and provides fate-frequency plots plus extrema / "termination" pseudotimes that
    define the wave-switch boundaries used by :class:`TFForceWaves`.
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
# Class 2: TF forces over pseudotime and wave classification of links
# ---------------------------------------------------------------------------

class TFForceWaves:
    """TF regulatory forces over pseudotime, with softmax-based wave assignment.

    Wraps :class:`SmoothedCurvesGRN` for one trajectory branch and exposes:
    expression / regulation trajectory plots, force computation for a set of
    links, the 3D single-link force landscape, the clustered force heatmap, the
    softmax peak-pseudotime per link, and the wave classification that bins each
    link by the cell-state termination pseudotimes from :class:`FateFrequency`.
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

    # Retain softmax helpers as part of this class (delegate to module fns).
    get_max_points = staticmethod(get_max_points)
    aggregate_max_points = staticmethod(aggregate_max_points)
    order_links_by_wave = staticmethod(order_links_by_wave)

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
    # Forces
    # ------------------------------------------------------------------

    def compute_forces(self, links, varname='w_in'):
        """Compute and cache beta / TF-expression / force curves for ``links``.

        Returns the force-curve DataFrame and stores ``beta_curves``,
        ``regulon_tf_expression``, ``force_curves`` and ``dtime`` on ``self``.
        """
        beta_curves, dtime = self.curves.get_beta_curves(links, varname=varname)
        tf_expression, _ = self.curves.get_smoothed_curves(mode='tf_expression')
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
    # Softmax peak pseudotimes and wave classification
    # ------------------------------------------------------------------

    def link_peak_pseudotimes(self, links=None, top_k=5, temperature=1.0,
                              method='weighted_mean'):
        """Softmax peak pseudotime per link.

        Returns the ``aggregate_max_points`` dict
        ``{(TF, Target): {'pseudotime': float, ...}}`` for ``links``
        (default: every link in ``force_curves``).
        """
        self._require_forces()
        df = self.force_curves if links is None else self.force_curves.loc[links]
        max_points = get_max_points(df, self.dtime, top_k=top_k, temperature=temperature)
        return aggregate_max_points(max_points, method=method)

    def classify_waves(self, switch_pseudotimes, links=None, top_k=5,
                       temperature=1.0, method='weighted_mean'):
        """Assign each link to a wave by its softmax peak pseudotime.

        ``switch_pseudotimes`` is the ordered list of cell-state termination
        pseudotimes that separate consecutive waves. ``N`` switches produce
        ``N + 1`` waves (1-indexed): a link with peak pseudotime ``p`` lands in
        wave ``k`` where ``switch[k-2] < p <= switch[k-1]``. So PB uses
        ``[ActB-4_termination, earlyPB_termination]`` (3 waves) and GC uses
        ``[ActB-3_termination]`` (2 waves).

        Returns
        -------
        pandas.DataFrame with columns ``TF, Target, peak_pseudotime, wave``,
        sorted by wave then peak pseudotime.
        """
        boundaries = np.sort(np.asarray(switch_pseudotimes, dtype=float))
        reg_pt = self.link_peak_pseudotimes(
            links=links, top_k=top_k, temperature=temperature, method=method
        )

        rows = []
        for (tf, target), info in reg_pt.items():
            peak = info['pseudotime']
            wave = int(np.digitize(peak, boundaries, right=True)) + 1
            rows.append({'TF': tf, 'Target': target,
                         'peak_pseudotime': peak, 'wave': wave})

        return (
            pd.DataFrame(rows)
            .sort_values(['wave', 'peak_pseudotime'])
            .reset_index(drop=True)
        )

    def classify_waves_from_states(self, fate_frequency, window_indices,
                                  boundary_states, links=None,
                                  termination_method='threshold', threshold_frac=0.1,
                                  prominence=10, distance=3, **softmax_kwargs):
        """Wave classification driven directly by cell-state termination pseudotimes.

        Computes the termination pseudotime of each state in ``boundary_states`` (in
        order) from a :class:`FateFrequency` instance, then bins links via
        :meth:`classify_waves`.

        Example
        -------
        PB (3 waves)::

            waves.classify_waves_from_states(
                ff, PB_post_bifurcation_window_indices,
                boundary_states=['ActB-4', 'earlyPB'])

        GC (2 waves)::

            waves.classify_waves_from_states(
                ff, GC_post_bifurcation_window_indices,
                boundary_states=['ActB-3'])
        """
        switch_pseudotimes = [
            fate_frequency.termination_pseudotime(
                state, window_indices,
                method=termination_method, threshold_frac=threshold_frac,
                prominence=prominence, distance=distance,
            )
            for state in boundary_states
        ]
        return self.classify_waves(switch_pseudotimes, links=links, **softmax_kwargs)


# ---------------------------------------------------------------------------
# Class 3: validation of wave-clustered links vs random links (skeleton)
# ---------------------------------------------------------------------------

class WaveValidation:
    """Validate wave-clustered FireFate links against random links.

    Intended to compare a metric (e.g. force magnitude / temporal coherence)
    between the prioritized FireFate links and random links drawn from the same
    force DataFrame, broken out by wave on the x-axis.

    Not yet implemented -- the validation procedure from the notebook is still
    to be written.
    """

    def __init__(self, force_curves, dtime, wave_assignments):
        """
        Parameters
        ----------
        force_curves : pandas.DataFrame
            Multi-indexed (TF, Target) force curves; the random-link pool is
            sampled from here.
        dtime : pandas.Series
            Pseudotime per column of ``force_curves``.
        wave_assignments : pandas.DataFrame
            Output of :meth:`TFForceWaves.classify_waves` (FireFate links + wave).
        """
        self.force_curves = force_curves
        self.dtime = dtime
        self.wave_assignments = wave_assignments

    def sample_random_links(self, n_per_wave, exclude=None, random_state=None):
        raise NotImplementedError("Random-link sampling not implemented yet.")

    def compare_by_wave(self):
        raise NotImplementedError("Per-wave comparison not implemented yet.")

    def plot(self):
        raise NotImplementedError("Validation plot (split by wave) not implemented yet.")
