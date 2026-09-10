"""Cell-state composition over pseudotime and the state-termination pseudotimes that delimit phases."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from focalfire.temporal._align import AlignTimeScales
from focalfire.utils.states import window_labels_to_count_df
import matplotlib.pyplot as plt


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
        return plot_state_composition_bars(df_plot, x, self.colors,
                                           n_bins=n_bins, figsize=figsize)

    def plot_composition_curves(self, window_indices, figsize=(15, 8), xlabel='branch'):
        """Line plot (shaded) of each cell-state count over pseudotime."""
        df_plot, x = self.fate_trajectory(window_indices)
        return plot_state_composition_curves(df_plot, x, self.colors,
                                             figsize=figsize, xlabel=xlabel)

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
        return plot_state_extrema(df_plot, x, extrema_info, self.colors, figsize=figsize)

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
# Figures: cell-state composition
# ---------------------------------------------------------------------------

def plot_state_composition_bars(df_plot, x, colors=None, n_bins=8, figsize=(6, 4)):
    """Stacked bar plot of average cell-state composition over binned windows.

    ``df_plot`` is a states x windows count table and ``x`` the matching pseudotime
    per window (``StateFrequency.fate_trajectory`` output); ``colors`` maps state
    name to colour.
    """
    colors = colors or {}
    x = np.asarray(x)
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
            color=colors.get(state),
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


def plot_state_composition_curves(df_plot, x, colors=None, figsize=(15, 8), xlabel='branch'):
    """Line plot (shaded) of each cell-state count over pseudotime."""
    colors = colors or {}
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    for state in df_plot.index:
        y = df_plot.loc[state]
        ax.plot(x, y, label=state, color=colors.get(state), linewidth=2)
        ax.fill_between(x, y, color=colors.get(state), alpha=0.25)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cell Count')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return fig, ax


def plot_state_extrema(df_plot, x, extrema_info, colors=None, figsize=(15, 8)):
    """Composition curves annotated with maxima (^) and minima (v) markers.

    ``extrema_info`` is the per-state dict returned by ``StateFrequency.find_extrema``.
    """
    colors = colors or {}
    fig, ax = plt.subplots(figsize=figsize)
    for state in df_plot.index:
        y = df_plot.loc[state].values
        color = colors.get(state)
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
