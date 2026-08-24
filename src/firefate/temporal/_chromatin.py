"""TF binding-score and OCR-count dynamics along the trajectory."""
from __future__ import annotations

from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Dict, Tuple, Optional, Union
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import math
from typing import Any, Dict, Optional, Tuple, Union
from plotly.subplots import make_subplots


class SmoothedCurvesChromatin:
    """
    A class to extract, process, and visualize Transcription Factor (TF) binding dynamics
    across genomic windows and pseudotime trajectories.
    """

    def __init__(self, tfs: Optional[List[str]], base_path: str):
        """
        Initialize the chromatin accessibility data analyzer.

        Parameters
        ----------
        tfs : list of str, or None
            Transcription factors to query. If None, all TFs found across the
            binding files will be loaded; the union is materialized as `self.tfs`
            after `extract_data()` runs.
        base_path : str
            Base path to the binding.tsv.gz files (e.g., 'path/to/tmp_dynamic').
        """
        self.tfs = tfs
        self.base_path = base_path
        
        # Data containers
        self.raw_scores: Dict[str, List[float]] = {}
        self.raw_counts: Dict[str, List[float]] = {}
        self.window_pseudotimes: Optional[np.ndarray] = None
        
        # Trajectory specific data
        self.pb_indices: Optional[List[int]] = None
        self.gc_indices: Optional[List[int]] = None
        self.pb_pseudotime: Optional[np.ndarray] = None
        self.gc_pseudotime: Optional[np.ndarray] = None
        
        # Processed series
        self.series_pb: Dict[str, np.ndarray] = {}
        self.series_gc: Dict[str, np.ndarray] = {}

    @staticmethod
    def _process_single_window(
        i: int,
        tfs: Optional[List[str]],
        base_path: str,
    ) -> Tuple[int, Dict[str, float], Dict[str, float]]:
        """
        Static worker for multiprocessing.

        If `tfs` is None, returns scores/counts for every TF present in this
        window's file. If `tfs` is a list, behaves as before: returns NaN/0 for
        TFs absent from the file.
        """
        try:
            file_path = f"{base_path}/Subset{i}/binding.tsv.gz"
            df = pd.read_csv(file_path, sep="\t", compression="gzip")

            if "loc" in df.columns:
                df[["chr", "start", "end"]] = df["loc"].str.split(":", expand=True)

            grouped = df.groupby("TF")

            # Decide which TFs this worker should emit
            if tfs is None:
                tfs_to_process = list(grouped.groups.keys())
                return_defaults_for_missing = False
            else:
                tfs_to_process = tfs
                return_defaults_for_missing = True

            window_scores: Dict[str, float] = {}
            window_counts: Dict[str, float] = {}

            for tf in tfs_to_process:
                if tf in grouped.groups:
                    tf_df = grouped.get_group(tf)
                    window_scores[tf] = (
                        tf_df.groupby("chr").agg({"score": "mean"}).mean().values[0]
                    )
                    window_counts[tf] = (
                        tf_df.groupby("chr").agg({"score": "count"}).mean().values[0]
                    )
                elif return_defaults_for_missing:
                    window_scores[tf] = float("nan")
                    window_counts[tf] = 0
                # if tfs is None and TF not in grouped: just skip — won't happen
                # since tfs_to_process came from grouped.groups in that branch.

            return (i, window_scores, window_counts)

        except Exception:
            # Safe defaults: unknown TFs in the None case → empty dicts;
            # the union step will simply not see this window's contributions.
            if tfs is None:
                return (i, {}, {})
            return (i, {tf: float("nan") for tf in tfs}, {tf: 0 for tf in tfs})

    @staticmethod
    def _to_float(v):
        if hasattr(v, "values"): # Handle Series/Array
            arr = v.values
            if len(arr) == 0: return np.nan
            return arr[0]
        return float(v)

    @staticmethod
    def _smooth(series, sigma):
        return gaussian_filter1d(series, sigma=sigma)

    # ------------------------------------------------------------------
    # Data Extraction Methods
    # ------------------------------------------------------------------

    def extract_data(self, n_windows: int = 194, n_processes: int = None):
        """
        Multiprocess the extraction of TF binding data across all windows.

        If `self.tfs` is None, discovers the full union of TFs present across
        all window files and stores it as `self.tfs` on completion.
        Populates `self.raw_scores` and `self.raw_counts`.
        """
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)

        print(f"Processing {n_windows} windows using {n_processes} processes...")

        process_func = partial(
            SmoothedCurvesChromatin._process_single_window,
            tfs=self.tfs,                   # None propagates to workers
            base_path=self.base_path,
        )

        with Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.imap(process_func, range(1, n_windows + 1)),
                total=n_windows,
                desc="Extracting Binding Data",
            ))

        # Fail loudly if no window produced any real data. The worker swallows
        # per-window errors (e.g. a missing binding.tsv.gz) into empty/all-NaN
        # results, which would otherwise yield empty series and surface as a
        # confusing error several steps downstream. (v == v is False only for NaN.)
        got_data = any(
            any(v == v for v in w_scores.values())
            for _, w_scores, _ in results
        )
        if not got_data:
            raise FileNotFoundError(
                f"No binding data extracted from any of {n_windows} windows. "
                f"Expected per-window files like "
                f"'{self.base_path}/Subset1/binding.tsv.gz'. Check that base_path is "
                f"correct and that the binding.tsv.gz files exist."
            )

        # If tfs was None, resolve the union of TFs seen across all windows.
        if self.tfs is None:
            all_tfs = set()
            for _, w_scores, _ in results:
                all_tfs.update(w_scores.keys())
            self.tfs = sorted(all_tfs)
            print(f"Discovered {len(self.tfs)} TFs across all windows.")

        # Allocate storage with sentinels (NaN for score, 0 for count) and fill.
        self.raw_scores = {tf: [np.nan] * n_windows for tf in self.tfs}
        self.raw_counts = {tf: [0] * n_windows for tf in self.tfs}

        for window_idx, w_scores, w_counts in results:
            idx = window_idx - 1
            if not (0 <= idx < n_windows):
                continue
            for tf in self.tfs:
                if tf in w_scores:
                    self.raw_scores[tf][idx] = w_scores[tf]
                    self.raw_counts[tf][idx] = w_counts.get(tf, 0)
                # else: leave the NaN/0 sentinel in place

        print("Extraction complete.")

    # ------------------------------------------------------------------
    # Trajectory & Processing Methods
    # ------------------------------------------------------------------

    def set_trajectory_info(self,
                            pb_indices: List[int],
                            gc_indices: List[int],
                            window_pseudotimes: Union[List, np.ndarray],
                            gc_window_pseudotimes: Optional[Union[List, np.ndarray]] = None):
        """
        Register trajectory indices and pseudotime values.

        Parameters:
        -----------
        pb_indices : List[int]
            0-based indices of windows belonging to the PB trajectory.
        gc_indices : List[int]
            0-based indices of windows belonging to the GC trajectory.
        window_pseudotimes : array-like
            Pseudotime per window ID in the PB-branch alignment frame (e.g.
            ``AlignTimeScales(..., (0, 2)).pseudotime_of_windows()``). Used for the
            PB branch, and for the GC branch too unless ``gc_window_pseudotimes``
            is given.
        gc_window_pseudotimes : array-like, optional
            Pseudotime per window ID in the GC-branch alignment frame (e.g.
            ``AlignTimeScales(..., (0, 3)).pseudotime_of_windows()``). Provide this
            when the two branches are aligned separately so the GC windows land on
            the GC pseudotime axis that the GC phase switches are defined in;
            otherwise the GC windows inherit the PB frame and won't line up with
            GC-frame switches. Defaults to ``window_pseudotimes``.
        """
        self.pb_indices = pb_indices
        self.gc_indices = gc_indices
        self.window_pseudotimes = np.array(window_pseudotimes)
        gc_window_pseudotimes = (self.window_pseudotimes if gc_window_pseudotimes is None
                                 else np.array(gc_window_pseudotimes))

        # Map indices to pseudotimes immediately (each branch in its own frame)
        self.pb_pseudotime = self.window_pseudotimes[self.pb_indices]
        self.gc_pseudotime = gc_window_pseudotimes[self.gc_indices]

    def process_dynamics(self, metric: str = 'score', smooth_sigma: float = 2.0, relative: bool = False):
        """
        Process raw data into ordered, smoothed trajectories for PB and GC.
        
        Parameters:
        -----------
        metric : str, 'score' or 'count'
            Which data source to process.
        smooth_sigma : float
            Sigma for Gaussian smoothing.
        relative : bool
            If True, min-max normalize each TF's series to [0, 1] using the
            global min/max across both trajectories combined, so both lines
            share the same reference scale.
        """
        if self.pb_indices is None or self.gc_indices is None:
            raise ValueError("Trajectories not set. Call set_trajectory_info() first.")

        source_data = self.raw_scores if metric == 'score' else self.raw_counts
        
        self.series_pb = {}
        self.series_gc = {}

        for tf in self.tfs:
            vals = [self._to_float(v) for v in source_data.get(tf, [])]
            vals_arr = np.array(vals)

            ordered_pb = vals_arr[self.pb_indices]
            ordered_gc = vals_arr[self.gc_indices]

            smoothed_pb = self._smooth(ordered_pb, sigma=smooth_sigma)
            smoothed_gc = self._smooth(ordered_gc, sigma=smooth_sigma)

            if relative:
                # Compute global min/max across both branches so they share the
                # same reference frame
                global_min = np.nanmin(np.concatenate([smoothed_pb, smoothed_gc]))
                global_max = np.nanmax(np.concatenate([smoothed_pb, smoothed_gc]))
                denom = global_max - global_min
                if denom == 0:
                    denom = 1.0  # avoid division by zero for flat signals
                smoothed_pb = (smoothed_pb - global_min) / denom
                smoothed_gc = (smoothed_gc - global_min) / denom

            self.series_pb[tf] = smoothed_pb
            self.series_gc[tf] = smoothed_gc

    # ------------------------------------------------------------------
    # Visualization Methods
    # ------------------------------------------------------------------

    def plot(self,
             categories: Dict[str, Dict[str, str]],
             y_label: str = "Binding Score",
             title: str = None,
             truncate_pb: bool = True) -> go.Figure:
        """
        Generate Plotly figure for the processed dynamics.

        Parameters:
        -----------
        categories : dict
            Structure: {"CategoryName": { "TF_Name": "ColorHex", ... }, ...}
            Example: {"Static": {"TF1": "red"}, "Episodic": {"TF2": "blue"}}
        y_label : str
            Label for Y-axis.
        truncate_pb : bool
            If True, cuts the PB line to match the max pseudotime of GC.
        """
        if not self.series_pb:
            raise ValueError("No processed data found. Call process_dynamics() first.")

        return plot_chromatin_tf_dynamics(
            self.pb_pseudotime,
            self.gc_pseudotime,
            self.series_pb,
            self.series_gc,
            categories,
            y_label=y_label,
            title=title,
            truncate_pb=truncate_pb,
        )

    def plot_score_vs_count_comparison(self,
        categories: Dict[str, Dict[str, str]],
        smooth_sigma: float = 2.0,
        subplot_cols: int = 3,
        title: str = None,
        ) -> go.Figure:
        """Per-TF comparison of GC binding score against OCR count (both min-max scaled)."""
        if not self.raw_scores or not self.raw_counts:
            raise ValueError("No raw data found. Call extract_data() first.")
        if self.gc_indices is None:
            raise ValueError("Trajectories not set. Call set_trajectory_info() first.")

        tf_color_map = {tf: color for cat in categories.values() for tf, color in cat.items()}

        series_by_tf = {}
        for tf in tf_color_map:
            score_vals = np.array([self._to_float(v) for v in self.raw_scores.get(tf, [])])
            count_vals = np.array([self._to_float(v) for v in self.raw_counts.get(tf, [])])
            series_by_tf[tf] = (
                self._smooth(score_vals[self.gc_indices], sigma=smooth_sigma),
                self._smooth(count_vals[self.gc_indices], sigma=smooth_sigma),
            )

        return plot_score_vs_count_subplots(
            self.gc_pseudotime,
            series_by_tf,
            tf_color_map,
            subplot_cols=subplot_cols,
            title=title,
        )


# ---------------------------------------------------------------------------
# Figures: chromatin binding dynamics
# ---------------------------------------------------------------------------

def plot_chromatin_tf_dynamics(
    pb_pseudotime,
    gc_pseudotime,
    series_pb: Dict[str, np.ndarray],
    series_gc: Dict[str, np.ndarray],
    categories: Dict[str, Dict[str, str]],
    y_label: str = "Binding Score",
    title: str = None,
    truncate_pb: bool = True,
) -> go.Figure:
    """Overlay per-TF PB (solid) and GC (dashed) traces against pseudotime.

    Parameters
    ----------
    series_pb, series_gc : dict
        TF name → processed values, one per pseudotime point of that branch.
    categories : dict
        ``{"CategoryName": {"TF_Name": "ColorHex", ...}, ...}``; sets legend groups.
    truncate_pb : bool
        If True, cuts the PB line to match the max pseudotime of GC.
    """
    # Determine truncation mask; as both branches are not same length, truncate the longer one
    mask_pb = np.ones(len(pb_pseudotime), dtype=bool)
    max_gc_time = np.nanmax(gc_pseudotime)

    if truncate_pb:
        mask_pb = pb_pseudotime <= max_gc_time

    fig = go.Figure()

    # Iterate through categories (e.g., Static, Episodic)
    for cat_name, tf_color_map in categories.items():
        first_in_cat = True

        for tf, color in tf_color_map.items():
            if tf not in series_pb:
                print(f"Warning: {tf} not found in processed data.")
                continue

            # Add GC Trace (Dashed)
            fig.add_trace(go.Scatter(
                x=gc_pseudotime,
                y=series_gc[tf],
                mode='lines',
                name=tf,
                line=dict(dash='dash', color=color, width=2.5),
                legendgroup=tf,
                legendgrouptitle_text=cat_name if first_in_cat else None,
                showlegend=True
            ))

            # Add PB Trace (Solid)
            fig.add_trace(go.Scatter(
                x=pb_pseudotime[mask_pb],
                y=series_pb[tf][mask_pb],
                mode='lines',
                name=tf,
                line=dict(dash='solid', color=color, width=2.5),
                legendgroup=tf,
                showlegend=False
            ))

            first_in_cat = False

    # Layout styling

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(
            title='Pseudotime',
            showgrid=True,
            range=[0, max_gc_time if truncate_pb else None]
        ),
        yaxis=dict(title=y_label),
        legend=dict(
            orientation='v', x=1.02, y=0.5,
            tracegroupgap=25,
            title_text="<b>TF Categories</b><br>(Solid=PB, Dashed=GC)"
        ),
        margin=dict(t=100, r=250),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif")
    )

    return fig


def plot_score_vs_count_subplots(
    x,
    series_by_tf: Dict[str, Tuple[np.ndarray, np.ndarray]],
    tf_color_map: Dict[str, str],
    subplot_cols: int = 3,
    title: str = None,
) -> go.Figure:
    """One subplot per TF comparing binding score against OCR count, both min-max scaled.

    Parameters
    ----------
    x : array-like
        Shared pseudotime axis for every subplot.
    series_by_tf : dict
        TF name → ``(score_values, count_values)``, already smoothed by the caller.
    tf_color_map : dict
        TF name → colour for its score trace; also fixes subplot order.
    """
    tf_list = list(tf_color_map.keys())

    n_cols = subplot_cols
    n_rows = math.ceil(len(tf_list) / n_cols)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=tf_list,
        shared_xaxes=False,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    def minmax(arr):
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        denom = hi - lo if (hi - lo) != 0 else 1.0
        return (arr - lo) / denom

    for idx, tf in enumerate(tf_list):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        color = tf_color_map[tf]

        score_vals, count_vals = series_by_tf[tf]

        norm_score = minmax(score_vals)
        norm_count = minmax(count_vals)

        show_legend = idx == 0

        fig.add_trace(go.Scatter(
            x=x, y=norm_score, mode="lines",
            name="TF Binding Score",
            line=dict(color=color, width=2.5, dash="solid"),
            legendgroup="score", showlegend=show_legend,
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=x, y=norm_count, mode="lines",
            name="OCR Count",
            line=dict(color="grey", width=2, dash="dash"),
            legendgroup="count", showlegend=show_legend,
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([
                np.where(norm_count > norm_score, norm_count, norm_score),
                np.where(norm_count > norm_score, norm_score, norm_score)[::-1],
            ]),
            fill="toself", fillcolor="rgba(180,180,180,0.18)",
            line=dict(width=0),
            name="Count > Score region",
            legendgroup="shade", showlegend=show_legend,
            hoverinfo="skip",
        ), row=row, col=col)

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        height=320 * n_rows,
        width=420 * n_cols,
        template="none",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=11),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.05,
            title_text="<b>— Binding Score &nbsp;&nbsp; -- OCR Count</b>",
        ),
    )

    # Global wipe first — must come before per-subplot calls
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    for i in range(1, n_rows * n_cols + 1):
        r, c = (i - 1) // n_cols + 1, (i - 1) % n_cols + 1
        fig.update_xaxes(
            title_text="Pseudotime" if i > (n_rows - 1) * n_cols else "",
            showline=True, linecolor="black", linewidth=1.5, mirror=False,
            zeroline=False,
            row=r, col=c,
        )
        fig.update_yaxes(
            title_text="Relative value [0–1]" if (i - 1) % n_cols == 0 else "",
            range=[-0.05, 1.1],
            showline=True, linecolor="black", linewidth=1.5, mirror=False,
            zeroline=False,
            row=r, col=c,
        )

    return fig
