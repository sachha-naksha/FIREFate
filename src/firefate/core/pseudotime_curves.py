import gc
import math
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import List, Dict, Tuple, Optional, Union

import dictys
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from dictys.net import stat
from dictys.utils.numpy import ArrayLike
from numpy.typing import NDArray
from joblib import Memory
from scipy import stats
from scipy.stats import hypergeom
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from firefate.core.stat_extensions import lcpm_tf
from firefate.utils.custom import *
from firefate.utils.plots import (
    plot_chromatin_tf_dynamics,
    plot_score_vs_count_subplots,
)


class SmoothedCurvesGRN:
    """
    provides methods to compute expression and regulation curves,
    and calculate regulatory forces.
    """
    
    def __init__(self, dictys_dynamic_object,
        trajectory_range,
        num_points=40,
        dist=0.001,
        sparsity=0.01,
        mode="expression",
        ):
        """
        initialize the analyzer
        """
        self.dictys_dynamic_object = dictys_dynamic_object
        self.trajectory_range = trajectory_range
        self.num_points = num_points
        self.dist = dist
        self.sparsity = sparsity
        self.mode = mode
    
    def get_smoothed_curves(
        self, mode=None, n_jobs=None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Compute expression (lcpm) and regulation (ltarget_count) curves over pseudotime
        for one branch.

        Returns
        -------
        tuple of (pandas.DataFrame, pandas.Series)
            ``(curves_dataframe, pseudotime_series)``.
        """

        # sample equispaced points and instantiate smoothing function
        pts, fsmooth = self.dictys_dynamic_object.linspace(self.trajectory_range[0], self.trajectory_range[1], self.num_points, self.dist)

        # Use provided mode or fall back to instance mode
        mode_to_use = mode if mode is not None else self.mode

        # Pseudo time values (x axis). First gene's pseudotime is returned as all
        # genes share the same pseudotime over the pseudo-bulked cells in the window.
        stat1_x = stat.pseudotime(self.dictys_dynamic_object, pts)
        dx = pd.Series(stat1_x.compute(pts)[0])

        if mode_to_use == "regulation":
            # Log number of targets. Parallel, NaN-aware reimplementation of the
            # dictys chain flnneighbor(fbinarize(fsmooth(net), sparsity)); the
            # per-point sparsity threshold + outdegree dominate runtime and are
            # independent across points, so they are spread over CPU cores.
            dy = self._regulation_curves_parallel(pts, fsmooth, n_jobs=n_jobs)
            return dy, dx

        if mode_to_use == "weighted_regulation":
            # Log weighted outdegree
            stat1_net = fsmooth(stat.net(self.dictys_dynamic_object))
            stat1_y = stat.flnneighbor(stat1_net, weighted_sparsity=self.sparsity)
        elif mode_to_use == "tf_expression":
            stat1_y = fsmooth(lcpm_tf(self.dictys_dynamic_object, cut=0))
        elif mode_to_use == "expression":
            stat1_y = fsmooth(stat.lcpm(self.dictys_dynamic_object, cut=0))
        else:
            raise ValueError(f"Unknown mode {mode_to_use}.")

        dy = pd.DataFrame(stat1_y.compute(pts), index=stat1_y.names[0])
        return dy, dx

    def _regulation_curves_parallel(self, pts, fsmooth, n_jobs=None) -> pd.DataFrame:
        """Parallel equivalent of ``flnneighbor(fbinarize(fsmooth(net), sparsity))``.

        Reproduces the dictys ``regulation`` chain exactly (verified bit-for-bit)
        but replaces two bottlenecks:

        * dictys' Gaussian smoothing rebuilds full-size ``isnan``/``nan_to_num``
          temporaries on every call; here the smoothing is a single NaN-aware
          matmul (``point.smoothen`` with ``nan='ignore'`` semantics).
        * the per-point top-``k`` sparsity threshold + outdegree run in a serial
          Python loop in ``stat.fbinarize``; each pseudotime point is independent,
          so they are spread across ``n_jobs`` threads (the heavy NumPy ops release
          the GIL and share the smoothed array without copying).

        Returns ``(n_regulator, n_point)`` DataFrame of ``log2(outdegree + 1)``.
        """
        if n_jobs is None:
            n_jobs = min(16, cpu_count())

        # Building the smoothing stat precomputes the node-filtered network array.
        stat1_net = fsmooth(stat.net(self.dictys_dynamic_object))
        fs = stat1_net.func_smooth
        pt = fs.func.__self__            # node-filtered dictys.traj.point
        data = fs.args[0]                # (n_reg, n_target, n_node)
        radius = fs.args[1]
        w = pt.weight_conv(pts, radius)  # (n_node, n_pts), column-normalised Gaussian

        n_reg, n_target, _ = data.shape
        n_pts = len(pts)
        # Number of strongest edges kept per point (matches stat.fbinarize).
        k = int(self.sparsity * n_reg * n_target)

        # Gaussian smoothing, NaN-aware (mirrors point.smoothen nan='ignore'):
        # nan entries contribute zero weight and a point is nan only if every
        # contributing node is nan. The no-nan branch is the identical result.
        if np.isnan(data).any():
            mask = (~np.isnan(data)).astype(data.dtype)
            den = mask @ w
            smoothed = (np.nan_to_num(data) @ w) / (den + 1e-300)
            smoothed[den == 0] = np.nan
        else:
            smoothed = data @ w          # (n_reg, n_target, n_pts), BLAS-threaded
        np.abs(smoothed, out=smoothed)   # signed binarisation ranks by |weight|

        # Per-point: keep top-k edges, count outdegree per regulator, log2(.+1).
        dy = np.empty((n_reg, n_pts), dtype=np.float64)

        def _fill(cols):
            for j in cols:
                arr = smoothed[:, :, j]
                cut = np.partition(arr.ravel(), -k)[-k]
                dy[:, j] = np.log2((arr >= cut).sum(axis=1) + 1.0)

        col_chunks = [c.tolist() for c in np.array_split(np.arange(n_pts), n_jobs) if len(c)]
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            list(ex.map(_fill, col_chunks))

        return pd.DataFrame(dy, index=stat1_net.names[0])

    def _subnetwork_curves(self, TF_indices, target_indices, varname):
        """Smoothed signed network for the given TF/target indices only.

        Computes just the requested sub-network instead of smoothing the whole
        GRN and slicing afterwards: the per-node network array is sliced to the
        queried TFs/targets *before* the Gaussian smoothing matmul, so the full
        ``(n_reg, n_target, n_pts)`` smoothed network is never materialised. The
        per-node data and weights are exactly those used by
        :meth:`_regulation_curves_parallel`; the signed smoothing here is the
        un-binarised version of that path (no ``abs``/top-``k``).

        Returns ``(subnetworks, dtime)`` with ``subnetworks`` of shape
        ``(len(TF_indices), len(target_indices), n_pts)``.
        """
        # sample evenly spaced points along the trajectory
        pts, fsmooth = self.dictys_dynamic_object.linspace(self.trajectory_range[0], self.trajectory_range[1], self.num_points, self.dist)
        stat1_net = fsmooth(stat.net(self.dictys_dynamic_object, varname=varname))
        fs = stat1_net.func_smooth
        pt = fs.func.__self__            # node-filtered dictys.traj.point
        data = fs.args[0]                # (n_reg, n_target, n_node), per-node network
        radius = fs.args[1]
        w = pt.weight_conv(pts, radius)  # (n_node, n_pts), column-normalised Gaussian

        # slice to the queried TFs/targets before smoothing
        sub = data[np.ix_(TF_indices, target_indices, range(data.shape[2]))]

        # Gaussian smoothing, NaN-aware (mirrors point.smoothen nan='ignore'); signed
        # -- these are the raw beta coefficients, so no abs/binarisation.
        if np.isnan(sub).any():
            mask = (~np.isnan(sub)).astype(sub.dtype)
            den = mask @ w
            subnetworks = (np.nan_to_num(sub) @ w) / (den + 1e-300)
            subnetworks[den == 0] = np.nan
        else:
            subnetworks = sub @ w        # (n_tf, n_target, n_pts)

        dtime = pd.Series(stat.pseudotime(self.dictys_dynamic_object, pts).compute(pts)[0])
        return subnetworks, dtime

    def get_beta_curves(self, specified_links: list, varname: str = 'w_in'):
        """
        get beta curves for specified links;
        varname: 'w_in' for normalized total effect network, 'w_n' for normalized direct effect network, 'w' for non-normalized direct effect network
        """

        # getting the TF and target indices for querying the network
        tf_list = list(set([link[0] for link in specified_links]))
        TF_indices, _, missing_tfs = get_tf_indices(self.dictys_dynamic_object, tf_list)
        target_list = list(set([link[1] for link in specified_links]))
        target_indices = get_gene_indices(self.dictys_dynamic_object, target_list)

        # compute only the queried sub-network (the full GRN is never smoothed)
        subnetworks, dtime = self._subnetwork_curves(TF_indices, target_indices, varname)

        # _subnetwork_curves keeps only the TFs/targets present in the network, so
        # build the index from the same found TFs/targets (in TF_indices /
        # target_indices order) to stay aligned with the data; otherwise the missing
        # genes make the index longer than the reshaped array.
        ndict = self.dictys_dynamic_object.ndict
        missing_tf_set = set(missing_tfs)
        found_tfs = [tf for tf in tf_list if tf not in missing_tf_set]
        found_targets = [target for target in target_list if target in ndict]
        n_missing_targets = len(target_list) - len(found_targets)
        if missing_tfs or n_missing_targets:
            print(f"get_beta_curves: skipping {len(missing_tfs)} TF(s) and "
                  f"{n_missing_targets} target(s) not present in the network.")

        # create multi-index tuples for all found TF-target combinations
        index_tuples = [(tf, target) for tf in found_tfs for target in found_targets]
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['TF', 'Target'])

        # reshape the subnetworks array to 2D (pairs × time points)
        n_tfs, n_targets, n_times = subnetworks.shape
        reshaped_data = subnetworks.reshape(-1, n_times)

        # create DataFrame with multi-index
        beta_dcurve = pd.DataFrame(
            reshaped_data,
            index=multi_index,
            columns=[f'time_{i}' for i in range(n_times)]
        )
        return beta_dcurve, dtime
    
    @staticmethod
    def calculate_force_curves(
        beta_curves: pd.DataFrame, 
        tf_expression: pd.Series
    ) -> pd.DataFrame:
        """
        calculates regulatory force curves using log transformation.
        
        force is calculated as beta * tf_expression
        
        args:
            beta_curves: DataFrame with regulatory coefficients (multi-indexed by tf and target)
            tf_expression: Series with tf expression values
            
        returns:
            dataframe with calculated force curves
        """
        # Count number of targets per tf from beta_curves multi-index
        targets_per_tf = beta_curves.index.get_level_values(0).value_counts()
        
        # Create a DataFrame with repeated tf expression values for each target
        expanded_tf_expr = pd.DataFrame(
            np.repeat(
                tf_expression.values, targets_per_tf.values, axis=0
            ),
            index=beta_curves.index,
            columns=beta_curves.columns,
        )
        
        # convert to numpy arrays for calculations
        beta_array = beta_curves.to_numpy()
        tf_array = expanded_tf_expr.to_numpy()
        
        # add small epsilon to avoid log(0)
        epsilon = 1e-10
        log_beta = np.log10(np.abs(beta_array) + epsilon)
        log_tf = np.log10(tf_array + epsilon)
        
        # peserve signs from original beta values
        signs = np.sign(beta_array)
        
        # calculate forces
        force_array = signs * np.exp(log_beta + log_tf)
        
        # convert back to DataFrame with original index/columns
        force_curves = pd.DataFrame(
            force_array, 
            index=beta_curves.index, 
            columns=beta_curves.columns
        )
        
        return force_curves

    @staticmethod
    def calculate_auc(dx: NDArray[float], dy: NDArray[float]) -> NDArray[float]:
        """
        computes area under the curves using trapezoidal rule.
        
        args:
            dx: X-axis values (must be increasing)
            dy: Y-axis values (2D array where each row is a curve)
            
        returns:
            array of AUC values for each curve
        """
        if len(dx) < 2 or not (dx[1:] > dx[:-1]).all():
            raise ValueError("dx must be increasing and have at least 2 values.")
        dxdiff = dx[1:] - dx[:-1]
        dymean = (dy[:, 1:] + dy[:, :-1]) / 2
        ans = dymean @ dxdiff
        return ans
    
    def calculate_transient_logfc(
        self, 
        dx: NDArray[float], 
        dy: NDArray[float]
    ) -> NDArray[float]:
        """
        computes transient log fold change for curves.
        
        args:
            dx: Pseudotime values
            dy: Expression/regulation values
            
        returns:
            Transient logFC values for each curve
        """
        n = dy.shape[1]
        dx = (dx - dx[0]) / (dx[-1] - dx[0])
        dy = dy - np.median(
            [dy, np.repeat(dy[:, [0]], n, axis=1), np.repeat(dy[:, [-1]], n, axis=1)],
            axis=0,
        )
        return self.calculate_auc(dx, dy)
    
    def calculate_switching_time(
        self, 
        dx: NDArray[float], 
        dy: NDArray[float]
    ) -> NDArray[float]:
        """
        measures when the main transition occurs by calculating the
        normalized area under the curve relative to the total change.
        
        args:
            dx: Pseudotime values
            dy: Expression/regulation values
            
        returns:
            switching time values for each curve
        """
        n = dy.shape[1]
        dx = (dx - dx[0]) / (dx[-1] - dx[0])
        dy = np.median(
            [dy, np.repeat(dy[:, [0]], n, axis=1), np.repeat(dy[:, [-1]], n, axis=1)],
            axis=0,
        )
        return (self.calculate_auc(dx, (dy.T - dy[:, -1]).T)) / (dy[:, 0] - dy[:, -1] + 1e-300)
    
    @staticmethod
    def calculate_terminal_logfc(
        dx: NDArray[float], 
        dy: NDArray[float]
    ) -> NDArray[float]:
        """
        computes difference between final and initial values.
        
        args:
            dx: Pseudotime values (must be increasing)
            dy: Expression/regulation values
            
        returns:
            terminal logFC values for each curve
        """
        if len(dx) < 2 or not (dx[1:] > dx[:-1]).all():
            raise ValueError("dx must be increasing and have at least 2 values.")
        return dy[:, -1] - dy[:, 0]

    def curve_characteristics(
        self,
        dx: NDArray[float],
        dy: NDArray[float],
        include_metrics: list = None
    ) -> pd.DataFrame:
        """
        Compute multiple characteristics for the given curves.

        Parameters
        ----------
        dx : ndarray
            Pseudotime values.
        dy : ndarray
            Expression or regulation values.
        include_metrics : list, optional
            Metrics to compute. Options: ``transient_logfc``, ``switching_time``,
            ``terminal_logfc``, ``auc``. If None, computes all metrics.

        Returns
        -------
        pandas.DataFrame
            One column per requested metric for each curve.
        """
        if include_metrics is None:
            include_metrics = ['transient_logfc', 'switching_time', 'terminal_logfc', 'auc']
        
        results = {}
        
        if 'transient_logfc' in include_metrics:
            results['transient_logfc'] = self.calculate_transient_logfc(dx, dy)
        if 'switching_time' in include_metrics:
            results['switching_time'] = self.calculate_switching_time(dx, dy)
        if 'terminal_logfc' in include_metrics:
            results['terminal_logfc'] = self.calculate_terminal_logfc(dx, dy)
        if 'auc' in include_metrics:
            results['auc'] = self.calculate_auc(dx, dy)
        
        return pd.DataFrame(results)

    def classify_tf_global_activity(self, dx, dy, terminal_col, transient_col):
        """
        add tf activity class to dataframe based on z-score normalized logfc comparison
        """
        df = self.curve_characteristics(dx, dy)
        # Z-score normalize both columns
        terminal_zscore = stats.zscore(df[terminal_col])
        transient_zscore = stats.zscore(df[transient_col])

        # classification function
        def get_class_name(terminal_z, transient_z):
            if abs(terminal_z) >= abs(transient_z):
                # terminal effect dominates
                return "Cumulative" if terminal_z > 0 else "Reductive"
            else:
                # transient effect dominates
                return "Bell wave" if transient_z > 0 else "U-shaped"

        # add class name column
        df["tf_class"] = [
            get_class_name(t_z, tr_z)
            for t_z, tr_z in zip(terminal_zscore, transient_zscore)
        ]
        # add z score columns
        df["terminal_z"] = terminal_zscore
        df["transient_z"] = transient_zscore
        df["terminal_rank"] = (
            df["terminal_z"].abs().rank(method="dense", ascending=False).astype(int)
        )
        df["transient_rank"] = (
            df["transient_z"].abs().rank(method="dense", ascending=False).astype(int)
        )
        return df

    # Map the internal global-activity classes to the four wave-pattern names.
    WAVE_PATTERN_NAMES = {
        "Cumulative": "up",
        "Reductive": "down",
        "Bell wave": "transiently_up",
        "U-shaped": "transiently_down",
    }

    def classify_wave_patterns(self, dx, dy, tf_list=None):
        """
        Classify each TF's curve into one of four wave patterns:
        ``up``, ``down``, ``transiently_up``, ``transiently_down``.

        The up/down vs transient decision compares z-scored terminal and
        transient logFC across the curves in ``dy`` (see
        ``classify_tf_global_activity``). Pass the full ``dy`` (all TFs) for a
        stable classification; ``tf_list`` only filters the returned rows.

        Parameters
        ----------
        dx : ndarray
            Pseudotime values.
        dy : pandas.DataFrame
            Smoothed curves, indexed by TF name (e.g. from
            ``get_smoothed_curves(mode="regulation")``).
        tf_list : list of str, optional
            TFs to keep in the output. If None, all TFs are returned.

        Returns
        -------
        pandas.DataFrame
            Indexed by TF, with a ``trajectory`` column and a ``wave_pattern``
            column (one of the four categories), alongside the diagnostic
            columns from ``classify_tf_global_activity``.
        """
        df = self.classify_tf_global_activity(
            dx, dy.values if hasattr(dy, "values") else dy,
            "terminal_logfc", "transient_logfc",
        )
        df.index = dy.index
        df["wave_pattern"] = df["tf_class"].map(self.WAVE_PATTERN_NAMES)
        df["trajectory"] = [self.trajectory_range] * len(df)

        if tf_list is not None:
            df = df.loc[df.index.intersection(tf_list)]

        return df

    def get_top_k_tfs_by_class(self, dx, dy, k=20):
        """
        get top k tfs from each class based on their relevant ranks
        """
        df = self.classify_tf_global_activity(dx, dy, "terminal_logfc", "transient_logfc")
        # determine which rank to use for each tf based on their class
        def get_relevant_rank(row):
            # if terminal effect dominates (Activating/Inactivating or similar), use terminal_rank
            # if transient effect dominates, use transient_rank
            if abs(row["terminal_z"]) >= abs(row["transient_z"]):
                return row["terminal_rank"]
            else:
                return row["transient_rank"]

        df["relevant_rank"] = df.apply(get_relevant_rank, axis=1)

        # get unique classes
        classes = df["tf_class"].unique()

        # dictionary to store top k tfs for each class
        top_tfs_dict = {}

        for class_name in classes:
            class_df = df[df["tf_class"] == class_name].copy()
            # sort by relevant rank and take top k
            top_k = class_df.nsmallest(k, "relevant_rank")
            # extract tf names (assuming index contains tf names)
            top_tfs_dict[class_name] = top_k.index.tolist()

        # create result dataframe with classes as columns
        # pad shorter lists with None to make all columns same length
        max_len = max(len(v) for v in top_tfs_dict.values())

        for class_name in top_tfs_dict:
            while len(top_tfs_dict[class_name]) < max_len:
                top_tfs_dict[class_name].append(None)

        result_df = pd.DataFrame(top_tfs_dict)

        return result_df

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
