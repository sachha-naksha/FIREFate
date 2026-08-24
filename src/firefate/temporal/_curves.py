"""Gaussian-smoothed expression, regulation and force curves over pseudotime."""
from __future__ import annotations

from multiprocessing import Pool, cpu_count
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Union
import dictys
import numpy as np
import pandas as pd
from dictys.net import stat
from numpy.typing import NDArray
from scipy import stats
from firefate.backends.dictys._stats import lcpm_tf
from firefate.utils.genes import get_gene_indices, get_tf_indices
import matplotlib
import matplotlib.pyplot as plt
import math
from typing import Any, Dict, Optional, Tuple, Union
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


# ---------------------------------------------------------------------------
# Figures: expression and regulation curves
# ---------------------------------------------------------------------------

def plot_expression_for_multiple_genes(
    targets_in_lf, lcpm_dcurve, dtime, ncols=3, figsize=(18, 15)
):
    """
    Plots expression curves for multiple target genes in a single figure.
    """
    # Calculate number of rows needed
    n_targets = len(targets_in_lf)
    nrows = math.ceil(n_targets / ncols)

    # Create figure and subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_targets > 1 else [axes]  # Handle case of single subplot

    # Loop through each target gene
    for i, gene in enumerate(targets_in_lf):
        ax = axes[i]

        # Check if gene exists in lcpm_dcurve
        if gene in lcpm_dcurve.index:
            # Plot expression curve
            line = ax.plot(dtime, lcpm_dcurve.loc[gene], linewidth=2, color="green")

            # Add label at the end of the line
            ax.text(
                dtime.iloc[-1],
                lcpm_dcurve.loc[gene].iloc[-1],
                f" {gene}",
                color="green",
                verticalalignment="center",
            )

            # Set title and labels
            ax.set_title(gene)
            ax.set_xlabel("Pseudotime")
            ax.set_ylabel("Log CPM")

            # Remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        else:
            ax.text(
                0.5,
                0.5,
                f"{gene} not found",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.axis("off")
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    # Adjust layout
    plt.tight_layout()
    plt.suptitle("Expression Curves for Target Genes", fontsize=16, y=1.02)
    return fig


def plot_gene_expression_subplots(gene_list, lcpm_data, time_data, 
                                 figsize=(15, 10), color='#0077b6', 
                                 ncols=3, save_path=None):
    """
    Plot expression trajectories for a list of genes in separate subplots.
    """
    
    # Filter genes that are actually in the data
    available_genes = [gene for gene in gene_list if gene in lcpm_data.index]
    missing_genes = [gene for gene in gene_list if gene not in lcpm_data.index]
    
    if missing_genes:
        print(f"Warning: The following genes were not found in the data: {missing_genes}")
    
    if not available_genes:
        print("No genes found in the data!")
        return None
    
    # Calculate subplot dimensions
    n_genes = len(available_genes)
    nrows = (n_genes + ncols - 1) // ncols  # Ceiling division
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Handle case where we have only one row or column
    if nrows == 1:
        axes = axes.reshape(1, -1) if n_genes > 1 else [axes]
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.flatten() if n_genes > 1 else [axes]
    
    # Plot each gene
    for i, gene in enumerate(available_genes):
        if nrows == 1 and ncols == 1:
            ax = axes
        elif nrows == 1:
            ax = axes[i]
        else:
            ax = axes[i] if n_genes > 1 else axes
            
        # Plot expression trajectory
        ax.plot(time_data, lcpm_data.loc[gene], linewidth=2, color=color)
        
        # Formatting
        ax.set_title(gene, fontsize=12, fontweight='bold')
        ax.set_ylabel('Log CPM', fontsize=10)
        ax.set_xlabel('Time', fontsize=10)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    if n_genes < len(axes):
        for i in range(n_genes, len(axes)):
            if nrows == 1:
                axes[i].set_visible(False)
            else:
                axes[i].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format='pdf')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    return fig


def fig_regulation_heatmap(
    network: dictys.net.dynamic_network,
    start: int,
    stop: int,
    regulations: list[Tuple[str, str]],
    num: int = 100,
    dist: float = 1.5,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    figsize: Tuple[float, float] = (2, 0.15),
    vmax: Optional[float] = None,
) -> Tuple[
    matplotlib.pyplot.Figure, matplotlib.axes.Axes, matplotlib.cm.ScalarMappable
]:
    """
    Draws pseudo-time dependent heatmap of regulation strengths without clustering.
    """
    # Get dynamic network edge strength
    pts, fsmooth = network.linspace(start, stop, num, dist)
    stat1_net = fsmooth(stat.net(network))
    stat1_x = stat.pseudotime(network, pts)
    tmp = stat1_x.compute(pts)[0]
    dx = pd.Series(tmp)
    # Test regulation existence and extract regulations
    tdict = [dict(zip(x, range(len(x)))) for x in stat1_net.names]
    t1 = [[x[y] for x in regulations if x[y] not in tdict[y]] for y in range(2)]
    if len(t1[0]) > 0:
        raise ValueError(
            "Regulator gene(s) {} not found in network.".format("/".join(t1[0]))
        )
    if len(t1[1]) > 0:
        raise ValueError(
            "Target gene(s) {} not found in network.".format("/".join(t1[1]))
        )
    # Extract regulations to draw
    dnet = stat1_net.compute(pts)
    t1 = np.array([[tdict[0][x[0]], tdict[1][x[1]]] for x in regulations]).T
    dnet = dnet[t1[0], t1[1]]
    # Create figure and axes
    if ax is None:
        figsize = (figsize[0], figsize[1] * dnet.shape[0])
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        if figsize is not None:
            raise ValueError("figsize should not be set if ax is set.")
        fig = ax.get_figure()
        figsize = fig.get_size_inches()
    aspect = (figsize[1] / dnet.shape[0]) / (figsize[0] / dnet.shape[1])
    # Determine and apply colormap
    if isinstance(cmap, str):
        if vmax is None:
            vmax = np.quantile(np.abs(dnet).ravel(), 0.95)
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax), cmap=cmap
        )
    elif vmax is not None:
        raise ValueError(
            "vmax should not be set if cmap is a matplotlib.cm.ScalarMappable."
        )
    if hasattr(cmap, "to_rgba"):
        im = ax.imshow(cmap.to_rgba(dnet), aspect=aspect, interpolation="none")
    else:
        im = ax.imshow(dnet, aspect=aspect, interpolation="none", cmap=cmap)
        plt.colorbar(im, label="Regulation strength")
    # Set pseudotime labels as x axis labels
    ax.set_xlabel("Pseudotime")
    num_ticks = 10
    tick_positions = np.linspace(0, dnet.shape[1] - 1, num_ticks, dtype=int)
    tick_labels = dx.iloc[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{x:.6f}" for x in tick_labels], rotation=45, ha="right")
    # Set regulation pair labels
    ax.set_yticks(list(range(len(regulations))))
    ax.set_yticklabels(["-".join(x) for x in regulations])
    # Add grid lines to separate rows
    ax.set_yticks(np.arange(dnet.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    return fig, ax, dnet


def fig_expression_gradient_heatmap(
    network: dictys.net.dynamic_network,
    start: int,
    stop: int,
    genes_or_regulations: Union[list[str], list[Tuple[str, str]]],
    num: int = 100,
    dist: float = 1.5,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    figsize: Tuple[float, float] = (2, 0.15),
) -> Tuple[
    matplotlib.pyplot.Figure, matplotlib.axes.Axes, matplotlib.cm.ScalarMappable
]:
    """
    Draws pseudo-time dependent heatmap of expression gradients.
    """
    # Get expression data
    pts, fsmooth = network.linspace(start, stop, num, dist)
    stat1_y = fsmooth(stat.lcpm(network, cut=0))
    stat1_x = stat.pseudotime(network, pts)
    dy = pd.DataFrame(stat1_y.compute(pts), index=stat1_y.names[0])
    dx = pd.Series(
        stat1_x.compute(pts)[0]
    )  # gene1's pseudotime is used as all genes have the same pseudotime
    # Determine if input is gene list or regulation list
    if isinstance(genes_or_regulations[0], tuple):
        # Extract target genes from regulations
        target_genes = [target for _, target in genes_or_regulations]
        # Remove duplicates while preserving order
        target_genes = list(dict.fromkeys(target_genes))
    else:
        # Use gene list directly
        target_genes = list(dict.fromkeys(genes_or_regulations))
    # Calculate gradients for target genes
    gradients = np.vstack(
        [np.gradient(dy.loc[gene].values, dx.values) for gene in target_genes]
    )
    # Create figure and axes
    if ax is None:
        figsize = (figsize[0], figsize[1] * len(target_genes))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
        figsize = fig.get_size_inches()
    aspect = (figsize[1] / len(target_genes)) / (figsize[0] / gradients.shape[1])
    # Determine and apply colormap
    if isinstance(cmap, str):
        vmax = np.quantile(np.abs(gradients).ravel(), 0.95)
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax), cmap=cmap
        )
    if hasattr(cmap, "to_rgba"):
        im = ax.imshow(cmap.to_rgba(gradients), aspect=aspect, interpolation="none")
    else:
        im = ax.imshow(gradients, aspect=aspect, interpolation="none", cmap=cmap)
        plt.colorbar(im, label="Expression gradient (Δ Log CPM/Δ Pseudotime)")
    # Set pseudotime labels as x axis labels
    ax.set_xlabel("Pseudotime")
    num_ticks = 10
    tick_positions = np.linspace(0, gradients.shape[1] - 1, num_ticks, dtype=int)
    tick_labels = dx.iloc[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{x:.6f}" for x in tick_labels], rotation=45, ha="right")
    # Set target gene labels
    ax.set_yticks(list(range(len(target_genes))))
    ax.set_yticklabels(target_genes)
    # Add grid lines to separate rows
    ax.set_yticks(np.arange(len(target_genes) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    return fig, ax, cmap


def fig_expression_linear_heatmap(
    network: dictys.net.dynamic_network,
    start: int,
    stop: int,
    genes_or_regulations: Union[list[str], list[Tuple[str, str]]],
    num: int = 100,
    dist: float = 1.5,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    figsize: Tuple[float, float] = (2, 0.15),
) -> Tuple[matplotlib.pyplot.Figure, matplotlib.axes.Axes, matplotlib.cm.ScalarMappable]:
    """
    Draws pseudo-time dependent heatmap of log2 CPM expression values.
    """
    # Get expression data
    pts, fsmooth = network.linspace(start, stop, num, dist)
    stat1_y = fsmooth(stat.lcpm(network, cut=0))
    stat1_x = stat.pseudotime(network, pts)

    # Get log2 CPM values
    dy = pd.DataFrame(stat1_y.compute(pts), index=stat1_y.names[0])
    dx = pd.Series(stat1_x.compute(pts)[0])
    dy_linear = dy.apply(lambda x: 2**x - 1)

    # Get target genes
    if isinstance(genes_or_regulations[0], tuple):
        target_genes = [target for _, target in genes_or_regulations]
        target_genes = list(dict.fromkeys(target_genes))
    else:
        target_genes = list(dict.fromkeys(genes_or_regulations))

    # Stack expression values
    expression_matrix = np.vstack([dy_linear.loc[gene].values for gene in target_genes])

    # Create figure and axes
    if ax is None:
        figsize = (figsize[0], figsize[1] * len(target_genes))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Create colormap
    if isinstance(cmap, str):
        vmax = np.quantile(expression_matrix.ravel(), 0.95)
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=0, vmax=vmax), cmap=cmap
        )

    # Create heatmap with auto aspect
    ax.imshow(
        cmap.to_rgba(expression_matrix),
        aspect='auto',          # ← key fix
        interpolation="none"
    )

    # Colorbar sized to match heatmap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(cmap, cax=cax)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Expression (log2 CPM)", fontsize=8)

    # Set pseudotime labels
    ax.set_xlabel("Pseudotime")
    num_ticks = 10
    tick_positions = np.linspace(0, expression_matrix.shape[1] - 1, num_ticks, dtype=int)
    tick_labels = dx.iloc[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{x:.3f}" for x in tick_labels], rotation=45, ha="right")

    # Set gene labels
    ax.set_yticks(list(range(len(target_genes))))
    ax.set_yticklabels(target_genes)

    # Add grid lines
    ax.set_yticks(np.arange(len(target_genes) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)

    return fig, ax, cmap, cb
