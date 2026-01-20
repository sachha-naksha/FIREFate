import gc
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Any, Optional, Tuple, Union

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

##################################### Data retrieval ############################################


def get_tf_indices(dictys_dynamic_object, tf_list):
    """
    Get the indices of transcription factors from a list, if present in ndict and nids[0].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    tf_mappings_to_gene_hashmap = dictys_dynamic_object.nids[0]
    tf_indices = []
    tf_gene_indices = []
    missing_tfs = []
    for gene in tf_list:
        # Check if the gene is in the gene_hashmap
        if gene in gene_hashmap:
            gene_index = gene_hashmap[gene]  # Get the index in gene_hashmap
            # Check if the gene index is present in tf_mappings_to_gene_hashmap
            match = np.where(tf_mappings_to_gene_hashmap == gene_index)[0]
            if match.size > 0:  # If a match is found
                tf_indices.append(int(match[0]))  # Append the position of the match
                tf_gene_indices.append(int(gene_index))  # Also append the gene index
            else:
                missing_tfs.append(gene)  # Gene exists but not as a TF
        else:
            missing_tfs.append(gene)  # Gene not found at all
    return tf_indices, tf_gene_indices, missing_tfs


def get_gene_indices(dictys_dynamic_object, gene_list):
    """
    Get the indices of target genes from a list, if present in ndict and nids[1].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    gene_indices = []
    for gene in gene_list:
        if gene in gene_hashmap:
            gene_indices.append(gene_hashmap[gene])
    return gene_indices


def check_if_gene_in_ndict(dictys_dynamic_object, gene_name, return_index=False):
    """
    Check if a gene is in the ndict of the dynamic object.
    """
    # Input validation
    if not hasattr(dictys_dynamic_object, "ndict"):
        raise AttributeError("Dynamic object does not have ndict attribute")
    # Handle single gene case
    if isinstance(gene_name, str):
        is_present = gene_name in dictys_dynamic_object.ndict
        if return_index:
            return dictys_dynamic_object.ndict.get(gene_name, None)
        return is_present
    # Handle list of genes case
    elif isinstance(gene_name, (list, tuple, set)):
        results = {
            "present": [],
            "missing": [],
            "indices": {} if return_index else None,
        }
        for gene in gene_name:
            if gene in dictys_dynamic_object.ndict:
                results["present"].append(gene)
                if return_index:
                    results["indices"][gene] = dictys_dynamic_object.ndict[gene]
            else:
                results["missing"].append(gene)
        # Add summary statistics
        results["stats"] = {
            "total_genes": len(gene_name),
            "found": len(results["present"]),
            "missing": len(results["missing"]),
            "percent_found": (len(results["present"]) / len(gene_name) * 100),
        }
        return results
    else:
        raise TypeError("gene_name must be a string or a list-like object of strings")


def curvature_of_expression(dcurve: pd.DataFrame, dtime: pd.Series):
    """
    Calculate the curvature of expression curves.
    """
    # First derivative (dx/dt)
    dx_dt = pd.DataFrame(
        np.gradient(dcurve, dtime, axis=1), index=dcurve.index, columns=dcurve.columns
    )
    # Second derivative (d2x/dt2)
    d2x_dt2 = pd.DataFrame(
        np.gradient(dx_dt, dtime, axis=1), index=dcurve.index, columns=dcurve.columns
    )
    return d2x_dt2

    
##################################### Window labels ############################################


def get_state_labels_in_window(dictys_dynamic_object, cell_labels):
    """
    Creates a mapping of window indices to their constituent cells' labels
    """
    # get cell assignment matrix from dictys_dynamic_object.prop['sc']['w']
    cell_assignment_matrix = dictys_dynamic_object.prop["sc"]["w"]
    state_labels_in_window = {}
    for window_idx in range(cell_assignment_matrix.shape[0]):
        indices_of_cells_present_in_window = np.where(
            cell_assignment_matrix[window_idx] == 1
        )[0] #these indices start from 0
        state_labels_in_window[window_idx] = [
            cell_labels[idx] for idx in indices_of_cells_present_in_window
        ]
    return state_labels_in_window


def get_state_total_counts(cell_labels):
    """
    Get total number of cells for each state in the dataset
    """
    state_counts = {}
    for label in cell_labels:
        state_counts[label] = state_counts.get(label, 0) + 1
    return state_counts


def get_top_k_fraction_labels(dictys_dynamic_object, window_idx, cell_labels, k=2):
    """
    Returns the k labels with both fractions for a given window
    """
    # Get state labels for all windows
    state_labels_dict = get_state_labels_in_window(dictys_dynamic_object, cell_labels)
    # Get window labels for specified window
    window_labels = state_labels_dict[window_idx]
    # Get total counts across all states
    state_total_counts = get_state_total_counts(cell_labels)
    # Count cells per state in this window
    window_counts = {}
    for label in window_labels:
        window_counts[label] = window_counts.get(label, 0) + 1
    total_cells_in_window = len(window_labels)
    # Calculate both fractions for each state
    state_metrics = {}
    for state in window_counts:
        window_composition = window_counts[state] / total_cells_in_window
        state_distribution = window_counts[state] / state_total_counts[state]
        state_metrics[state] = (window_composition, state_distribution)
    # Sort primarily by state_distribution, then by window_composition
    sorted_states = sorted(
        state_metrics.items(), key=lambda x: (x[1][1], x[1][0]), reverse=True
    )
    return sorted_states[:k]


def window_labels_to_count_df(window_labels_dict):
    """
    Converts a dictionary of window indices to cell labels into a DataFrame
    with counts of each label per window.
    """
    from collections import Counter

    import pandas as pd

    # Get all unique labels
    all_labels = set()
    for labels in window_labels_dict.values():
        all_labels.update(labels)

    # Sort labels for consistency
    all_labels = sorted(all_labels)

    # Get all window indices
    window_indices = sorted(window_labels_dict.keys())

    # Initialize DataFrame with zeros
    count_df = pd.DataFrame(0, index=all_labels, columns=window_indices)

    # Fill in the counts for each window
    for window_idx, labels in window_labels_dict.items():
        # Count occurrences of each label in this window
        label_counts = Counter(labels)

        # Update the DataFrame
        for label, count in label_counts.items():
            count_df.loc[label, window_idx] = count

    return count_df


##################################### Plotting ############################################

def create_enriched_links_per_state(enriched_links, state_LF):
    # Create dictionaries to map genes to their states
    gene_to_state = dict(zip(state_LF['gene'], state_LF['color']))

    # Initialize lists to store links for each state
    state1_links = []  # For Red (PB)
    state2_links = []  # For Blue (GC)
    TFs = set[Any]()
    targets_in_lf = set[Any]()

    # Iterate over each row in the enriched_links DataFrame
    for _, row in enriched_links.iterrows():
        tf_str = row['TF']
        # Extract the TF name from the string representation of a tuple
        tf = tf_str.strip("(,)' ").replace("'", "")
        TFs.add(tf)
        
        # Handle the targets as a string representation of a list
        if isinstance(row['common'], str):
            # If it's a string representation of a list, convert it to a list
            targets_str = row['common'].strip("[]").replace("'", "")
            targets = [t.strip() for t in targets_str.split(",")]
        else:
            # If it's already a list
            targets = row['common']
    
        # Assign each TF-target pair to the appropriate state
        for target in targets:
            if target and target in gene_to_state:
                targets_in_lf.add(target)
                state = gene_to_state[target]
                link = (tf, target)
                if state == 'Red':
                    state1_links.append(link)
                elif state == 'Blue':
                    state2_links.append(link)
    TFs = list(TFs)
    targets_in_lf = list(targets_in_lf)

    return state1_links, state2_links, TFs, targets_in_lf

def extract_tf_gene_info(enriched_links_df):
    """
    Extract TF-Gene links and unique TF/Gene names from enriched links dataframe.
    """
    
    # Validate input
    if enriched_links_df is None or enriched_links_df.empty:
        return [], [], []
    
    required_cols = ['TF', 'Gene']
    for col in required_cols:
        if col not in enriched_links_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Remove any rows with NaN values in TF or Gene columns
    clean_df = enriched_links_df.dropna(subset=['TF', 'Gene'])
    
    if clean_df.empty:
        return [], [], []
    
    # Extract links as list of tuples
    links_list = list(zip(clean_df['TF'], clean_df['Gene']))
    
    # Get unique TF names (sorted for consistency)
    unique_tfs = sorted(clean_df['TF'].unique().tolist())
    
    # Get unique gene names (sorted for consistency)
    unique_genes = sorted(clean_df['Gene'].unique().tolist())
    
    return links_list, unique_tfs, unique_genes
   
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


def cluster_heatmap(
    d,
    optimal_ordering=True,
    method="ward",
    metric="euclidean",
    dshow=None,
    fig=None,
    cmap="coolwarm",
    aspect=0.1,
    figscale=0.02,
    dtop=0.3,
    dright=0,
    wcolorbar=0.03,
    wedge=0.03,
    xselect=None,
    yselect=None,
    xtick=False,
    ytick=True,
    vmin=None,
    vmax=None,
    inverty=True,
):
    """
    Draw 2-D hierachical clustering of pandas.DataFrame, with optional hierachical clustering on both axes.
    X/Y axis of figure corresponds to columns/rows of the dataframe.
    d:		Pandas.DataFrame 2D data with index & column names for clustering.
    optimal_ordering: passed to scipy.cluster.hierarchy.dendrogram
    method: Method of hierarchical clustering, passed to scipy.cluster.hierarchy.linkage.
                    Accepts single strs or a tuple of two strings for different method options for x and y.
    metric:	Metric to compute pariwise distance for clustering, passed to scipy.spatial.distance.pdist.
                    Accepts single strs or a tuple of two strings for different metric options for x and y.
    dshow:	Pandas.DataFrame 2D data with index & column names to draw heatmap. Defaults to d.
    fig:	Figure to plot on.
    cmap:	Colormap
    aspect:	Aspect ratio
    figscale:	Scale of figure compared to font.
    dtop,
    dright:	Top and right dendrogram size. Value from 0 to 1 values as proportion.
                    If 0, do not cluster on given axis.
    wcolorbar: Width of colorbar. Value from 0 to 1 values as proportion.
    wedge:	Width of edges and between colorbar and main figure.
                    Value from 0 to 1 values as proportion.
    xselect,
    yselect:np.array(bool) of coordinates to draw. Current only selected coordinates are used for clustering.
    xtick,
    ytick:	Whether to show ticks.
    vmin,
    vmax:	Minimum/maximum values of heatmap.
    inverty:Whether to invert direction of y.
    Return:
    figure:	plt.Figure drawn on
    x:		column IDs included
    y:		index IDs included.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram, linkage

    assert isinstance(xtick, bool) or (
        isinstance(xtick, list) and len(xtick) == d.shape[1]
    )
    assert isinstance(ytick, bool) or (
        isinstance(ytick, list) and len(ytick) == d.shape[0]
    )
    if isinstance(method, str):
        method = [method, method]
    if len(method) != 2:
        raise ValueError(
            'Parameter "method" must have size 2 for x and y respectively.'
        )
    if isinstance(metric, str):
        metric = [metric, metric]
    if metric is not None and len(metric) != 2:
        raise ValueError(
            'Parameter "metric" must have size 2 for x and y respectively.'
        )
    if metric is None:
        assert d.ndim == 2 and d.shape[0] == d.shape[1]
        assert (d.index == d.columns).all()
        assert method[0] == method[1]
        if xselect is not None:
            assert yselect is not None
            assert (xselect == yselect).all()
        else:
            assert yselect is None
    if dshow is None:
        dshow = d
    assert (
        dshow.shape == d.shape
        and (dshow.index == d.index).all()
        and (dshow.columns == d.columns).all()
    )
    xt0 = d.columns if isinstance(xtick, bool) else xtick
    yt0 = d.index if isinstance(ytick, bool) else ytick
    # Genes to highlight
    d2 = d.copy()
    if xselect is not None:
        d2 = d2.loc[:, xselect]
        dshow = dshow.loc[:, xselect]
        xt0 = [xt0[x] for x in np.nonzero(xselect)[0]]
    if yselect is not None:
        d2 = d2.loc[yselect]
        dshow = dshow.loc[yselect]
        yt0 = [yt0[x] for x in np.nonzero(yselect)[0]]

    wtop = dtop / (1 + d2.shape[0] / 8)
    wright = dright / (1 + d2.shape[1] * aspect / 8)
    iscolorbar = wcolorbar > 0
    t1 = np.array(d2.T.shape)
    t1 = t1 * figscale
    t1[1] /= aspect
    t1[1] /= 1 - wedge * 2 - wtop
    t1[0] /= 1 - wedge * (2 + iscolorbar) - wright - wcolorbar
    if fig is None:
        fig = plt.figure(figsize=t1)
    d3 = dshow.copy()
    if metric is not None:
        # Right dendrogram
        if dright > 0:
            ax1 = fig.add_axes(
                [
                    1 - wedge * (1 + iscolorbar) - wright - wcolorbar,
                    wedge,
                    wright,
                    1 - 2 * wedge - wtop,
                ]
            )
            tl1 = linkage(
                d2,
                method=method[1],
                metric=metric[1],
                optimal_ordering=optimal_ordering,
            )
            td1 = dendrogram(tl1, orientation="right")
            ax1.set_xticks([])
            ax1.set_yticks([])
            d3 = d3.iloc[td1["leaves"], :]
            yt0 = [yt0[x] for x in td1["leaves"]]
        else:
            ax1 = None
        # Top dendrogram
        if dtop > 0:
            ax2 = fig.add_axes(
                [
                    wedge,
                    1 - wedge - wtop,
                    1 - wedge * (2 + iscolorbar) - wright - wcolorbar,
                    wtop,
                ]
            )
            tl2 = linkage(
                d2.T,
                method=method[0],
                metric=metric[0],
                optimal_ordering=optimal_ordering,
            )
            td2 = dendrogram(tl2)
            ax2.set_xticks([])
            ax2.set_yticks([])
            d3 = d3.iloc[:, td2["leaves"]]
            xt0 = [xt0[x] for x in td2["leaves"]]
        else:
            ax2 = None
    else:
        if dright > 0 or dtop > 0:
            from scipy.spatial.distance import squareform

            tl1 = linkage(
                squareform(d2), method=method[0], optimal_ordering=optimal_ordering
            )
            # Right dendrogram
            if dright > 0:
                ax1 = fig.add_axes(
                    [
                        1 - wedge * (1 + iscolorbar) - wright - wcolorbar,
                        wedge,
                        wright,
                        1 - 2 * wedge - wtop,
                    ]
                )
                td1 = dendrogram(tl1, orientation="right")
                ax1.set_xticks([])
                ax1.set_yticks([])
            else:
                ax1 = None
                td1 = None
            # Top dendrogram
            if dtop > 0:
                ax2 = fig.add_axes(
                    [
                        wedge,
                        1 - wedge - wtop,
                        1 - wedge * (2 + iscolorbar) - wright - wcolorbar,
                        wtop,
                    ]
                )
                td2 = dendrogram(tl1)
                ax2.set_xticks([])
                ax2.set_yticks([])
            else:
                ax2 = None
                td2 = None
            td0 = td1["leaves"] if td1 is not None else td2["leaves"]
            d3 = d3.iloc[td0, :].iloc[:, td0]
            xt0, yt0 = [[y[x] for x in td0] for y in [xt0, yt0]]
    axmatrix = fig.add_axes(
        [
            wedge,
            wedge,
            1 - wedge * (2 + iscolorbar) - wright - wcolorbar,
            1 - 2 * wedge - wtop,
        ]
    )
    ka = {"aspect": 1 / aspect, "origin": "lower", "cmap": cmap}
    if vmin is not None:
        ka["vmin"] = vmin
    if vmax is not None:
        ka["vmax"] = vmax
    im = axmatrix.matshow(d3, **ka)
    if not isinstance(xtick, bool) or xtick:
        t1 = list(zip(range(d3.shape[1]), xt0))
        t1 = list(zip(*list(filter(lambda x: x[1] is not None, t1))))
        axmatrix.set_xticks(t1[0])
        axmatrix.set_xticklabels(t1[1], minor=False, rotation=90)
    else:
        axmatrix.set_xticks([])
    if not isinstance(ytick, bool) or ytick:
        t1 = list(zip(range(d3.shape[0]), yt0))
        t1 = list(zip(*list(filter(lambda x: x[1] is not None, t1))))
        axmatrix.set_yticks(t1[0])
        axmatrix.set_yticklabels(t1[1], minor=False)
    else:
        axmatrix.set_yticks([])
    axmatrix.tick_params(
        top=False,
        bottom=True,
        labeltop=False,
        labelbottom=True,
        left=True,
        labelleft=True,
        right=False,
        labelright=False,
    )
    if inverty:
        if ax1 is not None:
            ax1.set_ylim(ax1.get_ylim()[::-1])
        axmatrix.set_ylim(axmatrix.get_ylim()[::-1])
    if wcolorbar > 0:
        cax = fig.add_axes(
            [1 - wedge - wcolorbar, wedge, wcolorbar, 1 - 2 * wedge - wtop]
        )
        fig.colorbar(im, cax=cax)
    return fig, d3.columns, d3.index


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
) -> Tuple[
    matplotlib.pyplot.Figure, matplotlib.axes.Axes, matplotlib.cm.ScalarMappable
]:
    """
    Draws pseudo-time dependent heatmap of linear expression values.
    """
    # Get expression data
    pts, fsmooth = network.linspace(start, stop, num, dist)
    stat1_y = fsmooth(stat.lcpm(network, cut=0))
    stat1_x = stat.pseudotime(network, pts)
    # Get log2 expression values and convert to linear
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
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
        figsize = fig.get_size_inches()
    aspect = (figsize[1] / len(target_genes)) / (
        figsize[0] / expression_matrix.shape[1]
    )
    # Create heatmap
    if isinstance(cmap, str):
        vmax = np.quantile(expression_matrix.ravel(), 0.95)
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=0, vmax=vmax), cmap=cmap
        )
    if hasattr(cmap, "to_rgba"):
        im = ax.imshow(
            cmap.to_rgba(expression_matrix), aspect=aspect, interpolation="none"
        )
    else:
        im = ax.imshow(
            expression_matrix, aspect=aspect, interpolation="none", cmap=cmap
        )
        plt.colorbar(im, label="Expression (CPM)")
    # Set pseudotime labels with rounded values
    ax.set_xlabel("Pseudotime")
    num_ticks = 10
    tick_positions = np.linspace(
        0, expression_matrix.shape[1] - 1, num_ticks, dtype=int
    )
    tick_labels = dx.iloc[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [f"{x:.3f}" for x in tick_labels], rotation=45, ha="right"
    )  # Round to 1 decimal place
    # Set gene labels
    ax.set_yticks(list(range(len(target_genes))))
    ax.set_yticklabels(target_genes)
    # Add grid lines
    ax.set_yticks(np.arange(len(target_genes) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    return fig, ax, cmap


def plot_force_heatmap(
    force_df: pd.DataFrame,
    dtime: pd.Series,
    regulations=None,
    tf_to_targets_dict=None,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    figsize: Tuple[float, float] = (10, 4),
    vmax: Optional[float] = None,
) -> Tuple[matplotlib.pyplot.Figure, matplotlib.axes.Axes, np.ndarray]:
    """
    Draws pseudo-time dependent heatmap of force values.
    """
    # Process input parameters to generate regulation pairs
    reg_pairs = []
    reg_labels = []
    # Case 1: Dictionary of TF -> targets provided
    if tf_to_targets_dict is not None:
        for tf, targets in tf_to_targets_dict.items():
            for target in targets:
                reg_pairs.append((tf, target))
                reg_labels.append(f"{tf}->{target}")
    # Case 2: List of regulation pairs or list of targets for a single TF
    elif regulations is not None:
        # Check if first item is a string (target) or tuple/list (regulation pair)
        if regulations and isinstance(regulations[0], str):
            # It's a list of targets for a single TF
            # Extract TF name from the calling context (not ideal but works for the notebook)
            for key, value in locals().items():
                if (
                    isinstance(value, dict)
                    and "PRDM1" in value
                    and value["PRDM1"] == regulations
                ):
                    tf = "PRDM1"  # Found the TF
                    break
            else:
                # If we can't determine the TF, use the first item in regulations as TF
                # and the rest as targets (this is a fallback and might not be correct)
                tf = regulations[0]
                regulations = regulations[1:]

            for target in regulations:
                reg_pairs.append((tf, target))
                reg_labels.append(f"{tf}->{target}")
        else:
            # It's a list of regulation pairs
            reg_pairs = regulations
            reg_labels = [f"{tf}->{target}" for tf, target in regulations]
    # If no regulations provided, use non-zero regulations from force_df
    if not reg_pairs:
        non_zero_mask = (force_df != 0).any(axis=1)
        force_df_filtered = force_df[non_zero_mask]
        reg_pairs = list(force_df_filtered.index)
        reg_labels = [f"{tf}->{target}" for tf, target in reg_pairs]
    # Extract force values for the specified regulations
    force_values = []
    for pair in reg_pairs:
        tf, target = pair
        try:
            force_values.append(force_df.loc[(tf, target)].values)
        except KeyError:
            raise ValueError(f"Regulation {tf}->{target} not found in force DataFrame")
    # Convert to numpy array
    dnet = np.array(force_values)
    # Create figure and axes
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
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
        im = ax.imshow(cmap.to_rgba(dnet), aspect="auto", interpolation="none")
    else:
        im = ax.imshow(dnet, aspect="auto", interpolation="none", cmap=cmap)
        plt.colorbar(im, label="Force")
    # Set pseudotime labels
    ax.set_xlabel("Pseudotime")
    num_ticks = 10
    tick_positions = np.linspace(0, dnet.shape[1] - 1, num_ticks, dtype=int)
    tick_labels = dtime.iloc[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{x:.4f}" for x in tick_labels], rotation=45, ha="right")
    # Set regulation pair labels
    ax.set_yticks(list(range(len(reg_labels))))
    ax.set_yticklabels(reg_labels)
    # Add grid lines
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    return fig, ax, dnet

def plot_force_heatmap_with_clustering(
    force_df: pd.DataFrame,
    dtime: pd.Series,
    regulations=None,
    tf_to_targets_dict=None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 8),
    plot_figure: bool = True,
    perform_clustering: bool = True,
    cluster_method: str = "ward",
    dtop: float = 0,
    dright: float = 0.3,
    row_scaling: dict = None,  # New parameter for scaling specific rows
) -> Tuple[pd.DataFrame, list, pd.Series, Optional[matplotlib.figure.Figure]]:
    """
    Prepares force value data for clustering heatmap and optionally plots it.
    
    Parameters:
    -----------
    row_scaling: Dict[Tuple[str, str], float]
        Dictionary mapping (TF, target) tuples to scaling factors.
        Example: {('IRF4', 'PRDM1'): 0.5} will scale that specific link to 50% of its original values.
    """
    # Process input parameters to generate regulation pairs
    reg_pairs = []
    reg_labels = []
    # Case 1: Dictionary of TF -> targets provided
    if tf_to_targets_dict is not None:
        for tf, targets in tf_to_targets_dict.items():
            for target in targets:
                reg_pairs.append((tf, target))
                reg_labels.append(f"{tf}->{target}")
    # Case 2: List of regulation pairs or list of targets for a single TF
    elif regulations is not None:
        # Check if first item is a string (target) or tuple/list (regulation pair)
        if regulations and isinstance(regulations[0], str):
            # It's a list of targets for a single TF
            # Extract TF name from the calling context (not ideal but works for the notebook)
            for key, value in locals().items():
                if (
                    isinstance(value, dict)
                    and "PRDM1" in value
                    and value["PRDM1"] == regulations
                ):
                    tf = "PRDM1"  # Found the TF
                    break
            else:
                # If we can't determine the TF, use the first item in regulations as TF
                # and the rest as targets (this is a fallback and might not be correct)
                tf = regulations[0]
                regulations = regulations[1:]

            for target in regulations:
                reg_pairs.append((tf, target))
                reg_labels.append(f"{tf}->{target}")
        else:
            # It's a list of regulation pairs
            reg_pairs = regulations
            reg_labels = [f"{tf}->{target}" for tf, target in regulations]
    # If no regulations provided, use non-zero regulations from force_df
    if not reg_pairs:
        non_zero_mask = (force_df != 0).any(axis=1)
        force_df_filtered = force_df[non_zero_mask]
        reg_pairs = list(force_df_filtered.index)
        reg_labels = [f"{tf}->{target}" for tf, target in reg_pairs]
    
    # Extract force values for the specified regulations
    force_values = []
    for pair in reg_pairs:
        tf, target = pair
        try:
            values = force_df.loc[(tf, target)].values
            
            # Apply scaling factor if provided for this pair
            if row_scaling and (tf, target) in row_scaling:
                scale_factor = row_scaling[(tf, target)]
                values = values * scale_factor
                
            force_values.append(values)
        except KeyError:
            raise ValueError(f"Regulation {tf}->{target} not found in force DataFrame")
    
    # Convert to numpy array
    dnet = np.array(force_values)
    
    # Convert dnet to DataFrame with proper labels
    force_df_for_cluster = pd.DataFrame(
        dnet, 
        index=reg_labels,
        columns=[f"{x:.4f}" for x in dtime]
    )
    
    # Plotting logic
    fig = None
    if plot_figure:
        # Calculate max absolute value for symmetric color scaling
        vmax_val = float(force_df_for_cluster.abs().max().max()) if vmax is None else vmax
        
        if perform_clustering:
            # Use cluster_heatmap for visualization
            fig, cols, rows = cluster_heatmap(
                d=force_df_for_cluster,
                optimal_ordering=True,
                method=cluster_method,
                metric="euclidean",
                cmap=cmap,
                aspect=0.1,
                figscale=0.02,
                dtop=dtop,      # Set to > 0 to enable clustering on columns (pseudotime)
                dright=dright,  # Set to > 0 to enable clustering on rows (regulations)
                wcolorbar=0.03,
                wedge=0.03,
                ytick=True,
                vmin=-vmax_val,
                vmax=vmax_val,
                figsize=figsize
            )
            plt.title("Clustered Force Heatmap")
        else:
            # Simple heatmap without clustering
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(dnet, aspect='auto', interpolation='none', cmap=cmap,
                          vmin=-vmax_val, vmax=vmax_val)
            
            # Add colorbar
            cbar = plt.colorbar(im, label="Force")
            
            # Set pseudotime labels as x axis labels
            ax.set_xlabel("Pseudotime")
            num_ticks = 10
            tick_positions = np.linspace(0, dnet.shape[1] - 1, num_ticks, dtype=int)
            tick_labels = dtime.iloc[tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{x:.4f}" for x in tick_labels], rotation=45, ha="right")
            
            # Set regulation pair labels
            ax.set_yticks(list(range(len(reg_labels))))
            ax.set_yticklabels(reg_labels)
            
            # Add grid lines
            ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
            
        plt.tight_layout()
    
    return force_df_for_cluster, reg_labels, dtime, fig

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

def create_pathway_color_scheme(pathway_labels, selected_genes):
    """
    Create a sophisticated color scheme for pathways that groups similar pathways
    and uses gradients for related categories.
    """
    
    # Define pathway groups with related categories
    pathway_groups = {
    'Protein Processing': [
        'Protein folding', 'Protein quality control', 
        'Protein glycosylation', 'Protein translocation'
    ],
    'Protein Secretion': [
        'Protein secretion/secretory pathway'
    ],
    'ER Stress': [  
        'ER stress', 'UPR'
    ],
    'Ubiquitination': [
        'Ubiquitination'
    ],
    'Immune System': [
        'Immune signaling', 'Antigen presentation', 'B-cell development', 
        'B-cell differentiation', 'Plasma cell differentiation'
    ],
    'Transcriptional Regulation': [
        'Transcriptional regulation', 'Transcriptional repression', 
        'Chromatin remodeling', 'Circadian rhythm'  
    ],
    'Cell Cycle & Growth': [
        'Cell cycle regulation', 'Growth factor signaling', 'RTK signaling'
    ],
    'Signaling Pathways': [
        'MAPK signaling', 'PI3K/AKT', 'cAMP signaling', 'TNF signaling', 
        'Glucocorticoid signaling', 'Glucose metabolism'  
    ],
    'Cellular Transport': [
        'Endocytosis', 'Vesicle-mediated transport', 'Ciliary/centrosome trafficking'
    ],
    'Metabolism & Processing': [
        'Metabolism', 'RNA processing'
    ],
    'Cell Structure': [
        'Cytoskeleton remodeling', 'Cell adhesion', 'Cell migration'
    ],
    'Stress Response': [
        'Apoptosis', 'Vascular remodeling'
    ],
    'Development': [
        'Bone/osteoblast differentiation'
    ]}    
    
    # Define base colors for each group (using distinct, well-separated colors)
        # Define base colors for each group (using distinct, well-separated colors)
    group_base_colors = {
        'Protein Processing': '#E31A1C',      # Red
        'Protein Secretion': '#FF7F00',      # Orange
        'ER Stress': '#D62728',              # Dark red
        'Ubiquitination': '#FF8C00',         # Dark orange
        'Immune System': '#1F78B4',          # Blue
        'Transcriptional Regulation': '#33A02C', # Green
        'Cell Cycle & Growth': '#FFFF33',    # Yellow
        'Signaling Pathways': '#6A3D9A',     # Purple
        'Cellular Transport': '#FF69B4',     # Hot pink (more distinct from brown)
        'Metabolism & Processing': '#A6CEE3', # Light blue
        'Cell Structure': '#B2DF8A',         # Light green
        'Stress Response': '#FDBF6F',        # Light orange
        'Development': '#CAB2D6'             # Light purple
    }
    
    # Create pathway to group mapping
    pathway_to_group = {}
    for group, pathways in pathway_groups.items():
        for pathway in pathways:
            pathway_to_group[pathway] = group
    
    # Extract pathways for selected genes
    gene_pathways = {}
    pathway_counts = {}
    
    for gene in selected_genes:
        if gene in pathway_labels:
            # Take the first pathway category (before comma if multiple)
            pathway = pathway_labels[gene].split(',')[0].strip()
            gene_pathways[gene] = pathway
            
            # Count occurrences for gradient assignment
            if pathway not in pathway_counts:
                pathway_counts[pathway] = 0
            pathway_counts[pathway] += 1
        else:
            gene_pathways[gene] = "Other"
            if "Other" not in pathway_counts:
                pathway_counts["Other"] = 0
            pathway_counts["Other"] += 1
    
    # Group pathways and assign colors with gradients
    import matplotlib.colors as mcolors
    pathway_color_map = {}
    
    for group, base_color in group_base_colors.items():
        # Find pathways in this group
        group_pathways = [p for p in pathway_counts.keys() 
                         if pathway_to_group.get(p, 'Other') == group]
        
        if group_pathways:
            if len(group_pathways) == 1:
                # Single pathway gets the base color
                pathway_color_map[group_pathways[0]] = base_color
            else:
                # Multiple pathways get gradient colors
                base_rgb = mcolors.hex2color(base_color)
                # Create lighter and darker versions
                gradients = []
                for i, pathway in enumerate(sorted(group_pathways)):
                    # Create gradient from darker to lighter
                    factor = 0.4 + (i / max(1, len(group_pathways) - 1)) * 0.6
                    rgb = tuple(min(1.0, c * factor + (1 - factor) * 0.9) for c in base_rgb)
                    gradients.append(mcolors.rgb2hex(rgb))
                
                for pathway, color in zip(sorted(group_pathways), gradients):
                    pathway_color_map[pathway] = color
    
    # Handle "Other" category
    if "Other" in pathway_counts:
        pathway_color_map["Other"] = '#CCCCCC'  # Gray for unclassified
    
    return gene_pathways, pathway_color_map, list(pathway_color_map.keys())

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

def plot_tf_gene_coregulation_heatmap(
    df_ep1, df_ep2, df_ep3, df_ep4,
    episode_labels=['Ep 1', 'Ep 2', 'Ep 3', 'Ep 4'],
    min_tf_episodes=1,
    min_gene_tfs=1,
    figsize=None,
    cmap_name="Blues",
    cluster_genes=True,
    cluster_tfs=True,
    show_values=True,
    value_fontsize=8,
    show_plot=True,
    pathway_labels=None  # Dictionary mapping genes to pathways
):
    """
    Creates a heatmap showing TF-gene co-regulation patterns across episodes with pathway annotations.
    """    
    # Helper function to parse genes
    def parse_genes_in_lf(genes_str):
        try:
            if pd.isna(genes_str) or genes_str == '' or genes_str == '()':
                return set()
            genes_tuple = ast.literal_eval(genes_str)
            return set(genes_tuple) if isinstance(genes_tuple, tuple) else {genes_tuple} if isinstance(genes_tuple, str) else set()
        except:
            return set()
    
    # Collect TF-gene relationships across all episodes
    tf_gene_episodes = {}  # (tf, gene) -> set of episodes
    
    dfs = [df_ep1, df_ep2, df_ep3, df_ep4]
    for ep_idx, (df_ep, ep_label) in enumerate(zip(dfs, episode_labels)):
        if df_ep is None or df_ep.empty:
            continue
            
        df_clean = df_ep.dropna(subset=['TF', 'genes_in_lf'])
        
        for _, row in df_clean.iterrows():
            tf_name = row['TF']
            genes_set = parse_genes_in_lf(row.get('genes_in_lf', ''))
            
            for gene in genes_set:
                key = (tf_name, gene)
                if key not in tf_gene_episodes:
                    tf_gene_episodes[key] = set()
                tf_gene_episodes[key].add(ep_idx)
    
    if not tf_gene_episodes:
        print("No TF-gene relationships found.")
        return None, None, None, None
    
    # Filter TFs and genes based on minimum criteria
    tf_episode_counts = {}
    gene_tf_counts = {}
    
    for (tf, gene), episodes in tf_gene_episodes.items():
        if tf not in tf_episode_counts:
            tf_episode_counts[tf] = set()
        tf_episode_counts[tf].update(episodes)
        
        if gene not in gene_tf_counts:
            gene_tf_counts[gene] = set()
        gene_tf_counts[gene].add(tf)
    
    # Filter TFs and genes
    selected_tfs = [tf for tf, episodes in tf_episode_counts.items() 
                   if len(episodes) >= min_tf_episodes]
    selected_genes = [gene for gene, tfs in gene_tf_counts.items() 
                     if len(tfs) >= min_gene_tfs]
    
    if not selected_tfs or not selected_genes:
        print(f"No TFs or genes meet the filtering criteria")
        return None, None, None, None
    
    # Create TF-gene matrix with episode counts
    tf_gene_matrix = np.zeros((len(selected_tfs), len(selected_genes)))
    
    for i, tf in enumerate(selected_tfs):
        for j, gene in enumerate(selected_genes):
            key = (tf, gene)
            if key in tf_gene_episodes:
                tf_gene_matrix[i, j] = len(tf_gene_episodes[key])
    
    # Process pathway annotations
    pathway_info = None
    if pathway_labels:
        gene_pathways, pathway_color_map, unique_pathways = create_pathway_color_scheme(
            pathway_labels, selected_genes)
        
        pathway_info = {
            'gene_pathways': gene_pathways,
            'pathway_color_map': pathway_color_map,
            'unique_pathways': unique_pathways
        }
    
    # Clustering
    if cluster_tfs and len(selected_tfs) > 1:
        from sklearn.metrics.pairwise import cosine_distances
        try:
            tf_distances = cosine_distances(tf_gene_matrix)
            tf_condensed = squareform(tf_distances)
            tf_linkage = linkage(tf_condensed, method='ward')
            tf_order = leaves_list(tf_linkage)
            selected_tfs = [selected_tfs[i] for i in tf_order]
            tf_gene_matrix = tf_gene_matrix[tf_order, :]
        except Exception as e:
            print(f"TF clustering failed: {e}")

    # Gene clustering by pathway groups
    if cluster_genes and len(selected_genes) > 1:
        if pathway_info:
            # Pathway-based clustering
            print("Clustering genes by pathway groups...")
            
            # Create ordered gene list by pathway
            ordered_genes = []
            for pathway in pathway_info['unique_pathways']:
                # Get genes in this pathway
                genes_in_pathway = [gene for gene in selected_genes 
                                if pathway_info['gene_pathways'][gene] == pathway]
                
                if len(genes_in_pathway) > 1:
                    # Cluster genes within the same pathway
                    try:
                        gene_indices = [selected_genes.index(gene) for gene in genes_in_pathway]
                        pathway_gene_matrix = tf_gene_matrix[:, gene_indices].T
                        
                        gene_distances = cosine_distances(pathway_gene_matrix)
                        gene_condensed = squareform(gene_distances)
                        gene_linkage = linkage(gene_condensed, method='ward')
                        gene_order = leaves_list(gene_linkage)
                        
                        genes_in_pathway = [genes_in_pathway[i] for i in gene_order]
                    except Exception:
                        genes_in_pathway = sorted(genes_in_pathway)
                
                ordered_genes.extend(genes_in_pathway)
            
            # Reorder matrix columns
            gene_reorder_indices = [selected_genes.index(gene) for gene in ordered_genes]
            tf_gene_matrix = tf_gene_matrix[:, gene_reorder_indices]
            selected_genes = ordered_genes
            
            print(f"Genes reordered by {len(pathway_info['unique_pathways'])} pathway groups")
        
        else:
            # Original similarity-based clustering
            try:
                gene_tf_matrix = tf_gene_matrix.T  
                gene_distances = cosine_distances(gene_tf_matrix)
                gene_condensed = squareform(gene_distances)
                gene_linkage = linkage(gene_condensed, method='ward')
                gene_order = leaves_list(gene_linkage)
                selected_genes = [selected_genes[i] for i in gene_order]
                tf_gene_matrix = tf_gene_matrix[:, gene_order]
            except Exception as e:
                print(f"Gene clustering failed: {e}")

    # Update pathway info after reordering
    if pathway_info:
        pathway_info['gene_pathways'] = {gene: pathway_info['gene_pathways'][gene] 
                                    for gene in selected_genes}
    
    # Determine figure size
    if figsize is None:
        fig_width = len(selected_genes) * 0.4 + 3
        fig_height = len(selected_tfs) * 0.3 + 2
        if pathway_info:
            fig_height += 0.3  # Reduced space for thinner pathway annotation
        figsize = (max(8, min(fig_width, 20)), max(6, min(fig_height, 15)))
    
    # Create figure with subplots for pathway annotation
    if pathway_info:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[0.06, 1], hspace=0.02)  # Thinner pathway bar
        ax_pathway = fig.add_subplot(gs[0])
        ax_main = fig.add_subplot(gs[1])
    else:
        fig, ax_main = plt.subplots(figsize=figsize)
    
    # Plot pathway annotation bar with improved colors
    if pathway_info:
        pathway_colors_ordered = [pathway_info['pathway_color_map'][pathway_info['gene_pathways'][gene]] 
                                for gene in selected_genes]
        
        # Convert hex colors to RGB for imshow
        pathway_colors_rgb = [mcolors.hex2color(color) for color in pathway_colors_ordered]
        pathway_bar = np.array(pathway_colors_rgb).reshape(1, -1, 3)
        
        ax_pathway.imshow(pathway_bar, aspect='auto')
        ax_pathway.set_xlim(-0.5, len(selected_genes) - 0.5)
        ax_pathway.set_xticks([])
        ax_pathway.set_yticks([])
        ax_pathway.set_ylabel("Pathway", fontsize=9, rotation=0, ha='right', va='center')
    
    # Plot main heatmap with proper grid
    im = ax_main.imshow(tf_gene_matrix, cmap=cmap_name, aspect='auto', 
                       vmin=0, vmax=len(episode_labels))
    
    # Add grid lines to create boxes
    ax_main.set_xticks(np.arange(-0.5, len(selected_genes), 1), minor=True)
    ax_main.set_yticks(np.arange(-0.5, len(selected_tfs), 1), minor=True)
    ax_main.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    # Set ticks and labels
    ax_main.set_xticks(range(len(selected_genes)))
    ax_main.set_xticklabels(selected_genes, rotation=90, ha='center', fontsize=8)
    ax_main.set_xlabel("Target Genes", fontsize=10)
    
    ax_main.set_yticks(range(len(selected_tfs)))
    ax_main.set_yticklabels(selected_tfs, fontsize=8)
    ax_main.set_ylabel("Transcription Factors", fontsize=10)
    
    # Add values to cells
    if show_values:
        for i in range(len(selected_tfs)):
            for j in range(len(selected_genes)):
                value = tf_gene_matrix[i, j]
                if value > 0:
                    text_color = 'white' if value > len(episode_labels)/2 else 'black'
                    ax_main.text(j, i, f'{int(value)}', ha='center', va='center',
                               color=text_color, fontsize=value_fontsize, weight='bold')
    
    # Horizontal Colorbar at bottom
    cbar = plt.colorbar(im, ax=ax_main, orientation='horizontal', 
                    fraction=0.05, pad=0.1, shrink=0.6, aspect=30)
    cbar.set_label('Number of Episodes', fontsize=10)
    cbar.set_ticks(range(len(episode_labels) + 1))
    
    # Add pathway legend in middle right
    if pathway_info:
        legend_elements = []
        for pathway in pathway_info['unique_pathways']:
            color = pathway_info['pathway_color_map'][pathway]
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=pathway))
        
        # Position legend in middle right
        legend = ax_main.legend(handles=legend_elements, loc='center left', 
                            bbox_to_anchor=(1.02, 0.5),  # Middle right
                            fontsize=7, frameon=False, ncol=1)
        
        # Adjust legend to not overlap with horizontal colorbar
        legend.set_bbox_to_anchor((1.02, 0.65))  # Move up slightly to avoid colorbar

    # Title
    title = f'TF-Gene Co-regulation Across Episodes\n({len(selected_tfs)} TFs, {len(selected_genes)} genes)'
    if pathway_info:
        title += f', {len(pathway_info["unique_pathways"])} pathways'
    ax_main.set_title(title, fontsize=12, pad=20)
    
    # Set limits
    ax_main.set_xlim(-0.5, len(selected_genes) - 0.5)
    ax_main.set_ylim(-0.5, len(selected_tfs) - 0.5)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    # Return summary dataframe
    summary_data = []
    for i, tf in enumerate(selected_tfs):
        for j, gene in enumerate(selected_genes):
            episodes_count = tf_gene_matrix[i, j]
            if episodes_count > 0:
                pathway = pathway_info['gene_pathways'][gene] if pathway_info else "Unknown"
                summary_data.append({
                    'TF': tf,
                    'Gene': gene, 
                    'Episodes_Count': int(episodes_count),
                    'Pathway': pathway
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Print summary
    print(f"\nPlot Summary:")
    print(f"- {len(selected_tfs)} TFs included")
    print(f"- {len(selected_genes)} target genes included")
    if pathway_info:
        print(f"- {len(pathway_info['unique_pathways'])} pathway categories")
    print(f"- Total TF-gene relationships: {len(summary_data)}")
    
    return fig, summary_df, selected_genes, selected_tfs