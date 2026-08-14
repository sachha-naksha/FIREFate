"""Single home for every FIREFate figure.

Contents, in order:

* episodic enrichment dotplot and TF–target heatmap (migrated from ``episode_plots``);
* interactive 3D regulatory-force landscapes (``plot_force_landscape`` and friends);
* stacked TF enrichment bars coloured by LF correlation sign (from ``enrichment.lf_bar_plots``);
* pseudotime heatmaps and expression-curve panels (from ``utils.custom``);
* chromatin binding-dynamics traces (from ``core.pseudotime_curves``).

Data loading, table building and smoothing stay in their own modules; everything
here takes prepared data and returns a figure.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import dictys
import matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dictys.net import stat
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

def sort_tfs_by_gene_similarity(tf_genes_dict, method='jaccard_hierarchical', return_linkage=False):
    """Sorts TFs based on gene similarity using Jaccard similarity and hierarchical clustering."""
    all_tfs = list(tf_genes_dict.keys())
    
    if len(all_tfs) <= 1:
        sorted_tfs = sorted(all_tfs)
        if return_linkage:
            return sorted_tfs, None, all_tfs
        return sorted_tfs
    
    def jaccard_similarity(set1, set2):
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    n_tfs = len(all_tfs)
    similarity_matrix = np.zeros((n_tfs, n_tfs))
    
    for i, tf1 in enumerate(all_tfs):
        for j, tf2 in enumerate(all_tfs):
            similarity_matrix[i, j] = jaccard_similarity(tf_genes_dict[tf1], tf_genes_dict[tf2])
    
    distance_matrix = 1 - similarity_matrix
    
    try:
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='ward')
        cluster_order = leaves_list(linkage_matrix)
        sorted_tfs = [all_tfs[i] for i in cluster_order]
        
        if return_linkage:
            return sorted_tfs, linkage_matrix, all_tfs
    except Exception as e:
        print(f"Warning: Gene similarity clustering failed ({e}), using alphabetical sorting")
        sorted_tfs = sorted(all_tfs)
        if return_linkage:
            return sorted_tfs, None, all_tfs
    
    return sorted_tfs 

def plot_tf_episodic_enrichment_dotplot(
    dfs,
    episode_labels,
    figsize=(12, 8),
    min_dot_size=10,
    max_dot_size=300,
    p_value_threshold=0.05,
    min_significance_threshold=None,
    min_targets_in_lf=2,
    min_targets_dwnstrm=2,
    cmap_name="coolwarm",
    value_legend_title="ES",
    size_legend_title="P-val",
    sort_by_gene_similarity=False,
    show_dendrogram=False,
    dendrogram_ratio=0.2,
    figure_title=None,
    log_scale=False,
    show_plot=True,
    tf_order=None,
    horizontal_layout=False  # NEW PARAMETER
):
    """
    Plots a dotplot for TF episodic enrichment.
    
    Parameters:
    -----------
    horizontal_layout : bool, default False
        If True, episodes are on y-axis (top to bottom) and TFs on x-axis (left to right).
        If False (default), TFs are on y-axis and episodes on x-axis.
    When tf_order is None and sort_by_gene_similarity is False, TFs are ordered by the
    episode in which they peak: all TFs peaking in episode 1 first, then episode 2, etc.
    The peak episode is the one with the highest enrichment score among that TF's
    significant episodes (p_value <= p_value_threshold), falling back to all episodes if
    none are significant. Within an episode block, TFs are ordered by decreasing peak
    enrichment score.
    """
    # 1-5. [Same validation and filtering code as before]
    required_cols = ['TF', 'p_value', 'enrichment_score', 'genes_in_lf', 'genes_dwnstrm']
    
    for i, df in enumerate(dfs):
        if df is None or df.empty:
            print(f"Episode {i+1} dataframe is None or empty.")
            return None, None, None
        for col in required_cols:
            if col not in df.columns:
                print(f"Episode {i+1} dataframe missing required column: {col}")
                return None, None, None
    
    def parse_genes_in_lf(genes_str):
        try:
            if pd.isna(genes_str) or genes_str == '' or genes_str == '()':
                return set()
            genes_tuple = ast.literal_eval(genes_str)
            if isinstance(genes_tuple, tuple):
                return set(genes_tuple)
            elif isinstance(genes_tuple, str):
                return {genes_tuple}
            else:
                return set()
        except:
            return set()
                
    tf_genes_dict = {}
    tf_dwnstrm_genes_dict = {}
    plot_data_list = []
    
    for i, (df, episode_label) in enumerate(zip(dfs, episode_labels)):
        df_clean = df.dropna(subset=['TF', 'p_value', 'enrichment_score'])
        
        for _, row in df_clean.iterrows():
            tf_name = row['TF']
            genes_in_lf_set = parse_genes_in_lf(row.get('genes_in_lf', ''))
            genes_dwnstrm_set = parse_genes_in_lf(row.get('genes_dwnstrm', ''))
            
            if tf_name not in tf_genes_dict:
                tf_genes_dict[tf_name] = set()
                tf_dwnstrm_genes_dict[tf_name] = set()
            tf_genes_dict[tf_name].update(genes_in_lf_set)
            tf_dwnstrm_genes_dict[tf_name].update(genes_dwnstrm_set)
            
            plot_data_list.append({
                'episode': episode_label,
                'episode_idx': i,
                'TF': tf_name,
                'p_value': row['p_value'],
                'enrichment_score': row['enrichment_score']
            })
    
    if not plot_data_list:
        print("No valid data found across all episodes.")
        return None, None, None
    
    plot_data_df = pd.DataFrame(plot_data_list)
    
    valid_tfs = set()
    for tf_name in tf_genes_dict.keys():
        lf_gene_count = len(tf_genes_dict[tf_name])
        dwnstrm_gene_count = len(tf_dwnstrm_genes_dict[tf_name])
        
        if lf_gene_count >= min_targets_in_lf and dwnstrm_gene_count >= min_targets_dwnstrm:
            valid_tfs.add(tf_name)
    
    if not valid_tfs:
        print(f"No TFs meet the criteria: >= {min_targets_in_lf} LF genes AND >= {min_targets_dwnstrm} downstream genes")
        return None, None, None
    
    plot_data_df = plot_data_df[plot_data_df['TF'].isin(valid_tfs)]
    tf_genes_dict = {tf: genes for tf, genes in tf_genes_dict.items() if tf in valid_tfs}
    tf_dwnstrm_genes_dict = {tf: genes for tf, genes in tf_dwnstrm_genes_dict.items() if tf in valid_tfs}
    
    print(f"Filtered to {len(valid_tfs)} TFs that meet gene count criteria")
    
    if min_significance_threshold is not None:
        significant_tfs = set()
        tf_min_pvalues = plot_data_df.groupby('TF')['p_value'].min()
        significant_tfs = set(tf_min_pvalues[tf_min_pvalues < min_significance_threshold].index)
        
        if not significant_tfs:
            print(f"No TFs meet the minimum significance threshold of {min_significance_threshold}")
            return None, None, None

        plot_data_df = plot_data_df[plot_data_df['TF'].isin(significant_tfs)]
        tf_genes_dict = {tf: genes for tf, genes in tf_genes_dict.items() if tf in significant_tfs}
        
        print(f"Further filtered to {len(significant_tfs)} TFs that meet significance threshold < {min_significance_threshold}")

    # 6. Sort TFs
    if tf_order is not None:
        available_tfs = set(tf_genes_dict.keys())
        all_tfs_sorted = [tf for tf in tf_order if tf in available_tfs]
        remaining_tfs = sorted(available_tfs - set(all_tfs_sorted))
        all_tfs_sorted.extend(remaining_tfs)
        linkage_matrix = None
        original_tf_labels = None
        if remaining_tfs:
            print(f"Note: {len(remaining_tfs)} TFs not in custom order were added alphabetically")
    elif sort_by_gene_similarity:
        if show_dendrogram:
            all_tfs_sorted, linkage_matrix, original_tf_labels = sort_tfs_by_gene_similarity(
                tf_genes_dict, return_linkage=True)
        else:
            all_tfs_sorted = sort_tfs_by_gene_similarity(tf_genes_dict)
            linkage_matrix = None
            original_tf_labels = None
    else:
        # Order TFs by the episode where they peak (Ep1 block, then Ep2 block, ...),
        # and within each block by decreasing peak enrichment score.
        tf_peaks = []
        for tf_name in tf_genes_dict.keys():
            tf_rows = plot_data_df[plot_data_df['TF'] == tf_name]
            sig_rows = tf_rows[tf_rows['p_value'] <= p_value_threshold]
            rows = sig_rows if not sig_rows.empty else tf_rows
            peak_row = rows.loc[rows['enrichment_score'].idxmax()]
            tf_peaks.append((peak_row['episode_idx'], -peak_row['enrichment_score'], tf_name))
        all_tfs_sorted = [tf for _, _, tf in sorted(tf_peaks)]
        linkage_matrix = None
        original_tf_labels = None

    # 7. Map p-values to dot sizes
    def p_value_to_size(p_val):
        if p_val > p_value_threshold:
            return min_dot_size * 0.5
        min_p_cap = 1e-6
        log_p = -np.log10(max(p_val, min_p_cap))
        log_thresh = -np.log10(p_value_threshold)
        log_min_cap = -np.log10(min_p_cap)
        
        if log_min_cap == log_thresh:
            scaled_val = 1.0
        else:
            scaled_val = (log_p - log_thresh) / (log_min_cap - log_thresh)
        
        size = min_dot_size + (max_dot_size - min_dot_size) * min(scaled_val, 1.0)
        return size

    plot_data_df['dot_size'] = plot_data_df['p_value'].apply(p_value_to_size)

    if log_scale:
        min_enrichment = plot_data_df['enrichment_score'].min()
        if min_enrichment <= 0:
            offset = abs(min_enrichment) + 1e-6
            plot_data_df['enrichment_score_log'] = np.log2(plot_data_df['enrichment_score'] + offset)
            value_legend_title = f"log2({value_legend_title} + {offset:.1e})"
        else:
            plot_data_df['enrichment_score_log'] = np.log2(plot_data_df['enrichment_score'])
            value_legend_title = f"log2({value_legend_title})"
        color_values = plot_data_df['enrichment_score_log']
    else:
        color_values = plot_data_df['enrichment_score']

    # 8. Create coordinate mappings - MODIFIED FOR HORIZONTAL LAYOUT
    if horizontal_layout:
        # TFs on x-axis, Episodes on y-axis
        tf_x_coords = {tf: i for i, tf in enumerate(all_tfs_sorted)}
        episode_y_coords = {label: i for i, label in enumerate(episode_labels)}
        print(f"DEBUG horizontal_layout: episode_labels order: {episode_labels}")
        print(f"DEBUG horizontal_layout: episode_y_coords: {episode_y_coords}")
    else:
        # Original: Episodes on x-axis, TFs on y-axis
        episode_x_coords = {label: i for i, label in enumerate(episode_labels)}
        tf_y_coords = {tf: i for i, tf in enumerate(all_tfs_sorted)}
    
    # 9. Create figure with subplots
    if show_dendrogram and linkage_matrix is not None and sort_by_gene_similarity:
        fig = plt.figure(figsize=figsize)
        
        if horizontal_layout:
            # Dendrogram above the plot
            dendro_height = dendrogram_ratio
            main_height = 1 - dendro_height - 0.15
            
            ax_dendro = fig.add_subplot(2, 1, 1)
            ax_main = fig.add_subplot(2, 1, 2)
            
            dendro_bottom = 0.7
            main_bottom = 0.1
            
            ax_dendro.set_position([0.1, dendro_bottom, 0.7, dendro_height])
            ax_main.set_position([0.1, main_bottom, 0.7, main_height])

            dendro_plot = dendrogram(
                linkage_matrix, 
                ax=ax_dendro,
                orientation='top',
                labels=original_tf_labels,
                leaf_font_size=8,
                color_threshold=0.7*max(linkage_matrix[:,2])
            )
            ax_dendro.set_xlabel("TF Clustering")
            ax_dendro.set_ylabel("Distance")
            ax_dendro.spines['top'].set_visible(False)
            ax_dendro.spines['right'].set_visible(False)
            ax_dendro.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
            # Dendrogram to the left (original)
            dendro_width = dendrogram_ratio
            main_width = 1 - dendro_width - 0.15
            
            ax_dendro = fig.add_subplot(1, 2, 1)
            ax_main = fig.add_subplot(1, 2, 2)
            
            dendro_left = 0.05
            main_left = dendro_left + dendro_width + 0.02
            
            ax_dendro.set_position([dendro_left, 0.1, dendro_width, 0.8])
            ax_main.set_position([main_left, 0.1, main_width, 0.8])

            dendro_plot = dendrogram(
                linkage_matrix, 
                ax=ax_dendro,
                orientation='left',
                labels=original_tf_labels,
                leaf_font_size=8,
                color_threshold=0.7*max(linkage_matrix[:,2])
            )
            ax_dendro.invert_yaxis()
            ax_dendro.set_ylabel("TF Clustering")
            ax_dendro.set_xlabel("Distance")
            ax_dendro.spines['top'].set_visible(False)
            ax_dendro.spines['right'].set_visible(False)
            ax_dendro.tick_params(axis='y', which='both', left=False, labelleft=False)
    else:
        fig, ax_main = plt.subplots(figsize=figsize)
    
    # 10. Create scatter plot - MODIFIED FOR HORIZONTAL LAYOUT
    if horizontal_layout:
        # Debug: Check what TFs are in the data
        unique_tfs_in_data = plot_data_df['TF'].unique()
        print(f"DEBUG: Unique TFs in plot_data_df: {len(unique_tfs_in_data)}")
        print(f"DEBUG: TFs in data: {sorted(unique_tfs_in_data)}")
        
        scatter = ax_main.scatter(
            x=plot_data_df['TF'].map(tf_x_coords),
            y=plot_data_df['episode'].map(episode_y_coords),
            s=plot_data_df['dot_size'],
            c=color_values,
            cmap=cmap_name,
            edgecolors='gray',
            linewidths=0.5,
            alpha=0.8
        )
    else:
        scatter = ax_main.scatter(
            x=plot_data_df['episode'].map(episode_x_coords),
            y=plot_data_df['TF'].map(tf_y_coords),
            s=plot_data_df['dot_size'],
            c=color_values,
            cmap=cmap_name,
            edgecolors='gray',
            linewidths=0.5,
            alpha=0.8
        )
    
    # 11. Axis formatting - MODIFIED FOR HORIZONTAL LAYOUT
    if horizontal_layout:
        # X-axis: TFs (left to right)
        # Force exact tick positions
        from matplotlib.ticker import FixedLocator
        ax_main.xaxis.set_major_locator(FixedLocator(list(range(len(all_tfs_sorted)))))
        ax_main.set_xticklabels(all_tfs_sorted, rotation=90, ha="center", fontsize=9)
        ax_main.tick_params(axis='x', which='major', labelsize=9)
        x_pad = 0.5
        ax_main.set_xlim(-x_pad, len(all_tfs_sorted) - 1 + x_pad)
        ax_main.set_xlabel("TFs", fontsize=12, fontweight='bold')
        
        print(f"DEBUG: Plotting {len(all_tfs_sorted)} TFs on x-axis: {all_tfs_sorted}")
        
        # Y-axis: Episodes (top to bottom)
        ax_main.set_yticks(list(episode_y_coords.values()))
        ax_main.set_yticklabels(episode_labels, rotation=0, ha="right")
        y_pad = 0.5
        ax_main.set_ylim(-y_pad, len(episode_labels) - 1 + y_pad)
        ax_main.set_ylabel("Episodes", fontsize=12, fontweight='bold', labelpad=15)
    else:
        # X-axis: Episodes
        ax_main.set_xticks(list(episode_x_coords.values()))
        ax_main.set_xticklabels(episode_labels, rotation=0, ha="center")
        x_pad = 0.5
        ax_main.set_xlim(-x_pad, len(episode_labels) - 1 + x_pad)
        ax_main.set_xlabel("Episodes", fontsize=12, fontweight='bold', labelpad=15)
        
        # Y-axis: TFs
        ax_main.set_yticks(list(tf_y_coords.values()))
        ax_main.set_yticklabels(all_tfs_sorted)
        ax_main.set_ylabel("TFs", fontsize=12, fontweight='bold')
    
    # 12. Size legend for P-values
    legend_p_values = [0.001, 0.01]
    legend_dots = []
    
    for p_val in legend_p_values:
        size_val = p_value_to_size(p_val)
        if p_val > p_value_threshold:
            label_text = f"{p_value_threshold}"
        else:
            label_text = f"{p_val}"
        legend_dots.append(plt.scatter([], [], s=size_val, c='gray', label=label_text))
    
    if show_dendrogram and linkage_matrix is not None and sort_by_gene_similarity:
        if horizontal_layout:
            bbox_anchor = (1.15, 0.5)
        else:
            bbox_anchor = (1.25, 0.6)
    else:
        bbox_anchor = (1.18, 0.6)
        
    size_leg = ax_main.legend(
        handles=legend_dots, 
        title=size_legend_title,
        bbox_to_anchor=bbox_anchor, 
        loc='center left',
        labelspacing=1.5, 
        borderpad=1, 
        frameon=True,
        handletextpad=1.5,
        scatterpoints=1
    )
    
    # 13. Colorbar
    if show_dendrogram and linkage_matrix is not None and sort_by_gene_similarity:
        if horizontal_layout:
            cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02])
        else:
            cbar_ax = fig.add_axes([0.85, 0.15, 0.3, 0.03])
    else:
        cbar_ax = fig.add_axes([0.65, 0.4, 0.2, 0.02])
    
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(value_legend_title, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # 14. Final formatting
    ax_main.grid(False)
    ax_main.tick_params(axis='both', which='major', pad=5)
    
    # Always invert y-axis so first item is at top
    # - Vertical layout: First TF at top, progressing downward
    # - Horizontal layout: First episode at top, progressing downward
    ax_main.invert_yaxis()
    
    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=14, fontweight='bold', y=0.95)

    if show_plot:
        plt.tight_layout()
        plt.show()
    
    return fig, plot_data_df, all_tfs_sorted

def plot_tf_target_episodic_heatmap(
    csv_path: str,
    tf_col: str = "TF",
    target_col: str = "target",
    title: str = "TF–Target Edge Scores by Episode",
    episode_label: str = "Episode",
    figsize: tuple = (14, 7),
    cell_annot: bool = True,
    annot_fmt: str = ".2f",
    cmap_colors: list = None,
    vmin: float = None,
    vmax: float = None,
    cbar_label: str = "Edge Score",
    tf_group_line_color: str = "black",
    tf_group_line_width: float = 2.5,
    save_path: str = None,
    dpi: int = 150,
):
    """
    Plot a styled heatmap of TF-target edge scores across episodes.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file. Expected columns: tf_col, target_col, then one
        column per episode (numeric).
    tf_col : str
        Name of the column containing transcription factor names.
    target_col : str
        Name of the column containing target gene names.
    title : str
        Plot title shown above the heatmap.
    episode_label : str
        Label shown above the episode columns.
    figsize : tuple
        Figure size (width, height) in inches.
    cell_annot : bool
        Whether to annotate each cell with its value.
    annot_fmt : str
        Number format string for cell annotations (e.g. ".2f", ".3f").
    cmap_colors : list of str/colors
        Three colors defining the colormap [negative, zero/center, positive].
        Defaults to ['blue', 'white', 'gold'].
    vmin, vmax : float or None
        Color scale limits. If None, symmetric limits based on abs-max are used.
    cbar_label : str
        Label for the colorbar.
    tf_group_line_color : str
        Color of the horizontal divider lines between TF groups.
    tf_group_line_width : float
        Line width of TF group dividers.
    save_path : str or None
        If provided, saves the figure to this path instead of showing it.
    dpi : int
        Resolution when saving the figure.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # ------------------------------------------------------------------ #
    # 1. Load & validate data
    # ------------------------------------------------------------------ #
    df = pd.read_csv(csv_path)

    missing = {tf_col, target_col} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    episode_cols = [c for c in df.columns if c not in (tf_col, target_col)]
    if not episode_cols:
        raise ValueError("No episode columns found after removing TF and target columns.")

    # ------------------------------------------------------------------ #
    # 2. Build matrix — rows ordered by TF group then target name
    # ------------------------------------------------------------------ #
    df = df.sort_values([tf_col, target_col]).reset_index(drop=True)

    row_labels = df[target_col].tolist()
    tf_labels  = df[tf_col].tolist()
    matrix     = df[episode_cols].values.astype(float)

    n_rows, n_cols = matrix.shape

    # ------------------------------------------------------------------ #
    # 3. Color scale
    # ------------------------------------------------------------------ #
    if cmap_colors is None:
        cmap_colors = ["blue", "white", "gold"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("custom", cmap_colors, N=256)

    abs_max = np.max(np.abs(matrix))
    vmin = vmin if vmin is not None else -abs_max
    vmax = vmax if vmax is not None else  abs_max
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # ------------------------------------------------------------------ #
    # 4. Figure & axes
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------------------------------ #
    # 5. Draw cells
    # ------------------------------------------------------------------ #
    for r in range(n_rows):
        for c in range(n_cols):
            val = matrix[r, c]
            color = cmap(norm(val))
            rect = mpl.patches.Rectangle(
                (c, n_rows - r - 1), 1, 1,
                facecolor=color,
                edgecolor="#dddddd",
                linewidth=0.4,
            )
            ax.add_patch(rect)

            if cell_annot and val != 0:
                # choose dark or light text based on cell brightness
                rgba = mpl.colors.to_rgba(color)
                brightness = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                txt_color = "black" if brightness > 0.5 else "white"
                ax.text(
                    c + 0.5, n_rows - r - 0.5,
                    format(val, annot_fmt),
                    ha="center", va="center",
                    fontsize=8, color=txt_color,
                )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)

    # ------------------------------------------------------------------ #
    # 6. Tick labels
    # ------------------------------------------------------------------ #
    # x-axis on top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(episode_cols, fontsize=11)
    ax.set_xlabel(episode_label, fontsize=13, labelpad=8)

    # y-axis: target names
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(reversed(row_labels), fontsize=10)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_ylabel("")

    # ------------------------------------------------------------------ #
    # 7. TF group labels + divider lines
    # ------------------------------------------------------------------ #
    tfs_unique = list(dict.fromkeys(tf_labels))   # ordered, deduplicated

    for tf in tfs_unique:
        indices = [i for i, t in enumerate(tf_labels) if t == tf]
        # center in flipped y coords
        mid = n_rows - np.mean(indices) - 0.5
        ax.text(
            -0.3, mid, tf,
            ha="right", va="center",
            fontsize=11, fontweight="bold",
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )
        # divider below the last row of this group (in flipped coords: above)
        boundary_y = n_rows - (indices[-1] + 1)
        if boundary_y > 0:                        # skip the very last group
            ax.axhline(
                y=boundary_y,
                color=tf_group_line_color,
                linewidth=tf_group_line_width,
            )

    # ------------------------------------------------------------------ #
    # 8. Colorbar
    # ------------------------------------------------------------------ #
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(cbar_label, fontsize=11)

    # ------------------------------------------------------------------ #
    # 9. Title & layout
    # ------------------------------------------------------------------ #
    ax.set_title(title, fontsize=14, fontweight="bold", pad=40)
    plt.tight_layout()

    # ------------------------------------------------------------------ #
    # 10. Save or show
    # ------------------------------------------------------------------ #
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 3D regulatory force landscape (Plotly)
# =============================================================================
# Visualizes the regulatory force as a surface over (TF expression, beta) space,
# with actual TF–target link trajectories traced as 3D curves on the landscape.
#
# Force definition (mirrors SmoothedCurvesGRN.calculate_force_curves):
#     force = sign(β) · exp( log₁₀(|β| + ε) + log₁₀(TF_expr + ε) )
# ─────────────────────────────────────────────────────────────────────────────

EPSILON = 1e-10


def _force_fn(tf_expr, beta):
    """Compute force = sign(β) · exp(log₁₀(|β|+ε) + log₁₀(tf+ε))."""
    log_beta = np.log10(np.abs(beta) + EPSILON)
    log_tf   = np.log10(np.abs(tf_expr) + EPSILON)
    return np.sign(beta) * np.exp(log_beta + log_tf)


def _build_surface(
    tf_range: tuple,
    beta_range: tuple,
    sign: int = 1,
    resolution: int = 120,
):
    """
    Create a meshgrid surface of the force function for a given sign of β.

    Parameters
    ----------
    tf_range : (min, max) of TF expression values (log CPM)
    beta_range : (min, max) of |β| values
    sign : +1 for activating surface, -1 for repressing surface
    resolution : grid density
    """
    tf_vals   = np.linspace(tf_range[0], tf_range[1], resolution)
    beta_vals = np.linspace(beta_range[0], beta_range[1], resolution)
    TF, BETA  = np.meshgrid(tf_vals, beta_vals)
    FORCE     = _force_fn(TF, sign * BETA)   # sign determines activation / repression
    return TF, BETA * sign, FORCE


def plot_force_landscape(
    beta_curves: pd.DataFrame,
    regulon_tf_expression: pd.DataFrame,
    force_curves: pd.DataFrame,
    dtime: pd.Series,
    links_to_highlight: list = None,
    surface_opacity: float = 0.35,
    surface_resolution: int = 100,
    line_width: float = 5,
    marker_size: float = 3,
    colorscale_positive: str = "Purples",
    colorscale_negative: str = "Blues",
    title: str = "Regulatory Force Landscape",
    width: int = 1000,
    height: int = 750,
    show_surface: bool = True,
    camera: dict = None,
):
    """
    Build an interactive 3D Plotly figure of the force landscape.

    Parameters
    ----------
    beta_curves : pd.DataFrame
        Multi-indexed (TF, Target) × time_points. Edge strengths.
    regulon_tf_expression : pd.DataFrame
        TF × time_points. Log-CPM expression of each TF, already broadcast-
        ready (one row per unique TF present in beta_curves level 0).
    force_curves : pd.DataFrame
        Multi-indexed (TF, Target) × time_points. Pre-computed forces.
    dtime : pd.Series
        Pseudotime values for each time point.
    links_to_highlight : list of (TF, Target) tuples, optional
        Subset of links to draw. Default: all links.
    show_surface : bool
        Whether to render the analytical force surface behind the curves.
    """

    fig = go.Figure()

    # ── resolve links ────────────────────────────────────────────────────
    all_links = beta_curves.index.tolist()
    if links_to_highlight is None:
        links_to_highlight = all_links

    # ── data ranges for surface ──────────────────────────────────────────
    # Build the broadcast TF expression aligned to beta_curves index
    targets_per_tf = beta_curves.index.get_level_values(0).value_counts()
    expanded_tf = pd.DataFrame(
        np.repeat(regulon_tf_expression.values,
                  [targets_per_tf[tf] for tf in regulon_tf_expression.index], axis=0),
        index=beta_curves.index,
        columns=beta_curves.columns,
    )

    tf_vals_all   = expanded_tf.loc[links_to_highlight].values.ravel()
    beta_vals_all = beta_curves.loc[links_to_highlight].values.ravel()

    tf_min, tf_max     = np.nanmin(tf_vals_all), np.nanmax(tf_vals_all)
    beta_min, beta_max = np.nanmin(beta_vals_all), np.nanmax(beta_vals_all)
    tf_pad   = (tf_max - tf_min) * 0.1
    beta_pad = (beta_max - beta_min) * 0.1

    # ── analytical surface(s) ────────────────────────────────────────────
    if show_surface:
        # Positive-β surface (activation half)
        if beta_max > 0:
            TF_p, BETA_p, FORCE_p = _build_surface(
                tf_range=(tf_min - tf_pad, tf_max + tf_pad),
                beta_range=(1e-6, beta_max + beta_pad),
                sign=1,
                resolution=surface_resolution,
            )
            fig.add_trace(go.Surface(
                x=TF_p, y=BETA_p, z=FORCE_p,
                colorscale=colorscale_positive,
                opacity=surface_opacity,
                showscale=False,
                name="Activation surface",
                hoverinfo="skip",
            ))

        # Negative-β surface (repression half)
        if beta_min < 0:
            TF_n, BETA_n, FORCE_n = _build_surface(
                tf_range=(tf_min - tf_pad, tf_max + tf_pad),
                beta_range=(1e-6, np.abs(beta_min) + beta_pad),
                sign=-1,
                resolution=surface_resolution,
            )
            fig.add_trace(go.Surface(
                x=TF_n, y=BETA_n, z=FORCE_n,
                colorscale=colorscale_negative,
                opacity=surface_opacity,
                showscale=False,
                name="Repression surface",
                hoverinfo="skip",
            ))

    # ── link trajectories ────────────────────────────────────────────────
    # Build a qualitative colour palette
    n_links = len(links_to_highlight)
    cmap = _get_qualitative_colors(n_links)

    for i, (tf, target) in enumerate(links_to_highlight):
        if (tf, target) not in all_links:
            continue

        x = expanded_tf.loc[(tf, target)].values.astype(float)
        y = beta_curves.loc[(tf, target)].values.astype(float)
        z = force_curves.loc[(tf, target)].values.astype(float)
        t = dtime.values.astype(float)

        color = cmap[i % len(cmap)]

        # Trajectory line
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=color, width=line_width),
            name=f"{tf} → {target}",
            customdata=np.stack([t, x, y, z], axis=-1),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "pseudotime: %{customdata[0]:.3f}<br>"
                "TF expr (lcpm): %{customdata[1]:.3f}<br>"
                "β: %{customdata[2]:.5f}<br>"
                "force: %{customdata[3]:.5f}"
                "<extra></extra>"
            ),
        ))

        # Start marker (early pseudotime)
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode="markers",
            marker=dict(size=marker_size + 3, color=color, symbol="diamond"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── layout ───────────────────────────────────────────────────────────
    default_camera = dict(
        eye=dict(x=1.6, y=-1.6, z=0.9),
        up=dict(x=0, y=0, z=1),
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Helvetica Neue, Arial"),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text="TF Expression (log CPM)", font=dict(size=14)),
                backgroundcolor="rgba(240,240,245,0.5)",
                gridcolor="rgba(200,200,210,0.4)",
                showbackground=True,
            ),
            yaxis=dict(
                title=dict(text="β (edge strength)", font=dict(size=14)),
                backgroundcolor="rgba(240,240,245,0.5)",
                gridcolor="rgba(200,200,210,0.4)",
                showbackground=True,
            ),
            zaxis=dict(
                title=dict(text="Regulatory Force", font=dict(size=14)),
                backgroundcolor="rgba(240,240,245,0.5)",
                gridcolor="rgba(200,200,210,0.4)",
                showbackground=True,
            ),
            camera=camera or default_camera,
        ),
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            font=dict(size=11),
            itemsizing="constant",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(200,200,200,0.5)",
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def plot_single_link_landscape(
    tf_name: str,
    target_name: str,
    beta_curves: pd.DataFrame,
    regulon_tf_expression: pd.DataFrame,
    force_curves: pd.DataFrame,
    dtime: pd.Series,
    surface_resolution: int = 150,
    colorscale: str = "Viridis",
    width: int = 800,
    height: int = 650,
):
    """
    Focused 3D view of a single TF → Target link on its own force surface.
    The trajectory is coloured by pseudotime.
    """

    # ── extract link data ────────────────────────────────────────────────
    targets_per_tf = beta_curves.index.get_level_values(0).value_counts()
    expanded_tf = pd.DataFrame(
        np.repeat(regulon_tf_expression.values,
                  [targets_per_tf[tf] for tf in regulon_tf_expression.index], axis=0),
        index=beta_curves.index,
        columns=beta_curves.columns,
    )

    x = expanded_tf.loc[(tf_name, target_name)].values.astype(float)
    y = beta_curves.loc[(tf_name, target_name)].values.astype(float)
    z = force_curves.loc[(tf_name, target_name)].values.astype(float)
    t = dtime.values.astype(float)

    # ── surface ──────────────────────────────────────────────────────────
    tf_pad   = (x.max() - x.min()) * 0.25
    beta_pad = (np.abs(y).max()) * 0.25
    sign = 1 if np.mean(y) >= 0 else -1

    TF_s, BETA_s, FORCE_s = _build_surface(
        tf_range=(x.min() - tf_pad, x.max() + tf_pad),
        beta_range=(1e-6, np.abs(y).max() + beta_pad),
        sign=sign,
        resolution=surface_resolution,
    )

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=TF_s, y=BETA_s, z=FORCE_s,
        colorscale="Purples" if sign > 0 else "Blues",
        opacity=0.30,
        showscale=False,
        hoverinfo="skip",
    ))

    # ── trajectory coloured by pseudotime ────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines+markers",
        line=dict(color=t, colorscale=colorscale, width=6),
        marker=dict(size=2.5, color=t, colorscale=colorscale,
                    colorbar=dict(title="Pseudotime", thickness=15, len=0.5)),
        name=f"{tf_name} → {target_name}",
        customdata=np.stack([t, x, y, z], axis=-1),
        hovertemplate=(
            f"<b>{tf_name} → {target_name}</b><br>"
            "pseudotime: %{customdata[0]:.3f}<br>"
            "TF expr: %{customdata[1]:.3f}<br>"
            "β: %{customdata[2]:.5f}<br>"
            "force: %{customdata[3]:.5f}"
            "<extra></extra>"
        ),
    ))

    # Start / end markers
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode="markers",
        marker=dict(size=7, color="limegreen", symbol="diamond",
                    line=dict(color="black", width=1)),
        name="Start", showlegend=True,
    ))
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode="markers",
        marker=dict(size=7, color="red", symbol="x",
                    line=dict(color="black", width=1)),
        name="End", showlegend=True,
    ))

    fig.update_layout(
        title=dict(
            text=f"Force Landscape: {tf_name} → {target_name}",
            font=dict(size=18, family="Helvetica Neue, Arial"),
            x=0.5,
        ),
        scene=dict(
            xaxis_title="TF Expression (log CPM)",
            yaxis_title="β (edge strength)",
            zaxis_title="Regulatory Force",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        width=width, height=height,
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_force_by_tf(
    beta_curves: pd.DataFrame,
    regulon_tf_expression: pd.DataFrame,
    force_curves: pd.DataFrame,
    dtime: pd.Series,
    links: list = None,
    width: int = 1200,
    height: int = 900,
):
    """
    One 3D subplot per TF, showing all its target trajectories.
    Good for comparing how a single TF's different targets behave.
    """
    if links is None:
        links = beta_curves.index.tolist()

    # Group links by TF
    from collections import defaultdict
    tf_groups = defaultdict(list)
    for tf, tgt in links:
        tf_groups[tf].append((tf, tgt))

    tfs = sorted(tf_groups.keys())
    n_tfs = len(tfs)
    cols = min(3, n_tfs)
    rows = int(np.ceil(n_tfs / cols))

    specs = [[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)]
    subplot_titles = [tf for tf in tfs]

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
    )

    targets_per_tf = beta_curves.index.get_level_values(0).value_counts()
    expanded_tf = pd.DataFrame(
        np.repeat(regulon_tf_expression.values,
                  [targets_per_tf[tf] for tf in regulon_tf_expression.index], axis=0),
        index=beta_curves.index,
        columns=beta_curves.columns,
    )

    for idx, tf in enumerate(tfs):
        r = idx // cols + 1
        c = idx % cols + 1
        scene_name = f"scene{idx + 1}" if idx > 0 else "scene"
        targets = tf_groups[tf]
        cmap = _get_qualitative_colors(len(targets))

        for j, (tf_name, tgt_name) in enumerate(targets):
            x = expanded_tf.loc[(tf_name, tgt_name)].values.astype(float)
            y = beta_curves.loc[(tf_name, tgt_name)].values.astype(float)
            z = force_curves.loc[(tf_name, tgt_name)].values.astype(float)

            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode="lines",
                    line=dict(color=cmap[j], width=4),
                    name=f"{tf_name}→{tgt_name}",
                    legendgroup=tf,
                ),
                row=r, col=c,
            )

        fig.update_layout(**{
            scene_name: dict(
                xaxis_title="TF expr",
                yaxis_title="β",
                zaxis_title="Force",
                camera=dict(eye=dict(x=1.4, y=-1.4, z=0.8)),
            )
        })

    fig.update_layout(
        title="Force Landscapes by TF",
        width=width,
        height=height * rows / 2,
        paper_bgcolor="white",
    )
    return fig


def _get_qualitative_colors(n: int) -> list:
    """Return n visually distinct colours (hex strings)."""
    palette = [
        "#6A3D9A", "#1F78B4", "#E31A1C", "#33A02C", "#FF7F00",
        "#FB9A99", "#B2DF8A", "#A6CEE3", "#FDBF6F", "#CAB2D6",
        "#B15928", "#FFFF99", "#8DD3C7", "#BEBADA", "#FB8072",
        "#80B1D3", "#FDB462", "#BC80BD", "#CCEBC5", "#D9D9D9",
    ]
    if n <= len(palette):
        return palette[:n]
    # cycle if more links than palette entries
    return [palette[i % len(palette)] for i in range(n)]

# ─────────────────────────────────────────────────────────────────────────────
# Pseudotime heatmaps and expression curves (migrated from ``firefate.utils.custom``)
# ─────────────────────────────────────────────────────────────────────────────

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
    Draw a 2D hierarchically clustered heatmap from a DataFrame.

    The figure X/Y axes correspond to DataFrame columns/rows.

    Parameters
    ----------
    d : pandas.DataFrame
        2D data with index and column names used for clustering.
    optimal_ordering : bool
        Passed to ``scipy.cluster.hierarchy.dendrogram``.
    method : str or tuple of str
        Linkage method(s) for ``scipy.cluster.hierarchy.linkage``.
    metric : str or tuple of str
        Distance metric(s) for ``scipy.spatial.distance.pdist``.
    dshow : pandas.DataFrame, optional
        Data to render; defaults to ``d``.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on.
    cmap : str
        Colormap name.
    aspect, figscale, dtop, dright, wcolorbar, wedge : float
        Layout and colorbar geometry (fractions of the figure).
    xselect, yselect : array-like of bool, optional
        Mask of rows/columns to include in clustering and display.
    xtick, ytick : bool
        Whether to show axis ticks.
    vmin, vmax : float, optional
        Color scale limits.
    inverty : bool
        Whether to invert the y-axis.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure with dendrograms and heatmap.
    x, y : list
        Column and index labels included after clustering/selection.
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


# ─────────────────────────────────────────────────────────────────────────────
# TF enrichment bars coloured by LF correlation sign
# (migrated from ``firefate.enrichment.lf_bar_plots``; table building stays there)
# ─────────────────────────────────────────────────────────────────────────────

COLOR_MAP = {"red": "#c1272d", "blue": "#2b6cb0", "gray": "#b0b0b0"}
_COLOR_ORDER = ["red", "blue", "gray"]
_COLOR_LABELS = {
    "red": "positively correlated LF gene",
    "blue": "negatively correlated LF gene",
    "gray": "not an LF gene",
}


def plot_tf_enrichment_bars(
    plot_df: pd.DataFrame,
    title: str,
    out_path: str | Path | None = None,
    *,
    min_label_proportion: float = 0.08,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Stacked bar of enrichment score per TF, split by LF correlation sign.

    Segments smaller than ``min_label_proportion`` of the bar are left unlabelled so
    the in-bar percentages stay readable. Saves vector output (SVG/PDF) when
    ``out_path`` is given.
    """
    wide = (
        plot_df.pivot(index="source", columns="color", values="height")
        .reindex(columns=_COLOR_ORDER, fill_value=0.0)
        .fillna(0.0)
    )
    props = (
        plot_df.pivot(index="source", columns="color", values="proportion")
        .reindex(columns=_COLOR_ORDER, fill_value=0.0)
        .fillna(0.0)
    )
    tfs = list(wide.index)

    if figsize is None:
        figsize = (max(6.0, 0.28 * len(tfs) + 2.0), 4.5)
    fig, ax = plt.subplots(figsize=figsize)

    bottom = pd.Series(0.0, index=wide.index)
    for color in _COLOR_ORDER:
        heights = wide[color]
        if heights.sum() == 0:
            continue
        ax.bar(
            tfs,
            heights,
            bottom=bottom,
            color=COLOR_MAP[color],
            label=_COLOR_LABELS[color],
            width=0.8,
        )
        for tf in tfs:
            if props.loc[tf, color] >= min_label_proportion:
                ax.text(
                    tf,
                    bottom[tf] + heights[tf] / 2,
                    f"{props.loc[tf, color] * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                )
        bottom = bottom + heights

    ax.set_xlabel("Transcription factor (TF)")
    ax.set_ylabel("Enrichment score")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.set_ylim(0, float(wide.sum(axis=1).max()) * 1.08)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Downstream gene", frameon=False, fontsize=7, title_fontsize=7)
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    return fig, ax


def plotly_tf_enrichment_bars(plot_df: pd.DataFrame, title: str) -> Any:
    """Interactive plotly version of :func:`plot_tf_enrichment_bars` (notebook display).

    Static export via ``fig.write_image`` needs kaleido + Chrome (``plotly_get_chrome``);
    use :func:`plot_tf_enrichment_bars` for file output instead.
    """
    import plotly.express as px

    data = plot_df.copy()
    data.loc[data["proportion"] == 0, "text"] = ""
    fig = px.bar(
        data,
        x="source",
        y="height",
        color="color",
        text="text",
        color_discrete_map=COLOR_MAP,
        category_orders={"color": _COLOR_ORDER},
        labels={"height": "Enrichment score", "source": "TF"},
        title=title,
    )
    fig.update_traces(textposition="inside", insidetextanchor="middle")
    fig.update_layout(
        xaxis_title="Transcription factor (TF)",
        yaxis_title="Enrichment score",
        xaxis_tickangle=-90,
        font=dict(family="Arial", size=8, color="black"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True, linecolor="black", ticks="outside"),
        yaxis=dict(showgrid=False, showline=True, linecolor="black", ticks="outside"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chromatin binding dynamics (Plotly)
# (migrated from ``SmoothedCurvesChromatin``; smoothing stays on the class)
# ─────────────────────────────────────────────────────────────────────────────


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
