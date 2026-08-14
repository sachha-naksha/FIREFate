"""Episodic enrichment and TF–target heatmap plotting (migrated from ``episode_plots``).

Also provides interactive 3D regulatory-force landscape visualizations
(``plot_force_landscape`` and friends) built on Plotly.
"""

import numpy as np
import pandas as pd
import ast
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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