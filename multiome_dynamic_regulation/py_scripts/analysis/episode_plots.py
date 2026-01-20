import numpy as np
import pandas as pd
import ast
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

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
        all_tfs_sorted = sorted(tf_genes_dict.keys())
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
    ax_main.grid(True, linestyle='--', alpha=0.3, axis='both')
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