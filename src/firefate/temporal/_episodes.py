"""Episodic GRNs: edges whose TF-target force stays temporally invariant across an episode."""
from __future__ import annotations

import os
import time
import pickle
import glob
import re
import dictys
import numpy as np
import pandas as pd
from dictys.net import stat
from firefate.base.enrichment import calculate_tf_episodic_enrichment
from firefate.temporal._align import AlignTimeScales
from firefate.temporal._curves import SmoothedCurvesGRN
from firefate.temporal._forces import (
    calculate_force_curves_parallel,
    filter_edges_by_significance_and_direction,
)
from firefate.utils.genes import check_if_gene_in_ndict
import matplotlib
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import ast
from pathlib import Path
import matplotlib as mpl
import matplotlib.gridspec as gridspec


class EpisodeDynamics:
    """
    workflow for episodic grn extraction, filtering, force calculation, and enrichment.
    """

    def __init__(self, dictys_dynamic_object, output_folder, mode="expression",
                 trajectory_range=(1, 3), num_points=40, dist=0.001, sparsity=0.01, n_processes=16):
        
        # Core parameters
        self.dictys_dynamic_object = dictys_dynamic_object
        self.output_folder = output_folder
        self.mode = mode
        self.trajectory_range = trajectory_range
        self.num_points = num_points
        self.dist = dist
        self.sparsity = sparsity
        self.n_processes = n_processes
        
        # Initialize composed objects with same parameters
        self.curves = SmoothedCurvesGRN(
            dictys_dynamic_object=dictys_dynamic_object,
            trajectory_range=trajectory_range,
            num_points=num_points,
            dist=dist,
            sparsity=sparsity,
            mode=mode
        )
        
        self.time_aligner = AlignTimeScales(
            dictys_dynamic_object=dictys_dynamic_object,
            trajectory_range=trajectory_range,
            num_points=num_points,
            dist=dist,
            sparsity=sparsity
        )
        
        # State variables
        self.lcpm_dcurve = None
        self.dtime = None
        self.episode_beta_dcurve = None
        self.filtered_edges = None
        self.filtered_edges_p001 = None
        self.tf_lcpm_episode = None
        self.force_curves = None
        self.avg_force_df = None
        self.episodic_grn_edges = None
        self.lf_genes = None
        self.lf_in_object = None
        self.episodic_enrichment_df = None

    def compute_expression_curves(self, mode="expression"):
        """
        expression curves for all genes using the curves component.
        """
        # Update the mode in the curves object
        self.curves.mode = mode
        lcpm_dcurve, dtime = self.curves.get_smoothed_curves()
        self.lcpm_dcurve = lcpm_dcurve
        self.dtime = dtime
        return lcpm_dcurve, dtime

    def build_episode_grn(self, time_slice=slice(0, 5)):
        """
        build the episodic grn (weighted and binarized) for the specified episode/time window.
        """
        pts, fsmooth = self.dictys_dynamic_object.linspace(
            self.trajectory_range[0],
            self.trajectory_range[1],
            self.num_points,
            self.dist,
        )
        stat1_net = fsmooth(stat.net(self.dictys_dynamic_object))
        stat1_netbin = stat.fbinarize(stat1_net, sparsity=self.sparsity)
        dnet = stat1_net.compute(pts)
        dnetbin = stat1_netbin.compute(pts)
        dnet_episode = dnet[:, :, time_slice]
        dnetbin_episode = dnetbin[:, :, time_slice]

        # Map indices to gene names
        ndict = self.dictys_dynamic_object.ndict
        index_to_gene = {idx: name for name, idx in ndict.items()}
        target_names = [index_to_gene[idx] for idx in range(dnetbin_episode.shape[1])]
        tf_gene_indices = [
            self.dictys_dynamic_object.nids[0][tf_idx]
            for tf_idx in range(dnetbin_episode.shape[0])
        ]
        tf_names = [index_to_gene[idx] for idx in tf_gene_indices]

        # reshape to dataframe
        index_tuples = [(tf, target) for tf in tf_names for target in target_names]
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=["TF", "Target"])
        n_tfs, n_targets, n_times = dnet_episode.shape
        reshaped_data = dnet_episode.reshape(-1, n_times)
        episode_beta_dcurve = pd.DataFrame(
            reshaped_data,
            index=multi_index,
            columns=[f"time_{i}" for i in range(n_times)],
        )
        episode_beta_dcurve = episode_beta_dcurve[episode_beta_dcurve.sum(axis=1) != 0]

        # remove tfs with names starting with ZNF and ZBTB
        episode_beta_dcurve = episode_beta_dcurve[
            ~episode_beta_dcurve.index.get_level_values(0).str.startswith("ZNF")
            & ~episode_beta_dcurve.index.get_level_values(0).str.startswith("ZBTB")
        ]
        self.episode_beta_dcurve = episode_beta_dcurve
        return episode_beta_dcurve

    def filter_edges(
        self,
        min_nonzero_timepoints=3,
        alpha=0.05,
        min_observations=3,
        check_direction_invariance=True,
        n_processes=16,
        chunk_size=8000,
        pval_threshold=0.001,
    ):
        """
        filter episodic grn edges for significance and direction invariance.
        """
        filtered_edges = filter_edges_by_significance_and_direction(
            self.episode_beta_dcurve,
            min_nonzero_timepoints=min_nonzero_timepoints,
            alpha=alpha,
            min_observations=min_observations,
            check_direction_invariance=check_direction_invariance,
            n_processes=n_processes,
            chunk_size=chunk_size,
            save_intermediate=False,
            intermediate_path=self.output_folder,
        )
        self.filtered_edges = filtered_edges
        filtered_edges_p001 = filtered_edges[filtered_edges["p_value"] < pval_threshold]
        self.filtered_edges_p001 = filtered_edges_p001
        return filtered_edges_p001

    def compute_tf_expression(self):
        """
        compute tf expression for the episode (matching time window).
        """
        tf_names = self.filtered_edges_p001.index.get_level_values(0).unique()
        tf_lcpm_values = self.lcpm_dcurve.loc[tf_names]
        n_time_cols = len(
            [col for col in self.filtered_edges_p001.columns if col.startswith("time_")]
        )
        tf_lcpm_episode = tf_lcpm_values.iloc[:, 0:n_time_cols]
        tf_lcpm_episode.columns = [
            col for col in self.filtered_edges_p001.columns if col.startswith("time_")
        ][:n_time_cols]
        self.tf_lcpm_episode = tf_lcpm_episode
        return tf_lcpm_episode

    def calculate_forces(self, n_processes=20, chunk_size=30000, epsilon=1e-10):
        """
        calculate force curves for the filtered episodic grn.
        """
        beta_curves_for_force = self.filtered_edges_p001.drop("p_value", axis=1)
        force_curves = calculate_force_curves_parallel(
            beta_curves=beta_curves_for_force,
            tf_expression=self.tf_lcpm_episode,
            n_processes=n_processes,
            chunk_size=chunk_size,
            epsilon=epsilon,
            save_intermediate=False,
        )
        self.force_curves = force_curves
        avg_force = force_curves.mean(axis=1)
        avg_force_df = avg_force.to_frame(name="avg_force")
        self.avg_force_df = avg_force_df
        return avg_force_df

    def select_top_edges(self, percentile=98):
        """
        select the top k% of edges by absolute average force to build the episodic grn.
        """
        threshold = np.percentile(np.abs(self.avg_force_df["avg_force"]), percentile)
        top_percent_mask = np.abs(self.avg_force_df["avg_force"]) >= threshold
        episodic_grn_edges = self.avg_force_df[top_percent_mask].copy()
        self.episodic_grn_edges = episodic_grn_edges
        return episodic_grn_edges

    def select_top_activating_and_repressing_edges(
        self, percentile_positive=98.5, percentile_negative=0.5
    ):
        """
        select the top k% of edges by average force to build the episodic grn.
        selects top k% positive and top k% negative edges separately.
        returns the selected edges as a dataframe.
        """
        avg_force = self.avg_force_df["avg_force"]
        # Separate positive and negative selection
        positive_forces = avg_force[avg_force > 0]
        negative_forces = avg_force[avg_force < 0]
        # Top k% positive
        if len(positive_forces) > 0:
            pos_threshold = np.percentile(positive_forces, percentile_positive)
            top_pos_edges = positive_forces[positive_forces >= pos_threshold]
        else:
            top_pos_edges = pd.Series(dtype=avg_force.dtype)
        # Top k% negative (most negative)
        if len(negative_forces) > 0:
            neg_threshold = np.percentile(negative_forces, percentile_negative)
            top_neg_edges = negative_forces[negative_forces <= neg_threshold]
        else:
            top_neg_edges = pd.Series(dtype=avg_force.dtype)
        episodic_grn_edges = pd.concat([top_pos_edges, top_neg_edges]).to_frame(
            name="avg_force"
        )
        episodic_grn_edges = episodic_grn_edges.sort_values(
            by="avg_force", ascending=False
        )
        self.episodic_grn_edges = episodic_grn_edges
        return episodic_grn_edges

    def set_lf_genes(self, lf_genes):
        """
        check if the lf genes are in the dictys dynamic object.
        """
        self.lf_genes = lf_genes
        self.lf_in_object = check_if_gene_in_ndict(
            self.dictys_dynamic_object, lf_genes, return_index=True
        )
        return self.lf_in_object

    def annotate_lf_in_grn(self):
        """
        annotate which targets in the episodic grn are lf genes.
        """
        if self.lf_genes is None:
            raise ValueError("LF genes not set. Use set_lf_genes() first.")
        self.episodic_grn_edges["is_in_lf"] = (
            self.episodic_grn_edges.index.get_level_values(1).isin(self.lf_genes)
        )
        return self.episodic_grn_edges

    def calculate_enrichment(self):
        """
        calculate tf enrichment for lf genes in the episodic grn.
        """
        lf_in_episodic_grn = self.episodic_grn_edges[
            self.episodic_grn_edges["is_in_lf"]
        ]
        lf_genes_active_in_episode = lf_in_episodic_grn.index.get_level_values(
            1
        ).unique()
        target_genes_in_episodic_grn = self.episodic_grn_edges.index.get_level_values(
            1
        ).unique()
        enrichment_df = calculate_tf_episodic_enrichment(
            self.episodic_grn_edges,
            total_lf_genes=len(lf_genes_active_in_episode),
            total_genes_in_grn=len(target_genes_in_episodic_grn),
        )
        # sort the enrichment_score in descending order
        episodic_enrichment_df_sorted = enrichment_df.sort_values(
            by="enrichment_score", ascending=False
        )
        # drop rows with 0 enrichment score
        episodic_enrichment_df_sorted = episodic_enrichment_df_sorted[
            episodic_enrichment_df_sorted["enrichment_score"] != 0
        ]
        self.episodic_enrichment_df = episodic_enrichment_df_sorted
        return episodic_enrichment_df_sorted

    def episodic_composition(self):
        """
        Cellular composition of the episode is the union of all 
        window compositions present in the episode.
        """
        return None


def get_episodic_grn_subset(
    output_folder: str,
    tfs_of_interest: list,
    targets_of_interest: list,
    value_col: str = 'avg_force'
) -> pd.DataFrame:
    """
    Loop over all episode pickle files in ``output_folder``, subset edges to the
    TFs and targets of interest, and return a wide DataFrame.

    Rows are ``(tf, target)`` edges; columns are episode indices; values come from
    ``value_col`` (e.g. ``avg_force``), or zero if the edge is absent in that episode.

    Parameters
    ----------
    output_folder : str
        Directory containing ``episode_0.pkl``, ``episode_1.pkl``, etc.
    tfs_of_interest : list
        TF names to keep.
    targets_of_interest : list
        Target gene names to keep.
    value_col : str, optional
        Column in each episode dataframe holding the edge weight. Default ``avg_force``.

    Returns
    -------
    pandas.DataFrame
        Shape ``(n_edges, n_episodes)``.
    """
    # ------------------------------------------------------------------ #
    # 1. Discover all episode pkl files, sorted by episode index           #
    # ------------------------------------------------------------------ #
    pattern = os.path.join(output_folder, 'episode_*.pkl')
    episode_files = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r'episode_(\d+)\.pkl', p).group(1))
    )

    if not episode_files:
        raise FileNotFoundError(f"No episode pkl files found in: {output_folder}")

    # ------------------------------------------------------------------ #
    # 2. Build the union of edges present across all episodes              #
    # ------------------------------------------------------------------ #
    episode_series = {}   # {episode_idx: pd.Series indexed by (tf, target)}

    for fpath in episode_files:
        ep_idx = int(re.search(r'episode_(\d+)\.pkl', fpath).group(1))

        with open(fpath, 'rb') as f:
            ep_df = pickle.load(f)

        # Subset to TFs and targets of interest (assumes MultiIndex: level 0 = TF, level 1 = target)
        mask = (
            ep_df.index.get_level_values(0).isin(tfs_of_interest) &
            ep_df.index.get_level_values(1).isin(targets_of_interest)
        )
        subset = ep_df.loc[mask, value_col]   # pd.Series with (tf, target) MultiIndex

        episode_series[ep_idx] = subset

    # ------------------------------------------------------------------ #
    # 3. Concatenate into a wide DataFrame, fill missing edges with 0     #
    # ------------------------------------------------------------------ #
    result = (
        pd.DataFrame(episode_series)   # rows = edges, cols = episode indices
          .fillna(0)
          .sort_index(axis=1)          # ensure episode columns are ordered
    )

    result.index.names = ['TF', 'target']
    result.columns.name = 'episode'

    return result


# ---------------------------------------------------------------------------
# Figures: episodic enrichment dotplots and heatmaps
# ---------------------------------------------------------------------------

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
    
    Parameters
    ----------
    horizontal_layout : bool, default False
        If True, episodes are on y-axis (top to bottom) and TFs on x-axis (left to right).
        If False (default), TFs are on y-axis and episodes on x-axis.

    Notes
    -----
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
