"""**FIREFateTemporal** -- how TF regulation changes along a trajectory.

Covers capabilities 3 (transition-window GRNs), 4 (episodic GRNs) and the dynamic
half of 5 (episodic enrichment).

Start at :class:`TemporalManager`; the classes below are what it composes. Each
figure lives in the module that owns its subject -- force landscapes with
:class:`TFForceWaves` in ``_waves``, phase heatmaps with the phase classes in
``_phases``, and so on -- so there is no separate plotting namespace to keep in
sync.
"""

from firefate.temporal._align import AlignTimeScales, plot_main_trajectory_nodes
from firefate.temporal._chromatin import (
    SmoothedCurvesChromatin,
    plot_chromatin_tf_dynamics,
    plot_score_vs_count_subplots,
)
from firefate.temporal._curves import (
    SmoothedCurvesGRN,
    fig_expression_gradient_heatmap,
    fig_expression_linear_heatmap,
    fig_regulation_heatmap,
    plot_expression_for_multiple_genes,
    plot_gene_expression_subplots,
)
from firefate.temporal._episodes import (
    EpisodeDynamics,
    create_pathway_color_scheme,
    get_episodic_grn_subset,
    plot_tf_episodic_enrichment_dotplot,
    plot_tf_gene_coregulation_heatmap,
    plot_tf_target_episodic_heatmap,
    sort_tfs_by_gene_similarity,
)
from firefate.temporal._forces import (
    calculate_force_curves,
    calculate_force_curves_parallel,
    filter_edges_by_significance_and_direction,
    get_unique_regs_by_target,
)
from firefate.temporal._source import TFForceSource
from firefate.temporal._reductions import (
    abs_max_force,
    mean_force,
    softmax_peak,
)
from firefate.temporal._phases import (
    BindingPhases,
    ForceWavePhases,
    RegulatoryPhases,
    aggregate_max_points,
    get_max_points,
    order_links,
    order_links_by_phase,
    plot_force_heatmap_by_phase,
    plot_phase_binding_boxes,
    plot_phase_ordered_force_heatmap,
)
from firefate.temporal._states import (
    StateFrequency,
    plot_state_composition_bars,
    plot_state_composition_curves,
    plot_state_extrema,
)
from firefate.temporal._validation import (
    TFForceValidation,
    plot_force_validation_boxes,
    plot_force_validation_by_phase,
    plot_force_validation_multi,
    plot_force_validation_phase_cells,
)
from firefate.temporal._waves import (
    TFForceWaves,
    cluster_heatmap,
    plot_force_by_tf,
    plot_force_heatmap,
    plot_force_heatmap_with_clustering,
    plot_force_landscape,
    plot_gene_trajectories,
    plot_single_link_landscape,
)
from firefate.temporal.manager import (
    TemporalManager,
    run_episodic_construction,
    run_episodic_enrichment,
)

__all__ = [
    "AlignTimeScales",
    "BindingPhases",
    "EpisodeDynamics",
    "ForceWavePhases",
    "RegulatoryPhases",
    "SmoothedCurvesChromatin",
    "SmoothedCurvesGRN",
    "StateFrequency",
    "TFForceSource",
    "TFForceValidation",
    "TFForceWaves",
    "TemporalManager",
    "abs_max_force",
    "aggregate_max_points",
    "calculate_force_curves",
    "calculate_force_curves_parallel",
    "cluster_heatmap",
    "create_pathway_color_scheme",
    "fig_expression_gradient_heatmap",
    "fig_expression_linear_heatmap",
    "fig_regulation_heatmap",
    "filter_edges_by_significance_and_direction",
    "get_episodic_grn_subset",
    "get_max_points",
    "mean_force",
    "get_unique_regs_by_target",
    "order_links",
    "order_links_by_phase",
    "softmax_peak",
    "plot_chromatin_tf_dynamics",
    "plot_expression_for_multiple_genes",
    "plot_force_by_tf",
    "plot_force_heatmap",
    "plot_force_heatmap_by_phase",
    "plot_force_heatmap_with_clustering",
    "plot_force_landscape",
    "plot_force_validation_boxes",
    "plot_force_validation_by_phase",
    "plot_force_validation_multi",
    "plot_force_validation_phase_cells",
    "plot_gene_expression_subplots",
    "plot_gene_trajectories",
    "plot_main_trajectory_nodes",
    "plot_phase_binding_boxes",
    "plot_phase_ordered_force_heatmap",
    "plot_score_vs_count_subplots",
    "plot_single_link_landscape",
    "plot_state_composition_bars",
    "plot_state_composition_curves",
    "plot_state_extrema",
    "plot_tf_episodic_enrichment_dotplot",
    "plot_tf_gene_coregulation_heatmap",
    "plot_tf_target_episodic_heatmap",
    "run_episodic_construction",
    "run_episodic_enrichment",
    "sort_tfs_by_gene_similarity",
]
