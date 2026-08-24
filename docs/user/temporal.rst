Temporal
~~~~~~~~
How TF regulation changes *along* a trajectory: transition-window and episodic GRNs,
force waves, regulatory phases (capabilities 3, 4 and dynamic 5).

.. module:: firefate.temporal
.. currentmodule:: firefate

Manager
-------
.. autosummary::
    :toctree: genapi

    temporal.TemporalManager

Curves and time alignment
-------------------------
.. autosummary::
    :toctree: genapi

    temporal.SmoothedCurvesGRN
    temporal.SmoothedCurvesChromatin
    temporal.AlignTimeScales

Episodic GRNs
-------------
.. autosummary::
    :toctree: genapi

    temporal.EpisodeDynamics
    temporal.get_episodic_grn_subset
    temporal.filter_edges_by_significance_and_direction
    temporal.calculate_force_curves_parallel
    temporal.get_unique_regs_by_target
    temporal.run_episodic_construction
    temporal.run_episodic_enrichment

Force waves, states and phases
------------------------------
.. autosummary::
    :toctree: genapi

    temporal.TFForceWaves
    temporal.StateFrequency
    temporal.RegulatoryPhases
    temporal.ForceWavePhases
    temporal.BindingPhases
    temporal.order_links
    temporal.order_links_by_phase
    temporal.get_max_points
    temporal.aggregate_max_points

Validation
----------
.. autosummary::
    :toctree: genapi

    temporal.TFForceValidation

Figures
-------
Colocated with the classes whose output they draw, and re-exported here.

.. autosummary::
    :toctree: genapi

    temporal.plot_force_landscape
    temporal.plot_single_link_landscape
    temporal.plot_force_by_tf
    temporal.plot_force_heatmap
    temporal.plot_force_heatmap_with_clustering
    temporal.plot_force_heatmap_by_phase
    temporal.plot_phase_ordered_force_heatmap
    temporal.plot_phase_binding_boxes
    temporal.plot_force_validation_boxes
    temporal.plot_force_validation_multi
    temporal.plot_force_validation_by_phase
    temporal.plot_force_validation_phase_cells
    temporal.plot_state_composition_bars
    temporal.plot_state_composition_curves
    temporal.plot_state_extrema
    temporal.plot_gene_trajectories
    temporal.plot_expression_for_multiple_genes
    temporal.plot_gene_expression_subplots
    temporal.plot_chromatin_tf_dynamics
    temporal.plot_score_vs_count_subplots
    temporal.plot_main_trajectory_nodes
    temporal.plot_tf_episodic_enrichment_dotplot
    temporal.plot_tf_target_episodic_heatmap
    temporal.plot_tf_gene_coregulation_heatmap
    temporal.fig_regulation_heatmap
    temporal.fig_expression_gradient_heatmap
    temporal.fig_expression_linear_heatmap
    temporal.cluster_heatmap
    temporal.sort_tfs_by_gene_similarity
    temporal.create_pathway_color_scheme
