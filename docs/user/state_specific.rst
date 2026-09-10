State-specific
~~~~~~~~~~~~~~
What separates two *fixed* cell states, and what happens when the regulators of that
difference are perturbed (capabilities 2 and static 5; 1 and 6 to follow).

.. module:: focalfire.state_specific
.. currentmodule:: focalfire

Manager
-------
.. autosummary::
    :toctree: genapi

    state_specific.StateSpecificManager

GRN construction
----------------
.. autosummary::
    :toctree: genapi

    state_specific.combine_cluster_grn_links
    state_specific.grn_edges_from_combined_links
    state_specific.filter_network_scores_for_clusters
    state_specific.load_celloracle_grn_tables

Enrichment
----------
.. autosummary::
    :toctree: genapi

    state_specific.StateSpecificEnrichment
    state_specific.hypergeom_slide_grn_score
    state_specific.get_slide_grn_enrichment
    state_specific.build_enrichment_table
    state_specific.write_enrichment_table

Figures
-------
.. autosummary::
    :toctree: genapi

    state_specific.build_tf_color_bar_table
    state_specific.plot_tf_enrichment_bars
    state_specific.plotly_tf_enrichment_bars
    state_specific.plot_strength_key_distribution
    state_specific.plot_episode_from_csvs
