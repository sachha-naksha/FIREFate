"""**FIREFateStateSpecific** -- what separates two fixed cell states.

Covers capabilities 2 (state-specific and cross-state GRNs) and the static half of
5 (enrichment against those GRNs). Capabilities 1 (cellular-program discovery) and
6 (in-silico perturbation) land here next, alongside the celloracle and SLIDE-R
backends.

The SLIDE latent-factor enrichment bar plots live with the enrichment code that
produces their input, in ``_enrichment``.

Start at :class:`StateSpecificManager`.
"""

from firefate.state_specific._enrichment import (
    COLOR_MAP,
    StateSpecificEnrichment,
    build_tf_color_bar_table,
    plot_episode_from_csvs,
    plot_strength_key_distribution,
    plot_tf_enrichment_bars,
    plotly_tf_enrichment_bars,
)
from firefate.state_specific._grn import (
    combine_cluster_grn_links,
    filter_network_scores_for_clusters,
    grn_edges_from_combined_links,
    load_celloracle_grn_tables,
)
from firefate.state_specific._slide import (
    build_enrichment_table,
    get_slide_grn_enrichment,
    hypergeom_slide_grn_score,
    write_enrichment_table,
)
from firefate.state_specific.manager import StateSpecificManager

__all__ = [
    "COLOR_MAP",
    "StateSpecificEnrichment",
    "StateSpecificManager",
    "build_enrichment_table",
    "build_tf_color_bar_table",
    "combine_cluster_grn_links",
    "filter_network_scores_for_clusters",
    "get_slide_grn_enrichment",
    "grn_edges_from_combined_links",
    "hypergeom_slide_grn_score",
    "load_celloracle_grn_tables",
    "plot_episode_from_csvs",
    "plot_strength_key_distribution",
    "plot_tf_enrichment_bars",
    "plotly_tf_enrichment_bars",
    "write_enrichment_table",
]
