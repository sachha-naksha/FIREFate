"""Enrichment primitives (episodic ORA lives in ``firefate.core``; SLIDE–GRN here)."""

from firefate.enrichment.lf_bar_plots import (
    build_tf_color_bar_table,
    load_episode_enrichment,
    load_lf_gene_colors,
    load_tf_target_links,
    plot_episode_from_csvs,
    plot_tf_enrichment_bars,
    plotly_tf_enrichment_bars,
)
from firefate.enrichment.slide_grn import (
    build_enrichment_table,
    get_slide_grn_enrichment,
    hypergeom_slide_grn_score,
    write_enrichment_table,
)

__all__ = [
    "build_enrichment_table",
    "build_tf_color_bar_table",
    "get_slide_grn_enrichment",
    "hypergeom_slide_grn_score",
    "load_episode_enrichment",
    "load_lf_gene_colors",
    "load_tf_target_links",
    "plot_episode_from_csvs",
    "plot_tf_enrichment_bars",
    "plotly_tf_enrichment_bars",
    "write_enrichment_table",
]
