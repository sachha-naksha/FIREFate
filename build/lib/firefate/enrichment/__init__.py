"""Enrichment primitives (episodic ORA lives in ``firefate.core``; SLIDE–GRN here)."""

from firefate.enrichment.slide_grn import (
    build_enrichment_table,
    get_slide_grn_enrichment,
    hypergeom_slide_grn_score,
    write_enrichment_table,
)

__all__ = [
    "build_enrichment_table",
    "get_slide_grn_enrichment",
    "hypergeom_slide_grn_score",
    "write_enrichment_table",
]
