"""Helpers used by more than one module.

The rule from ``references/architecture.md`` §6: if two subpackages need it, it
lives here, not in whichever subpackage happened to need it first.
"""

from focalfire.utils.curves import curvature_of_expression
from focalfire.utils.genes import (
    check_if_gene_in_ndict,
    extract_tf_gene_info,
    get_gene_indices,
    get_tf_indices,
)
from focalfire.utils.parallel import create_balanced_chunks
from focalfire.utils.states import (
    create_enriched_links_per_state,
    get_state_labels_in_window,
    get_state_total_counts,
    get_top_k_fraction_labels,
    window_labels_to_count_df,
)

__all__ = [
    "check_if_gene_in_ndict",
    "create_balanced_chunks",
    "create_enriched_links_per_state",
    "curvature_of_expression",
    "extract_tf_gene_info",
    "get_gene_indices",
    "get_state_labels_in_window",
    "get_state_total_counts",
    "get_tf_indices",
    "get_top_k_fraction_labels",
    "window_labels_to_count_df",
]
