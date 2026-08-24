Utilities
~~~~~~~~~
Helpers used by more than one module. The rule: if two subpackages need it, it lives
here, not in whichever subpackage happened to need it first.

.. module:: firefate.utils
.. currentmodule:: firefate

Genes and networks
------------------
.. autosummary::
    :toctree: genapi

    utils.get_tf_indices
    utils.get_gene_indices
    utils.check_if_gene_in_ndict
    utils.extract_tf_gene_info

Cell states
-----------
.. autosummary::
    :toctree: genapi

    utils.get_state_labels_in_window
    utils.get_state_total_counts
    utils.get_top_k_fraction_labels
    utils.window_labels_to_count_df
    utils.create_enriched_links_per_state

Curves and parallelism
----------------------
.. autosummary::
    :toctree: genapi

    utils.curvature_of_expression
    utils.create_balanced_chunks
