FIREFate
========
.. module:: firefate

.. image:: _static/img/fig1_091026.png
    :width: 800px
    :align: center
    :alt: The interpretable modules of FIREFate

|

**FIREFate** (Functional and Interpretable Regulatory Encoding of cellular Fate) combines
mechanistic gene regulatory networks (GRNs) with interpretable machine learning to focus
dense state-specific and dynamic GRNs onto the regulatory components that govern cell
fate decisions. It works on single-cell RNA and ATAC data (sc/snRNA-seq, scATAC-seq),
matched or unmatched.

The framework is three modules in one package, one per product in the figure above:

.. grid:: 3
    :gutter: 2

    .. grid-item-card:: State-specific
        :link: user/state_specific
        :link-type: doc

        **Prioritized regulatory subnetworks.** Interpretable ML discovers sparse
        cellular programs (CPs) that separate contrasting cell states, and embeds them
        in state-specific and cross-state GRNs to surface the TFs whose regulons are
        enriched for each program, including in-silico perturbation of the enriched TFs.

    .. grid-item-card:: Temporal
        :link: user/temporal
        :link-type: doc

        **Phase-resolved dynamic regulation.** Transition-window and episodic GRNs
        along pseudotime assign TF–target edges to regulatory phases inferred from
        pseudotemporal clustering, ordering waves of TF regulation and retaining the
        forces that stay invariant within each episode.

    .. grid-item-card:: Cross-prediction
        :link: user/cross_prediction
        :link-type: doc

        **Fate predisposition by transfer learning.** CPs learned from fate-switching
        perturbations (e.g. TF knockouts) versus controls stratify uncommitted
        populations in unperturbed data by their predicted fate bias.

Getting started
---------------
- :doc:`Install <installation>` the package, then read the
  :doc:`seven capabilities <capabilities>` for what FIREFate does and why.
- :doc:`Methods <methods>` gives the formal description of each capability.
- The :doc:`notebooks <notebooks/index>` show the framework applied end to end on
  B-cell and T-cell multiome data.
- :doc:`User API <user/index>` documents the public surface;
  :doc:`Developer API <developer/index>` the shared machinery underneath it.

Important resources
-------------------
.. grid:: 2
    :gutter: 1

    .. grid-item-card:: Installation
        :link: installation
        :link-type: doc

        Install ``firefate``, with or without the notebooks submodule.

    .. grid-item-card:: Notebooks
        :link: notebooks/index
        :link-type: doc

        Analysis notebooks per module, from the companion
        `firefate_notebooks <https://github.com/sachha-naksha/firefate_notebooks>`_
        repository.

    .. grid-item-card:: Contributing
        :link: contributing
        :link-type: doc

        Where new code goes, and the test conventions.

    .. grid-item-card:: Repository
        :link: https://github.com/sachha-naksha/FIREFate
        :link-type: url

        Source code, issues, and development on GitHub.

.. toctree::
    :caption: User guide
    :maxdepth: 2
    :hidden:

    installation
    capabilities
    methods

.. toctree::
    :caption: Notebooks
    :maxdepth: 2
    :hidden:

    notebooks/index

.. toctree::
    :caption: API
    :maxdepth: 2
    :hidden:

    user/index
    developer/index

.. toctree::
    :caption: About
    :maxdepth: 1
    :hidden:

    contributing
    references
