FIREFate
========
.. module:: firefate

.. image:: _static/img/fig1_091026.png
    :width: 800px
    :align: center
    :alt: The interpretable modules of FIREFate

|

**FIREFate** (Functional and Interpretable Regulatory Encoding of cellular Fate) uses
interpretable machine learning to focus dense mechanistic gene regulatory networks
onto the components that actually govern cell fate decisions. It works on single-cell
RNA and ATAC data, matched or unmatched.

The framework is three modules in one package, each answering a different question
about regulation:

.. grid:: 3
    :gutter: 2

    .. grid-item-card:: Temporal
        :link: user/temporal
        :link-type: doc

        How does TF regulation change *along* a trajectory? Transition-window and
        episodic GRNs, force waves, regulatory phases.

    .. grid-item-card:: State-specific
        :link: user/state_specific
        :link-type: doc

        What separates two *fixed* states, and what happens if we perturb it?
        State and cross-state GRNs, enrichment, in-silico knockout.

    .. grid-item-card:: Cross-prediction
        :link: user/cross_prediction
        :link-type: doc

        Do programs learned on one dataset *transfer* to stratify another?
        Fate-bias stratification of uncommitted populations.

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
