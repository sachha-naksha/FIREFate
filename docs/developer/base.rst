Base
~~~~
Contracts and primitives shared by all three modules. Nothing here is domain-specific.

:func:`~firefate.base.calculate_tf_episodic_enrichment` is the single
over-representation primitive that both the Temporal and StateSpecific enrichment paths
run; :class:`~firefate.base.BaseManager` is the bookkeeping every module manager
inherits.

.. module:: firefate.base
.. currentmodule:: firefate

.. autosummary::
    :toctree: genapi

    base.BaseManager
    base.calculate_tf_episodic_enrichment
