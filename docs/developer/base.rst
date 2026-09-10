Base
~~~~
Contracts and primitives shared by all three modules. Nothing here is domain-specific.

:func:`~focalfire.base.calculate_tf_episodic_enrichment` is the single
over-representation primitive that both the Temporal and StateSpecific enrichment paths
run; :class:`~focalfire.base.BaseManager` is the bookkeeping every module manager
inherits.

.. module:: focalfire.base
.. currentmodule:: focalfire

.. autosummary::
    :toctree: genapi

    base.BaseManager
    base.calculate_tf_episodic_enrichment
