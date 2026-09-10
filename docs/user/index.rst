User API
########

.. module:: focalfire.user

The public surface of the three FocalFire modules. Import :mod:`focalfire` as::

    import focalfire as ff

Each module's entry point is its manager: :class:`~focalfire.temporal.TemporalManager`,
:class:`~focalfire.state_specific.StateSpecificManager`. Figures are not a separate
namespace — every plotting function is exported by the module whose results it draws.

.. toctree::
    :maxdepth: 2

    temporal
    state_specific
    cross_prediction
    io
