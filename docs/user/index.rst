User API
########

.. module:: firefate.user

The public surface of the three FIREFate modules. Import :mod:`firefate` as::

    import firefate as ff

Each module's entry point is its manager: :class:`~firefate.temporal.TemporalManager`,
:class:`~firefate.state_specific.StateSpecificManager`. Figures are not a separate
namespace — every plotting function is exported by the module whose results it draws.

.. toctree::
    :maxdepth: 2

    temporal
    state_specific
    cross_prediction
    io
