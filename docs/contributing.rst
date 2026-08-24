Contributing
############

Development install
===================

.. code-block:: bash

    git clone --recurse-submodules https://github.com/sachha-naksha/FIREFate
    cd FIREFate
    pip install -e ".[dev,docs]"

``--recurse-submodules`` brings in ``docs/notebooks``. If you cloned without it,
``git submodule update --init docs/notebooks``.

Where code goes
===============

FIREFate is three modules in one package. A new function belongs in the module whose
question it answers, not in a layer named after its technique:

``firefate.temporal``
    How does regulation change *along* a trajectory?

``firefate.state_specific``
    What separates two *fixed* states, and what happens if we perturb it?

``firefate.cross_prediction``
    Do programs learned on one dataset *transfer* to another?

Two rules follow from that:

* **Figures are colocated.** A plotting function lives in the file that owns its
  subject — force landscapes with :class:`~firefate.temporal.TFForceWaves`, phase
  heatmaps with the phase classes. There is no ``firefate.plotting``.
* **Shared code moves down, never sideways.** If two modules need the same helper it
  goes in :mod:`firefate.utils` or :mod:`firefate.base`, not into whichever module
  needed it first. Third-party engines go behind :mod:`firefate.backends`.

Files are private (``_curves.py``, ``_phases.py``); the public surface is whatever the
subpackage ``__init__`` re-exports. See ``references/module_structure_plan.md`` for the
full layout and the reasoning behind it.

Tests
=====

.. code-block:: bash

    pytest tests/temporal -q

Conventions carried over from ``tests/temporal/README.md``:

* every expected value is derived on paper or computed by an *independent*
  implementation — never by calling the function under test;
* a test named ``test_currently_*`` pins behaviour that is wrong but load-bearing, with
  a matching ``xfail(strict=True)`` asserting what it should be. Fixing the bug turns
  the xfail into an ``XPASS`` failure, which is the prompt to delete the ``currently``
  test.

Notebooks
=========

Notebooks live in `firefate_notebooks
<https://github.com/sachha-naksha/firefate_notebooks>`_, not in this repository.
Commit them **with their outputs** — the docs render stored outputs and never execute
a cell, so a stripped notebook renders as an empty page. Run ``add_titles.py`` there
after adding one, so it gets a page title and a sidebar link.

Documentation
=============

.. code-block:: bash

    sphinx-build -b html docs docs/_build/html

The build needs no data and no GPU: ``nb_execution_mode = "off"``, and ``dictys`` is
mocked for autodoc.
