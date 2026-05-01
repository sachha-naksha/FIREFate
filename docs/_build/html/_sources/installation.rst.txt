Installation
============

Python **3.10+** is required.

From a clone of the repository:

.. code-block:: bash

   pip install .

Optional dependency groups (defined in ``pyproject.toml``):

.. code-block:: bash

   pip install ".[scanpy]"
   pip install ".[celloracle]"

Documentation tools (also used on Read the Docs):

.. code-block:: bash

   pip install ".[docs]"

Build the HTML docs locally:

.. code-block:: bash

   sphinx-build -W -b html docs docs/_build/html

Then open ``docs/_build/html/index.html`` in a browser.
