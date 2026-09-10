Command line
~~~~~~~~~~~~
One module per pipeline step, each runnable as ``python -m focalfire.cli.<name>``. The
SLURM scripts under ``multiome_dynamic_regulation/bash_scripts/`` invoke them this way.

.. code-block:: bash

    python -m focalfire.cli.expression_to_tsv MATRIX_DIR expression.tsv.gz
    python -m focalfire.cli.validate_inputs --dir_data data --dir_makefiles makefiles
    python -m focalfire.cli.reconstruct_networks START END WORK_DIR
    python -m focalfire.cli.motifs_to_homer IN.meme OUT.motif

.. automodule:: focalfire.cli.expression_to_tsv
    :members:

.. automodule:: focalfire.cli.validate_inputs
    :members:

.. automodule:: focalfire.cli.reconstruct_networks
    :members:

.. automodule:: focalfire.cli.motifs_to_homer
    :members:
