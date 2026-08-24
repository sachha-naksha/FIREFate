Command line
~~~~~~~~~~~~
One module per pipeline step, each runnable as ``python -m firefate.cli.<name>``. The
SLURM scripts under ``multiome_dynamic_regulation/bash_scripts/`` invoke them this way.

.. code-block:: bash

    python -m firefate.cli.expression_to_tsv MATRIX_DIR expression.tsv.gz
    python -m firefate.cli.validate_inputs --dir_data data --dir_makefiles makefiles
    python -m firefate.cli.reconstruct_networks START END WORK_DIR
    python -m firefate.cli.motifs_to_homer IN.meme OUT.motif

.. automodule:: firefate.cli.expression_to_tsv
    :members:

.. automodule:: firefate.cli.validate_inputs
    :members:

.. automodule:: firefate.cli.reconstruct_networks
    :members:

.. automodule:: firefate.cli.motifs_to_homer
    :members:
