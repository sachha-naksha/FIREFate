Backends
~~~~~~~~
Third-party engines FocalFire drives, each behind its own subpackage. Every
``import dictys`` lives here rather than in the domain modules, so the engines stay
swappable and the domain code stays testable against fakes.

The celloracle and SLIDE-R backends land with the StateSpecific port.

.. module:: focalfire.backends
.. currentmodule:: focalfire

dictys
------
.. autosummary::
    :toctree: genapi

    backends.dictys.lcpm_tf
    backends.dictys.ChromatinGRNStat
    backends.dictys.read_h5_file
    backends.dictys.read_adata_from_pkl
    backends.dictys.reconstruct_subset
    backends.dictys.reconstruct_range
    backends.dictys.subset_dir
