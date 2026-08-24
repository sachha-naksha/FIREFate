"""Readers for the on-disk artefacts dictys produces."""
from __future__ import annotations

import h5py


def read_h5_file(file_path):
    """
    Read HDF5 file and return its contents as a dictionary for easier debugging
    """
    data_dict = {}
    with h5py.File(file_path, "r") as f:
        # Recursively read groups and datasets
        def read_group(group, dict_obj):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    # Convert dataset to numpy array for easier inspection
                    dict_obj[key] = item[()]
                elif isinstance(item, h5py.Group):
                    dict_obj[key] = {}
                    read_group(item, dict_obj[key])

        read_group(f, data_dict)

    return data_dict


def read_adata_from_pkl(pkl_path, workdir):
    """
    Read an AnnData object from a PKL file using stream
    """
    import stream as st

    adata = st.read(file_name=pkl_path, workdir=workdir)
    return adata
