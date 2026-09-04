"""Memory-lean loader for a dictys ``dynamic.h5``.

``dictys.net.dynamic_network.from_file`` reads *every* dataset under ``prop`` eagerly with
``np.array(...)``. For the B-cell network that is

    es/w, es/w_in, es/w_n   3 x (551, 11907, 194) float64  = 30.6 GB
    es/mask, mask_in, mask_n 3 x                    bool   =  3.8 GB
    nc/readcount            (11907, 28494) int16           =  0.7 GB

~35 GB before any smoothing happens, which does not fit the 64 GB the validation job
asks for.

The episodic pipeline only ever touches ``es/w`` (``stat.net`` defaults to
``varname='w'``), ``ns/cpm`` (``lcpm`` / ``lcpm_tf``) and the trajectory/point objects.
``network.check()` validates only the shapes of whatever properties are present, so
omitting the rest is safe. Peak resident drops to ~11 GB.

This lives in the test folder, not in the package: it is a harness convenience, and it
deliberately returns an object that would be *wrong* for any analysis using ``w_in``
(force waves, validation) or the edge masks.
"""
from __future__ import annotations

from collections import defaultdict

import h5py
import numpy as np

from dictys.net import dynamic_network
from dictys.traj import point, trajectory

#: Property datasets the episodic pipeline actually reads.
KEEP = {
    "es": ("w",),           # stat.net(varname='w') -- the episodic GRN source
    "ns": ("cpm",),         # stat.lcpm / lcpm_tf
    "sc": ("w",),           # cell -> window assignment (StateFrequency)
    "c": ("coord",),        # small, kept so plotting helpers still work
    "ss": ("traj-neighbor",),
}


def load_slim(path: str, keep: dict | None = None, verbose: bool = True) -> dynamic_network:
    """Load ``path`` with only the property datasets in ``keep``.

    Parameters
    ----------
    path
        Path to ``dynamic.h5``.
    keep
        ``{prop_group: (dataset, ...)}``. Defaults to :data:`KEEP`. A group listed with
        an empty tuple is skipped entirely; a group absent from the mapping is skipped.
    verbose
        Print what was loaded and what was skipped, with sizes.

    Returns
    -------
    dictys.net.dynamic_network
    """
    keep = KEEP if keep is None else keep
    params: dict = {}
    loaded, skipped = [], []

    with h5py.File(path, "r") as f:
        for name in f:
            if name in {"prop", "traj", "point", "_version_"}:
                continue
            arr = np.array(f[name])
            if arr.dtype.char == "S":
                arr = arr.astype(str)
            params[name] = arr

        params["prop"] = defaultdict(dict)
        for group in f["prop"]:
            wanted = keep.get(group, ())
            for dset in f["prop"][group]:
                nbytes = f["prop"][group][dset].nbytes
                if dset not in wanted:
                    skipped.append((f"{group}/{dset}", nbytes))
                    continue
                arr = np.array(f["prop"][group][dset])
                if arr.dtype.char == "S":
                    arr = arr.astype(str)
                params["prop"][group][dset] = arr
                loaded.append((f"{group}/{dset}", nbytes))

        if "traj" in f:
            params["traj"] = trajectory.from_fileobj(f["traj"])
        if "point" in f:
            params["point"] = {
                k: point.from_fileobj(params["traj"], f["point"][k]) for k in f["point"]
            }

    params["nids"] = [params.pop("nids1"), params.pop("nids2")]

    if verbose:
        gb = lambda b: b / 1e9  # noqa: E731
        print(f"[slim_loader] {path}")
        for n, b in loaded:
            print(f"[slim_loader]   loaded  prop/{n:<20s} {gb(b):7.2f} GB")
        for n, b in skipped:
            print(f"[slim_loader]   skipped prop/{n:<20s} {gb(b):7.2f} GB")
        print(
            f"[slim_loader]   total loaded {gb(sum(b for _, b in loaded)):.2f} GB, "
            f"skipped {gb(sum(b for _, b in skipped)):.2f} GB"
        )

    return dynamic_network(**params)
