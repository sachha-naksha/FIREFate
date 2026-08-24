"""Drive ``dictys.network.reconstruct`` over the per-window subset directories.

Called as a library function rather than through the dictys CLI so a single SLURM
task can walk a range of subsets without re-importing torch each time.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: Files dictys ``reconstruct`` reads and writes inside ``tmp_dynamic/Subset<i>/``.
_INPUTS = {"fi_exp": "expression0.tsv.gz", "fi_mask": "binlinking.tsv.gz"}
_OUTPUTS = {
    "fo_weight": "net_weight.tsv.gz",
    "fo_meanvar": "net_meanvar.tsv.gz",
    "fo_covfactor": "net_covfactor.tsv.gz",
    "fo_loss": "net_loss.tsv.gz",
    "fo_stats": "net_stats.tsv.gz",
}


def subset_dir(work_dir: str, subset_num: int) -> str:
    """Directory holding one trajectory window's reconstruction inputs/outputs."""
    return os.path.join(work_dir, "tmp_dynamic", f"Subset{subset_num}")


def reconstruct_subset(
    work_dir: str,
    subset_num: int,
    *,
    device: str = "cuda",
    nth: int = 5,
) -> bool:
    """Reconstruct the network for one subset. Returns whether it succeeded.

    ``device='cuda'`` deliberately does not pin a GPU index, so CUDA picks up the
    device SLURM assigned to the task.
    """
    from dictys.network import reconstruct

    d = subset_dir(work_dir, subset_num)
    paths = {k: os.path.join(d, v) for k, v in {**_INPUTS, **_OUTPUTS}.items()}
    try:
        reconstruct(device=device, nth=nth, **paths)
    except Exception as e:  # a failed window should not abort the remaining range
        logger.error("Reconstruction failed for Subset %s: %s", subset_num, e)
        return False
    return True


def reconstruct_range(
    work_dir: str,
    start_subset: int,
    end_subset: int,
    *,
    device: str = "cuda",
    nth: int = 5,
) -> list[int]:
    """Reconstruct subsets ``start_subset..end_subset`` inclusive.

    Returns the subset numbers that failed, so the caller can requeue them.
    """
    failed = []
    for subset_num in range(start_subset, end_subset + 1):
        if not reconstruct_subset(work_dir, subset_num, device=device, nth=nth):
            failed.append(subset_num)
    return failed
