"""Reconstruct a range of per-window networks.

Replaces ``py_scripts/dynamic_grn/network_reconstruct_batch.py``. Run as::

    python -m firefate.cli.reconstruct_networks START END WORK_DIR

which is what ``bash_scripts/network/1_network_reconstruct.sbatch`` calls.
"""
from __future__ import annotations

import argparse
import logging
import sys

from firefate.backends.dictys._reconstruct import reconstruct_range


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("start_subset", type=int, help="First subset (window) to reconstruct.")
    parser.add_argument("end_subset", type=int, help="Last subset (window), inclusive.")
    parser.add_argument("work_dir", help="Directory containing tmp_dynamic/Subset<i>/.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device. 'cuda' leaves the GPU index to SLURM. Default: cuda.",
    )
    parser.add_argument(
        "--nth", type=int, default=5, help="Threads per reconstruction. Default: 5."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    failed = reconstruct_range(
        args.work_dir,
        args.start_subset,
        args.end_subset,
        device=args.device,
        nth=args.nth,
    )
    if failed:
        logging.error("%d subset(s) failed: %s", len(failed), failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
