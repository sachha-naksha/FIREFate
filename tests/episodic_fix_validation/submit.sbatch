#!/bin/bash
#SBATCH --job-name=ff_episodic_fix
#SBATCH --account=bhdw-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Re-runs the episodic enrichment pipeline on the B-cell PB and GC branches with the
# pre-fix and post-fix code, from identical inputs, and reports whether the enriched
# TF set changes.
#
# Resources: matches the interactive `srun` template except --time, raised from 2h to
# 8h. The one-sample t-test in filter_edges_by_significance_and_direction is a Python
# loop over ~10^6 edges per episode and runs 8 times (2 branches x 4 episodes); 2h is
# not a safe budget on 4 cores. Raising --cpus-per-task shortens it roughly linearly
# (the driver reads SLURM_CPUS_PER_TASK), and 64G is comfortable because
# slim_loader.py skips the w_in / w_n / mask* / readcount datasets that
# dictys.net.network.from_file would otherwise load (~25 GB of the 35 GB).
#
# Submit with:
#     cd FocalFire/tests/episodic_fix_validation && sbatch submit.sbatch

set -euo pipefail

# SLURM copies the batch script into /var/spool/slurmd/job<id>/, so ${BASH_SOURCE[0]}
# does NOT point at the submit directory. Prefer SLURM_SUBMIT_DIR, and fall back to
# the script's own location when run directly from a shell.
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$HERE"
if [ ! -f "$HERE/run_validation.py" ]; then
    echo "ERROR: run_validation.py not found in $HERE" >&2
    exit 1
fi

CONDA_ENV=/projects/bgdb/asachan/.conda/envs/dictys
PY="$CONDA_ENV/bin/python"

echo "host        : $(hostname)"
echo "job id      : ${SLURM_JOB_ID:-<none>}"
echo "cpus        : ${SLURM_CPUS_PER_TASK:-?}"
echo "mem         : ${SLURM_MEM_PER_NODE:-?} MB"
echo "started     : $(date -Is)"
echo "python      : $PY"
"$PY" -c "import focalfire; print('focalfire     :', focalfire.__file__)"
"$PY" -c "import dictys;   print('dictys       :', dictys.__file__)"
echo

# Keep BLAS single-threaded: the smoothing matmul is already large and the outer
# multiprocessing pool provides the parallelism.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PY" run_validation.py \
    --outdir "$HERE/results" \
    --branches pb,gc \
    --num-points 20 \
    --points-per-episode 5 \
    --percentile 98 \
    --pval-threshold 0.001 \
    --n-processes "${SLURM_CPUS_PER_TASK:-4}"

echo
echo "finished    : $(date -Is)"
echo "results in  : $HERE/results"
