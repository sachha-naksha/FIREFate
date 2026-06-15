#!/bin/bash
#SBATCH --job-name=simulation_randomTFs2
#SBATCH --output=/ocean/projects/cis240075p/heidarir/CellOracle_Tonsil_Bcells/logs/05_simulation_randomTFs2.log
#SBATCH --error=/ocean/projects/cis240075p/heidarir/CellOracle_Tonsil_Bcells/logs/05_simulation_randomTFs2.err
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 1-12:30:00
#SBATCH --mem=100G

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

/ocean/projects/cis240075p/skeshari/.conda/envs/celloracle_env/bin/python \
  /ocean/projects/cis240075p/heidarir/CellOracle_Tonsil_Bcells/Notebooks/05_simulation/05_simulation_randomTFs.py