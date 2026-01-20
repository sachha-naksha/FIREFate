#!/bin/bash
#SBATCH --job-name="subset_traj"
#SBATCH --output="windows_rbpj_ntc.log"

#SBATCH -p RM-shared
#SBATCH -N 1 # by default for gpu-shared you can only request 1 node
#SBATCH -t 24:00:00 # wall-time
#SBATCH --ntasks-per-node 32 # how many cards do you want?

# Set the working directory to where data and makefiles are located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/outs/dictys/rbpj_ntc"
# Change directory to the working directory
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

################ RUN dictys in dynamic mode ################

module load anaconda3/2022.10
source activate dictys
mkdir -p tmp_dynamic

# Calculate parameters
total_cells_in_dataset=2088 # Rbpj-ntc
#ncell=$(( (${total_cells_in_dataset} * 155 / 1000) / 100 * 100 ))
ncell=500
#noverlap=$(( (${ncell} * 875 / 1000) / 50 * 50 ))
noverlap=400
max_pseudotime_in_traj=0.0156
# dmax=$(awk -v max=$max_pseudotime_in_traj 'BEGIN {
#     raw = 0.004 * max;
#     scaled = raw * 1000000;
#     rounded = int(scaled / 5) * 5;
#     printf "%.6f", rounded / 1000000;
# }')
dmax=0.005

python -c "
from dictys.dynamic import subsets_rna

# get the subsets
subsets_rna(
    fi_traj='$WORK_DIR/data/traj_node.h5',
    fi_traj_cell_rna='$WORK_DIR/data/traj_cell_rna.h5',
    fi_coord_rna='$WORK_DIR/data/coord_rna.tsv.gz',
    diro_subsets='$WORK_DIR/tmp_dynamic',
    fo_subsets='$WORK_DIR/tmp_dynamic/subsets.txt',
    fo_subset_locs='$WORK_DIR/tmp_dynamic/subset_locs.h5',
    fo_subset_edges='$WORK_DIR/tmp_dynamic/subset_edges.tsv.gz',
    ncell=$ncell,
    noverlap=$noverlap,
    dmax=$dmax)
"

echo "Trajectory subsets made for Rbpj-ntc"