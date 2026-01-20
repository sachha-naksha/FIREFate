#!/bin/bash
#SBATCH --job-name="network_to_file"
#SBATCH --output="network_to_file.out"
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128 #for memory to contain all subsets and write to a single file
#SBATCH -t 2:00:00  # Adjust time as needed

# Set the working directory
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/outs/dictys/rbpj_ntc"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Ensure the output directory exists
OUTPUT_DIR="output"
mkdir -p $OUTPUT_DIR  # Create the output directory if it does not exist

# Load necessary modules and activate the environment
module load anaconda3/2022.10
source activate dictys

python -c "
from dictys.network import tofile

tofile(
    diri_data='data',
    diri_work='tmp_dynamic',
    fi_subsets='tmp_dynamic/subsets.txt',
    fo_networks='output/dynamic.h5',
    dynamic=True
)
"

# Log completion
echo "Done"
