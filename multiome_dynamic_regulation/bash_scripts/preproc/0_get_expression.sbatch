#!/bin/bash
#SBATCH --job-name="get_expression"
#SBATCH --output="get_expression_%A_%a.log" # Log file for each array job
#SBATCH -p RM-shared
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks-per-node=48 # number of cores per node; memory is 2gb per node for rm-shared
#SBATCH -t 12:00:00

module load anaconda3/2022.10

# Define all input paths if you don't have an aggr cell-ranger output (by creating a list of matrix_dir and output_files and using array jobs to get per-day expression.tsv.gz)
MATRIX_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/outs/filtered_feature_bc_matrix"

# Define the output reference expression file
OUTPUT_FILE="expression.tsv.gz"
OUTPUT_PATH="$LOCAL/$OUTPUT_FILE"

# Python script
PYTHON_FILE="/ocean/projects/cis240075p/asachan/bio_informatics_analysis/B_Cells_human_analysis/analysis_repo/multiome_dynamic_regulation/py_scripts/dynamic_grn/get_main_expression.py"

# Destination directory for the output
DESTINATION_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2/data"

# Trap to ensure output file is copied back on script exit
trap 'cp "$OUTPUT_PATH" "$DESTINATION_DIR/"' EXIT

# Create scratch directory if it doesn't exist
mkdir -p $LOCAL

# Transfer the input files to scratch
cp -r "$MATRIX_DIR"/* $LOCAL/

# Activate the dictys environment
source activate dictys
set -eo pipefail
cd $LOCAL/

# Run the python script with correct arguments
python "$PYTHON_FILE" "$LOCAL" "$OUTPUT_FILE"

echo "get expression done"
