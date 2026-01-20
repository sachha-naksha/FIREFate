#!/bin/bash
#SBATCH --job-name="QC"
#SBATCH --output="QC.out"
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node=16 #mem=2gb per core
#SBATCH --array=0-193 # no of subsets

# Set the working directory to where data is located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2"
INPUT_MASK="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/other_files/combinatorial_control/genes.txt"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Calculate the subset number based on the job array ID
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# Construct the input and output file paths
INPUT_GEX="tmp_dynamic/Subset${SUBSET_NUM}/expression0.tsv.gz"
INPUT_ATAC="tmp_dynamic/Subset${SUBSET_NUM}/names_atac0.txt"
OUTPUT_GEX="tmp_dynamic/Subset${SUBSET_NUM}/expression.tsv.gz"
OUTPUT_ATAC="tmp_dynamic/Subset${SUBSET_NUM}/names_atac.txt"

####################################### RUN COMMAND #######################################
module load anaconda3/2022.10
source activate dictys

# Check if input file exists and run QC using Python directly
if [[ -f "$INPUT_GEX" ]]; then
    python -c "
import dictys.preproc
dictys.preproc.qc_reads('$INPUT_GEX', '$INPUT_MASK', '$OUTPUT_GEX', 50, 10, 0, 200, 10, 0)
"
else
    echo "Error: Input file $INPUT_GEX not found for Subset${SUBSET_NUM}"
    exit 1
fi

# Process ATAC names list using Python directly
if [[ -f "$INPUT_ATAC" ]]; then
    python -c "
import dictys.preproc
dictys.preproc.selects_atac('$OUTPUT_GEX', '$INPUT_ATAC', '$OUTPUT_ATAC')
"
else
    echo "Error: Input file $INPUT_ATAC not found for Subset${SUBSET_NUM}"
    exit 1
fi