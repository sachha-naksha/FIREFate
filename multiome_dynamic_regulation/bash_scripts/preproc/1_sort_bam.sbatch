#!/bin/bash
#SBATCH --job-name="sort_atac_bams"
#SBATCH --output="sort_bam_all_days_%A_%a.log"
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 6:00:00
#SBATCH --array=0-6

# Error handling
set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Set environment variables before loading modules
export CONDA_ROOT="/opt/packages/anaconda3-2022.10"
export UDUNITS2_XML_PATH="${CONDA_ROOT}/share/udunits/udunits2.xml"

# Load modules and initialize conda
module purge  # Clear any existing modules
module load anaconda3/2022.10

# Initialize conda properly
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda init bash
source ~/.bashrc  # Re-source bashrc to apply conda initialization

conda activate dictys

# Function to log messages with timestamps
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to clean up directories
cleanup_directories() {
    local dir=$1
    log_message "Cleaning up directory: $dir"
    rm -rf "${dir}" "${dir}_text" "${dir}_header" || log_message "Warning: Cleanup failed for $dir"
}

# Function to handle errors
error_handler() {
    log_message "Error occurred in script at line: $1"
    cleanup_directories "${OUTPUT_FOLDERS[$SLURM_ARRAY_TASK_ID]}"
    exit 1
}

# Set error trap
trap 'error_handler ${LINENO}' ERR

# Define arrays
BAM_FILES=(
    "/ocean/projects/cis240075p/skeshari/igvf/bcell2/female_donor/in_data_required/donor2_stanford/GEM1/atac_possorted_bam.bam"
    "/ocean/projects/cis240075p/skeshari/igvf/bcell2/female_donor/in_data_required/donor2_stanford/GEM2/atac_possorted_bam.bam"
    "/ocean/projects/cis240075p/skeshari/igvf/bcell2/female_donor/in_data_required/donor2_stanford/GEM3/atac_possorted_bam.bam"
    "/ocean/projects/cis240075p/skeshari/igvf/bcell2/female_donor/in_data_required/donor2_stanford/GEM4/atac_possorted_bam.bam"
    "/ocean/projects/cis240075p/skeshari/igvf/bcell2/female_donor/in_data_required/donor2_stanford/GEM5/atac_possorted_bam.bam"
    "/ocean/projects/cis240075p/skeshari/igvf/bcell2/female_donor/in_data_required/donor2_stanford/GEM6/atac_possorted_bam.bam"
    "/ocean/projects/cis240075p/skeshari/igvf/bcell2/female_donor/in_data_required/donor2_stanford/GEM7/atac_possorted_bam.bam"
)

REF_EXPRESSIONS=(
    "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_2nd_donor/expression_mtx/day1_expression.tsv.gz"
    "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_2nd_donor/expression_mtx/day2_expression.tsv.gz"
    "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_2nd_donor/expression_mtx/day3_expression.tsv.gz"
    "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_2nd_donor/expression_mtx/day4_expression.tsv.gz"
    "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_2nd_donor/expression_mtx/day5_expression.tsv.gz"
    "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_2nd_donor/expression_mtx/day6_expression.tsv.gz"
    "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_2nd_donor/expression_mtx/day7_expression.tsv.gz"
)

OUTPUT_FOLDERS=(
    "$LOCAL/sorted_bams_day1"
    "$LOCAL/sorted_bams_day2"
    "$LOCAL/sorted_bams_day3"
    "$LOCAL/sorted_bams_day4"
    "$LOCAL/sorted_bams_day5"
    "$LOCAL/sorted_bams_day6"
    "$LOCAL/sorted_bams_day7"
)

DESTINATION_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_2nd_donor/sorting_atac_outs"

# Clean up and create directories
log_message "Cleaning up existing directories"
cleanup_directories "${OUTPUT_FOLDERS[$SLURM_ARRAY_TASK_ID]}"

log_message "Creating output directory"
mkdir -p "${OUTPUT_FOLDERS[$SLURM_ARRAY_TASK_ID]}"

# Copy input files
log_message "Copying input files to local storage"
cp "${BAM_FILES[$SLURM_ARRAY_TASK_ID]}" "$LOCAL/" || { log_message "Failed to copy BAM file"; exit 1; }
cp "${REF_EXPRESSIONS[$SLURM_ARRAY_TASK_ID]}" "$LOCAL/" || { log_message "Failed to copy reference file"; exit 1; }

# Set up trap for copying results
trap 'log_message "Copying results to destination"; cp -r "${OUTPUT_FOLDERS[$SLURM_ARRAY_TASK_ID]}" "$DESTINATION_DIR/"' EXIT

# Change to local directory
cd "$LOCAL"

# Prepare input files
INPUT_BAM=$(basename "${BAM_FILES[$SLURM_ARRAY_TASK_ID]}")
REF_EXPRESSION=$(basename "${REF_EXPRESSIONS[$SLURM_ARRAY_TASK_ID]}")
OUTPUT_FOLDER="${OUTPUT_FOLDERS[$SLURM_ARRAY_TASK_ID]}"

# Run dictys_helper
log_message "Running dictys_helper for $INPUT_BAM"
dictys_helper split_bam.sh "$INPUT_BAM" "$OUTPUT_FOLDER" --section "CB:Z:" --ref_expression "$REF_EXPRESSION" || { 
    log_message "dictys_helper failed"
    exit 1
}

log_message "Successfully completed processing for $INPUT_BAM"