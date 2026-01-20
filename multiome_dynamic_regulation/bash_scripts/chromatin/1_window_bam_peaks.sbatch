#!/bin/bash
#SBATCH --job-name="bam_peaks"
#SBATCH --output="window_bam_peaks.out"
#SBATCH -p RM
#SBATCH -N 1
#SBATCH -t 00:30:00 # wall-time
#SBATCH --ntasks-per-node=128
#SBATCH --array=0-193

# Set the working directory to where data and makefiles are located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2"
# Change directory to the working directory
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Calculate the subset number based on the job array ID
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# Construct the input and output file paths
INPUT_ATAC="tmp_dynamic/Subset${SUBSET_NUM}/names_atac.txt"
OUTPUT_BAM="tmp_dynamic/Subset${SUBSET_NUM}/reads.bam"
OUTPUT_BAI="tmp_dynamic/Subset${SUBSET_NUM}/reads.bai"
OUTPUT_PEAKS="tmp_dynamic/Subset${SUBSET_NUM}/peaks.bed"

################ RUN dictys in dynamic mode ################
# bash runs all commands sequentially
module load anaconda3/2022.10
source activate dictys

# Run peak calling using direct Python call
python -c "
from dictys.chromatin import macs2
macs2(
    fi_names='$INPUT_ATAC',
    fi_bam='data/bams',
    fo_bam='$OUTPUT_BAM',
    fo_bai='$OUTPUT_BAI',
    fo_bed='$OUTPUT_PEAKS',
    genome_size='hs',
    nth=100
)
"

echo "Done with peak calling for Subset${SUBSET_NUM}"