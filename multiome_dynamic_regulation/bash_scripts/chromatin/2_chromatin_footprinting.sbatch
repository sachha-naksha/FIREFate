#!/bin/bash
#SBATCH --job-name="footprinting"
#SBATCH --output="footprint_windows.out"
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -t 00:30:00 # wall-time
#SBATCH --ntasks-per-node=32
#SBATCH --array=0-193

# Set the working directory to where data and makefiles are located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2"
# Change directory to the working directory
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Calculate the subset number based on the job array ID
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# Construct the input and output file paths
INPUT_BAM="tmp_dynamic/Subset${SUBSET_NUM}/reads.bam"
INPUT_BAI="tmp_dynamic/Subset${SUBSET_NUM}/reads.bai"
INPUT_PEAKS="tmp_dynamic/Subset${SUBSET_NUM}/peaks.bed"
OUTPUT_FOOTPRINTS="tmp_dynamic/Subset${SUBSET_NUM}/footprints.bed"

################ RUN dictys in dynamic mode ################

module load anaconda3/2022.10
source activate dictys

# Run footprinting using direct Python call
python -c "
from dictys.chromatin import wellington
wellington(
    fi_bam='$INPUT_BAM',
    fi_bai='$INPUT_BAI',
    fi_bed='$INPUT_PEAKS',
    fo_bed='$OUTPUT_FOOTPRINTS',
    fi_blacklist='data/blacklist.bed',
    nth=30
)
"
echo "Done with footprinting for Subset${SUBSET_NUM}"

