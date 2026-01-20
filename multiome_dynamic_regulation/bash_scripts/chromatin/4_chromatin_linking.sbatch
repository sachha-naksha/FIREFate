#!/bin/bash
#SBATCH --job-name="priors"
#SBATCH --output="tf_tg_priors.out"
#SBATCH -p RM
#SBATCH -N 1
#SBATCH -t 00:30:00 
#SBATCH --ntasks-per-node=128
#SBATCH --array=1-193

# Set the working directory to where data and makefiles are located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2"
# Change directory to the working directory
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Calculate the subset number based on the job array ID
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# Construct the input and output file paths
INPUT_GEX="tmp_dynamic/Subset${SUBSET_NUM}/expression.tsv.gz"
INPUT_WELLINGTON="tmp_dynamic/Subset${SUBSET_NUM}/wellington.tsv.gz"
INPUT_HOMER="tmp_dynamic/Subset${SUBSET_NUM}/homer.tsv.gz"
BINDING="tmp_dynamic/Subset${SUBSET_NUM}/binding.tsv.gz"
TSSDIST="tmp_dynamic/Subset${SUBSET_NUM}/tssdist.tsv.gz"
LINKING="tmp_dynamic/Subset${SUBSET_NUM}/linking.tsv.gz"
BINLINKING="tmp_dynamic/Subset${SUBSET_NUM}/binlinking.tsv.gz"

################ RUN dictys in dynamic mode ################
# bash runs all commands sequentially, so does python
module load anaconda3/2022.10
source activate dictys

python -c "
# make sure to import all functions
from dictys.chromatin import binding, tssdist, linking, binlinking

# Get binding score by using footprinting and motif scanning scores
binding(
    fi_wellington='$INPUT_WELLINGTON',
    fi_homer='$INPUT_HOMER',
    fo_bind='$BINDING'
)

# Calculate TSS distances of bound TFs with expressed genes
tssdist(
    fi_exp='$INPUT_GEX',
    fi_wellington='$INPUT_WELLINGTON',
    fi_tss='data/gene.bed',
    fo_dist='$TSSDIST'
)

# Generate linking data
linking(
    fi_binding='$BINDING',
    fi_dist='$TSSDIST',
    fo_linking='$LINKING'
)

# Create binary linking data
binlinking(
    fi_linking='$LINKING',
    fo_binlinking='$BINLINKING',
    n=20
)
"
echo "TF_TG priors made for ${SUBSET_NUM}"
