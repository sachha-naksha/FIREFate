#!/bin/bash
#SBATCH --job-name="normalization"
#SBATCH --output="normalize_rbpj.log"  
#SBATCH -p RM-shared           
#SBATCH -N 1            
#SBATCH --ntasks-per-node=32 #for memory     
#SBATCH -t 00:30:00 
#SBATCH --array=0-23

# Set working directory
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/outs/dictys/rbpj_ntc"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Load necessary modules and activate the environment
module load anaconda3/2022.10
source activate dictys

# Calculate the window number based on the array task ID
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# I/O file names
INPUT_WEIGHT="tmp_dynamic/Subset${SUBSET_NUM}/net_weight.tsv.gz"
INPUT_MEANVAR="tmp_dynamic/Subset${SUBSET_NUM}/net_meanvar.tsv.gz"
INPUT_COVFACTOR="tmp_dynamic/Subset${SUBSET_NUM}/net_covfactor.tsv.gz"
NWEIGHT="tmp_dynamic/Subset${SUBSET_NUM}/net_nweight.tsv.gz"
IWEIGHT="tmp_dynamic/Subset${SUBSET_NUM}/net_iweight.tsv.gz"
INWEIGHT="tmp_dynamic/Subset${SUBSET_NUM}/net_inweight.tsv.gz"

# Run normalization 
echo "Running normalization for Subset${SUBSET_NUM}" 
python -c "
import sys
from dictys.network import normalize, indirect

# Run normalization on direct effect
normalize(
    fi_weight='$INPUT_WEIGHT',
    fi_meanvar='$INPUT_MEANVAR',
    fi_covfactor='$INPUT_COVFACTOR',
    fo_nweight='$NWEIGHT',
    nth=32
)

# Run indirect effect
indirect(
    fi_weight='$INPUT_WEIGHT',
    fi_covfactor='$INPUT_COVFACTOR',
    fo_iweight='$IWEIGHT',
    fi_meanvar='$INPUT_MEANVAR',
    nth=32
)

# Normalize indirect effect
normalize(
    fi_weight='$IWEIGHT',
    fi_meanvar='$INPUT_MEANVAR',
    fi_covfactor='$INPUT_COVFACTOR',
    fo_nweight='$INWEIGHT',
    nth=32
)
"

# Log completion of the subset
echo "Done with Subset${SUBSET_NUM}"
