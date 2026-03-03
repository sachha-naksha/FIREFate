#!/bin/bash
#SBATCH --job-name="epi_grn_GC_loose_threshold"
#SBATCH --output="epi_grn_GC_loose_threshold.log"  
#SBATCH -p RM-shared           
#SBATCH -N 1            
#SBATCH --ntasks-per-node=64 #for memory
#SBATCH -t 00:30:00 
#SBATCH --array=1-8

# Set working directory where the python script is located
WORK_DIR="/ocean/projects/cis240075p/asachan/methods/FIREFate/multiome_dynamic_regulation/py_scripts/analysis"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Load necessary modules and activate the environment
module load anaconda3/2022.10
source activate dictys

# I/O file names
dictys_dynamic_object_path="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2/output/dynamic.h5"
output_folder="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2/output/episodic_grns"

# Calculate the window number based on the array task ID
episode_idx=$((SLURM_ARRAY_TASK_ID))
time_slice_start=$(( (episode_idx - 1) * 5 )) # N windows per episode
time_slice_end=$(( episode_idx * 5 ))
traj_start=1
traj_end=3
num_points=40
dist=0.001
sparsity=0.01
percentile=94

# Run episodic construction
echo "Running episodic construction for episode ${episode_idx}" 

########################## Python script ##########################
python -c "
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import dictys
from dictys.net import stat
import joblib
import pickle
from scipy.stats import median_abs_deviation, hypergeom
import math

from utils_custom import *
from episodic_dynamics import *
from pseudotime_curves import *

# Run episodic construction
run_episodic_construction(
    episode_idx=${episode_idx},
    dictys_dynamic_object_path=\"${dictys_dynamic_object_path}\",
    output_folder=\"${output_folder}\",
    trajectory_range=(${traj_start}, ${traj_end}),
    num_points=${num_points},
    time_slice_start=${time_slice_start},
    time_slice_end=${time_slice_end},
    dist=${dist},
    sparsity=${sparsity},
    percentile=${percentile}
)
"

# Log completion of the subset
echo "Done with episodic construction for episode ${episode_idx}"
