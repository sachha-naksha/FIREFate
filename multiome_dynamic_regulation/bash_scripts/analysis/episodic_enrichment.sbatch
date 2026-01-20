#!/bin/bash
#SBATCH --job-name="ee_ikzf1_ko"
#SBATCH --output="ee_ikzf1_ko.log"  
#SBATCH -p RM-shared           
#SBATCH -N 1            
#SBATCH --ntasks-per-node=64 #for memory
#SBATCH -t 00:30:00 
#SBATCH --array=1-6

# Set working directory where the python script is located
WORK_DIR="/ocean/projects/cis240075p/asachan/bio_informatics_analysis/B_Cells_human_analysis/src/multiome_dynamic_regulation/py_scripts/analysis"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Load necessary modules and activate the environment
module load anaconda3/2022.10
source activate dictys

# I/O file names
dictys_dynamic_object_path="/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/outs/dictys/rbpj_ntc/output/dynamic.h5"
output_folder="/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/outs/dictys/rbpj_ntc/output/ikzf1/sig_lfs"

# Calculate the window number based on the array task ID
episode_idx=$((SLURM_ARRAY_TASK_ID))
time_slice_start=$(( (episode_idx - 1) * 4 )) # N windows per episode
time_slice_end=$(( episode_idx * 4 ))
traj_start=0
traj_end=1
num_points=24
dist=0.001
sparsity=0.01
percentile=98

# Run enrichment 
echo "Running enrichment for episode ${episode_idx}" 

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

# ETS1 KO
# ets1_lf_sig1 = pd.read_csv(\"/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/Data/latent_factors/ets1_lfs/feature_list_Z7.txt\", sep='\t')['names'].tolist()
# ets1_lf_sig2 = pd.read_csv(\"/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/Data/latent_factors/ets1_lfs/feature_list_Z28.txt\", sep='\t')['names'].tolist()
# ets1_lf_int1 = pd.read_csv(\"/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/Data/latent_factors/ets1_lfs/feature_list_Z16.txt\", sep='\t')['names'].tolist()
# ets1_lf_int2 = pd.read_csv(\"/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/Data/latent_factors/ets1_lfs/feature_list_Z26.txt\", sep='\t')['names'].tolist()

# #create a set of all latent factors
# ets1_all_lfs = set(ets1_lf_sig1 + ets1_lf_sig2 + ets1_lf_int1 + ets1_lf_int2)

# IKZF1 KO
ikzf1_lf_sig = pd.read_csv(\"/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/Data/latent_factors/ikzf1_lfs/feature_list_Z7.txt\", sep='\t')['names'].tolist()
ikzf1_lf_int1 = pd.read_csv(\"/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/Data/latent_factors/ikzf1_lfs/feature_list_Z18.txt\", sep='\t')['names'].tolist()
ikzf1_lf_int2 = pd.read_csv(\"/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/Data/latent_factors/ikzf1_lfs/feature_list_Z29.txt\", sep='\t')['names'].tolist()

#create a set of all latent factors
ikzf1_all_lfs = set(ikzf1_lf_sig + ikzf1_lf_int1 + ikzf1_lf_int2)

run_episode(
    episode_idx=${episode_idx},
    dictys_dynamic_object_path=\"${dictys_dynamic_object_path}\",
    output_folder=\"${output_folder}\",
    trajectory_range=(${traj_start}, ${traj_end}),
    num_points=${num_points},
    time_slice_start=${time_slice_start},
    time_slice_end=${time_slice_end},
    dist=${dist},
    sparsity=${sparsity},
    percentile=${percentile},
    lf_genes=ikzf1_lf_sig
)
"

# Log completion of the subset
echo "Done with enrichment for episode ${episode_idx}"
