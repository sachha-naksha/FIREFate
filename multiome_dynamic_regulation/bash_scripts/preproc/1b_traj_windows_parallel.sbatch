#!/bin/bash
#SBATCH --job-name="process_edges_and_combine"
#SBATCH --output="process_edges_and_combine_%j.log"
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00

# Set working directory
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }
# Create tmp_dynamic folder if it doesn't exist
mkdir -p tmp_dynamic

# Load necessary modules and activate the environment
module load anaconda3/2022.10
source activate dictys

# Set parameters
FI_TRAJ="data/traj_node.h5"
FI_TRAJ_CELL_RNA="data/traj_cell_rna.h5"
FI_COORD_RNA="data/coord_rna.tsv.gz"
FO_SUBSETS="tmp_dynamic/subsets.txt"
FO_SUBSET_LOCS="tmp_dynamic/subset_locs.h5"
DIRO_SUBSETS="tmp_dynamic"
FO_SUBSET_EDGES="tmp_dynamic/subset_edges.tsv.gz"

# Calculate parameters
total_cells_in_dataset=24210
ncell=$(( (${total_cells_in_dataset} * 155 / 1000) / 100 * 100 ))
noverlap=$(( (${ncell} * 875 / 1000) / 50 * 50 ))
max_pseudotime_in_traj=0.008
dmax=$(awk -v max=$max_pseudotime_in_traj 'BEGIN {
    raw = 0.004 * max;
    scaled = raw * 1000000;
    rounded = int(scaled / 5) * 5;
    printf "%.6f", rounded / 1000000;
}')

# Get number of edges
num_edges=$(h5dump -d /edges $FI_TRAJ | grep "(0)" | wc -l)
echo "Number of edges: $num_edges"

# Function to submit a job for processing a single edge
submit_edge_job() {
    local edge=$1
    sbatch --job-name="edge_${edge}" --output="edge_${edge}_%j.log" --partition=RM --time=02:00:00 --ntasks-per-node=128 <<EOF
#!/bin/bash
#SBATCH -N 1

cd $WORK_DIR
module load anaconda3/2022.10
source activate dictys
python3 /ocean/projects/cis240075p/asachan/bio_informatics_analysis/B_Cells_human_analysis/analysis_repo/multiome_dynamic_regulation/py_scripts/trajectory/process_single_edge.py $edge $FI_TRAJ $FI_TRAJ_CELL_RNA $FI_COORD_RNA $ncell $noverlap $dmax tmp_dynamic/edge_${edge}_output.h5
EOF
}

# Submit jobs for each edge
for edge in $(seq 0 $((num_edges-1))); do
    submit_edge_job $edge
done

# Wait for all edge processing jobs to complete
echo "Waiting for all edge processing jobs to complete..."
while [ $(squeue -h -u $USER -n "edge_*" | wc -l) -gt 0 ]; do
    sleep 60
done

# Combine results
echo "Combining results..."
python3 <<EOF
import h5py
import numpy as np
import pandas as pd
from dictys.traj import trajectory, point
import os
from os import makedirs, linesep
from os.path import join as pjoin

def combine_results(num_edges, fo_subsets, fo_subset_locs, diro_subsets, fo_subset_edges, fi_traj):
    all_edges = []
    all_locs = []
    all_subsets = []
    all_neighbors = []

    for edge in range(num_edges):
        try:
            with h5py.File(f"tmp_dynamic/edge_{edge}_output.h5", "r") as f:
                all_edges.extend(f["edges"][:])
                all_locs.extend(f["locs"][:])
                all_subsets.extend(f["subsets"][:])
                all_neighbors.append(f["neighbors"][:])
        except (OSError, KeyError) as e:
            print(f"Error processing edge {edge}: {str(e)}")

    # Validation of combined edges, locs and subsets
    n = len(all_edges)
    assert len(all_locs) == n and len(all_subsets) == n
    all_edges = np.array(all_edges).astype('u2')
    all_locs = np.array(all_locs)
    all_subsets = np.array(all_subsets)

    # Combine neighbor matrices
    neighbor_matrix = np.zeros((n, n), dtype=bool)
    offset = 0
    for edge_neighbors in all_neighbors:
        size = edge_neighbors.shape[0]
        neighbor_matrix[offset:offset+size, offset:offset+size] = edge_neighbors
        offset += size

    assert (neighbor_matrix >= 0).all() and (neighbor_matrix < n).all()

    names_subset = [f'Subset{i+1}' for i in range(n)]
    
    with open(fo_subsets, 'w') as f:
        f.write(linesep.join(names_subset) + linesep)
    
    traj = trajectory.from_file(fi_traj)
    subsets = point(traj, all_edges, all_locs)
    subsets.to_file(fo_subset_locs)
    
    nodegraph = pd.DataFrame(neighbor_matrix, index=names_subset, columns=names_subset)
    nodegraph.to_csv(fo_subset_edges, sep='\t', header=True, index=True)
    
    for i, subset in enumerate(all_subsets):
        subset_dir = pjoin(diro_subsets, names_subset[i])
        makedirs(subset_dir, exist_ok=True)
        with open(pjoin(subset_dir, "names_rna.txt"), "w") as f:
            f.write(linesep.join(subset) + linesep)

combine_results($num_edges, "$FO_SUBSETS", "$FO_SUBSET_LOCS", "$DIRO_SUBSETS", "$FO_SUBSET_EDGES", "$FI_TRAJ")
print("Results combined successfully")
EOF

echo "All processing complete."