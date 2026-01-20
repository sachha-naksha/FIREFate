#!/bin/bash
#SBATCH --job-name="subset_barcodes" 
#SBATCH --output="subset_barcodes.log" 
#SBATCH -p RM
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks-per-node=128 # number of cores per node; memory is 2gb per node for rm-shared
#SBATCH -t 2:00:00


source activate dictys

cd /ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/data
# Get the unique subsets
subsets="$(tail -n +2 clusters.csv | awk -F , '{print $2}' | sort -u)"
echo "$subsets" | awk '{print $1}' > subsets.txt

# Define a function for processing each subset/cell-type for loop
process_subset() {
    local x=$1
    mkdir -p "subsets/$x"
    grep ",$x"'$' clusters.csv | awk -F , '{print $1}' > "subsets/$x/names_rna.txt"
    cp "subsets/$x/names_rna.txt" "subsets/$x/names_atac.txt"
}

export -f process_subset # Export the function so that it's available to parallel

# Run the process_subset function in parallel
echo "$subsets" | parallel -j $SLURM_NTASKS_PER_NODE process_subset {}

