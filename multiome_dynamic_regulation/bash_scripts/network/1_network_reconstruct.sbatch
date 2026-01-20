#!/bin/bash
#SBATCH --job-name="auto_reconstruct_submit"
#SBATCH --output="rbpj_reconstruct_windows_auto.log"  # Single log file for the submission job
#SBATCH -p RM-shared           # Use a CPU-only partition for job submission script
#SBATCH -N 1            # Request 1 node
#SBATCH  --ntasks-per-node=1     # Single task for submission
#SBATCH -t 1:00:00 # Time for the submission script (can be adjusted)

# Set working directory
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/outs/dictys/rbpj_ntc"
SCRIPT_PATH="/ocean/projects/cis240075p/asachan/bio_informatics_analysis/B_Cells_human_analysis/src/multiome_dynamic_regulation/py_scripts/dynamic_grn/network_reconstruct_batch.py"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Load necessary modules and activate the environment (if required)
module load anaconda3/2022.10
source activate dictys

# Function to extract available GPU nodes from sinfo
get_available_gpu_nodes() {
    sinfo -p GPU-shared --format="%N %G" | grep v100 | awk '{print $1}' | wc -l
}

# Get the number of available GPU nodes (N)
available_nodes=$(get_available_gpu_nodes)
echo "Available GPU nodes: $available_nodes"

#################################### Define windows and batch size ####################################
start_window=1
end_window=24
batch_size=4  # Maximum batch size is 4 windows (1.75 hours per window) (this can change based on available gpu nodes)

#################################### Submit jobs in batches ####################################
# Calculate total windows to process
total_windows=$((end_window - start_window + 1))
current_window=$start_window

# Loop through windows and submit jobs in batches
while [ $current_window -le $end_window ]; do
    # Calculate the upper limit for this batch (adjust batch size if fewer windows remain)
    upper_limit=$((current_window + batch_size - 1))
    if [ $upper_limit -gt $end_window ]; then
        upper_limit=$end_window
    fi

    # Calculate time required for this batch (1.75 hours per window)
    num_windows_in_batch=$((upper_limit - current_window + 1))
    time_required=$(echo "scale=2; $num_windows_in_batch * 1.75" | bc)
    hours=$(echo "$time_required" | cut -d'.' -f1)
    minutes=$(echo "$time_required" | cut -d'.' -f2)
    if [ -z "$minutes" ]; then minutes="00"; fi

    # Submit the SLURM job for this batch to the GPU-shared partition
    echo "Submitting batch for windows $current_window to $upper_limit (Time: $hours:$minutes:00)"
    sbatch --partition=GPU-shared --time="${hours}:${minutes}:00" --output="reconstruct_windows_${current_window}_to_${upper_limit}.log" <<-EOF
#!/bin/bash
#SBATCH --job-name="rbpj_reconstruct_batch_${current_window}_to_${upper_limit}"
#SBATCH --output=/dev/null
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=5
#SBATCH --constraint="v100-32|v100-16"
#SBATCH --time=${hours}:${minutes}:00
#SBATCH -N 1

# Change to working directory
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Load modules and activate the environment
module load anaconda3/2022.10
source activate dictys

# Run Python script for this batch of windows
python3 $SCRIPT_PATH $current_window $upper_limit $WORK_DIR
EOF

    # Update current window for the next batch
    current_window=$((upper_limit + 1))

    # Sleep for a short while before checking for the next batch (optional)
    sleep 2
done

echo "All batches submitted."
