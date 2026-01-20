import os
import logging
import sys
from subprocess import Popen
from dictys.network import reconstruct

# Working directory
if len(sys.argv) < 4:
    print("Usage: python script.py start_window end_window work_dir")
    sys.exit(1)
    
WORK_DIR = sys.argv[3]
os.chdir(WORK_DIR)

def run_reconstruction(subset_num):
    input_expression = f"tmp_dynamic/Subset{subset_num}/expression0.tsv.gz"
    input_binlinking = f"tmp_dynamic/Subset{subset_num}/binlinking.tsv.gz"
    output_net_weight = f"tmp_dynamic/Subset{subset_num}/net_weight.tsv.gz"
    output_net_meanvar = f"tmp_dynamic/Subset{subset_num}/net_meanvar.tsv.gz"
    output_net_covfactor = f"tmp_dynamic/Subset{subset_num}/net_covfactor.tsv.gz"
    output_net_loss = f"tmp_dynamic/Subset{subset_num}/net_loss.tsv.gz"
    output_net_stats = f"tmp_dynamic/Subset{subset_num}/net_stats.tsv.gz"
    
    # Run dictys reconstruction (not CLI)
    # Note: We don't specify the GPU ID, letting CUDA use the SLURM-assigned GPU
    try:
        # Direct function call instead of CLI
        reconstruct(
            fi_exp=input_expression,
            fi_mask=input_binlinking,
            fo_weight=output_net_weight,
            fo_meanvar=output_net_meanvar,
            fo_covfactor=output_net_covfactor,
            fo_loss=output_net_loss,
            fo_stats=output_net_stats,
            device='cuda',  # Let CUDA use the GPU that SLURM assigned
            nth=5
        )

    except Exception as e:
        print(f"Error running reconstruction for Subset {subset_num}: {e}")

def main():
    start_subset = int(sys.argv[1])
    end_subset = int(sys.argv[2])
    # Process each subset sequentially
    for subset_num in range(start_subset, end_subset + 1):
        run_reconstruction(subset_num)

if __name__ == "__main__":
    main()