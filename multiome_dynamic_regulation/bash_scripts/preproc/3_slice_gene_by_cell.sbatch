#!/bin/bash
#SBATCH --job-name="gene_by_cell"
#SBATCH --output="slice_rbpj_tcell.out"
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node=16
#SBATCH --array=0-23

# Set the working directory to where data is located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/outs/dictys/rbpj_ntc"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Calculate the subset number based on the job array ID
SUBSET_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# Construct the input and output file paths
INPUT_RNA="tmp_dynamic/Subset${SUBSET_NUM}/names_rna.txt"
OUTPUT_EXPR="tmp_dynamic/Subset${SUBSET_NUM}/expression0.tsv.gz"

# Check if expression data file exists
if [ ! -f "$WORK_DIR/data/expression.tsv.gz" ]; then
    echo "ERROR: Expression data file not found: $WORK_DIR/data/expression.tsv.gz" >&2
    exit 1
fi

# Check if subset directory exists
SUBSET_DIR="tmp_dynamic/Subset${SUBSET_NUM}"
if [ ! -d "$SUBSET_DIR" ]; then
    echo "ERROR: Subset directory does not exist: $SUBSET_DIR" >&2
    echo "Available subsets:" >&2
    ls -d tmp_dynamic/Subset* 2>/dev/null || echo "No subset directories found" >&2
    exit 1
fi

# Check if input RNA file exists
if [ ! -f "$INPUT_RNA" ]; then
    echo "ERROR: Input file not found for Subset${SUBSET_NUM}: $INPUT_RNA" >&2
    echo "Contents of Subset${SUBSET_NUM} directory:" >&2
    ls -lh "$SUBSET_DIR" >&2
    exit 1
fi

# Note: If output file exists, it will be overwritten
if [ -f "$OUTPUT_EXPR" ]; then
    echo "WARNING: Output file already exists: $OUTPUT_EXPR"
    echo "It will be overwritten"
fi

# Verify file is not empty
if [ ! -s "$INPUT_RNA" ]; then
    echo "ERROR: Input file is empty: $INPUT_RNA" >&2
    exit 1
fi

echo "All input files verified ✓"
echo ""

####################################### RUN COMMAND #######################################
# the cmd loads the whole gene_by_cell in memory and then slices the loaded df
module load anaconda3/2022.10
source activate dictys

python -c "
from dictys.preproc import selects_rna
selects_rna(
    fi_reads='$WORK_DIR/data/expression.tsv.gz',
    fi_names='$INPUT_RNA',
    fo_reads='$OUTPUT_EXPR'
)
"

echo "Processing completed for Subset${SUBSET_NUM}"