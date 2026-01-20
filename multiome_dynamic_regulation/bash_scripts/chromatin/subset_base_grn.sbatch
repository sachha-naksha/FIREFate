#!/bin/bash
#SBATCH --job-name="base_grn_filter"
#SBATCH --output="base_grn_filter.out"
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -t 00:30:00 
#SBATCH --ntasks-per-node=32
#SBATCH --array=1-24

# Set the working directory to where data and makefiles are located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/outs/dictys/rbpj_ntc"

# Change directory to the working directory
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Calculate the subset number (array starts at 1, so SUBSET_NUM = SLURM_ARRAY_TASK_ID)
SUBSET_NUM=$SLURM_ARRAY_TASK_ID

# Define paths
BASE_GRN="$WORK_DIR/data/mouse_base_grn.tsv.gz"
SUBSET_DIR="$WORK_DIR/tmp_dynamic/Subset${SUBSET_NUM}"
EXPRESSION_FILE="$SUBSET_DIR/expression0.tsv.gz"
OUTPUT_FILE="$SUBSET_DIR/binlinking.tsv.gz"

# Check if base GRN exists
if [ ! -f "$BASE_GRN" ]; then
    echo "ERROR: Base GRN file not found: $BASE_GRN" >&2
    exit 1
fi

# Check if subset directory exists
if [ ! -d "$SUBSET_DIR" ]; then
    echo "ERROR: Subset directory does not exist: $SUBSET_DIR" >&2
    exit 1
fi

# Check if expression file exists
if [ ! -f "$EXPRESSION_FILE" ]; then
    echo "ERROR: Expression file not found: $EXPRESSION_FILE" >&2
    exit 1
fi

echo "Processing Subset${SUBSET_NUM}..."
echo "Base GRN: $BASE_GRN"
echo "Expression file: $EXPRESSION_FILE"
echo "Output file: $OUTPUT_FILE"

####################################### RUN COMMAND #######################################
module load anaconda3/2022.10
source activate dictys

python -c "
import pandas as pd

# Read expression matrix
expression_mtx = pd.read_csv('$EXPRESSION_FILE', sep='\t')
expressed_genes = set(expression_mtx['Unnamed: 0'])

print(f'Expression matrix has {len(expressed_genes)} genes')
print(f'Expression matrix shape: {expression_mtx.shape}')

# Read binlinking file
binlinking = pd.read_csv('$BASE_GRN', sep='\t')
print(f'Base GRN has {binlinking.shape[0]} TFs and {binlinking.shape[1]-1} targets')

# Filter TFs (rows) - only keep TFs where the first column value is in expressed_genes
binlinking_filtered = binlinking[binlinking.iloc[:, 0].isin(expressed_genes)]

# Reset the index to be sequential
binlinking_filtered = binlinking_filtered.reset_index(drop=True)

print(f'After filtering TFs: {binlinking_filtered.shape[0]} TFs remain')

# Filter targets (columns) - keep the first column (TF names) and only target columns that are in expression data
cols_to_keep = [binlinking_filtered.columns[0]]  # Keep the unnamed TF name column
cols_to_keep.extend([col for col in binlinking_filtered.columns[1:] if col in expressed_genes])

binlinking_filtered = binlinking_filtered[cols_to_keep]

print(f'After filtering targets: {binlinking_filtered.shape[1]-1} target genes remain')
print(f'Final filtered GRN shape: {binlinking_filtered.shape}')

# Check if any TFs remain
if binlinking_filtered.shape[0] == 0:
    raise RuntimeError('No regulators found in expression data.')

# Save the filtered GRN
binlinking_filtered.to_csv('$OUTPUT_FILE', sep='\t', index=False, compression='gzip')

print(f'Saved filtered GRN to: $OUTPUT_FILE')
"

echo "Processing completed for Subset${SUBSET_NUM}"