#!/usr/bin/env python3

# the final op to write the csv after processing all chunks takes the most amount of time (so increasing the process calls/number of cores doesn't help)
import argparse
import logging
from multiprocessing import Pool
from os.path import join as pjoin
import pandas as pd
import numpy as np
from scipy.io import mmread
from functools import partial

def remove_duplicates_chunk(chunk_data, names_chunk):
    """Process a chunk of cells to remove duplicates
    
    Args:
        chunk_data: numpy array of shape (chunk_size, n_features/cells)
        names_chunk: array of names (features or cells) corresponding to the chunk
    """
    td = set()
    ids = []
    dups = []
    
    for xj in range(len(names_chunk)):
        if names_chunk[xj] not in td:
            td.add(names_chunk[xj])
            ids.append(xj)
        else:
            dups.append(names_chunk[xj])
            
    if len(dups) > 0:
        logging.warning(f"Skipped duplicate names: {','.join(dups)}")
        
    return ids, chunk_data[ids] if len(ids) > 0 else None

def process_data_parallel(data, names, chunk_size=1000, n_processes=None):
    """Process data in parallel using chunks
    
    Args:
        data: numpy array gene by cell matrix to process (or cell by gene if transpose is passed)
        names: array of names (features or cells are passed)
        chunk_size: size of chunks to process (number of cells)
        n_processes: number of processes to use (cpu cores)
    """
    # Split data and names into chunks
    n_samples = len(names)
    chunks = [(i, min(i + chunk_size, n_samples)) for i in range(0, n_samples, chunk_size)]
    
    # Prepare chunks of data and names
    data_chunks = [data[start:end] for start, end in chunks]
    name_chunks = [names[start:end] for start, end in chunks]
    
    # Process chunks in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.starmap(remove_duplicates_chunk, zip(data_chunks, name_chunks))
    
    # Combine results
    all_ids = []
    processed_chunks = []
    
    # maintain global index of ids and processed chunks
    for (chunk_start, _), (ids, chunk_data) in zip(chunks, results):
        if ids:
            all_ids.extend([i + chunk_start for i in ids])
            processed_chunks.append(chunk_data)
    
    if processed_chunks:
        processed_data = np.vstack(processed_chunks)
        return np.array(all_ids), processed_data
    return None, None


def main(args):
    diri = args.input_folder
    fo = args.output_file
    colid = args.column
    num_processes = args.processes
    chunk_size = args.chunk_size

    # Read files
    d = mmread(pjoin(diri, "matrix.mtx.gz"))
    d = d.toarray()
    names = [
        pd.read_csv(pjoin(diri, x + ".tsv.gz"), header=None, index_col=None, sep="\t")
        for x in ["features", "barcodes"]
    ]
    # Check that barcodes file has only one column
    assert names[1].shape[1] == 1
    # extract the column of interest (feature name, leave the feature type column)
    names[0] = names[0][colid].values
    names[1] = names[1][0].values
    # check that the expression matrix is of the shape feature by cell (no features filtered)
    assert d.shape == tuple(len(x) for x in names)

    # First filter: Remove features with colons in names (keep only genes, also genes with ".")
    logging.info("Filtering out features with colons, which are peaks from multiome data...")
    feature_filter = np.array([":" not in x for x in names[0]])
    names[0] = names[0][feature_filter]
    d = d[feature_filter, :]

    # Second filter: Remove specified genes
    logging.info("Filtering out specified genes...")
    genes_to_remove = {"TBCE", "LINC01238", "CYB561D2", "MATR3", "LINC01505", 
                      "HSPA14", "GOLGA8M", "GGT1", "ARMCX5-GPRASP2", "TMSB15B"}
    gene_filter = np.array([gene not in genes_to_remove for gene in names[0]])
    names[0] = names[0][gene_filter]
    d = d[gene_filter, :]
    
    # Process genes (rows) and cells (columns) in parallel
    logging.info("Processing genes...")
    gene_ids, processed_genes = process_data_parallel(
        d, names[0], chunk_size=chunk_size, n_processes=num_processes
    )
    
    logging.info("Processing cells...")
    cell_ids, processed_cells = process_data_parallel(
        d.T, names[1], chunk_size=chunk_size, n_processes=num_processes
    )
    
    if gene_ids is not None and cell_ids is not None:
        # Create DataFrame with processed data (ops take the longest time)
        d = pd.DataFrame(
            processed_genes[:, cell_ids].T,
            index=names[1][cell_ids],
            columns=names[0][gene_ids]
        )
        
        # Sort and save
        d = d.loc[sorted(d.index), sorted(d.columns)]
        d.to_csv(fo, header=True, index=True, sep="\t")
    else:
        logging.error("Processing failed - no valid data after filtering")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts mtx.gz format expression file to tsv.gz format."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Input folder that contains matrix.mtx.gz, features.tsv.gz, and barcodes.tsv.gz.",
    )
    parser.add_argument(
        "output_file", 
        type=str,
        help="Output file in tsv.gz format"
    )
    parser.add_argument(
        "--column",
        default=1,
        type=int,
        help="Column ID in features.tsv.gz for gene name. Starts with 0. Default: 1.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=8,
        help="Number of processes to use for multiprocessing. Default: 8",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Size of chunks for parallel processing. Default: 5000",
    )

    # class Args:
    #     input_folder = "/ocean/projects/cis240075p/skeshari/igvf/bcell2/female_donor/in_data_required/donor2_stanford/GEM1/filtered_feature_bc_matrix"
    #     output_file = "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_2nd_donor/expression_mtx/day1_expression.tsv.gz"
    #     column = 1
    #     processes = 8
    #     chunk_size = 5000

    ####### Run the script with args here #######
    args = parser.parse_args()
    #args = Args() # if you don't want to run using sbatch, just pass args to the class and run locally on the current node
    logging.basicConfig(level=logging.INFO)
    main(args)