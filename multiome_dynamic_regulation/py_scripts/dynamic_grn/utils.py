import itertools
import math
import os
import re
import sys
from copy import deepcopy

import h5py
import matplotlib as mpl
import matplotlib.patches as Patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D


def read_h5_file(file_path):
    """
    Read HDF5 file and return its contents as a dictionary for easier debugging
    """
    data_dict = {}
    with h5py.File(file_path, "r") as f:
        # Recursively read groups and datasets
        def read_group(group, dict_obj):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    # Convert dataset to numpy array for easier inspection
                    dict_obj[key] = item[()]
                elif isinstance(item, h5py.Group):
                    dict_obj[key] = {}
                    read_group(item, dict_obj[key])

        read_group(f, data_dict)

    return data_dict


def read_adata_from_pkl(pkl_path, workdir):
    """
    Read an AnnData object from a PKL file using stream
    """
    import stream as st

    adata = st.read(file_name=pkl_path, workdir=workdir)
    return adata


def qc_reads(
    fi_reads: str,
    fi_mask: str,
    fo_reads: str,
    n_gene: int,
    nc_gene: int,
    ncp_gene: float,
    n_cell: int,
    nt_cell: int,
    ntp_cell: float,
) -> None:
    """
    Quality control by bounding read counts.
    """
    import logging
    import numpy as np
    import pandas as pd

    logging.info(f"Reading file {fi_reads}.")
    reads0 = pd.read_csv(fi_reads, header=0, index_col=0, sep="\t")
    reads = reads0.values

    # Read masked genes
    logging.info(f"Reading mask file {fi_mask}.")
    with open(fi_mask, "r") as f:
        masked_genes = set(line.strip() for line in f)

    # Create initial mask for genes to preserve
    gene_mask = reads0.index.isin(masked_genes)
    logging.info(f"Found {sum(gene_mask)} genes in mask file.")

    if reads.ndim != 2:
        raise ValueError("reads must have 2 dimensions.")
    if not np.all([x >= 0 for x in [n_gene, nc_gene, ncp_gene, n_cell, nt_cell, ntp_cell]]):
        raise ValueError("All parameters must be non-negative.")
    if not np.all([x <= 1 for x in [ncp_gene, ntp_cell]]):
        raise ValueError("Proportional parameters must be no greater than 1.")

    dt = reads
    nt, ns = dt.shape
    nt0 = ns0 = 0
    st = np.arange(nt)
    ss = np.arange(ns)

    while nt0 != nt or ns0 != ns:
        nt0 = nt
        ns0 = ns
        st1 = np.ones(len(st), dtype=bool)
        ss1 = np.ones(len(ss), dtype=bool)

        # Filter genes
        if n_gene > 0 or nc_gene > 0 or ncp_gene > 0:
            # Create gene filter with same size as current genes
            gene_filter = np.ones(len(st), dtype=bool)
            if n_gene > 0:
                gene_filter &= dt.sum(axis=1) >= n_gene
            if nc_gene > 0 or ncp_gene > 0:
                t1 = (dt > 0).sum(axis=1)
                if nc_gene > 0:
                    gene_filter &= t1 >= nc_gene
                if ncp_gene > 0:
                    gene_filter &= t1 >= ncp_gene * ns
            
            # Create mask for current genes
            current_mask = gene_mask[st]
            # Apply filter while preserving masked genes
            st1 &= (gene_filter | current_mask)

        # Filter cells
        if n_cell > 0:
            ss1 &= dt.sum(axis=0) >= n_cell
        if nt_cell > 0 or ntp_cell > 0:
            t1 = (dt > 0).sum(axis=0)
            if nt_cell > 0:
                ss1 &= t1 >= nt_cell
            if ntp_cell > 0:
                ss1 &= t1 >= ntp_cell * nt

        # Removals
        st = st[st1]
        ss = ss[ss1]
        dt = dt[st1][:, ss1]
        nt = len(st)
        ns = len(ss)

        if nt == 0:
            raise RuntimeError("All genes removed in QC.")
        if ns == 0:
            raise RuntimeError("All cells removed in QC.")

    removed_genes = reads.shape[0] - len(st)
    removed_cells = reads.shape[1] - len(ss)
    preserved_genes = sum(gene_mask[st])  # Count preserved genes in final set
    logging.info(f"Removed {removed_genes}/{reads.shape[0]} genes and {removed_cells}/{reads.shape[1]} cells in QC.")
    logging.info(f"Preserved {preserved_genes} masked genes.")
    reads0 = reads0.iloc[st, ss]
    logging.info(f"Writing file {fo_reads}.")
    reads0.to_csv(fo_reads, header=True, index=True, sep="\t")


def plot_main_trajectory_nodes(
    adata, n_components=2, comp1=0, comp2=1, fig_size=(8, 6), save_path=None
):
    """
    Plot and label the main trajectory nodes (S0, S1, S2, S3) on the elastic principal graph
    """
    # Use 'Agg' backend for non-interactive environments
    import matplotlib

    matplotlib.use("Agg")

    # Create figure
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    # Plot all EPG edges
    epg = adata.uns["epg"]
    epg_node_pos = nx.get_node_attributes(epg, "pos")

    for edge_i in epg.edges():
        start_pos = epg_node_pos[edge_i[0]]
        end_pos = epg_node_pos[edge_i[1]]

        ax.plot(
            [start_pos[comp1], end_pos[comp1]],
            [start_pos[comp2], end_pos[comp2]],
            "k-",
            alpha=0.3,
            linewidth=1,
        )
        ax.plot(
            [start_pos[comp1], end_pos[comp1]],
            [start_pos[comp2], end_pos[comp2]],
            "ko",
            ms=3,
            alpha=0.3,
        )

    # Get flat tree nodes
    flat_tree = adata.uns["flat_tree"]
    ft_nodes = list(flat_tree.nodes())
    main_nodes = {
        "S0": ft_nodes[0],
        "S3": ft_nodes[1],
        "S2": ft_nodes[2],
        "S1": ft_nodes[3],
    }

    # Plot and label main trajectory nodes
    for label, node_idx in main_nodes.items():
        pos = epg_node_pos[node_idx]
        ax.scatter(pos[comp1], pos[comp2], c="red", s=100, zorder=10)
        ax.text(
            pos[comp1],
            pos[comp2],
            label,
            color="red",
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="bottom",
        )

    ax.set_xlabel(f"Dim{comp1+1}")
    ax.set_ylabel(f"Dim{comp2+1}")
    plt.tight_layout()

    # Save the plot
    if save_path is None:
        save_path = "trajectory_nodes.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {save_path}")


def parse_cisBP_motifs(file_content):
    motifs = []
    current_motif = None
    lines = file_content.split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("MOTIF"):
            # Start new motif
            if current_motif:
                motifs.append(current_motif)
            current_motif = {
                "name": line.split()[1],  # Get motif ID after "MOTIF"
                "matrix": [],
                "w": None,
            }
        elif line.startswith("letter-probability matrix"):
            if current_motif:
                # Extract width from the line
                w_match = re.search(r"w=\s*(\d+)", line)
                current_motif["w"] = int(w_match.group(1)) if w_match else None
        elif current_motif and current_motif["w"] is not None:
            # Parse probability lines (skip empty lines and URL lines)
            if line and not line.startswith("URL"):
                try:
                    probs = [float(x) for x in line.split()]
                    if len(probs) == 4:  # Check for 4 probabilities (A,C,G,T)
                        current_motif["matrix"].append(probs)
                except ValueError:
                    continue

    # Don't forget the last motif
    if current_motif and current_motif["matrix"]:
        motifs.append(current_motif)

    # Debug print
    print(f"Parsed {len(motifs)} motifs")
    for i, motif in enumerate(motifs, 1):
        print(
            f"Motif {i}: {motif['name']}, width={motif['w']}, matrix rows={len(motif['matrix'])}"
        )

    return motifs

def process_motif_file_in_homer_format(input_file, output_file=None):
    with open(input_file, "r") as f:
        content = f.read()
    motifs = parse_cisBP_motifs(content)
    print(f"Found {len(motifs)} motifs")  # Debug print
    
    # Set default output file if none provided
    if output_file is None:
        output_file = input_file + ".motif"
    
    # Track seen TF names to skip duplicates
    seen_tfs = set()  # Use a set to track unique TF names
    
    # Constant score for all motifs
    CONSTANT_SCORE = 6.9
        
    with open(output_file, "w") as f:
        for i, motif in enumerate(motifs, 1):
            # Skip if we've seen this TF before
            base_name = motif['name']
            if base_name in seen_tfs:
                print(f"Skipping duplicate motif for {base_name}")
                continue
                
            seen_tfs.add(base_name)  # Add to seen set
            
            # Debug prints
            print(f"Processing motif {i}: {base_name}")
            print(
                f"Matrix size: {len(motif['matrix'])}x{len(motif['matrix'][0]) if motif['matrix'] else 0}"
            )
            if not motif["matrix"]:
                print(f"Skipping motif {i}: empty matrix")
                continue
                
            # Calculate only consensus sequence
            consensus = "".join(["ACGT"[row.index(max(row))] for row in motif["matrix"]])
            
            # Add _MEME1 suffix
            modified_name = f"{base_name}_MEME1"
            
            # Write header line
            header = f">{consensus}\t{modified_name}\t{CONSTANT_SCORE:.5f}\t-10\n"
            f.write(header)
            print(f"Wrote header: {header.strip()}")  # Debug print
            
            # Write probability matrix
            for row in motif["matrix"]:
                line = "\t".join(map(str, row)) + "\n"
                f.write(line)
                print(f"Wrote matrix row: {line.strip()}")  # Debug print
            
            # Add blank line between motifs
            f.write("\n")
            # Ensure writing to disk
            f.flush()
    
    print(f"\nProcessed {len(seen_tfs)} unique TFs")
    print(f"Skipped {len(motifs) - len(seen_tfs)} duplicate motifs")
    
    # Verify file was written
    if os.path.exists(output_file):
        print(f"File created successfully at: {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
    else:
        print("Error: File was not created!")
    return output_file

if __name__ == "__main__":
    input_file = "/ocean/projects/cis240075p/asachan/datasets/TF_motif_files/CisBP_Human_FigR_meme"
    output_file = "/ocean/projects/cis240075p/asachan/datasets/TF_motif_files/CisBP_Human_FigR_meme.motif"
    process_motif_file_in_homer_format(input_file, output_file)
