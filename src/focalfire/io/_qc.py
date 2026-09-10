"""Read-count quality control with a preserved-gene mask."""
from __future__ import annotations

import numpy as np
import pandas as pd


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
