"""Gene and TF lookups against a dictys network, and TF-target table parsing."""
from __future__ import annotations

import numpy as np
from scipy import stats


def get_tf_indices(dictys_dynamic_object, tf_list):
    """
    Get the indices of transcription factors from a list, if present in ndict and nids[0].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    tf_mappings_to_gene_hashmap = dictys_dynamic_object.nids[0]
    tf_indices = []
    tf_gene_indices = []
    missing_tfs = []
    for gene in tf_list:
        # Check if the gene is in the gene_hashmap
        if gene in gene_hashmap:
            gene_index = gene_hashmap[gene]  # Get the index in gene_hashmap
            # Check if the gene index is present in tf_mappings_to_gene_hashmap
            match = np.where(tf_mappings_to_gene_hashmap == gene_index)[0]
            if match.size > 0:  # If a match is found
                tf_indices.append(int(match[0]))  # Append the position of the match
                tf_gene_indices.append(int(gene_index))  # Also append the gene index
            else:
                missing_tfs.append(gene)  # Gene exists but not as a TF
        else:
            missing_tfs.append(gene)  # Gene not found at all
    return tf_indices, tf_gene_indices, missing_tfs


def get_gene_indices(dictys_dynamic_object, gene_list):
    """
    Get the indices of target genes from a list, if present in ndict and nids[1].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    gene_indices = []
    for gene in gene_list:
        if gene in gene_hashmap:
            gene_indices.append(gene_hashmap[gene])
    return gene_indices


def check_if_gene_in_ndict(dictys_dynamic_object, gene_name, return_index=False):
    """
    Check if a gene is in the ndict of the dynamic object.
    """
    # Input validation
    if not hasattr(dictys_dynamic_object, "ndict"):
        raise AttributeError("Dynamic object does not have ndict attribute")
    # Handle single gene case
    if isinstance(gene_name, str):
        is_present = gene_name in dictys_dynamic_object.ndict
        if return_index:
            return dictys_dynamic_object.ndict.get(gene_name, None)
        return is_present
    # Handle list of genes case
    elif isinstance(gene_name, (list, tuple, set)):
        results = {
            "present": [],
            "missing": [],
            "indices": {} if return_index else None,
        }
        for gene in gene_name:
            if gene in dictys_dynamic_object.ndict:
                results["present"].append(gene)
                if return_index:
                    results["indices"][gene] = dictys_dynamic_object.ndict[gene]
            else:
                results["missing"].append(gene)
        # Add summary statistics
        results["stats"] = {
            "total_genes": len(gene_name),
            "found": len(results["present"]),
            "missing": len(results["missing"]),
            "percent_found": (len(results["present"]) / len(gene_name) * 100),
        }
        return results
    else:
        raise TypeError("gene_name must be a string or a list-like object of strings")


def extract_tf_gene_info(enriched_links_df):
    """
    Extract TF-Gene links and unique TF/Gene names from enriched links dataframe.
    """
    
    # Validate input
    if enriched_links_df is None or enriched_links_df.empty:
        return [], [], []
    
    required_cols = ['TF', 'Gene']
    for col in required_cols:
        if col not in enriched_links_df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Remove any rows with NaN values in TF or Gene columns
    clean_df = enriched_links_df.dropna(subset=['TF', 'Gene'])
    
    if clean_df.empty:
        return [], [], []
    
    # Extract links as list of tuples
    links_list = list(zip(clean_df['TF'], clean_df['Gene']))
    
    # Get unique TF names (sorted for consistency)
    unique_tfs = sorted(clean_df['TF'].unique().tolist())
    
    # Get unique gene names (sorted for consistency)
    unique_genes = sorted(clean_df['Gene'].unique().tolist())
    
    return links_list, unique_tfs, unique_genes
