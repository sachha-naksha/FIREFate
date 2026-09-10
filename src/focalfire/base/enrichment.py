"""The hypergeometric over-representation primitive shared by every FocalFire module."""
from __future__ import annotations

import pandas as pd
from scipy.stats import hypergeom


def calculate_tf_episodic_enrichment(df, total_lf_genes, total_genes_in_grn):
    """
    Calculate TF enrichment scores using a hypergeometric test.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with MultiIndex (TF, Target) and columns ``avg_force``, ``is_in_lf``.
    total_lf_genes : int
        Total number of active LF genes in the episode.
    total_genes_in_grn : int
        Total number of genes in the episodic GRN.

    Returns
    -------
    pandas.DataFrame
        Columns include TF, p_value, enrichment_score, genes_in_lf, genes_dwnstrm, weights.
    """

    results = []

    # Group by TF (level 0 of the MultiIndex)
    for tf in df.index.get_level_values(0).unique():
        tf_data = df.loc[tf]

        # Calculate metrics for this TF
        tf_lf_targets = tf_data[
            "is_in_lf"
        ].sum()  # Number of LF targets for this TF (k)
        tf_total_targets = len(tf_data)  # Total targets for this TF (n)

        # Get LF gene names and their weights for this TF
        lf_mask = tf_data["is_in_lf"]
        genes_in_lf = tuple(str(gene) for gene in tf_data[lf_mask].index.tolist())
        weights = tuple(tf_data[lf_mask]["avg_force"].tolist())

        # Get downstream genes that are NOT in LF (False for is_in_lf)
        non_lf_mask = ~tf_data["is_in_lf"]
        genes_dwnstrm = tuple(str(gene) for gene in tf_data[non_lf_mask].index.tolist())

        # Hypergeometric test parameters:
        # N = total_genes_in_grn (population size)
        # K = total_lf_genes (number of success states in population)
        # n = tf_total_targets (sample size)
        # k = tf_lf_targets (number of observed successes)

        # Calculate p-value using hypergeometric distribution
        # P(X >= k) = 1 - P(X <= k-1)
        if tf_total_targets > 0 and total_lf_genes > 0:
            p_value = hypergeom.sf(
                tf_lf_targets - 1, total_genes_in_grn, total_lf_genes, tf_total_targets
            )

            # Calculate enrichment score (fold enrichment)
            expected = (tf_total_targets * total_lf_genes) / total_genes_in_grn
            enrichment_score = tf_lf_targets / expected if expected > 0 else 0
        else:
            p_value = 1.0
            enrichment_score = 0

        results.append(
            {
                "TF": str(tf),
                "p_value": p_value,
                "enrichment_score": enrichment_score,
                "genes_in_lf": genes_in_lf,
                "genes_dwnstrm": genes_dwnstrm,
                "weights": weights,
            }
        )

    return pd.DataFrame(results)
