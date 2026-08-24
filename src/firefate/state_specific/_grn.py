"""State-specific GRN construction from CellOracle outputs (capability 2).

Structured after ``TA_muscle_ageing/py_scripts/grn_inference`` — cluster fusion,
edge quantile thresholding, and NetworkX graph views. The enrichment that consumes
these networks lives in :mod:`firefate.state_specific._enrichment`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def combine_cluster_grn_links(
    links_after_fit: dict[Any, pd.DataFrame],
    cluster_fusion: tuple[Any, ...],
    quantile: float = 0.9,
    histogram_path: str | Path | None = None,
) -> tuple[pd.DataFrame, float]:
    """Concatenate per-cluster edge tables and binarize ``coef_abs`` at a quantile threshold.

    Parameters
    ----------
    links_after_fit
        Maps cluster id → DataFrame with columns ``source``, ``target``, ``coef_mean``, ``coef_abs``.
    cluster_fusion
        Clusters whose edges are merged (order preserved).
    quantile
        Quantile of the ``coef_abs`` column used as cutoff; edges at or above are
        ``strength == 1``.
    histogram_path
        If set, save coefficient histogram with cutoff line.
    """
    combined_links = pd.DataFrame()
    for cluster in cluster_fusion:
        combined_links_og = links_after_fit[cluster].copy()
        combined_links_og["cluster"] = cluster
        combined_links = pd.concat([combined_links, combined_links_og], axis=0)
        if histogram_path is not None:
            combined_links_og["coef_abs"].plot(kind="hist", bins=20, alpha=0.5)

    threshold = float(abs(combined_links["coef_abs"]).quantile(quantile))
    combined_links["strength"] = (combined_links["coef_abs"] >= threshold).astype(int)

    if histogram_path is not None:
        plt.axvline(x=threshold, color="red", linestyle="--", linewidth=1)
        plt.title(f"Threshold for edge strength: {threshold}")
        plt.xlabel("Absolute coefficient value")
        plt.ylabel("Frequency")
        Path(histogram_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(histogram_path)
        plt.close()

    return combined_links, threshold


def grn_edges_from_combined_links(
    combined_links: pd.DataFrame,
) -> tuple[nx.MultiDiGraph, pd.DataFrame]:
    """Build a :class:`networkx.MultiDiGraph` and a flat edge list with MultiDiGraph keys."""
    grn: nx.MultiDiGraph = nx.MultiDiGraph()
    for _, row in combined_links.iterrows():
        grn.add_edge(
            row["source"],
            row["target"],
            weight=row["coef_mean"],
            cluster=row["cluster"],
            strength=row["strength"],
        )
    edges_df = pd.DataFrame(
        [{"source": u, "target": v, "key": k, **d} for u, v, k, d in grn.edges(keys=True, data=True)]
    )
    return grn, edges_df


def filter_network_scores_for_clusters(
    cluster_fusion: tuple[Any, ...],
    network_scores: pd.DataFrame,
    *,
    cluster_column: str = "cluster",
) -> pd.DataFrame:
    """Subset and de-duplicate network scores across fused clusters (first row per index)."""
    parts = [network_scores[network_scores[cluster_column] == int(cid)] for cid in cluster_fusion]
    combined = pd.concat(parts).sort_values(by="degree_centrality_out", ascending=False)
    return combined.loc[lambda df: (~df.index.duplicated(keep="first"))]


def load_celloracle_grn_tables(
    grn_workdir: str | Path,
    oracle_object_name: str,
    *,
    network_scores_filename: str = "ridge_fitted_2_merged_network_scores.csv",
) -> tuple[dict[Any, pd.DataFrame], pd.DataFrame, list[str]]:
    """Load CellOracle HDF5 and per-cluster coefficient edges (nonzero links only)."""
    import celloracle as co

    grn_workdir = Path(grn_workdir)
    oracle = co.load_hdf5(str(grn_workdir / "out_files" / oracle_object_name))
    network_scores = pd.read_csv(grn_workdir / "out_files" / network_scores_filename, index_col=0)
    grn_tfs = list(oracle.all_regulatory_genes_in_TFdict)
    links_after_fit: dict[Any, pd.DataFrame] = {key: pd.DataFrame() for key in oracle.coef_matrix_per_cluster.keys()}
    for cluster in oracle.coef_matrix_per_cluster.keys():
        links = oracle.coef_matrix_per_cluster[cluster].stack().reset_index()
        links.columns = ["source", "target", "coef_mean"]
        links = links[links["coef_mean"] != 0].reset_index(drop=True)
        links["coef_abs"] = np.abs(links["coef_mean"])
        links_after_fit[cluster] = links
    return links_after_fit, network_scores, grn_tfs
