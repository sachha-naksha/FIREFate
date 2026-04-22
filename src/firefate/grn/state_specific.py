"""State-specific GRN construction from CellOracle outputs (capability 2).

Structured after ``TA_muscle_ageing/py_scripts/grn_inference`` — cluster fusion,
edge quantile thresholding, and NetworkX graph views for SLIDE–GRN enrichment.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from firefate.enrichment.slide_grn import (
    build_enrichment_table,
    get_slide_grn_enrichment,
    write_enrichment_table,
)

if TYPE_CHECKING:
    pass


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
        Quantile of |coef_abs| used as cutoff; edges at or above are ``strength == 1``.
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


class StateSpecificEnrichment:
    """Orchestrate SLIDE feature loading, cluster fusion, and SLIDE–GRN enrichment.

    Mirrors ``grn_inference/state_lf_enrich.py`` with pathlib-based paths and optional
    figure I/O (no global scanpy style side effects).
    """

    def __init__(
        self,
        grn_workdir: str | Path,
        oracle_object_name: str,
        feature_folder: str | Path,
        out_path: str | Path,
        slide_starting_genes: int,
        clusters_of_interest: list[Any],
        order_fr_clust: list[int],
        order_fr_tfcomb: list[int],
        *,
        quantile: float = 0.70,
        weight: str = "strength",
        experiment_label: str = "",
    ):
        self.grn_workdir = Path(grn_workdir)
        self.oracle_object_name = oracle_object_name
        self.feature_folder = Path(feature_folder)
        self.out_path = Path(out_path)
        self.slide_starting_genes = slide_starting_genes
        self.clusters_of_interest = clusters_of_interest
        self.order_fr_clust = order_fr_clust
        self.order_fr_tfcomb = order_fr_tfcomb
        self.quantile = quantile
        self.weight = weight
        self.experiment_label = experiment_label

        self.out_path.joinpath("figures").mkdir(parents=True, exist_ok=True)
        self.out_path.joinpath("out_files", "SLIDE_LF_enrichment").mkdir(parents=True, exist_ok=True)

        self.GRN_links_after_fit: dict[Any, pd.DataFrame] | None = None
        self.GRN_network_scores: pd.DataFrame | None = None
        self.GRN_TFs: list[str] | None = None
        self.slide_features: set[str] | None = None
        self.cluster_fusions: list[tuple[Any, ...]] = []
        self.cc_dicts: dict[tuple[Any, ...], dict[int, list]] = {}
        self.enrichment_dfs: dict[tuple[Any, ...], dict[int, pd.DataFrame]] = {}

    def load_grn_data(self) -> None:
        links, scores, tfs = load_celloracle_grn_tables(self.grn_workdir, self.oracle_object_name)
        self.GRN_links_after_fit = links
        self.GRN_network_scores = scores
        self.GRN_TFs = tfs

    def load_slide_features_from_files(
        self,
        feature_files: list[str | Path] | None = None,
        *,
        a_loading_threshold: float | None = None,
    ) -> None:
        if feature_files is None:
            feature_files = sorted(self.feature_folder.glob("*feature_list*"))
        data = pd.concat([pd.read_csv(f, sep="\t", header=0) for f in feature_files])
        if a_loading_threshold is not None:
            data = data[data["A_loading"] >= a_loading_threshold]
        self.slide_features = set(data["names"])

    def build_cluster_fusions(self) -> None:
        self.cluster_fusions = []
        for ord_clus in self.order_fr_clust:
            self.cluster_fusions += list(itertools.combinations(self.clusters_of_interest, ord_clus))

    def run_enrichment(
        self,
        *,
        save_pickle: bool = True,
        show_progress: bool = True,
        save_strength_diagnostics: bool = True,
    ) -> None:
        import pickle

        if self.GRN_links_after_fit is None or self.GRN_TFs is None or self.slide_features is None:
            raise RuntimeError("Call load_grn_data() and load_slide_features_from_files() first.")

        self.build_cluster_fusions()
        suffix = f"_{self.experiment_label}" if self.experiment_label else ""

        for cluster_fusion in self.cluster_fusions:
            combined_links, threshold = combine_cluster_grn_links(
                self.GRN_links_after_fit,
                cluster_fusion,
                quantile=self.quantile,
                histogram_path=self.out_path / "figures" / f"combined_links_cutoff_histogram_{cluster_fusion}.pdf",
            )
            grn, edges_df = grn_edges_from_combined_links(combined_links)

            if save_strength_diagnostics:
                fig = edges_df.groupby(["strength", "key"]).size().unstack().plot(kind="bar", stacked=True)
                plt.xlabel("Strength")
                plt.ylabel("Count")
                plt.title("Distribution of key and strength")
                plt.savefig(
                    self.out_path / "figures" / f"combined_links_key_strength_{cluster_fusion}{suffix}.pdf"
                )
                plt.close()

            slide_in_graph = self.slide_features.intersection(set(grn.nodes))
            neighbors: list[str] = []
            for gene in slide_in_graph:
                neighbors.extend(list(grn.predecessors(gene)))
            slide_tf_candidates = (slide_in_graph.union(set(neighbors))).intersection(self.GRN_TFs)

            for ord_tf in self.order_fr_tfcomb:
                cc_dict: dict[tuple[Any, ...], dict[int, list]] = {}
                get_slide_grn_enrichment(
                    edges_df,
                    cc_dict,
                    cluster_fusion,
                    ord_tf,
                    slide_in_graph,
                    self.slide_starting_genes,
                    slide_tf_candidates,
                    "slide",
                    show_progress=show_progress,
                )
                self.cc_dicts.setdefault(cluster_fusion, {})[ord_tf] = cc_dict

                enrichment_df = build_enrichment_table(cc_dict, cluster_fusion, ord_tf)
                self.enrichment_dfs.setdefault(cluster_fusion, {})[ord_tf] = enrichment_df
                csv_path = (
                    self.out_path / "out_files" / "SLIDE_LF_enrichment" / f"enriched_df_{ord_tf}_TFs_{cluster_fusion}{suffix}.csv"
                )
                write_enrichment_table(enrichment_df, csv_path)

                if save_pickle:
                    pkl_path = (
                        self.out_path / "out_files" / "SLIDE_LF_enrichment" / f"cc_dict_{ord_tf}_TFs_{cluster_fusion}{suffix}.pickle"
                    )
                    with open(pkl_path, "wb") as f:
                        pickle.dump(cc_dict, f)
