"""SLIDE latent-factor enrichment against state-specific GRN edges (capabilities 2 and 5).

This module ports the statistical and combinatorial core from
``grn_inference/utils.py`` (TA muscle ageing workflow): hypergeometric overrepresentation
of SLIDE feature genes among GRN downstream targets, over TF combinations and
binary edge-strength patterns.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from scipy.stats import hypergeom
from tqdm.auto import tqdm


def hypergeom_slide_grn_score(
    population_size: int,
    successes_in_population: int,
    sample_size: int,
    successes_in_sample: int,
) -> tuple[float, float]:
    """Hypergeometric log2 fold-enrichment and one-sided p-value.

    Parameters
    ----------
    population_size
        ``P`` — e.g. total genes considered in SLIDE / universe.
    successes_in_population
        ``p`` — e.g. number of downstream targets in the GRN slice.
    sample_size
        ``S`` — e.g. SLIDE latent-factor genes overlapping the network.
    successes_in_sample
        ``s`` — e.g. LF genes that are also downstream targets.

    Returns
    -------
    score, p_value
        log2((s/S)/(p/P)) and P(X >= s) under Hypergeom(P, p, S).
    """
    if sample_size <= 0 or population_size <= 0:
        return 0.0, 1.0
    if successes_in_sample <= 0:
        return 0.0, 1.0
    p_value = float(1.0 - hypergeom.cdf(successes_in_sample - 1, population_size, successes_in_population, sample_size))
    expected = (sample_size * successes_in_population) / population_size
    score = math.log2((successes_in_sample / sample_size) / (successes_in_population / population_size)) if expected > 0 else 0.0
    return score, p_value


def get_slide_grn_enrichment(
    edges_df: pd.DataFrame,
    cc_dict: dict[tuple[Any, ...], dict[int, list]],
    cluster_fusion: tuple[Any, ...],
    ord_tf: int,
    slide_features: set[str] | Iterable[str],
    slide_starting_genes: int,
    total_source: set[str] | Iterable[str],
    case: str,
    *,
    show_progress: bool = True,
) -> dict[tuple[Any, ...], dict[int, list]]:
    """Enumerate TF combinations and strength conditions; fill ``cc_dict`` (in-place).

    ``edges_df`` columns must include ``source``, ``target``, ``strength`` (0/1),
    and ``weight`` (used to dedupe by max absolute ``weight`` per source–target).

    The nested list structure matches the TA muscle pipeline for backward compatibility
    with :func:`build_enrichment_table` / pickling.
    """
    slide_features = set(slide_features)
    total_source = set(total_source)
    possible_tf_combinations = list(itertools.combinations(sorted(total_source), ord_tf))
    strength_condition_tf_comb = list(itertools.product([0, 1], repeat=ord_tf))
    cc_dict.setdefault(cluster_fusion, {}).setdefault(ord_tf, [])

    for tf_comb in tqdm(
        possible_tf_combinations,
        desc=f"SLIDE–GRN enrichment {cluster_fusion}",
        disable=not show_progress,
    ):
        edges_grouped = edges_df[edges_df["source"].isin(tf_comb)].groupby("source").agg(list)
        if edges_grouped.empty:
            continue
        common_targets_from_grn = set.intersection(*map(set, edges_grouped["target"].values))
        for condition in strength_condition_tf_comb:
            if len(common_targets_from_grn) == 0:
                cc_dict[cluster_fusion][ord_tf].append(
                    [tf_comb, condition, (0, 1), ([], []), case, (pd.DataFrame(), pd.DataFrame())]
                )
                continue

            filter_df = pd.DataFrame({"source": tf_comb, "strength": condition})
            filtered = edges_df.merge(filter_df, on=["source", "strength"])
            filtered = filtered[filtered["target"].isin(common_targets_from_grn)]

            common_targets = (
                filtered.groupby("target")[["source", "strength"]].nunique().eq(len(filter_df)).all(axis=1)
            )
            filtered_edges_df = filtered[filtered["target"].isin(common_targets[common_targets].index)]
            if filtered_edges_df.empty:
                cc_dict[cluster_fusion][ord_tf].append(
                    [tf_comb, condition, (0, 1), ([], []), case, (filtered_edges_df, pd.DataFrame())]
                )
                continue

            idx = filtered_edges_df.groupby(["source", "target"])["weight"].apply(lambda x: x.abs().idxmax())
            filtered_edges_df_unique = filtered_edges_df.loc[idx]
            dwn_list = list(filtered_edges_df_unique["target"].unique())
            dwngene = len(dwn_list)
            cmn_list = list(set(filtered_edges_df_unique["target"]).intersection(slide_features))
            common = len(cmn_list)
            if common == 0:
                cc_dict[cluster_fusion][ord_tf].append(
                    [tf_comb, condition, (0, 1), ([], [dwn_list]), case, (filtered_edges_df, filtered_edges_df_unique)]
                )
            else:
                enrich = hypergeom_slide_grn_score(slide_starting_genes, dwngene, len(slide_features), common)
                cc_dict[cluster_fusion][ord_tf].append(
                    [tf_comb, condition, enrich, (cmn_list, dwn_list), case, (filtered_edges_df, filtered_edges_df_unique)]
                )
    return cc_dict


def _records_to_enrichment_df(
    cc_dict: dict[tuple[Any, ...], dict[int, list]],
    cluster_fusion: tuple[Any, ...],
    order_of_combination: int,
    filter_conditions: list[tuple[int, ...]] | None,
) -> pd.DataFrame:
    enrichment_df = pd.DataFrame(
        cc_dict[cluster_fusion][order_of_combination],
        columns=["TF", "condition", "ES", "Genes", "case", "dfs"],
    )
    enrichment_df[["score", "p_value"]] = pd.DataFrame(enrichment_df["ES"].tolist(), index=enrichment_df.index)
    enrichment_df[["common", "dwnstrm"]] = pd.DataFrame(enrichment_df["Genes"].tolist(), index=enrichment_df.index)
    enrichment_df = enrichment_df.drop(columns=["ES", "Genes"])
    if filter_conditions is not None:
        enrichment_df = enrichment_df[enrichment_df["condition"].isin(filter_conditions)]
    enrichment_df = enrichment_df.sort_values(by="score", ascending=False)

    if order_of_combination == 1:
        mask = (
            (enrichment_df["p_value"] < 0.05)
            & (enrichment_df["dwnstrm"].apply(len) > 2)
            & (enrichment_df["common"].apply(len) > 1)
            & (enrichment_df["score"] > 0)
            & (enrichment_df["case"].isin(["slide", "net", "rnd"]))
        )
        enrichment_df = enrichment_df[mask].reset_index(drop=True)
    elif order_of_combination == 2:
        mask = (
            (enrichment_df["p_value"] < 0.05)
            & (enrichment_df["dwnstrm"].apply(len) > 2)
            & (enrichment_df["common"].apply(len) > 1)
            & (enrichment_df["score"] > 0)
            & (enrichment_df["case"].isin(["slide", "net", "rnd"]))
        )
        enrichment_df = enrichment_df.copy()
        enrichment_df.loc[~mask, "score"] = 0
        enrichment_df.loc[~mask, "p_value"] = 1.0
    else:
        raise ValueError("order_of_combination must be 1 or 2")

    return enrichment_df


def build_enrichment_table(
    cc_dict: dict[tuple[Any, ...], dict[int, list]],
    cluster_fusion: tuple[Any, ...],
    order_of_combination: int,
    filter_conditions: list[tuple[int, ...]] | None = None,
) -> pd.DataFrame:
    """Turn in-memory ``cc_dict`` records into a filtered enrichment table."""
    if order_of_combination == 1 and filter_conditions is None:
        filter_conditions = [(1,)]
    elif order_of_combination == 2 and filter_conditions is None:
        filter_conditions = [(1, 1), (0, 1), (1, 0)]
    elif order_of_combination not in (1, 2):
        raise ValueError("order_of_combination must be 1 or 2")
    return _records_to_enrichment_df(cc_dict, cluster_fusion, order_of_combination, filter_conditions)


def write_enrichment_table(
    enrichment_df: pd.DataFrame,
    path: str | Path,
    *,
    serialize_dfs: bool = True,
) -> None:
    """Write enrichment table to CSV; optionally JSON-serialize embedded DataFrame tuples."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = enrichment_df.copy()
    if serialize_dfs and "dfs" in out.columns:
        out["dfs"] = out["dfs"].apply(
            lambda x: x.to_json() if isinstance(x, pd.DataFrame) else str(x)
        )
    out.to_csv(path, index=False)
