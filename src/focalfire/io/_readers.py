"""Readers for the flat files FocalFire consumes: SLIDE feature lists, link tables, enrichment CSVs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import pandas as pd


def load_lf_gene_colors(
    feature_files: Iterable[str | Path],
    *,
    a_loading_threshold: float | None = None,
) -> dict[str, str]:
    """Map each SLIDE LF gene to ``"red"`` (corrs >= 0) or ``"blue"`` (corrs < 0).

    Genes appearing in several LFs are resolved by majority vote, ties going to red
    (matching the ``corrs >= 0`` convention in the original class).
    """
    frames = [pd.read_csv(f, sep="\t", header=0) for f in feature_files]
    if not frames:
        raise ValueError("No feature files supplied.")
    data = pd.concat(frames, ignore_index=True)
    if a_loading_threshold is not None:
        data = data[data["A_loading"] >= a_loading_threshold]

    votes = (
        data.assign(_c=lambda d: (d["corrs"] >= 0).map({True: "red", False: "blue"}))
        .groupby(["names", "_c"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["red", "blue"], fill_value=0)
    )
    return {g: ("red" if r >= b else "blue") for g, r, b in votes.itertuples(index=True)}


def load_tf_target_links(
    path: str | Path,
    *,
    tf_col: str = "TF",
    target_col: str | None = None,
) -> pd.DataFrame:
    """Read a TF → downstream-target links CSV into ``source``/``target`` columns.

    ``target_col`` defaults to whichever of ``Target``/``Gene``/``target`` is present.
    """
    links = pd.read_csv(path)
    if target_col is None:
        for candidate in ("Target", "Gene", "target"):
            if candidate in links.columns:
                target_col = candidate
                break
        else:
            raise KeyError(
                f"No target column found in {path}; columns are {list(links.columns)}. "
                "Pass target_col explicitly."
            )
    out = links[[tf_col, target_col]].rename(columns={tf_col: "source", target_col: "target"})
    return out.drop_duplicates().reset_index(drop=True)


def load_episode_enrichment(
    path: str | Path,
    *,
    p_max: float | None = 0.05,
    score_col: str = "enrichment_score",
) -> pd.DataFrame:
    """Read a per-episode enrichment CSV, optionally keeping only ``p_value < p_max``."""
    enrichment = pd.read_csv(path)
    if p_max is not None:
        enrichment = enrichment[enrichment["p_value"] < p_max]
    return (
        enrichment[["TF", score_col, "p_value"]]
        .rename(columns={score_col: "score"})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
