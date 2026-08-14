"""TF enrichment bar plots built from CSV inputs (no CellOracle object required).

Adapts ``StateSpecificEnrichment.plot_enrichment_scores_with_color_proportions`` to
work from flat CSVs:

* **links CSV** — ``TF`` → downstream target gene (``Target`` or ``Gene`` column).
  Supplies each TF's downstream gene set, which is split by SLIDE LF correlation sign.
* **enrichment CSV** — per-episode ``TF,p_value,enrichment_score,...``. Supplies the
  bar height (ES). Nothing is recomputed here.
* **SLIDE ``*feature_list*`` TSVs** — ``names``/``corrs`` give each gene its sign:
  red = positively correlated with the LF, blue = negative, gray = not an LF gene.

Bar height is the TF's enrichment score; each bar is stacked into red/blue/gray
segments proportional to that TF's downstream gene composition. The figures
themselves live in :mod:`firefate.utils.plots`; this module builds their input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from firefate.utils.plots import (  # noqa: F401  (re-exported for existing call sites)
    COLOR_MAP,
    _COLOR_ORDER,
    plot_tf_enrichment_bars,
    plotly_tf_enrichment_bars,
)


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


def build_tf_color_bar_table(
    links: pd.DataFrame,
    enrichment: pd.DataFrame,
    gene_colors: dict[str, str],
) -> pd.DataFrame:
    """Join links to scores and split each TF's bar into color-proportioned segments.

    Returns a long table with one row per (TF, color): ``proportion`` is the share of
    that TF's downstream genes with that color, ``height`` is ``proportion * score``.
    TFs in ``enrichment`` with no links are dropped.
    """
    scored = links.merge(enrichment[["TF", "score"]], left_on="source", right_on="TF")
    if scored.empty:
        raise ValueError("No TF overlap between the links CSV and the enrichment CSV.")
    scored["color"] = scored["target"].map(gene_colors).fillna("gray")

    proportions = (
        scored.groupby("source")["color"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=_COLOR_ORDER, fill_value=0.0)
    )
    scores = scored.drop_duplicates("source").set_index("source")["score"]
    order = scores.loc[proportions.index].sort_values(ascending=False).index

    plot_df = (
        proportions.loc[order]
        .stack()
        .rename("proportion")
        .reset_index()
        .rename(columns={"level_1": "color"})
    )
    plot_df["height"] = plot_df["proportion"] * plot_df["source"].map(scores)
    plot_df["text"] = (plot_df["proportion"] * 100).round(1).astype(str) + "%"
    plot_df["source"] = pd.Categorical(plot_df["source"], categories=order, ordered=True)
    return plot_df.sort_values(["source", "color"]).reset_index(drop=True)


def plot_episode_from_csvs(
    links_csv: str | Path,
    enrichment_csv: str | Path,
    feature_files: Iterable[str | Path],
    title: str,
    out_path: str | Path | None = None,
    *,
    p_max: float | None = 0.05,
    a_loading_threshold: float | None = None,
    verbose: bool = True,
) -> tuple[Any, pd.DataFrame]:
    """End-to-end: links + enrichment + LF signs → stacked TF enrichment bar plot.

    Returns the matplotlib figure and the long plot table (one row per TF × color).
    """
    gene_colors = load_lf_gene_colors(feature_files, a_loading_threshold=a_loading_threshold)
    links = load_tf_target_links(links_csv)
    enrichment = load_episode_enrichment(enrichment_csv, p_max=p_max)
    plot_df = build_tf_color_bar_table(links, enrichment, gene_colors)

    if verbose:
        dropped = sorted(set(enrichment["TF"]) - set(links["source"]))
        print(
            f"{title}: {plot_df['source'].nunique()} TFs plotted, "
            f"{len(dropped)} of {len(enrichment)} scored TFs dropped (no links): {dropped}"
        )
    fig, _ = plot_tf_enrichment_bars(plot_df, title, out_path)
    return fig, plot_df
