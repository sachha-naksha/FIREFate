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
segments proportional to that TF's downstream gene composition.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

COLOR_MAP = {"red": "#c1272d", "blue": "#2b6cb0", "gray": "#b0b0b0"}
_COLOR_ORDER = ["red", "blue", "gray"]
_COLOR_LABELS = {
    "red": "positively correlated LF gene",
    "blue": "negatively correlated LF gene",
    "gray": "not an LF gene",
}


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


def plot_tf_enrichment_bars(
    plot_df: pd.DataFrame,
    title: str,
    out_path: str | Path | None = None,
    *,
    min_label_proportion: float = 0.08,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Stacked bar of enrichment score per TF, split by LF correlation sign.

    Segments smaller than ``min_label_proportion`` of the bar are left unlabelled so
    the in-bar percentages stay readable. Saves vector output (SVG/PDF) when
    ``out_path`` is given.
    """
    wide = (
        plot_df.pivot(index="source", columns="color", values="height")
        .reindex(columns=_COLOR_ORDER, fill_value=0.0)
        .fillna(0.0)
    )
    props = (
        plot_df.pivot(index="source", columns="color", values="proportion")
        .reindex(columns=_COLOR_ORDER, fill_value=0.0)
        .fillna(0.0)
    )
    tfs = list(wide.index)

    if figsize is None:
        figsize = (max(6.0, 0.28 * len(tfs) + 2.0), 4.5)
    fig, ax = plt.subplots(figsize=figsize)

    bottom = pd.Series(0.0, index=wide.index)
    for color in _COLOR_ORDER:
        heights = wide[color]
        if heights.sum() == 0:
            continue
        ax.bar(
            tfs,
            heights,
            bottom=bottom,
            color=COLOR_MAP[color],
            label=_COLOR_LABELS[color],
            width=0.8,
        )
        for tf in tfs:
            if props.loc[tf, color] >= min_label_proportion:
                ax.text(
                    tf,
                    bottom[tf] + heights[tf] / 2,
                    f"{props.loc[tf, color] * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                )
        bottom = bottom + heights

    ax.set_xlabel("Transcription factor (TF)")
    ax.set_ylabel("Enrichment score")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.set_ylim(0, float(wide.sum(axis=1).max()) * 1.08)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Downstream gene", frameon=False, fontsize=7, title_fontsize=7)
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    return fig, ax


def plotly_tf_enrichment_bars(plot_df: pd.DataFrame, title: str) -> Any:
    """Interactive plotly version of :func:`plot_tf_enrichment_bars` (notebook display).

    Static export via ``fig.write_image`` needs kaleido + Chrome (``plotly_get_chrome``);
    use :func:`plot_tf_enrichment_bars` for file output instead.
    """
    import plotly.express as px

    data = plot_df.copy()
    data.loc[data["proportion"] == 0, "text"] = ""
    fig = px.bar(
        data,
        x="source",
        y="height",
        color="color",
        text="text",
        color_discrete_map=COLOR_MAP,
        category_orders={"color": _COLOR_ORDER},
        labels={"height": "Enrichment score", "source": "TF"},
        title=title,
    )
    fig.update_traces(textposition="inside", insidetextanchor="middle")
    fig.update_layout(
        xaxis_title="Transcription factor (TF)",
        yaxis_title="Enrichment score",
        xaxis_tickangle=-90,
        font=dict(family="Arial", size=8, color="black"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, showline=True, linecolor="black", ticks="outside"),
        yaxis=dict(showgrid=False, showline=True, linecolor="black", ticks="outside"),
    )
    return fig


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
