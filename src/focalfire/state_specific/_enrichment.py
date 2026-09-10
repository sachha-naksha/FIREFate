"""State-specific SLIDE–GRN enrichment orchestration — the ``sse`` object.

:class:`StateSpecificEnrichment` drives the workflow end to end:

1. load CellOracle GRN tables and SLIDE latent-factor gene sets;
2. fuse clusters and threshold edges into a state-specific network — the GRN
   construction itself stays in :mod:`focalfire.state_specific._grn`;
3. run hypergeometric SLIDE–GRN enrichment over TF combinations
   (:mod:`focalfire.state_specific._slide`);
4. post-process into per-TF used-weight and enriched-link tables;
5. plot them with the bar-chart helpers further down this module.

Loading SLIDE features is explicit: pass the ``*feature_list*`` TSVs you want, via
:meth:`load_slide_features_from_files` (single set) or
:meth:`load_slide_features_combined` (several LFs fused into one cellular program,
with per-gene correlation signs). Experiment-specific file presets belong in the
analysis project, not here.
"""
from __future__ import annotations

import ast
import itertools
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from focalfire.state_specific._slide import (
    build_enrichment_table,
    get_slide_grn_enrichment,
    write_enrichment_table,
)
from focalfire.state_specific._grn import (
    combine_cluster_grn_links,
    grn_edges_from_combined_links,
    load_celloracle_grn_tables,
)
import matplotlib
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Tuple, Union
from focalfire.io._readers import (
    load_episode_enrichment,
    load_lf_gene_colors,
    load_tf_target_links,
)


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
        self.slide_features_provenance: dict[str, list[str]] = {}
        self.positive_corr_names: set[str] | None = None
        self.negative_corr_names: set[str] | None = None
        self.cluster_fusions: list[tuple[Any, ...]] = []
        self.cc_dicts: dict[tuple[Any, ...], dict[int, dict]] = {}
        self.enrichment_dfs: dict[tuple[Any, ...], dict[int, pd.DataFrame]] = {}

    # ── paths ──────────────────────────────────────────────────────────────

    @property
    def _suffix(self) -> str:
        return f"_{self.experiment_label}" if self.experiment_label else ""

    @property
    def _enrichment_dir(self) -> Path:
        return self.out_path / "out_files" / "SLIDE_LF_enrichment"

    def _stem(self, cluster_fusion: tuple[Any, ...], ord_tf: int) -> str:
        return f"{ord_tf}_TFs_{cluster_fusion}{self._suffix}"

    def _records(self, cluster_fusion: tuple[Any, ...], ord_tf: int) -> list:
        """Flat record list for one (cluster fusion, TF order).

        ``self.cc_dicts[cf][ord]`` holds the whole ``cc_dict`` — itself keyed by
        cluster fusion then TF order — so the layout of pickles written by
        :meth:`run_enrichment` matches the TA muscle pipeline. Hence the double index.
        """
        return self.cc_dicts[cluster_fusion][ord_tf][cluster_fusion][ord_tf]

    # ── 1. Data loading ────────────────────────────────────────────────────

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

    def load_slide_features_combined(
        self,
        lf_files: list[str | Path],
        a_loading_thresholds: list[float] | float | None = None,
        keep_corr_sign: bool = True,
        sign_resolution: str = "majority",
        verbose: bool = True,
    ) -> None:
        """Fuse multiple SLIDE LF lists into one combined cellular program.

        Each SLIDE latent factor is stored as a ``*feature_list*`` TSV. This method
        unions the gene sets across the supplied LFs, applies per-LF ``A_loading``
        cutoffs, and resolves a single correlation sign per gene when
        ``keep_corr_sign=True``.

        Parameters
        ----------
        lf_files
            Paths to ``*feature_list*`` TSVs. One file = one LF.
        a_loading_thresholds
            Per-LF ``A_loading`` cutoff. Scalar broadcasts to all files. ``None``
            keeps every row of every file.
        keep_corr_sign
            If True, populates ``self.positive_corr_names`` /
            ``self.negative_corr_names`` (used by
            :meth:`plot_enrichment_scores_with_color_proportions`).
        sign_resolution
            How to assign a gene's sign when it appears in multiple LFs with
            conflicting signs:

            - ``"majority"``: sign with the most LF votes (ties → positive).
            - ``"first"``:    sign from the first LF in ``lf_files`` containing it.
            - ``"mixed"``:    drop conflicting genes from both sign sets.
        verbose
            Print per-LF row counts and sign-conflict diagnostics.

        Notes
        -----
        ``self.slide_features`` (the union) is the combined CP that
        :meth:`run_enrichment` consumes downstream. ``self.slide_features_provenance``
        maps each gene → list of originating LF filenames.
        """
        # ---- normalize thresholds ------------------------------------------------
        if a_loading_thresholds is None:
            thresholds: list[float | None] = [None] * len(lf_files)
        elif isinstance(a_loading_thresholds, (int, float)):
            thresholds = [float(a_loading_thresholds)] * len(lf_files)
        else:
            if len(a_loading_thresholds) != len(lf_files):
                raise ValueError(
                    f"Length mismatch: {len(lf_files)} files vs "
                    f"{len(a_loading_thresholds)} thresholds."
                )
            thresholds = list(a_loading_thresholds)

        if sign_resolution not in {"majority", "first", "mixed"}:
            raise ValueError(
                f"sign_resolution must be one of 'majority', 'first', 'mixed'; "
                f"got {sign_resolution!r}."
            )

        # ---- read + threshold each LF, stamp provenance --------------------------
        lf_dfs: list[pd.DataFrame] = []
        for path, thr in zip(lf_files, thresholds):
            df = pd.read_csv(path, sep="\t", header=0)
            if thr is not None:
                df = df[df["A_loading"] >= thr]
            lf_dfs.append(df.assign(_lf=Path(path).name))

        combined = pd.concat(lf_dfs, ignore_index=True)

        # ---- union → combined CP -------------------------------------------------
        self.slide_features = set(combined["names"])
        self.slide_features_provenance = (
            combined.groupby("names")["_lf"].apply(list).to_dict()
        )

        if verbose:
            print(
                f"Combined CP: {len(self.slide_features)} unique genes "
                f"from {len(lf_files)} LFs"
            )
            for path, thr, df in zip(lf_files, thresholds, lf_dfs):
                thr_str = f"A_loading ≥ {thr}" if thr is not None else "no threshold"
                print(f"  {Path(path).name:<40s} {len(df):>5d} genes  ({thr_str})")

        if not keep_corr_sign:
            return

        if "corrs" not in combined.columns:
            raise KeyError(
                "keep_corr_sign=True but 'corrs' column missing from feature files."
            )

        # ---- per-gene sign resolution across LFs ---------------------------------
        sign_counts = (
            combined.assign(_sign=np.where(combined["corrs"] >= 0, "pos", "neg"))
                    .groupby(["names", "_sign"]).size()
                    .unstack(fill_value=0)
                    .reindex(columns=["pos", "neg"], fill_value=0)
        )

        if sign_resolution == "majority":
            # ties → pos, matching the `corrs >= 0` convention used across the package
            pos = set(sign_counts.index[sign_counts["pos"] >= sign_counts["neg"]])
            neg = set(sign_counts.index[sign_counts["neg"] > sign_counts["pos"]])
        elif sign_resolution == "first":
            first = (
                combined.drop_duplicates("names", keep="first")
                        .assign(_sign=lambda d: np.where(d["corrs"] >= 0, "pos", "neg"))
                        .set_index("names")["_sign"]
            )
            pos = set(first.index[first == "pos"])
            neg = set(first.index[first == "neg"])
        else:  # "mixed": conflicting genes are dropped from both sets
            pure_pos = (sign_counts["pos"] > 0) & (sign_counts["neg"] == 0)
            pure_neg = (sign_counts["neg"] > 0) & (sign_counts["pos"] == 0)
            pos = set(sign_counts.index[pure_pos])
            neg = set(sign_counts.index[pure_neg])

        self.positive_corr_names = pos
        self.negative_corr_names = neg

        if verbose:
            conflicts = int(((sign_counts["pos"] > 0) & (sign_counts["neg"] > 0)).sum())
            print(
                f"sign_resolution={sign_resolution!r}: "
                f"{len(pos)} positive, {len(neg)} negative, "
                f"{conflicts} genes with conflicting signs across LFs"
            )

    # ── 2. Run enrichment ──────────────────────────────────────────────────

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
        if self.GRN_links_after_fit is None or self.GRN_TFs is None or self.slide_features is None:
            raise RuntimeError("Call load_grn_data() and a load_slide_features_* method first.")

        self.build_cluster_fusions()

        for cluster_fusion in self.cluster_fusions:
            combined_links, threshold = combine_cluster_grn_links(
                self.GRN_links_after_fit,
                cluster_fusion,
                quantile=self.quantile,
                histogram_path=self.out_path / "figures" / f"combined_links_cutoff_histogram_{cluster_fusion}.pdf",
            )
            grn, edges_df = grn_edges_from_combined_links(combined_links)

            if save_strength_diagnostics:
                plot_strength_key_distribution(
                    edges_df,
                    out_path=self.out_path
                    / "figures"
                    / f"combined_links_key_strength_{cluster_fusion}{self._suffix}.pdf",
                )

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
                write_enrichment_table(
                    enrichment_df,
                    self._enrichment_dir / f"enriched_df_{self._stem(cluster_fusion, ord_tf)}.csv",
                )

                if save_pickle:
                    pkl_path = self._enrichment_dir / f"cc_dict_{self._stem(cluster_fusion, ord_tf)}.pickle"
                    with open(pkl_path, "wb") as f:
                        pickle.dump(cc_dict, f)

    # ── 3. Post-processing ─────────────────────────────────────────────────

    def load_results(self, cluster_fusion: tuple[Any, ...], ord_tf: int) -> None:
        """Restore ``cc_dicts`` / ``enrichment_dfs`` from a previous :meth:`run_enrichment`."""
        stem = self._stem(cluster_fusion, ord_tf)
        with open(self._enrichment_dir / f"cc_dict_{stem}.pickle", "rb") as f:
            cc_dict = pickle.load(f)
        self.cc_dicts.setdefault(cluster_fusion, {})[ord_tf] = cc_dict
        self.enrichment_dfs.setdefault(cluster_fusion, {})[ord_tf] = pd.read_csv(
            self._enrichment_dir / f"enriched_df_{stem}.csv"
        )

    def build_used_weight_df(self, cluster_fusion: tuple[Any, ...], ord_tf: int) -> pd.DataFrame:
        """Concatenate the per-record deduplicated edge tables into one ``used_weights`` CSV."""
        frames = []
        for record in self._records(cluster_fusion, ord_tf):
            w_df = record[5][1]          # (filtered edges, deduplicated edges)
            if w_df.empty:
                continue
            w_df = w_df.copy()
            w_df["case"] = record[4]
            frames.append(w_df)

        concatenated = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        concatenated.to_csv(
            self._enrichment_dir / f"{self._stem(cluster_fusion, ord_tf)}_used_weights.csv", index=False
        )
        return concatenated

    def build_slide_lf_enriched(self, cluster_fusion: tuple[Any, ...], ord_tf: int) -> pd.DataFrame:
        """Explode enriched TF→target links and join the strong (``strength == 1``) edge weights."""
        stem = self._stem(cluster_fusion, ord_tf)
        slide_lf = pd.read_csv(self._enrichment_dir / f"enriched_df_{stem}.csv")
        used_w = pd.read_csv(self._enrichment_dir / f"{stem}_used_weights.csv")
        slide_lf = slide_lf[slide_lf["case"] == "slide"]
        used_w = used_w[used_w["case"] == "slide"]

        slide_lf = slide_lf[["TF", "common"]]
        slide_lf["TF"] = slide_lf["TF"].apply(ast.literal_eval).apply(tuple)
        slide_lf["common"] = slide_lf["common"].apply(ast.literal_eval).apply(list)
        slide_lf = slide_lf.explode("common").explode("TF")
        slide_lf.columns = ["source", "target"]
        used_w = used_w[used_w["strength"] == 1]
        slide_lf = slide_lf.merge(used_w, on=["source", "target"], how="left")

        slide_lf.to_csv(
            self._enrichment_dir / f"enriched_df_{stem}_used_weights_strong.csv", index=False
        )
        return slide_lf

    def tf_enrichment_scores(self, cluster_fusion: tuple[Any, ...], ord_tf: int) -> dict[str, float]:
        """Per-TF enrichment score, taken from the all-edges-strong condition."""
        if ord_tf != 1:
            raise ValueError(
                f"Per-TF scores are only defined for ord_tf=1; got {ord_tf}. "
                "Higher orders score TF combinations, not single TFs."
            )
        return {
            record[0][0]: record[2][0]
            for record in self._records(cluster_fusion, ord_tf)
            if record[1] == (1,)
        }

    # ── 4. Visualization ───────────────────────────────────────────────────

    def plot_strength_key_distribution(
        self,
        slide_lf_enriched: pd.DataFrame,
        cluster_fusion: tuple[Any, ...],
        ord_tf: int,
    ):
        """Edge-strength/key breakdown of the enriched links (figure written to ``figures/``)."""
        return plot_strength_key_distribution(
            slide_lf_enriched,
            out_path=self.out_path
            / "figures"
            / f"enriched_key_strength_{self._stem(cluster_fusion, ord_tf)}_used_weights_strong.pdf",
        )

    def plot_enrichment_scores_with_color_proportions(
        self,
        slide_lf_enriched: pd.DataFrame,
        cluster_fusion: tuple[Any, ...],
        ord_tf: int,
        *,
        out_path: str | Path | None = None,
    ):
        """Stacked TF enrichment bars, each split by the LF correlation sign of its targets.

        Bar height is the TF's enrichment score; the red/blue/gray split is the share of
        its downstream targets that are positively correlated LF genes, negatively
        correlated ones, or not LF genes. Needs correlation signs, so call
        :meth:`load_slide_features_combined` with ``keep_corr_sign=True`` first.
        """
        if self.positive_corr_names is None:
            raise RuntimeError(
                "Correlation signs unknown. Call load_slide_features_combined(keep_corr_sign=True) first."
            )

        scores = self.tf_enrichment_scores(cluster_fusion, ord_tf)
        gene_colors = {
            **{g: "red" for g in self.positive_corr_names},
            **{g: "blue" for g in self.negative_corr_names},
        }
        plot_df = build_tf_color_bar_table(
            slide_lf_enriched[["source", "target"]],
            pd.DataFrame({"TF": list(scores), "score": list(scores.values())}),
            gene_colors,
        )

        fig = plotly_tf_enrichment_bars(
            plot_df,
            f"TF enrichment scores with colour proportions for "
            f"{self.experiment_label} - {cluster_fusion} - {ord_tf} order",
        )

        if out_path is None:
            out_path = (
                self.out_path
                / "figures"
                / f"0_7_TF_Enrichment_Scores_with_color_Proportions_{self._stem(cluster_fusion, ord_tf)}.svg"
            )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(out_path), format="svg")
        return fig


# ---------------------------------------------------------------------------
# Figures: SLIDE latent-factor enrichment bars
# ---------------------------------------------------------------------------

COLOR_MAP = {"red": "#c1272d", "blue": "#2b6cb0", "gray": "#b0b0b0"}


_COLOR_ORDER = ["red", "blue", "gray"]


_COLOR_LABELS = {
    "red": "high in case",  # corrs >= 0 with the LF (e.g. PRDM1/XBP1/MZB1 in Z11 = PB)
    "blue": "high in control",  # corrs < 0 (e.g. PAX5/AFF3 in Z11 = GC)
    "gray": "not an LF gene",
}


def plot_tf_enrichment_bars(
    plot_df: pd.DataFrame,
    title: str,
    out_path: str | Path | None = None,
    *,
    min_label_proportion: float = 0.08,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Stacked horizontal bar of enrichment score per TF, split by LF correlation sign.

    TFs run down the y axis, highest score at the top. Segments smaller than
    ``min_label_proportion`` of the bar are left unlabelled so the in-bar percentages
    stay readable. Saves vector output (SVG/PDF) when ``out_path`` is given.
    """
    wide = (
        plot_df.pivot(index="source", columns="color", values="height")
        .reindex(columns=_COLOR_ORDER, fill_value=0.0)
        .fillna(0.0)
        .iloc[::-1]  # barh draws the first row at the bottom
    )
    props = (
        plot_df.pivot(index="source", columns="color", values="proportion")
        .reindex(columns=_COLOR_ORDER, fill_value=0.0)
        .fillna(0.0)
        .reindex(wide.index)
    )
    tfs = list(wide.index)

    if figsize is None:
        figsize = (6.0, max(3.0, 0.22 * len(tfs) + 1.5))
    fig, ax = plt.subplots(figsize=figsize)

    left = pd.Series(0.0, index=wide.index)
    for color in _COLOR_ORDER:
        widths = wide[color]
        if widths.sum() == 0:
            continue
        ax.barh(
            tfs,
            widths,
            left=left,
            color=COLOR_MAP[color],
            label=_COLOR_LABELS[color],
            height=0.8,
        )
        for tf in tfs:
            if props.loc[tf, color] >= min_label_proportion:
                ax.text(
                    left[tf] + widths[tf] / 2,
                    tf,
                    f"{props.loc[tf, color] * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                )
        left = left + widths

    ax.set_ylabel("Transcription factor (TF)")
    ax.set_xlabel("Enrichment score")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlim(0, float(wide.sum(axis=1).max()) * 1.08)
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


def plot_strength_key_distribution(
    edges: pd.DataFrame,
    title: str = "Distribution of key and strength",
    out_path: str | Path | None = None,
    *,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Stacked bar of edge counts per (strength, key) — the state-specific GRN edge diagnostic.

    ``edges`` is a flat edge table with ``strength`` (0/1) and ``key`` (MultiDiGraph
    edge key) columns, as returned by ``grn_edges_from_combined_links``. Saves and
    closes the figure when ``out_path`` is given.
    """
    ax = (
        edges.groupby(["strength", "key"])
        .size()
        .unstack()
        .plot(kind="bar", stacked=True, figsize=figsize)
    )
    ax.set_xlabel("Strength")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig = ax.get_figure()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return fig, ax


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
