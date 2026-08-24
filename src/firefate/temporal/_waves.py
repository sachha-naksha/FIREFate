"""TF force waves: per-link regulatory force curves along a lineage."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from firefate.temporal._curves import SmoothedCurvesGRN
import matplotlib
import numpy as np
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TFForceWaves:
    """TF regulatory force waves over pseudotime.

    Wraps :class:`SmoothedCurvesGRN` for one trajectory branch and exposes:
    expression / regulation trajectory plots, force-wave computation for a set of
    links, the 3D single-link force landscape, and the clustered force heatmap.

    Phase classification of links lives in :class:`RegulatoryPhases`; the nested
    :class:`ForceSelector` picks per-link forces either lineage-specific or
    combined across lineages.
    """

    def __init__(
        self,
        dictys_dynamic_object,
        trajectory_range=(0, 2),
        num_points=100,
        dist=0.0005,
        sparsity=0.01,
    ):
        self.dictys_dynamic_object = dictys_dynamic_object
        self.trajectory_range = trajectory_range
        self.curves = SmoothedCurvesGRN(
            dictys_dynamic_object,
            trajectory_range=trajectory_range,
            num_points=num_points,
            dist=dist,
            sparsity=sparsity,
        )

        # Cached smoothed curves
        self._exp_curves = None      # (dy, dx)
        self._reg_curves = None      # (dy, dx)

        # Populated by compute_forces()
        self.beta_curves = None
        self.regulon_tf_expression = None
        self.force_curves = None
        self.dtime = None

    # ------------------------------------------------------------------
    # Expression / regulation trajectories
    # ------------------------------------------------------------------

    def expression_curves(self):
        """``(dy, dx)`` smoothed log-CPM expression curves (cached)."""
        if self._exp_curves is None:
            self._exp_curves = self.curves.get_smoothed_curves(mode="expression")
        return self._exp_curves

    def regulation_curves(self):
        """``(dy, dx)`` smoothed regulation (log target-count) curves (cached)."""
        if self._reg_curves is None:
            self._reg_curves = self.curves.get_smoothed_curves(mode="regulation")
        return self._reg_curves

    def _plot_gene_trajectories(self, dy, dx, genes, colors, ylabel, figsize):
        return plot_gene_trajectories(dy, dx, genes, colors, ylabel, figsize=figsize)

    def plot_expression(self, genes, colors, ylabel='Log (CPM)', figsize=(8, 6)):
        """Expression trajectories for ``genes`` (coloured, labelled at the end)."""
        dy, dx = self.expression_curves()
        return self._plot_gene_trajectories(dy, dx, genes, colors, ylabel, figsize)

    def plot_regulation(self, genes, colors, ylabel='Log (target counts)', figsize=(8, 6)):
        """Regulation trajectories for ``genes`` (coloured, labelled at the end)."""
        dy, dx = self.regulation_curves()
        return self._plot_gene_trajectories(dy, dx, genes, colors, ylabel, figsize)

    # ------------------------------------------------------------------
    # Force waves
    # ------------------------------------------------------------------

    def compute_forces(self, links, varname='w_in'):
        """Compute and cache beta / TF-expression / force curves for ``links``.

        Returns the force-curve DataFrame and stores ``beta_curves``,
        ``regulon_tf_expression``, ``force_curves`` and ``dtime`` on ``self``.
        """
        # beta-network curves and TF-expression curves are two independent dictys
        # `.compute` passes over the trajectory; overlap them (both release the GIL
        # in their NumPy/BLAS work).
        with ThreadPoolExecutor(max_workers=2) as ex:
            beta_future = ex.submit(self.curves.get_beta_curves, links, varname=varname)
            tf_future = ex.submit(self.curves.get_smoothed_curves, mode='tf_expression')
            beta_curves, dtime = beta_future.result()
            tf_expression, _ = tf_future.result()
        regulon_tf_expression = tf_expression.loc[
            beta_curves.index.get_level_values(0).unique()
        ]
        force_curves = SmoothedCurvesGRN.calculate_force_curves(
            beta_curves, regulon_tf_expression
        )

        self.beta_curves = beta_curves
        self.regulon_tf_expression = regulon_tf_expression
        self.force_curves = force_curves
        self.dtime = dtime
        return force_curves

    def _require_forces(self):
        if self.force_curves is None:
            raise RuntimeError("Call compute_forces(links) first.")

    def plot_landscape(self, tf_name, target_name, **kwargs):
        """3D force landscape of a single ``tf_name -> target_name`` link."""
        self._require_forces()
        return plot_single_link_landscape(
            tf_name, target_name,
            self.beta_curves, self.regulon_tf_expression,
            self.force_curves, self.dtime,
            **kwargs,
        )

    def plot_force_heatmap(self, links=None, perform_clustering=False, **kwargs):
        """Clustered (or plain) force heatmap for ``links`` (default: all)."""
        self._require_forces()
        if links is None:
            links = list(self.force_curves.index)
        df_plot = self.force_curves.loc[links]
        return plot_force_heatmap_with_clustering(
            force_df=df_plot,
            dtime=self.dtime,
            regulations=links,
            perform_clustering=perform_clustering,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Nested: pick lineage-specific or combined forces
    # ------------------------------------------------------------------

    class ForceSelector:
        """Pick per-link TF forces, lineage-specific or combined across lineages.

        A single TF-Target *force wave* exists on each lineage branch. Given one
        or more fitted :class:`TFForceWaves` branches, this returns, per link,
        either the force on one lineage (``mode='lineage'``) or the combined
        force across lineages (``mode='combined'`` -- the stronger lineage, i.e.
        ``max`` of ``max_t |force|`` taken over branches). Links already present
        in a branch's cached ``force_curves`` reuse them; others are scored on
        demand via the same GRN.
        """

        def __init__(self, waves_by_branch, varname='w_in'):
            """
            Parameters
            ----------
            waves_by_branch : TFForceWaves or dict {name: TFForceWaves}
                A single fitted branch, or a mapping of lineage name -> branch
                (e.g. ``{'PB': waves_pb, 'GC': waves_gc}``). The branches are
                assumed to share the same dictys universe.
            varname : str
                Network variable used for on-demand force curves. Default ``'w_in'``.
            """
            if isinstance(waves_by_branch, TFForceWaves):
                waves_by_branch = {'lineage': waves_by_branch}
            if not waves_by_branch:
                raise ValueError("ForceSelector needs at least one branch.")
            self.waves_by_branch = dict(waves_by_branch)
            self.varname = varname

        @property
        def branches(self):
            return list(self.waves_by_branch)

        def any_waves(self):
            """The first branch (its dictys universe is shared across branches)."""
            return next(iter(self.waves_by_branch.values()))

        def _branch_force_curves(self, waves, links):
            """``(force_curves, dtime)`` for ``links`` on one branch.

            Reuses ``waves.force_curves`` when it already holds all ``links``;
            otherwise scores them on demand via the GRN (without mutating
            ``waves``).
            """
            links = [tuple(l) for l in links]
            cached = waves.force_curves
            if cached is not None and all(l in cached.index for l in links):
                return cached.loc[links], waves.dtime
            beta_curves, dtime = waves.curves.get_beta_curves(links, varname=self.varname)
            tf_expression, _ = waves.curves.get_smoothed_curves(mode='tf_expression')
            regulon_tf_expression = tf_expression.loc[
                beta_curves.index.get_level_values(0).unique()
            ]
            force_curves = SmoothedCurvesGRN.calculate_force_curves(
                beta_curves, regulon_tf_expression
            )
            return force_curves, dtime

        @staticmethod
        def abs_max(force_curves, links=None):
            """``max_t |force(t)|`` per link as a dict {(TF, Target): float}."""
            idx = force_curves.index if links is None else links
            return {l: float(force_curves.loc[l].abs().max())
                    for l in idx if l in force_curves.index}

        def force_curves(self, links, branch=None):
            """``(force_curves, dtime)`` for ``links`` on one lineage (default: first)."""
            name = branch or self.branches[0]
            return self._branch_force_curves(self.waves_by_branch[name], links)

        def branch_abs_max_force(self, links):
            """Per-branch abs-max force: ``{branch: {link: force}}``."""
            return {name: self.abs_max(self._branch_force_curves(w, links)[0], links)
                    for name, w in self.waves_by_branch.items()}

        def abs_max_force(self, links, mode='lineage', branch=None):
            """Per-link abs-max force.

            ``mode='lineage'`` -> force on ``branch`` (default: the first branch).
            ``mode='combined'`` -> ``max`` across branches per link (the lineage
            where the link is strongest).
            """
            if mode == 'lineage':
                fc, _ = self.force_curves(links, branch=branch)
                return self.abs_max(fc, [tuple(l) for l in links])
            if mode == 'combined':
                return {l: d['abs_max_force']
                        for l, d in self.combined_abs_max_force(links).items()}
            raise ValueError("mode must be 'lineage' or 'combined'.")

        def combined_abs_max_force(self, links):
            """Per-link combined force together with the winning lineage.

            Returns ``{link: {'abs_max_force': float, 'branch': name}}`` where
            ``branch`` is the lineage on which the link's ``max_t |force|`` is
            largest (ties go to the first branch).
            """
            out = {}
            for name, am in self.branch_abs_max_force(links).items():
                for l, v in am.items():
                    if l not in out or v > out[l]['abs_max_force']:
                        out[l] = {'abs_max_force': v, 'branch': name}
            return out


# ---------------------------------------------------------------------------
# Figures: force landscapes and force heatmaps
# ---------------------------------------------------------------------------

EPSILON = 1e-10


def _force_fn(tf_expr, beta):
    """Compute force = sign(β) · exp(log₁₀(|β|+ε) + log₁₀(tf+ε))."""
    log_beta = np.log10(np.abs(beta) + EPSILON)
    log_tf   = np.log10(np.abs(tf_expr) + EPSILON)
    return np.sign(beta) * np.exp(log_beta + log_tf)


def _build_surface(
    tf_range: tuple,
    beta_range: tuple,
    sign: int = 1,
    resolution: int = 120,
):
    """
    Create a meshgrid surface of the force function for a given sign of β.

    Parameters
    ----------
    tf_range : (min, max) of TF expression values (log CPM)
    beta_range : (min, max) of |β| values
    sign : +1 for activating surface, -1 for repressing surface
    resolution : grid density
    """
    tf_vals   = np.linspace(tf_range[0], tf_range[1], resolution)
    beta_vals = np.linspace(beta_range[0], beta_range[1], resolution)
    TF, BETA  = np.meshgrid(tf_vals, beta_vals)
    FORCE     = _force_fn(TF, sign * BETA)   # sign determines activation / repression
    return TF, BETA * sign, FORCE


def _get_qualitative_colors(n: int) -> list:
    """Return n visually distinct colours (hex strings)."""
    palette = [
        "#6A3D9A", "#1F78B4", "#E31A1C", "#33A02C", "#FF7F00",
        "#FB9A99", "#B2DF8A", "#A6CEE3", "#FDBF6F", "#CAB2D6",
        "#B15928", "#FFFF99", "#8DD3C7", "#BEBADA", "#FB8072",
        "#80B1D3", "#FDB462", "#BC80BD", "#CCEBC5", "#D9D9D9",
    ]
    if n <= len(palette):
        return palette[:n]
    # cycle if more links than palette entries
    return [palette[i % len(palette)] for i in range(n)]


def plot_force_landscape(
    beta_curves: pd.DataFrame,
    regulon_tf_expression: pd.DataFrame,
    force_curves: pd.DataFrame,
    dtime: pd.Series,
    links_to_highlight: list = None,
    surface_opacity: float = 0.35,
    surface_resolution: int = 100,
    line_width: float = 5,
    marker_size: float = 3,
    colorscale_positive: str = "Purples",
    colorscale_negative: str = "Blues",
    title: str = "Regulatory Force Landscape",
    width: int = 1000,
    height: int = 750,
    show_surface: bool = True,
    camera: dict = None,
):
    """
    Build an interactive 3D Plotly figure of the force landscape.

    Parameters
    ----------
    beta_curves : pd.DataFrame
        Multi-indexed (TF, Target) × time_points. Edge strengths.
    regulon_tf_expression : pd.DataFrame
        TF × time_points. Log-CPM expression of each TF, already broadcast-
        ready (one row per unique TF present in beta_curves level 0).
    force_curves : pd.DataFrame
        Multi-indexed (TF, Target) × time_points. Pre-computed forces.
    dtime : pd.Series
        Pseudotime values for each time point.
    links_to_highlight : list of (TF, Target) tuples, optional
        Subset of links to draw. Default: all links.
    show_surface : bool
        Whether to render the analytical force surface behind the curves.
    """

    fig = go.Figure()

    # ── resolve links ────────────────────────────────────────────────────
    all_links = beta_curves.index.tolist()
    if links_to_highlight is None:
        links_to_highlight = all_links

    # ── data ranges for surface ──────────────────────────────────────────
    # Build the broadcast TF expression aligned to beta_curves index
    targets_per_tf = beta_curves.index.get_level_values(0).value_counts()
    expanded_tf = pd.DataFrame(
        np.repeat(regulon_tf_expression.values,
                  [targets_per_tf[tf] for tf in regulon_tf_expression.index], axis=0),
        index=beta_curves.index,
        columns=beta_curves.columns,
    )

    tf_vals_all   = expanded_tf.loc[links_to_highlight].values.ravel()
    beta_vals_all = beta_curves.loc[links_to_highlight].values.ravel()

    tf_min, tf_max     = np.nanmin(tf_vals_all), np.nanmax(tf_vals_all)
    beta_min, beta_max = np.nanmin(beta_vals_all), np.nanmax(beta_vals_all)
    tf_pad   = (tf_max - tf_min) * 0.1
    beta_pad = (beta_max - beta_min) * 0.1

    # ── analytical surface(s) ────────────────────────────────────────────
    if show_surface:
        # Positive-β surface (activation half)
        if beta_max > 0:
            TF_p, BETA_p, FORCE_p = _build_surface(
                tf_range=(tf_min - tf_pad, tf_max + tf_pad),
                beta_range=(1e-6, beta_max + beta_pad),
                sign=1,
                resolution=surface_resolution,
            )
            fig.add_trace(go.Surface(
                x=TF_p, y=BETA_p, z=FORCE_p,
                colorscale=colorscale_positive,
                opacity=surface_opacity,
                showscale=False,
                name="Activation surface",
                hoverinfo="skip",
            ))

        # Negative-β surface (repression half)
        if beta_min < 0:
            TF_n, BETA_n, FORCE_n = _build_surface(
                tf_range=(tf_min - tf_pad, tf_max + tf_pad),
                beta_range=(1e-6, np.abs(beta_min) + beta_pad),
                sign=-1,
                resolution=surface_resolution,
            )
            fig.add_trace(go.Surface(
                x=TF_n, y=BETA_n, z=FORCE_n,
                colorscale=colorscale_negative,
                opacity=surface_opacity,
                showscale=False,
                name="Repression surface",
                hoverinfo="skip",
            ))

    # ── link trajectories ────────────────────────────────────────────────
    # Build a qualitative colour palette
    n_links = len(links_to_highlight)
    cmap = _get_qualitative_colors(n_links)

    for i, (tf, target) in enumerate(links_to_highlight):
        if (tf, target) not in all_links:
            continue

        x = expanded_tf.loc[(tf, target)].values.astype(float)
        y = beta_curves.loc[(tf, target)].values.astype(float)
        z = force_curves.loc[(tf, target)].values.astype(float)
        t = dtime.values.astype(float)

        color = cmap[i % len(cmap)]

        # Trajectory line
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=color, width=line_width),
            name=f"{tf} → {target}",
            customdata=np.stack([t, x, y, z], axis=-1),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "pseudotime: %{customdata[0]:.3f}<br>"
                "TF expr (lcpm): %{customdata[1]:.3f}<br>"
                "β: %{customdata[2]:.5f}<br>"
                "force: %{customdata[3]:.5f}"
                "<extra></extra>"
            ),
        ))

        # Start marker (early pseudotime)
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode="markers",
            marker=dict(size=marker_size + 3, color=color, symbol="diamond"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── layout ───────────────────────────────────────────────────────────
    default_camera = dict(
        eye=dict(x=1.6, y=-1.6, z=0.9),
        up=dict(x=0, y=0, z=1),
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Helvetica Neue, Arial"),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text="TF Expression (log CPM)", font=dict(size=14)),
                backgroundcolor="rgba(240,240,245,0.5)",
                gridcolor="rgba(200,200,210,0.4)",
                showbackground=True,
            ),
            yaxis=dict(
                title=dict(text="β (edge strength)", font=dict(size=14)),
                backgroundcolor="rgba(240,240,245,0.5)",
                gridcolor="rgba(200,200,210,0.4)",
                showbackground=True,
            ),
            zaxis=dict(
                title=dict(text="Regulatory Force", font=dict(size=14)),
                backgroundcolor="rgba(240,240,245,0.5)",
                gridcolor="rgba(200,200,210,0.4)",
                showbackground=True,
            ),
            camera=camera or default_camera,
        ),
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            font=dict(size=11),
            itemsizing="constant",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(200,200,200,0.5)",
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def plot_single_link_landscape(
    tf_name: str,
    target_name: str,
    beta_curves: pd.DataFrame,
    regulon_tf_expression: pd.DataFrame,
    force_curves: pd.DataFrame,
    dtime: pd.Series,
    surface_resolution: int = 150,
    colorscale: str = "Viridis",
    width: int = 800,
    height: int = 650,
):
    """
    Focused 3D view of a single TF → Target link on its own force surface.
    The trajectory is coloured by pseudotime.
    """

    # ── extract link data ────────────────────────────────────────────────
    targets_per_tf = beta_curves.index.get_level_values(0).value_counts()
    expanded_tf = pd.DataFrame(
        np.repeat(regulon_tf_expression.values,
                  [targets_per_tf[tf] for tf in regulon_tf_expression.index], axis=0),
        index=beta_curves.index,
        columns=beta_curves.columns,
    )

    x = expanded_tf.loc[(tf_name, target_name)].values.astype(float)
    y = beta_curves.loc[(tf_name, target_name)].values.astype(float)
    z = force_curves.loc[(tf_name, target_name)].values.astype(float)
    t = dtime.values.astype(float)

    # ── surface ──────────────────────────────────────────────────────────
    tf_pad   = (x.max() - x.min()) * 0.25
    beta_pad = (np.abs(y).max()) * 0.25
    sign = 1 if np.mean(y) >= 0 else -1

    TF_s, BETA_s, FORCE_s = _build_surface(
        tf_range=(x.min() - tf_pad, x.max() + tf_pad),
        beta_range=(1e-6, np.abs(y).max() + beta_pad),
        sign=sign,
        resolution=surface_resolution,
    )

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=TF_s, y=BETA_s, z=FORCE_s,
        colorscale="Purples" if sign > 0 else "Blues",
        opacity=0.30,
        showscale=False,
        hoverinfo="skip",
    ))

    # ── trajectory coloured by pseudotime ────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines+markers",
        line=dict(color=t, colorscale=colorscale, width=6),
        marker=dict(size=2.5, color=t, colorscale=colorscale,
                    colorbar=dict(title="Pseudotime", thickness=15, len=0.5)),
        name=f"{tf_name} → {target_name}",
        customdata=np.stack([t, x, y, z], axis=-1),
        hovertemplate=(
            f"<b>{tf_name} → {target_name}</b><br>"
            "pseudotime: %{customdata[0]:.3f}<br>"
            "TF expr: %{customdata[1]:.3f}<br>"
            "β: %{customdata[2]:.5f}<br>"
            "force: %{customdata[3]:.5f}"
            "<extra></extra>"
        ),
    ))

    # Start / end markers
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode="markers",
        marker=dict(size=7, color="limegreen", symbol="diamond",
                    line=dict(color="black", width=1)),
        name="Start", showlegend=True,
    ))
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode="markers",
        marker=dict(size=7, color="red", symbol="x",
                    line=dict(color="black", width=1)),
        name="End", showlegend=True,
    ))

    fig.update_layout(
        title=dict(
            text=f"Force Landscape: {tf_name} → {target_name}",
            font=dict(size=18, family="Helvetica Neue, Arial"),
            x=0.5,
        ),
        scene=dict(
            xaxis_title="TF Expression (log CPM)",
            yaxis_title="β (edge strength)",
            zaxis_title="Regulatory Force",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        width=width, height=height,
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_force_by_tf(
    beta_curves: pd.DataFrame,
    regulon_tf_expression: pd.DataFrame,
    force_curves: pd.DataFrame,
    dtime: pd.Series,
    links: list = None,
    width: int = 1200,
    height: int = 900,
):
    """
    One 3D subplot per TF, showing all its target trajectories.
    Good for comparing how a single TF's different targets behave.
    """
    if links is None:
        links = beta_curves.index.tolist()

    # Group links by TF
    from collections import defaultdict
    tf_groups = defaultdict(list)
    for tf, tgt in links:
        tf_groups[tf].append((tf, tgt))

    tfs = sorted(tf_groups.keys())
    n_tfs = len(tfs)
    cols = min(3, n_tfs)
    rows = int(np.ceil(n_tfs / cols))

    specs = [[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)]
    subplot_titles = [tf for tf in tfs]

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
    )

    targets_per_tf = beta_curves.index.get_level_values(0).value_counts()
    expanded_tf = pd.DataFrame(
        np.repeat(regulon_tf_expression.values,
                  [targets_per_tf[tf] for tf in regulon_tf_expression.index], axis=0),
        index=beta_curves.index,
        columns=beta_curves.columns,
    )

    for idx, tf in enumerate(tfs):
        r = idx // cols + 1
        c = idx % cols + 1
        scene_name = f"scene{idx + 1}" if idx > 0 else "scene"
        targets = tf_groups[tf]
        cmap = _get_qualitative_colors(len(targets))

        for j, (tf_name, tgt_name) in enumerate(targets):
            x = expanded_tf.loc[(tf_name, tgt_name)].values.astype(float)
            y = beta_curves.loc[(tf_name, tgt_name)].values.astype(float)
            z = force_curves.loc[(tf_name, tgt_name)].values.astype(float)

            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode="lines",
                    line=dict(color=cmap[j], width=4),
                    name=f"{tf_name}→{tgt_name}",
                    legendgroup=tf,
                ),
                row=r, col=c,
            )

        fig.update_layout(**{
            scene_name: dict(
                xaxis_title="TF expr",
                yaxis_title="β",
                zaxis_title="Force",
                camera=dict(eye=dict(x=1.4, y=-1.4, z=0.8)),
            )
        })

    fig.update_layout(
        title="Force Landscapes by TF",
        width=width,
        height=height * rows / 2,
        paper_bgcolor="white",
    )
    return fig


def plot_gene_trajectories(dy, dx, genes, colors, ylabel, figsize=(8, 6)):
    """Per-gene curves over pseudotime, each labelled at its right end."""
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for gene, color in zip(genes, colors):
        if gene in dy.index:
            ax.plot(dx, dy.loc[gene], linewidth=2, color=color)
            ax.text(dx.iloc[-1], dy.loc[gene].iloc[-1], f' {gene}',
                    color=color, verticalalignment='center')
    ax.set_xlabel('Pseudotime')
    ax.set_ylabel(ylabel)
    return fig, ax


def plot_force_heatmap(
    force_df: pd.DataFrame,
    dtime: pd.Series,
    regulations=None,
    tf_to_targets_dict=None,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    figsize: Tuple[float, float] = (10, 4),
    vmax: Optional[float] = None,
) -> Tuple[matplotlib.pyplot.Figure, matplotlib.axes.Axes, np.ndarray]:
    """
    Draws pseudo-time dependent heatmap of force values.
    """
    # Process input parameters to generate regulation pairs
    reg_pairs = []
    reg_labels = []
    # Case 1: Dictionary of TF -> targets provided
    if tf_to_targets_dict is not None:
        for tf, targets in tf_to_targets_dict.items():
            for target in targets:
                reg_pairs.append((tf, target))
                reg_labels.append(f"{tf}->{target}")
    # Case 2: List of regulation pairs or list of targets for a single TF
    elif regulations is not None:
        # Check if first item is a string (target) or tuple/list (regulation pair)
        if regulations and isinstance(regulations[0], str):
            # It's a list of targets for a single TF
            # Extract TF name from the calling context (not ideal but works for the notebook)
            for key, value in locals().items():
                if (
                    isinstance(value, dict)
                    and "PRDM1" in value
                    and value["PRDM1"] == regulations
                ):
                    tf = "PRDM1"  # Found the TF
                    break
            else:
                # If we can't determine the TF, use the first item in regulations as TF
                # and the rest as targets (this is a fallback and might not be correct)
                tf = regulations[0]
                regulations = regulations[1:]

            for target in regulations:
                reg_pairs.append((tf, target))
                reg_labels.append(f"{tf}->{target}")
        else:
            # It's a list of regulation pairs
            reg_pairs = regulations
            reg_labels = [f"{tf}->{target}" for tf, target in regulations]
    # If no regulations provided, use non-zero regulations from force_df
    if not reg_pairs:
        non_zero_mask = (force_df != 0).any(axis=1)
        force_df_filtered = force_df[non_zero_mask]
        reg_pairs = list(force_df_filtered.index)
        reg_labels = [f"{tf}->{target}" for tf, target in reg_pairs]
    # Extract force values for the specified regulations
    force_values = []
    for pair in reg_pairs:
        tf, target = pair
        try:
            force_values.append(force_df.loc[(tf, target)].values)
        except KeyError:
            raise ValueError(f"Regulation {tf}->{target} not found in force DataFrame")
    # Convert to numpy array
    dnet = np.array(force_values)
    # Create figure and axes
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    # Determine and apply colormap
    if isinstance(cmap, str):
        if vmax is None:
            vmax = np.quantile(np.abs(dnet).ravel(), 0.95)
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax), cmap=cmap
        )
    elif vmax is not None:
        raise ValueError(
            "vmax should not be set if cmap is a matplotlib.cm.ScalarMappable."
        )
    if hasattr(cmap, "to_rgba"):
        im = ax.imshow(cmap.to_rgba(dnet), aspect="auto", interpolation="none")
    else:
        im = ax.imshow(dnet, aspect="auto", interpolation="none", cmap=cmap)
        plt.colorbar(im, label="Force")
    # Set pseudotime labels
    ax.set_xlabel("Pseudotime")
    num_ticks = 10
    tick_positions = np.linspace(0, dnet.shape[1] - 1, num_ticks, dtype=int)
    tick_labels = dtime.iloc[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{x:.4f}" for x in tick_labels], rotation=45, ha="right")
    # Set regulation pair labels
    ax.set_yticks(list(range(len(reg_labels))))
    ax.set_yticklabels(reg_labels)
    # Add grid lines
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    return fig, ax, dnet


def cluster_heatmap(
    d,
    optimal_ordering=True,
    method="ward",
    metric="euclidean",
    dshow=None,
    fig=None,
    cmap="coolwarm",
    aspect=0.1,
    figscale=0.02,
    dtop=0.3,
    dright=0,
    wcolorbar=0.03,
    wedge=0.03,
    xselect=None,
    yselect=None,
    xtick=False,
    ytick=True,
    vmin=None,
    vmax=None,
    inverty=True,
):
    """
    Draw a 2D hierarchically clustered heatmap from a DataFrame.

    The figure X/Y axes correspond to DataFrame columns/rows.

    Parameters
    ----------
    d : pandas.DataFrame
        2D data with index and column names used for clustering.
    optimal_ordering : bool
        Passed to ``scipy.cluster.hierarchy.dendrogram``.
    method : str or tuple of str
        Linkage method(s) for ``scipy.cluster.hierarchy.linkage``.
    metric : str or tuple of str
        Distance metric(s) for ``scipy.spatial.distance.pdist``.
    dshow : pandas.DataFrame, optional
        Data to render; defaults to ``d``.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on.
    cmap : str
        Colormap name.
    aspect, figscale, dtop, dright, wcolorbar, wedge : float
        Layout and colorbar geometry (fractions of the figure).
    xselect, yselect : array-like of bool, optional
        Mask of rows/columns to include in clustering and display.
    xtick, ytick : bool
        Whether to show axis ticks.
    vmin, vmax : float, optional
        Color scale limits.
    inverty : bool
        Whether to invert the y-axis.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure with dendrograms and heatmap.
    x, y : list
        Column and index labels included after clustering/selection.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram, linkage

    assert isinstance(xtick, bool) or (
        isinstance(xtick, list) and len(xtick) == d.shape[1]
    )
    assert isinstance(ytick, bool) or (
        isinstance(ytick, list) and len(ytick) == d.shape[0]
    )
    if isinstance(method, str):
        method = [method, method]
    if len(method) != 2:
        raise ValueError(
            'Parameter "method" must have size 2 for x and y respectively.'
        )
    if isinstance(metric, str):
        metric = [metric, metric]
    if metric is not None and len(metric) != 2:
        raise ValueError(
            'Parameter "metric" must have size 2 for x and y respectively.'
        )
    if metric is None:
        assert d.ndim == 2 and d.shape[0] == d.shape[1]
        assert (d.index == d.columns).all()
        assert method[0] == method[1]
        if xselect is not None:
            assert yselect is not None
            assert (xselect == yselect).all()
        else:
            assert yselect is None
    if dshow is None:
        dshow = d
    assert (
        dshow.shape == d.shape
        and (dshow.index == d.index).all()
        and (dshow.columns == d.columns).all()
    )
    xt0 = d.columns if isinstance(xtick, bool) else xtick
    yt0 = d.index if isinstance(ytick, bool) else ytick
    # Genes to highlight
    d2 = d.copy()
    if xselect is not None:
        d2 = d2.loc[:, xselect]
        dshow = dshow.loc[:, xselect]
        xt0 = [xt0[x] for x in np.nonzero(xselect)[0]]
    if yselect is not None:
        d2 = d2.loc[yselect]
        dshow = dshow.loc[yselect]
        yt0 = [yt0[x] for x in np.nonzero(yselect)[0]]

    wtop = dtop / (1 + d2.shape[0] / 8)
    wright = dright / (1 + d2.shape[1] * aspect / 8)
    iscolorbar = wcolorbar > 0
    t1 = np.array(d2.T.shape)
    t1 = t1 * figscale
    t1[1] /= aspect
    t1[1] /= 1 - wedge * 2 - wtop
    t1[0] /= 1 - wedge * (2 + iscolorbar) - wright - wcolorbar
    if fig is None:
        fig = plt.figure(figsize=t1)
    d3 = dshow.copy()
    if metric is not None:
        # Right dendrogram
        if dright > 0:
            ax1 = fig.add_axes(
                [
                    1 - wedge * (1 + iscolorbar) - wright - wcolorbar,
                    wedge,
                    wright,
                    1 - 2 * wedge - wtop,
                ]
            )
            tl1 = linkage(
                d2,
                method=method[1],
                metric=metric[1],
                optimal_ordering=optimal_ordering,
            )
            td1 = dendrogram(tl1, orientation="right")
            ax1.set_xticks([])
            ax1.set_yticks([])
            d3 = d3.iloc[td1["leaves"], :]
            yt0 = [yt0[x] for x in td1["leaves"]]
        else:
            ax1 = None
        # Top dendrogram
        if dtop > 0:
            ax2 = fig.add_axes(
                [
                    wedge,
                    1 - wedge - wtop,
                    1 - wedge * (2 + iscolorbar) - wright - wcolorbar,
                    wtop,
                ]
            )
            tl2 = linkage(
                d2.T,
                method=method[0],
                metric=metric[0],
                optimal_ordering=optimal_ordering,
            )
            td2 = dendrogram(tl2)
            ax2.set_xticks([])
            ax2.set_yticks([])
            d3 = d3.iloc[:, td2["leaves"]]
            xt0 = [xt0[x] for x in td2["leaves"]]
        else:
            ax2 = None
    else:
        if dright > 0 or dtop > 0:
            from scipy.spatial.distance import squareform

            tl1 = linkage(
                squareform(d2), method=method[0], optimal_ordering=optimal_ordering
            )
            # Right dendrogram
            if dright > 0:
                ax1 = fig.add_axes(
                    [
                        1 - wedge * (1 + iscolorbar) - wright - wcolorbar,
                        wedge,
                        wright,
                        1 - 2 * wedge - wtop,
                    ]
                )
                td1 = dendrogram(tl1, orientation="right")
                ax1.set_xticks([])
                ax1.set_yticks([])
            else:
                ax1 = None
                td1 = None
            # Top dendrogram
            if dtop > 0:
                ax2 = fig.add_axes(
                    [
                        wedge,
                        1 - wedge - wtop,
                        1 - wedge * (2 + iscolorbar) - wright - wcolorbar,
                        wtop,
                    ]
                )
                td2 = dendrogram(tl1)
                ax2.set_xticks([])
                ax2.set_yticks([])
            else:
                ax2 = None
                td2 = None
            td0 = td1["leaves"] if td1 is not None else td2["leaves"]
            d3 = d3.iloc[td0, :].iloc[:, td0]
            xt0, yt0 = [[y[x] for x in td0] for y in [xt0, yt0]]
    axmatrix = fig.add_axes(
        [
            wedge,
            wedge,
            1 - wedge * (2 + iscolorbar) - wright - wcolorbar,
            1 - 2 * wedge - wtop,
        ]
    )
    ka = {"aspect": 1 / aspect, "origin": "lower", "cmap": cmap}
    if vmin is not None:
        ka["vmin"] = vmin
    if vmax is not None:
        ka["vmax"] = vmax
    im = axmatrix.matshow(d3, **ka)
    if not isinstance(xtick, bool) or xtick:
        t1 = list(zip(range(d3.shape[1]), xt0))
        t1 = list(zip(*list(filter(lambda x: x[1] is not None, t1))))
        axmatrix.set_xticks(t1[0])
        axmatrix.set_xticklabels(t1[1], minor=False, rotation=90)
    else:
        axmatrix.set_xticks([])
    if not isinstance(ytick, bool) or ytick:
        t1 = list(zip(range(d3.shape[0]), yt0))
        t1 = list(zip(*list(filter(lambda x: x[1] is not None, t1))))
        axmatrix.set_yticks(t1[0])
        axmatrix.set_yticklabels(t1[1], minor=False)
    else:
        axmatrix.set_yticks([])
    axmatrix.tick_params(
        top=False,
        bottom=True,
        labeltop=False,
        labelbottom=True,
        left=True,
        labelleft=True,
        right=False,
        labelright=False,
    )
    if inverty:
        if ax1 is not None:
            ax1.set_ylim(ax1.get_ylim()[::-1])
        axmatrix.set_ylim(axmatrix.get_ylim()[::-1])
    if wcolorbar > 0:
        cax = fig.add_axes(
            [1 - wedge - wcolorbar, wedge, wcolorbar, 1 - 2 * wedge - wtop]
        )
        fig.colorbar(im, cax=cax)
    return fig, d3.columns, d3.index


def plot_force_heatmap_with_clustering(
    force_df: pd.DataFrame,
    dtime: pd.Series,
    regulations=None,
    tf_to_targets_dict=None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 8),
    plot_figure: bool = True,
    perform_clustering: bool = True,
    cluster_method: str = "ward",
    dtop: float = 0,
    dright: float = 0.3,
    row_scaling: dict = None,  # New parameter for scaling specific rows
) -> Tuple[pd.DataFrame, list, pd.Series, Optional[matplotlib.figure.Figure]]:
    """
    Prepares force value data for clustering heatmap and optionally plots it.
    
    Parameters:
    -----------
    row_scaling: Dict[Tuple[str, str], float]
        Dictionary mapping (TF, target) tuples to scaling factors.
        Example: {('IRF4', 'PRDM1'): 0.5} will scale that specific link to 50% of its original values.
    """
    # Process input parameters to generate regulation pairs
    reg_pairs = []
    reg_labels = []
    # Case 1: Dictionary of TF -> targets provided
    if tf_to_targets_dict is not None:
        for tf, targets in tf_to_targets_dict.items():
            for target in targets:
                reg_pairs.append((tf, target))
                reg_labels.append(f"{tf}->{target}")
    # Case 2: List of regulation pairs or list of targets for a single TF
    elif regulations is not None:
        # Check if first item is a string (target) or tuple/list (regulation pair)
        if regulations and isinstance(regulations[0], str):
            # It's a list of targets for a single TF
            # Extract TF name from the calling context (not ideal but works for the notebook)
            for key, value in locals().items():
                if (
                    isinstance(value, dict)
                    and "PRDM1" in value
                    and value["PRDM1"] == regulations
                ):
                    tf = "PRDM1"  # Found the TF
                    break
            else:
                # If we can't determine the TF, use the first item in regulations as TF
                # and the rest as targets (this is a fallback and might not be correct)
                tf = regulations[0]
                regulations = regulations[1:]

            for target in regulations:
                reg_pairs.append((tf, target))
                reg_labels.append(f"{tf}->{target}")
        else:
            # It's a list of regulation pairs
            reg_pairs = regulations
            reg_labels = [f"{tf}->{target}" for tf, target in regulations]
    # If no regulations provided, use non-zero regulations from force_df
    if not reg_pairs:
        non_zero_mask = (force_df != 0).any(axis=1)
        force_df_filtered = force_df[non_zero_mask]
        reg_pairs = list(force_df_filtered.index)
        reg_labels = [f"{tf}->{target}" for tf, target in reg_pairs]
    
    # Extract force values for the specified regulations
    force_values = []
    for pair in reg_pairs:
        tf, target = pair
        try:
            values = force_df.loc[(tf, target)].values
            
            # Apply scaling factor if provided for this pair
            if row_scaling and (tf, target) in row_scaling:
                scale_factor = row_scaling[(tf, target)]
                values = values * scale_factor
                
            force_values.append(values)
        except KeyError:
            raise ValueError(f"Regulation {tf}->{target} not found in force DataFrame")
    
    # Convert to numpy array
    dnet = np.array(force_values)
    
    # Convert dnet to DataFrame with proper labels
    force_df_for_cluster = pd.DataFrame(
        dnet, 
        index=reg_labels,
        columns=[f"{x:.4f}" for x in dtime]
    )
    
    # Plotting logic
    fig = None
    if plot_figure:
        # Calculate max absolute value for symmetric color scaling
        vmax_val = float(force_df_for_cluster.abs().max().max()) if vmax is None else vmax
        
        if perform_clustering:
            # Use cluster_heatmap for visualization
            fig, cols, rows = cluster_heatmap(
                d=force_df_for_cluster,
                optimal_ordering=True,
                method=cluster_method,
                metric="euclidean",
                cmap=cmap,
                aspect=0.1,
                figscale=0.02,
                dtop=dtop,      # Set to > 0 to enable clustering on columns (pseudotime)
                dright=dright,  # Set to > 0 to enable clustering on rows (regulations)
                wcolorbar=0.03,
                wedge=0.03,
                ytick=True,
                vmin=-vmax_val,
                vmax=vmax_val,
                figsize=figsize
            )
            plt.title("Clustered Force Heatmap")
        else:
            # Simple heatmap without clustering
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(dnet, aspect='auto', interpolation='none', cmap=cmap,
                          vmin=-vmax_val, vmax=vmax_val)
            
            # Add colorbar
            cbar = plt.colorbar(im, label="Force")
            
            # Set pseudotime labels as x axis labels
            ax.set_xlabel("Pseudotime")
            num_ticks = 10
            tick_positions = np.linspace(0, dnet.shape[1] - 1, num_ticks, dtype=int)
            tick_labels = dtime.iloc[tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{x:.4f}" for x in tick_labels], rotation=45, ha="right")
            
            # Set regulation pair labels
            ax.set_yticks(list(range(len(reg_labels))))
            ax.set_yticklabels(reg_labels)
            
            # Add grid lines
            ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
            
        plt.tight_layout()
    
    return force_df_for_cluster, reg_labels, dtime, fig
