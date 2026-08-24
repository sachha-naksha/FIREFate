# FIREFate Architecture

> **Sections 6 and 10 are superseded** by `references/module_structure_plan.md`, which
> restructures the package along the three FIREFate modules (Temporal, StateSpecific,
> CrossPrediction) instead of the capability-per-subpackage split described below.
> Sections 1-5, 7 and 8 (design principles, src-layout, composition over inheritance,
> managers, types, logging) still hold and the new layout follows them.

Design principles and code structure for FIREFate. Read before any refactor, module addition, or
"how should I organize this" question.

## Table of Contents
1. [Design principles](#1-design-principles)
2. [src/-layout vs flat-layout](#2-src-layout-vs-flat-layout)
3. [Composition over inheritance](#3-composition-over-inheritance)
4. [Class hierarchy (actual)](#4-class-hierarchy-actual)
5. [Manager classes](#5-manager-classes)
6. [Module organization](#6-module-organization)
7. [Type system](#7-type-system)
8. [Logging and errors](#8-logging-and-errors)
9. [Current state of the repo](#9-current-state-of-the-repo)
10. [Migration plan](#10-migration-plan)

---

## 1. Design principles

FIREFate follows **library mode** from the `coding-style` skill (moscot-style), not research mode.
This is because FIREFate is a public-facing package that others will import, not a one-off
experiment.

The three commitments this entails:

- **Public surface is the seven capabilities.** Everything else is `_private`. A user should be
  able to `import firefate as ff` and call `ff.discover_cps(...)`, `ff.build_episodic_grn(...)`,
  `ff.ora(...)`, etc. without ever touching an underscore-prefixed symbol.
- **Types on every public signature.** `def f(adata: AnnData, key: str) -> pd.DataFrame:` — never
  untyped public functions. Internals can be untyped if they're obvious.
- **No magic absolute paths, no HPC-specific defaults in the package.** Cluster paths live in
  tutorials and SLURM scripts, never in `src/firefate/`.

## 2. src/-layout vs flat-layout

Use **src/-layout**:
```
FIREFate/
├── src/firefate/
│   └── __init__.py
├── tests/
└── pyproject.toml
```

Not the flat layout (`firefate/` directly under repo root alongside `pyproject.toml`). Reasons:

- **Forces correct installation.** With `src/`-layout, you cannot accidentally import from the repo
  root — only from the installed package. This catches "works on my laptop, fails when pip
  installed" bugs early.
- **Cleaner test isolation.** `tests/` doesn't accidentally live inside the importable package.
- **Standard for moscot, scanpy, scvi-tools** — users expect this layout for scientific packages.

This also means during development you `pip install -e .` from the repo root, which adds `src/`
to the path.

## 3. Composition over inheritance

**Established decision** (carried over from `pseudotime_curves.py` / `episodic_dynamics.py`):
`EpisodeDynamics` composes `SmoothedCurvesGRN` and `AlignTimeScales` rather than inheriting from them.

```python
# GOOD - composition
class EpisodeDynamics:
    def __init__(self, dictys_dynamic_object, ...):
        self.curves = SmoothedCurvesGRN(dictys_dynamic_object=dictys_dynamic_object, ...)
        self.time_aligner = AlignTimeScales(dictys_dynamic_object=dictys_dynamic_object, ...)

    def compute_expression_curves(self, mode="expression"):
        self.curves.mode = mode
        return self.curves.get_smoothed_curves()

# BAD - multiple inheritance
class EpisodeDynamics(SmoothedCurvesGRN, AlignTimeScales):  # MRO hell, unclear ownership
    ...
```

The **same composition pattern** extends to the new Manager classes — `EnrichmentManager` composes
`EpisodeDynamics` instances and ORA primitives; `GRNManager` composes curve builders and edge
filters. Neither inherits from domain classes.

## 4. Class hierarchy (actual)

Based on the current implemented codebase:

```
firefate.core
├── SmoothedCurvesGRN                # β(t), E_TF(t), LCPM(t) over pseudotime
│   ├── get_smoothed_curves()        # modes: expression / regulation / tf_expression / weighted
│   ├── get_beta_curves()            # β-curves for specified TF→target links
│   ├── calculate_force_curves()     # F = sign(β)·exp(log|β| + log(E_TF))
│   ├── curve_characteristics()      # transient_logfc, switching_time, terminal_logfc, auc
│   ├── classify_tf_global_activity()  # 4-class: Cumulative/Reductive/Bell/U-shaped
│   └── get_top_k_tfs_by_class()
├── SmoothedCurvesChromatin          # TF binding score / OCR count dynamics
│   ├── extract_data()               # multiprocess binding.tsv.gz extraction
│   ├── set_trajectory_info()        # PB/GC indices + pseudotime
│   ├── process_dynamics()           # ordered, smoothed trajectories
│   ├── plot()                       # Plotly comparative (PB solid, GC dashed)
│   └── plot_score_vs_count_comparison()
├── AlignTimeScales                  # window ↔ sampled points ↔ episode mapping
│   ├── pseudotime_of_windows()
│   └── pseudotime_of_sampled_points()
└── EpisodeDynamics                  # composes SmoothedCurvesGRN + AlignTimeScales
    ├── compute_expression_curves()
    ├── build_episode_grn()          # time_slice → filtered edge DataFrame
    ├── filter_edges()               # significance + direction invariance (parallel)
    ├── compute_tf_expression()
    ├── calculate_forces()           # parallel force calculation
    ├── select_top_edges()           # percentile-based (unsigned)
    ├── select_top_activating_and_repressing_edges()  # signed tails
    ├── set_lf_genes()
    ├── annotate_lf_in_grn()
    └── calculate_enrichment()       # hypergeometric ORA

firefate.managers (NEW — unification layer)
├── GRNManager                       # unified GRN construction across resolutions
│   ├── build_transition_window()
│   ├── build_episodic()
│   └── build_all_episodes()
└── EnrichmentManager                # unified ORA across GRN types
    ├── enrich_episodic()
    ├── enrich_state_specific()
    ├── enrich_all_episodes()
    └── batch_from_config()

firefate.enrichment
└── ora()                            # primitive: hypergeometric test (M, n, N, X)

firefate.utils
├── gene_utils                       # get_tf_indices, get_gene_indices, check_if_gene_in_ndict
└── plots                            # regulation heatmaps, force heatmaps, dotplots, clustering
```

Naming rule: **GRN classes are nouns for the output**, not the algorithm. `EpisodicGRN` is the
object users manipulate; the construction logic lives in `EpisodeDynamics.build_episode_grn(...)`.
This mirrors how scanpy exposes `AnnData` as the noun and keeps algorithms as functions/methods.

## 5. Manager classes

Managers are the **unification layer** — they provide a single API surface for tasks that share the
same primitive logic but differ in their data source. The ORA primitive is identical whether the
input is a state-specific GRN edge table, an episodic GRN, or a cross-state GRN. Managers hide
the plumbing.

### 5.1 EnrichmentManager

**Problem it solves**: Enrichment of TF activity in cellular programs appears in three settings:
1. **Episodic enrichment** — per-episode, per-CP, using episodic GRN edges (capability 5 on cap 4)
2. **State-specific enrichment** — using state-specific GRN edges (capability 5 on cap 2)
3. **Batch enrichment** — across all episodes, loading from config paths (paper workflow)

Without a manager, you'd write three separate loops with nearly identical ORA + annotation logic.
The `EnrichmentManager` centralizes this.

**Design**:

```python
class EnrichmentManager:
    """Unified API for TF-activity enrichment across GRN types.

    Composes EpisodeDynamics and ORA primitives. Does NOT inherit from either.

    Parameters
    ----------
    lf_genes : list[str]
        Cellular program gene set (from SLIDE latent factor).
    dictys_dynamic_object : dynamic_network, optional
        Required for episodic enrichment. Not needed for pre-computed edge tables.
    trajectory_range : tuple[float, float]
        Trajectory endpoints for episodic analysis.
    num_points : int
        Number of sampled pseudotime points.
    dist : float
        Smoothing bandwidth.
    sparsity : float
        Network sparsity threshold.
    """

    def __init__(
        self,
        lf_genes: list[str],
        dictys_dynamic_object=None,
        trajectory_range: tuple[float, float] = (1, 3),
        num_points: int = 40,
        dist: float = 0.001,
        sparsity: float = 0.01,
    ):
        self.lf_genes = lf_genes
        self._dictys_obj = dictys_dynamic_object
        self._traj_range = trajectory_range
        self._num_points = num_points
        self._dist = dist
        self._sparsity = sparsity

    # --- Episodic enrichment (capability 5 on episodic GRN) ---

    def enrich_episodic(
        self,
        episode_idx: int,
        time_slice: slice,
        percentile: float = 98,
        pval_threshold: float = 0.001,
        n_processes: int = 16,
    ) -> pd.DataFrame:
        """Run full episodic enrichment pipeline for one episode.

        Delegates to EpisodeDynamics internally.

        Returns
        -------
        pd.DataFrame
            Columns: TF, p_value, enrichment_score, genes_in_lf, genes_dwnstrm, weights
        """
        if self._dictys_obj is None:
            raise ValueError("dictys_dynamic_object required for episodic enrichment.")

        epi = EpisodeDynamics(
            dictys_dynamic_object=self._dictys_obj,
            output_folder="",  # not saving intermediate
            trajectory_range=self._traj_range,
            num_points=self._num_points,
            dist=self._dist,
            sparsity=self._sparsity,
        )
        epi.compute_expression_curves()
        epi.set_lf_genes(self.lf_genes)
        epi.build_episode_grn(time_slice=time_slice)
        epi.filter_edges(pval_threshold=pval_threshold, n_processes=n_processes)
        epi.compute_tf_expression()
        epi.calculate_forces()
        epi.select_top_edges(percentile)
        epi.annotate_lf_in_grn()
        return epi.calculate_enrichment()

    # --- State-specific enrichment (capability 5 on state GRN) ---

    def enrich_state_specific(
        self,
        grn_edges: pd.DataFrame,
        force_col: str = "avg_force",
    ) -> pd.DataFrame:
        """Enrich TF activity against a pre-computed state-specific GRN edge table.

        Parameters
        ----------
        grn_edges : pd.DataFrame
            MultiIndex (TF, Target), must have ``force_col`` column.

        Returns
        -------
        pd.DataFrame
            Same schema as enrich_episodic output.
        """
        # Annotate which targets are LF genes
        grn_edges = grn_edges.copy()
        grn_edges["is_in_lf"] = grn_edges.index.get_level_values(1).isin(self.lf_genes)

        lf_active = grn_edges[grn_edges["is_in_lf"]].index.get_level_values(1).unique()
        all_targets = grn_edges.index.get_level_values(1).unique()

        return calculate_tf_episodic_enrichment(
            grn_edges,
            total_lf_genes=len(lf_active),
            total_genes_in_grn=len(all_targets),
        )

    # --- Batch across episodes ---

    def enrich_all_episodes(
        self,
        total_episodes: int = 8,
        points_per_episode: int = 5,
        percentile: float = 98,
        output_folder: str = None,
        n_processes: int = 16,
    ) -> dict[int, pd.DataFrame]:
        """Run enrichment for all episodes sequentially.

        Parameters
        ----------
        total_episodes : int
            Number of episodes to partition pseudotime into.
        points_per_episode : int
            Number of sampled points per episode window.
        output_folder : str, optional
            If provided, saves each episode result as CSV.

        Returns
        -------
        dict[int, pd.DataFrame]
            Mapping episode_idx → enrichment DataFrame.
        """
        results = {}
        for ep_idx in range(1, total_episodes + 1):
            start = (ep_idx - 1) * points_per_episode
            end = ep_idx * points_per_episode
            time_sl = slice(start, end)

            enrichment_df = self.enrich_episodic(
                episode_idx=ep_idx,
                time_slice=time_sl,
                percentile=percentile,
                n_processes=n_processes,
            )
            results[ep_idx] = enrichment_df

            if output_folder:
                import os
                os.makedirs(output_folder, exist_ok=True)
                enrichment_df.to_csv(
                    os.path.join(output_folder, f"enrichment_episode_{ep_idx}.csv"),
                    index=False,
                )

        return results

    # --- Load from config paths ---

    @staticmethod
    def batch_from_config(
        config_paths: dict[str, str],
        p_value_threshold: float = 0.05,
    ) -> list[pd.DataFrame]:
        """Load pre-computed enrichment CSVs from a Config-style path dict.

        Parameters
        ----------
        config_paths : dict
            e.g. ``config.BLIMP1`` — keys like 'ep1', values are file paths.
        p_value_threshold : float
            Filter loaded DataFrames to this significance.

        Returns
        -------
        list[pd.DataFrame]
            One DataFrame per episode, ordered by key.
        """
        dfs = []
        for key in sorted(config_paths.keys()):
            path = config_paths[key]
            try:
                df = pd.read_csv(path)
                dfs.append(df[df["p_value"] < p_value_threshold] if "p_value" in df.columns else df)
            except FileNotFoundError:
                dfs.append(pd.DataFrame())
        return dfs
```

### 5.2 GRNManager

**Problem it solves**: Building GRNs at different resolutions (transition-window, episodic, all
episodes) shares the same Dictys smoothing + force computation core. `GRNManager` provides a
single construction entry point.

**Design** (pseudocode aligned with `SmoothedCurvesGRN.get_beta_curves` and
`SmoothedCurvesGRN.calculate_force_curves(beta_curves, tf_expression)`):

```python
class GRNManager:
    """Unified API for GRN construction at different temporal resolutions.

    Parameters
    ----------
    dictys_dynamic_object : dynamic_network
        Loaded Dictys dynamic network.
    trajectory_range : tuple[float, float]
        Trajectory endpoints.
    num_points : int
        Number of sampled pseudotime points.
    dist : float
        Smoothing bandwidth.
    sparsity : float
        Network sparsity threshold.
    """

    def __init__(
        self,
        dictys_dynamic_object,
        trajectory_range: tuple[float, float] = (1, 3),
        num_points: int = 40,
        dist: float = 0.001,
        sparsity: float = 0.01,
    ):
        self._dictys_obj = dictys_dynamic_object
        self._traj_range = trajectory_range
        self._num_points = num_points
        self._dist = dist
        self._sparsity = sparsity

    def build_transition_window(
        self,
        links: list[tuple[str, str]] = None,
        mode: str = "expression",
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.Series]:
        """Build transition-window GRN (capability 3).

        Returns
        -------
        beta_curves : pd.DataFrame
        force_curves : pd.DataFrame or None
            None when ``links`` is omitted (no TF–target β slice to pair with TF expression).
        dtime : pd.Series
        """
        curves = SmoothedCurvesGRN(
            dictys_dynamic_object=self._dictys_obj,
            trajectory_range=self._traj_range,
            num_points=self._num_points,
            dist=self._dist,
            sparsity=self._sparsity,
            mode=mode,
        )
        if links is None:
            beta_curves, dtime = curves.get_smoothed_curves(mode="regulation")
            return beta_curves, None, dtime

        beta_curves, dtime = curves.get_beta_curves(links)
        tf_expression_df, _ = curves.get_smoothed_curves(mode="tf_expression")
        tfs = beta_curves.index.get_level_values(0).unique()
        # One scalar TF activity per TF for force composition (e.g. terminal pseudotime column).
        tf_expression = tf_expression_df.loc[tfs].iloc[:, -1]
        force_curves = SmoothedCurvesGRN.calculate_force_curves(beta_curves, tf_expression)
        return beta_curves, force_curves, dtime

    def build_episodic(
        self,
        time_slice: slice,
        percentile: float = 98,
        pval_threshold: float = 0.001,
        n_processes: int = 16,
        output_folder: str = "",
    ) -> pd.DataFrame:
        """Build episodic GRN (capability 4).

        Returns
        -------
        pd.DataFrame
            Episodic GRN edges with avg_force column.
        """
        epi = EpisodeDynamics(
            dictys_dynamic_object=self._dictys_obj,
            output_folder=output_folder,
            trajectory_range=self._traj_range,
            num_points=self._num_points,
            dist=self._dist,
            sparsity=self._sparsity,
        )
        epi.compute_expression_curves()
        epi.build_episode_grn(time_slice=time_slice)
        epi.filter_edges(pval_threshold=pval_threshold, n_processes=n_processes)
        epi.compute_tf_expression()
        epi.calculate_forces()
        return epi.select_top_edges(percentile)

    def build_all_episodes(
        self,
        total_episodes: int = 8,
        points_per_episode: int = 5,
        percentile: float = 98,
        output_folder: str = None,
        n_processes: int = 16,
    ) -> dict[int, pd.DataFrame]:
        """Build episodic GRNs for all episodes.

        Returns
        -------
        dict[int, pd.DataFrame]
            Mapping episode_idx → episodic GRN edge DataFrame.
        """
        results = {}
        for ep_idx in range(1, total_episodes + 1):
            start = (ep_idx - 1) * points_per_episode
            end = ep_idx * points_per_episode
            grn = self.build_episodic(
                time_slice=slice(start, end),
                percentile=percentile,
                n_processes=n_processes,
                output_folder=output_folder or "",
            )
            results[ep_idx] = grn

            if output_folder:
                import os
                os.makedirs(output_folder, exist_ok=True)
                grn.to_parquet(os.path.join(output_folder, f"episode_{ep_idx}.parquet"))

        return results
```

## 6. Module organization

Each subpackage under `firefate/` corresponds to one or more capabilities:

| Subpackage | Capabilities | Main entry points |
|---|---|---|
| `core/` | — (building blocks) | `SmoothedCurvesGRN`, `SmoothedCurvesChromatin`, `AlignTimeScales`, `EpisodeDynamics` |
| `grn/` | 2, 3, 4 | `build_state_specific`, `build_transition_window`, `build_episodic` |
| `programs/` | 1 | `discover_cps`, `annotate_cp_with_gsea` |
| `enrichment/` | 5 | `ora`, `enrichment_on_grn`, `get_SLIDE_GRN_enrichment` |
| `managers/` | 4+5 (unified) | `GRNManager`, `EnrichmentManager` |
| `perturbation/` | 6 | `in_silico_ko`, `phenotypic_shift` |
| `fate_bias/` | 7 | `stratify_uncommitted`, `transfer_cps_to_query` |
| `utils/` | — | plotting, I/O, gene helpers |

**Rule**: if a function is needed by multiple subpackages, it goes in `utils/`, not in whichever
subpackage first needed it. This prevents circular imports.

## 7. Type system

Create `src/firefate/_types.py` (moscot pattern):

```python
"""Type aliases used across firefate."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData
    from dictys.net import dynamic_network

# Paths
PathLike_t = Union[str, Path]

# Arrays
ArrayLike_t = Union[np.ndarray, pd.DataFrame]

# Data objects
AnnData_t = "AnnData"
DictysNet_t = "dynamic_network"

# Enumerated modes
ForceMode_t = Literal["signed", "abs"]
CurveMode_t = Literal["expression", "regulation", "tf_expression", "weighted_regulation"]
TrajectoryRange_t = Tuple[float, float]

# GRN edge format
EdgeTuple_t = Tuple[str, str]   # (TF, target)
EdgeTable_t = pd.DataFrame       # MultiIndex (TF, Target), columns include avg_force

# Enrichment result
EnrichmentResult_t = pd.DataFrame  # columns: TF, p_value, enrichment_score, genes_in_lf, ...

# Episode definition
EpisodeSlice_t = slice
```

## 8. Logging and errors

Create `src/firefate/_logging.py`:

```python
"""Rich logger for firefate.

Users can silence with logging.getLogger("firefate").setLevel(logging.WARNING).
"""
import logging

from rich.logging import RichHandler

logger = logging.getLogger("firefate")
if not logger.handlers:
    handler = RichHandler(rich_tracebacks=True, show_time=False, show_path=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
```

**Error messages** follow the moscot pattern — describe what was expected and what was found,
never just "invalid input":

```python
# GOOD
if not isinstance(adata, AnnData):
    raise TypeError(
        f"Expected `adata` to be AnnData, found {type(adata).__name__}."
    )
if "pseudotime" not in adata.obs:
    raise KeyError(
        f"Column 'pseudotime' not found in adata.obs. "
        f"Available columns: {list(adata.obs.columns)}"
    )

# BAD
assert isinstance(adata, AnnData)
assert "pseudotime" in adata.obs
```

## 9. Current state of the repo

As of the latest snapshot, the analysis code lives under
`multiome_dynamic_regulation/py_scripts/analysis/` with five modules:

```
multiome_dynamic_regulation/py_scripts/analysis/
├── config.py                  # Config class with HPC paths (stays in tutorials)
├── utils_custom.py            # Gene helpers + plotting functions
├── pseudotime_curves.py       # SmoothedCurvesGRN, SmoothedCurvesChromatin
├── episodic_dynamics.py       # EpisodeDynamics, AlignTimeScales, parallel force/filter
├── episode_plots.py           # Dotplot + heatmap visualizations for enrichment
├── LF_global_dynamics.ipynb   # Global regulation analysis notebook
└── LF_local_dynamics.ipynb    # Episodic analysis notebook
```

Legacy SLIDE / ESCAPE / TF-dynamic scaffolding may live under a `zarifeh_code/` tree in some
checkouts (not present in every snapshot). When it exists, it typically mirrors:

```
zarifeh_code/
├── ESCAPE/
├── SLIDE/
├── TF_Dynamic_Activity/
└── README.md
```

## 10. Migration plan

| Step | Source | Target | Notes |
|---|---|---|---|
| 1 | `pseudotime_curves.py` | `src/firefate/core/pseudotime_curves.py` | Both `SmoothedCurvesGRN` + `SmoothedCurvesChromatin` |
| 2 | `episodic_dynamics.py` → `AlignTimeScales` | `src/firefate/core/align_time.py` | Own module for reuse |
| 3 | `episodic_dynamics.py` → `EpisodeDynamics` | `src/firefate/core/episodic_dynamics.py` | Imports from core siblings |
| 4 | `episodic_dynamics.py` → parallel functions | `src/firefate/core/force.py` | `calculate_force_curves_parallel`, `filter_edges_by_significance_and_direction` |
| 5 | `episodic_dynamics.py` → `calculate_tf_episodic_enrichment` | `src/firefate/enrichment/ora.py` | The ORA primitive |
| 6 | `episodic_dynamics.py` → `run_episodic_*` orchestrators | `src/firefate/managers/grn_manager.py` + `enrichment_manager.py` | Unified managers |
| 7 | `utils_custom.py` → gene helpers | `src/firefate/utils/gene_utils.py` | `get_tf_indices`, `get_gene_indices`, `check_if_gene_in_ndict` |
| 8 | `utils_custom.py` + `episode_plots.py` → plots | `src/firefate/utils/plots.py` | Merge all plotting code |
| 9 | `config.py` | `tutorials/config.py` | HPC paths = tutorial-only, NOT in package |
| 10 | `zarifeh_code/ESCAPE/*` | `src/firefate/enrichment/slide_grn.py` | SLIDE-GRN enrichment |
| 11 | `zarifeh_code/SLIDE/*` | `src/firefate/programs/slide_interface.py` | Thin wrapper importing `loveslide` |
| 12 | Create | `src/firefate/__init__.py` | Seven public entry points + manager classes |
| 13 | Create | `pyproject.toml` | See `references/pip-packaging.md` |
| 14 | Create | `docs/` Sphinx skeleton | See `references/documentation.md` |
| 15 | Archive | `zarifeh_code/` → branch `legacy-zarifeh` | Preserve lineage |

Use `git mv` (not `rm` + `add`) so `git log --follow` traces functions to their original home.
