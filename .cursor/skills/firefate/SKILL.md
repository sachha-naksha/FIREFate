---
name: firefate
description: >
  FIREFate codebase, documentation, and pip-packaging assistant. FIREFate (Functional and
  Interpretable Regulatory Encoding of Fate determination) is Akanksha's flagship framework
  using interpretable ML to focus mechanistic GRNs onto components governing cell fate
  decisions. Trigger whenever the user mentions FIREFate, FIREFate-NP, cellular programs (CPs),
  episodic GRN, SLIDE-GRN enrichment, TF force curves, fate-bias stratification, in-silico TF
  KO on CPs, EpisodeDynamics, SmoothedCurvesGRN, SmoothedCurvesChromatin, EnrichmentManager,
  GRNManager, the github.com/sachha-naksha/FIREFate repo, or any FIREFate API/README/
  pyproject/docs/read-the-docs work. Also fire for packaging FIREFate for pip/PyPI, drafting
  FIREFate documentation or tutorials, structuring/refactoring FIREFate modules, implementing
  force-curve computation, building the seven-capability API, or writing paper text about
  FIREFate methods. Casual triggers like "revamp FIREFate," "add a module to firefate,"
  "what's the API for capability 4," or "package firefate for pypi" should fire this.
---

# FIREFate Codebase & Documentation

FIREFate is Akanksha's first-author framework. This skill is the central reference for the project:
what it is, how it's structured, how to extend it, how to package it, and how to document it. Think
of it as "project manager + senior engineer + technical writer" for FIREFate specifically.

---

## What FIREFate is, in one sentence

**FIREFate is a framework that uses interpretable ML to focus dense mechanistic GRNs onto the
components governing cell fate decisions.**

The "focusing" verb is load-bearing — this is the Overrepresentation Analysis (ORA) operator that
takes a universe of genes, a GRN-downstream set, and a SLIDE-derived cellular program, and returns
a log₂ fold-enrichment score + hypergeometric p-value. The magnifying glass is the operator
symbol; `ORA_Ω(A, B)` is the textual form. See `references/math-operators.md` for the formal
definition.

---

## The seven capabilities (the public API surface)

FIREFate exposes seven capabilities. These are the scientific contract — the README enumerates
them, the docs organize around them, and the package structure maps onto them.

1. **Discover cellular programs (CPs)** underlying contrasting cell states using interpretable ML,
   characterize their phenotypic roles via LLM-augmented GSEA.
2. **Construct state-specific GRNs** from sc-multiomics or scRNA-seq, plus a combined cross-state
   GRN whose connectivity spans both states (matching CP scope from capability 1).
3. **Construct transition-window-specific GRNs** from sc-RNA + sc-ATAC (matched or unmatched),
   cluster regulatory edges to temporally order waves of TF regulation.
4. **Construct episodic GRNs** by retaining TF–target forces that remain temporally invariant
   across an episode AND fall in top-percentile tails of activating/repressive edges.
5. **Quantify enrichment of dynamic TF activity** within each CP using state-specific or episodic
   GRN connectivity — this is the ORA / magnifying glass operator applied.
6. **Quantify in-silico perturbation effects** from the phenotypic shift induced by perturbing
   enriched TFs within each CP.
7. **Stratify uncommitted populations for fate-bias** using CPs inferred from intervention
   (gene-KO) vs unperturbed control populations.

**FIREFate-NP (Network Prioritization)** is the downstream task combining these: static + dynamic
GRNs + interpretable ML → sparse minimal gene sets (CPs) that distinguish cell fates → embedded in
high-resolution GRNs to reveal TF-centric regulons.

For each capability's inputs/outputs, formulas, and implementation notes, see
`references/capabilities.md`.

---

## Current implementation state (as of codebase snapshot)

The analysis code lives under `multiome_dynamic_regulation/py_scripts/analysis/` with five
production modules and two analysis notebooks:

### Implemented modules

| Module | Key classes / functions | Maps to capability |
|---|---|---|
| `pseudotime_curves.py` | `SmoothedCurvesGRN`, `SmoothedCurvesChromatin` | 3 (transition-window GRNs) |
| `episodic_dynamics.py` | `EpisodeDynamics`, `AlignTimeScales`, `calculate_tf_episodic_enrichment`, `filter_edges_by_significance_and_direction`, `calculate_force_curves_parallel` | 4, 5 (episodic GRNs + enrichment) |
| `utils_custom.py` | `get_tf_indices`, `fig_regulation_heatmap`, `plot_force_heatmap_with_clustering`, `cluster_heatmap` | Shared utilities |
| `episode_plots.py` | `plot_tf_episodic_enrichment_dotplot`, `plot_tf_target_episodic_heatmap` | Visualization for 5 |
| `config.py` | `Config` | Path management |

### Implemented class hierarchy (actual)

```
pseudotime_curves.py
├── SmoothedCurvesGRN             # β(t), E_TF(t), LCPM(t), force curves, curve characteristics
│   ├── get_smoothed_curves()     # expression / regulation / tf_expression / weighted modes
│   ├── get_beta_curves()         # β-curves for specified TF→target links
│   ├── calculate_force_curves()  # F = sign(β)·exp(log|β|+log(E_TF))
│   ├── curve_characteristics()   # transient_logfc, switching_time, terminal_logfc, auc
│   ├── classify_tf_global_activity()  # 4-class: Cumulative/Reductive/Bell/U-shaped
│   └── get_top_k_tfs_by_class()
└── SmoothedCurvesChromatin       # TF binding score / OCR count dynamics
    ├── extract_data()            # multiprocess binding.tsv.gz extraction
    ├── set_trajectory_info()     # PB/GC indices + pseudotime
    ├── process_dynamics()        # ordered, smoothed trajectories
    ├── plot()                    # Plotly comparative (PB solid, GC dashed)
    └── plot_score_vs_count_comparison()  # score vs OCR count per TF

episodic_dynamics.py
├── AlignTimeScales               # window ↔ sampled points pseudotime mapping
│   ├── pseudotime_of_windows()
│   └── pseudotime_of_sampled_points()
├── EpisodeDynamics               # composes SmoothedCurvesGRN + AlignTimeScales
│   ├── compute_expression_curves()
│   ├── build_episode_grn()       # time_slice → filtered edge DataFrame
│   ├── filter_edges()            # significance + direction invariance (parallel)
│   ├── compute_tf_expression()
│   ├── calculate_forces()        # parallel force calculation
│   ├── select_top_edges()        # percentile-based
│   ├── select_top_activating_and_repressing_edges()  # signed tails
│   ├── set_lf_genes()            # register cellular program genes
│   ├── annotate_lf_in_grn()
│   └── calculate_enrichment()    # hypergeometric ORA
├── calculate_tf_episodic_enrichment()    # standalone ORA function
├── filter_edges_by_significance_and_direction()  # chunked multiprocessing
├── calculate_force_curves_parallel()     # chunked multiprocessing
├── run_episodic_enrichment()     # top-level orchestrator (for joblib/SLURM)
├── run_episodic_construction()   # top-level orchestrator (for joblib/SLURM)
└── get_episodic_grn_subset()     # load + subset across episode pkl files
```

### Analysis notebooks

- `LF_global_dynamics.ipynb` — Global regulation analysis: Dictys dynamic network loading,
  custom TF visualization, beta-curve heatmaps, force-curve computation + clustering,
  network animations, branch-specific Dictys animations.
- `LF_local_dynamics.ipynb` — Episodic analysis: episode construction + enrichment workflow,
  parallel edge filtering, force calculation, top-edge selection, LF gene enrichment
  (hypergeometric test per TF per episode).

---

## Target repository layout (pip package)

```
FIREFate/
├── src/firefate/
│   ├── __init__.py
│   ├── _types.py                     # Type aliases (moscot-style)
│   ├── _logging.py                   # Rich logger
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pseudotime_curves.py      # SmoothedCurvesGRN, SmoothedCurvesChromatin
│   │   ├── align_time.py             # AlignTimeScales
│   │   ├── episodic_dynamics.py      # EpisodeDynamics (composition)
│   │   └── force.py                  # Parallel force computation
│   ├── grn/
│   │   ├── __init__.py
│   │   ├── state_specific.py         # Capability 2
│   │   ├── transition_window.py      # Capability 3
│   │   └── episodic.py               # Capability 4
│   ├── programs/
│   │   ├── __init__.py
│   │   ├── slide_interface.py        # SLIDE latent factors = CPs (capability 1)
│   │   └── gsea_llm.py              # LLM-augmented GSEA
│   ├── enrichment/
│   │   ├── __init__.py
│   │   ├── ora.py                    # Hypergeometric ORA primitive (capability 5)
│   │   ├── manager.py                # EnrichmentManager (unified API)
│   │   └── slide_grn.py             # get_SLIDE_GRN_enrichment
│   ├── perturbation/
│   │   ├── __init__.py
│   │   └── in_silico.py             # Capability 6 (CellOracle interface)
│   ├── fate_bias/
│   │   ├── __init__.py
│   │   └── stratify.py              # Capability 7
│   ├── managers/
│   │   ├── __init__.py
│   │   ├── grn_manager.py            # GRNManager — unified GRN construction
│   │   └── enrichment_manager.py     # EnrichmentManager — unified enrichment
│   └── utils/
│       ├── __init__.py
│       ├── plots.py                  # Migrated from utils_custom.py + episode_plots.py
│       ├── io.py                     # I/O helpers
│       └── gene_utils.py            # get_tf_indices, check_if_gene_in_ndict, etc.
├── tests/
├── docs/                             # Sphinx source
├── tutorials/                        # Jupyter notebooks per capability
├── README.md
├── pyproject.toml                    # See references/pip-packaging.md
├── LICENSE                           # MIT
└── .github/workflows/
```

---

## Manager classes — the unification layer

The codebase currently has enrichment logic scattered across `EpisodeDynamics.calculate_enrichment()`
(episodic) and the standalone `calculate_tf_episodic_enrichment()`. The same ORA primitive applies
to state-specific GRNs, episodic GRNs, and cross-state GRNs — only the edge source differs.

**Two manager classes** unify these:

### `EnrichmentManager`
A single entry point for TF-activity enrichment regardless of GRN type:

```python
from firefate.managers import EnrichmentManager

mgr = EnrichmentManager(
    lf_genes=lf_blimp1,
    dictys_dynamic_object=dictys_obj,
    trajectory_range=(1, 3),
    num_points=40,
)

# Episodic enrichment (capability 5 on episodic GRN): manager builds EpisodeDynamics internally
result = mgr.enrich_episodic(
    episode_idx=1,
    time_slice=slice(0, 5),
    percentile=98,
)

# State-specific enrichment (capability 5 on state GRN)
result = mgr.enrich_state_specific(grn_edges=state_grn_edges)

# Batch across all episodes
results = mgr.enrich_all_episodes(total_episodes=8, points_per_episode=5)
```

### `GRNManager`
A single entry point for GRN construction regardless of resolution:

```python
from firefate.managers import GRNManager

mgr = GRNManager(dictys_obj, trajectory_range=(1, 3), num_points=40)

# Transition-window GRN (capability 3): beta, forces (or None if links omitted), pseudotime
beta_curves, force_curves, dtime = mgr.build_transition_window(links=tf_target_links)

# Episodic GRN (capability 4)
episodic_grn = mgr.build_episodic(time_slice=slice(0, 5), percentile=98)

# Full pipeline: all episodes
all_episodes = mgr.build_all_episodes(total_episodes=8)
```

For the full manager implementations, see `references/architecture.md`.

---

## Migration plan: current code → pip package

| Current file | Target location | Notes |
|---|---|---|
| `pseudotime_curves.py` → `SmoothedCurvesGRN` | `src/firefate/core/pseudotime_curves.py` | Rename class to `SmoothedCurvesGRN` (already done in code) |
| `pseudotime_curves.py` → `SmoothedCurvesChromatin` | `src/firefate/core/pseudotime_curves.py` | Keep colocated with GRN curves |
| `episodic_dynamics.py` → `AlignTimeScales` | `src/firefate/core/align_time.py` | Extract to own module |
| `episodic_dynamics.py` → `EpisodeDynamics` | `src/firefate/core/episodic_dynamics.py` | Composes SmoothedCurvesGRN + AlignTimeScales |
| `episodic_dynamics.py` → parallel functions | `src/firefate/core/force.py` | `calculate_force_curves_parallel`, `filter_edges_by_significance_and_direction` |
| `episodic_dynamics.py` → `calculate_tf_episodic_enrichment` | `src/firefate/enrichment/ora.py` | The ORA primitive |
| `episodic_dynamics.py` → `run_episodic_enrichment/construction` | `src/firefate/managers/grn_manager.py` + `enrichment_manager.py` | Orchestration methods |
| `utils_custom.py` → gene helpers | `src/firefate/utils/gene_utils.py` | `get_tf_indices`, `get_gene_indices`, `check_if_gene_in_ndict` |
| `utils_custom.py` → plotting functions | `src/firefate/utils/plots.py` | All `fig_*`, `plot_*`, `cluster_heatmap` |
| `episode_plots.py` | `src/firefate/utils/plots.py` | Merge with other plotting code |
| `config.py` | `tutorials/config.py` (NOT in src/) | HPC paths don't belong in the package |

---

## Routing: which reference to read

When the user's request touches FIREFate, match to one of these references and read it before
writing anything substantial:

| User request type | Read |
|---|---|
| "How does capability N work?" / "What does the episodic GRN code do?" | `references/capabilities.md` |
| "Refactor this class" / "Restructure the modules" / "How should I organize..." | `references/architecture.md` + the `coding-style` skill's library-mode reference |
| "Package for pip/PyPI" / "pyproject.toml" / "pip install -e ." / "publish to pypi" | `references/pip-packaging.md` |
| "Write the README" / "Set up read-the-docs" / "Docstrings" / "Sphinx" / "Tutorial" | `references/documentation.md` |
| "Force curve formula" / "ORA" / "hypergeometric" / "magnifying glass" | `references/math-operators.md` |
| "B-cell results" / "ERCC1" / "T-cell exhaustion" / "which TFs matter" | `references/biological-context.md` + the `gene-pathways` skill |
| "Write a paper paragraph about FIREFate" | `references/paper-prose.md` + the `comp-bio-writing` skill |
| "EnrichmentManager" / "GRNManager" / "unify enrichment" / "class managers" | `references/architecture.md` (§ Manager classes) |

Simple in-chat questions ("what is capability 4?") can be answered from SKILL.md alone. Anything
that produces code, prose, or commits deserves loading the matching reference.

---

## Composition with other skills

FIREFate sits at the intersection of several Akanksha skills. The division of labor:

- **`coding-style`** — how to write FIREFate code (library mode from moscot for the package, research
  mode from loco-vis for tutorials/scripts).
- **`single-cell-analysis`** — scanpy/AnnData patterns FIREFate builds on (AnnData inputs, HVG
  selection, scVI integration, DEG analysis).
- **`flow-matching-ot`** — trajectory inference upstream of Dictys GRN calls; not currently a
  FIREFate dependency but useful for tutorials.
- **`gene-pathways`** — biological interpretation of TFs FIREFate nominates (BACH2/IRF4/PRDM1 etc.).
- **`comp-bio-writing`** — methods sections and paper prose for FIREFate.

### Project metadata

| Field | Value |
|-------|-------|
| Repository | https://github.com/sachha-naksha/FIREFate |
| Python | 3.12 |
| Web app | https://pitt-csi.shinyapps.io/firefate/ |
| Docs site | `<READ-THE-DOCS>` (placeholder, not yet built) |
| Dependencies upstream | `dictys`, `loveslide` (Akanksha's fork of `alw399/SLIDE_py`), `anndata`, `scanpy`, `scvi-tools`, `cellOracle` (for capability 6) |
| License | MIT |
| Authors | Akanksha Sachan (first author), Jishnu Das (PI) |

---

## Style commitments

These are carried over from the paper prose iteration and should apply to every FIREFate
deliverable (README, docs, paper text, comments, commit messages):

- **Never "convex combination"** unless FIREFate literally outputs αX + (1−α)Y with α∈[0,1] and
  weights live somewhere explicit. Default word: **"integrates" / "couples"**.
- **Never "discriminative"** in the "distinguishes two states" sense — that word means
  "models p(y|x)" in ML. Use **"contrasting" / "distinct"**.
- **Never "API products"** — say **"capabilities"** (preferred) or **"modules"**.
- **Never "comprehensive framework"** — "comprehensive" is padding. Just **"framework"**.
- **Top-percentile tails** not "top-percentile of activating and repressive" — the tails are what
  matter, and the phrasing disambiguates magnitude-threshold vs signed-force-threshold.
- **"Nominate TFs" / "downstream targets"** not "offer downstream links" — nominate is from the
  bridge lexicon (ML importance ↔ TF nomination).
- **Parallel imperatives** for capability bullets (all imperative verbs: Discover, Construct,
  Construct, Construct, Quantify, Quantify, Stratify).
- **Live text in SVG** (Illustrator-editable), `pdf.fonttype = 42` in matplotlib, per the
  paper-schematics skill.

---

## Common first asks

- **"Revamp the repo"** → Read `references/architecture.md` first. Current state: analysis modules
  in `multiome_dynamic_regulation/py_scripts/analysis/`. Target: migrate into `src/firefate/`
  with the layout above, add `pyproject.toml` (see `references/pip-packaging.md`), README
  (see `references/documentation.md`).
- **"Package for pip"** → `references/pip-packaging.md` has the full pyproject.toml template
  for FIREFate's actual dependency tree and a ready-to-run publish workflow.
- **"Unify enrichment"** → Read `references/architecture.md` § Manager classes. The
  `EnrichmentManager` provides a single API surface for episodic, state-specific, and cross-state
  enrichment, delegating to the same ORA primitive underneath.
- **"Set up read-the-docs"** → `references/documentation.md` has the Sphinx config, docstring
  style (NumPy), and autoapi setup tuned for FIREFate's seven-capability organization.
- **"Write the methods section"** → Read both `references/paper-prose.md` and invoke the
  `comp-bio-writing` skill.
