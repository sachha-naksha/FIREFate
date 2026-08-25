# FIREFate module structure — proposal (moscot-inspired)

Status: **ACCEPTED and implemented for the Temporal pass.** Supersedes §6 and §10 of
`architecture.md`. See §7 for what was built and where it deviates from this plan.

## 0. The organising idea

FIREFate is three modules in one package:

| Module | Capabilities (docs/capabilities.rst) | Question it answers |
|---|---|---|
| **Temporal** | 3, 4, 5-dynamic | How does TF regulation change *along* a trajectory? |
| **StateSpecific** | 1, 2, 5-static, 6 | What separates two *fixed* states, and what happens if we perturb it? |
| **CrossPrediction** | 7 | Do programs learned in one dataset *transfer* to stratify another? |

moscot's layout maps onto this almost one-to-one:

| moscot | FIREFate | why |
|---|---|---|
| `base/problems/{problem,manager,_mixins}.py` | `base/{problem,manager,mixins}.py` | one abstract contract, N concrete modules |
| `problems/{time,space,cross_modality}/` | `temporal/`, `state_specific/`, `cross_prediction/` | the domain split |
| `problems/*/​_mixins.py` | `*/_mixins.py` | *analysis* methods separated from *construction* methods |
| `backends/ott/` | `backends/{dictys,celloracle,slide}/` | third-party engines isolated behind one door |
| `plotting/` | *(none — figures colocated)* | a figure lives with the class whose output it draws |
| `_types.py`, `_constants.py`, `_logging.py` | same | shared vocabulary |

Two moscot conventions worth copying verbatim:

1. **Private modules, public `__init__`.** Files are `_curves.py`, `_phases.py`; the only public
   names are those re-exported by `__init__.py`. Renaming a file then never breaks a user.
2. **Mixin split.** A `Problem` knows how to *build* itself; a `Mixin` knows how to *interrogate*
   the built result. In moscot, `TemporalProblem` = `CompoundProblem` + `TemporalMixin`. For us,
   `TFForceWaves` (construction) vs. the phase/validation/plotting methods hung off it (analysis).

## 1. Proposed tree

```
src/firefate/
├── __init__.py                  # version + `from firefate import temporal, state_specific, cross_prediction`
├── _types.py                    # PathLike_t, EdgeTable_t, CurveMode_t, DictysNet_t, ...
├── _constants.py                # column names ('avg_force', 'p_value'), mode enums, defaults
├── _logging.py                  # logging.getLogger("firefate")
├── py.typed
│
├── base/                        # contracts every module obeys
│   ├── __init__.py
│   ├── problem.py               # BaseProblem: prepare() → run() → result, state machine + validation
│   ├── manager.py               # BaseManager: owns {key: Problem}, batch run / save / load / to_frame
│   ├── curves.py                # BaseCurves: pseudotime-indexed frame contract + smoothing hook
│   ├── network.py               # BaseGRN: MultiIndex (TF, target) edge-table contract
│   ├── enrichment.py            # ora(M, n, N, X) — THE hypergeometric primitive, used by 2 modules
│   └── mixins.py                # ResultCacheMixin, SerialisationMixin
│
├── backends/                    # every third-party import lives here and nowhere else
│   ├── __init__.py
│   ├── dictys/
│   │   ├── __init__.py
│   │   ├── _io.py               # load dynamic net, read binding.tsv.gz, h5 dump   [dynamic_grn/utils.py]
│   │   ├── _stats.py            # lcpm_tf, ChromatinGRNStat                        [core/stat_extensions.py]
│   │   └── _reconstruct.py      # reconstruct() wrapper                    [network_reconstruct_batch.py]
│   ├── celloracle/              # links → GRN tables, in-silico KO         [grn/state_specific.py, 05_sim]
│   └── slide/                   # R SLIDE / ESCAPE subprocess wrappers  [Cross_prediction/{SLIDE,ESCAPE}/*.R]
│
├── temporal/                    # ============ FIREFateTemporal ============
│   ├── __init__.py
│   ├── _align.py                # AlignTimeScales
│   ├── _curves.py               # SmoothedCurvesGRN
│   ├── _chromatin.py            # SmoothedCurvesChromatin
│   ├── _forces.py               # force math + parallel force/filter kernels
│   ├── _episodes.py             # EpisodeDynamics
│   ├── _waves.py                # TFForceWaves
│   ├── _states.py               # StateFrequency
│   ├── _phases.py               # RegulatoryPhases, ForceWavePhases, BindingPhases, order_links*
│   ├── _validation.py           # TFForceValidation
│   │   # NB: each module above also holds ITS OWN figures — no plotting/ package
│   └── manager.py               # TemporalManager — the module entry point
│
├── state_specific/              # ========= FIREFateStateSpecific =========
│   ├── __init__.py
│   ├── _programs.py             # CP / latent-factor loading + gene-set annotation
│   ├── _grn.py                  # state + cross-state GRN construction
│   ├── _enrichment.py           # StateSpecificEnrichment, SLIDE-GRN hypergeometric
│   ├── _perturbation.py         # in-silico KO, phenotypic shift / perturbation score
│   ├── _mixins.py
│   └── manager.py               # StateSpecificManager
│
├── cross_prediction/            # ======== FIREFateCrossPrediction ========
│   ├── __init__.py
│   ├── _transfer.py             # move CPs from intervention → query dataset
│   ├── _fate_bias.py            # stratify uncommitted populations
│   ├── _mixins.py
│   └── manager.py               # CrossPredictionManager
│
├── io/
│   ├── __init__.py
│   ├── _readers.py              # h5 dump, mtx → tsv, AnnData-from-pkl
│   ├── _qc.py                   # qc_reads
│   └── _paths.py                # DatasetPaths dataclass  ← replaces analysis/config.py
│
├── utils/
│   ├── __init__.py
│   ├── genes.py                 # get_tf_indices, get_gene_indices, check_if_gene_in_ndict
│   ├── curves.py                # curvature_of_expression, AUC, logFC helpers
│   ├── states.py                # window/state label helpers
│   └── parallel.py              # create_balanced_chunks + Pool wrappers
│
└── cli/                         # argparse entry points the .sbatch files call
    ├── __init__.py
    ├── expression_to_tsv.py     # ← get_main_expression.py
    ├── validate_inputs.py       # ← debug_subset_cell.py
    ├── reconstruct_networks.py  # ← network_reconstruct_batch.py
    └── motifs_to_homer.py       # ← the motif half of dynamic_grn/utils.py
```

## 2. File-by-file mapping

### 2a. `py_scripts/` → `src/` (the requested move)

| Source | Destination | Notes |
|---|---|---|
| `analysis/config.py` | `io/_paths.py` + `py_scripts/datasets.yaml` | class keeps its shape; the **absolute HPC paths move out of Python into YAML** (architecture.md §1: no HPC paths in the package) |
| `analysis/ensure_firefate_path.py` | **deleted** | replaced by `pip install -e .` |
| `analysis/{episodic_dynamics,pseudotime_curves,state_dynamics,dynamic_validation,episode_plots,utils_custom}.py` | **deleted** | already pure `import *` shims |
| `dynamic_grn/get_main_expression.py` | `cli/expression_to_tsv.py` + `io/_readers.py` | `main()` splits into an importable function + a `__main__` guard |
| `dynamic_grn/debug_subset_cell.py` | `cli/validate_inputs.py` | currently top-level statements; wrap in `main(args)` |
| `dynamic_grn/network_reconstruct_batch.py` | `cli/reconstruct_networks.py` + `backends/dictys/_reconstruct.py` | drops the `os.chdir` |
| `dynamic_grn/utils.py` → `read_h5_file`, `read_adata_from_pkl` | `io/_readers.py` | |
| `dynamic_grn/utils.py` → `qc_reads` | `io/_qc.py` | |
| `dynamic_grn/utils.py` → `plot_main_trajectory_nodes` | `temporal/_align.py` | drops the `matplotlib.use("Agg")` side effect |
| `dynamic_grn/utils.py` → `parse_cisBP_motifs`, `process_motif_file_in_homer_format` | `cli/motifs_to_homer.py` | drops the hard-coded `__main__` paths |

Result: `py_scripts/` holds **only `.ipynb`** plus one `datasets.yaml`.

### 2b. `src/firefate/` internal re-shuffle

| Current | Destination |
|---|---|
| `core/episodic_dynamics.py` | `temporal/_align.py` (AlignTimeScales) · `temporal/_episodes.py` (EpisodeDynamics) · `temporal/_forces.py` (chunking + parallel force/filter) · `base/enrichment.py` (`calculate_tf_episodic_enrichment` → `ora`) · `temporal/manager.py` (`run_episodic_*`) |
| `core/pseudotime_curves.py` | `temporal/_curves.py` + `temporal/_chromatin.py` + `temporal/_mixins.py` (the `curve_characteristics` / `classify_*` / `get_top_k_*` block) |
| `core/state_dynamics.py` | `temporal/_states.py` · `temporal/_waves.py` · `temporal/_phases.py` (figures included) |
| `core/validation.py` | `temporal/_validation.py`, with its `plot_*` staticmethods |
| `core/stat_extensions.py` | `backends/dictys/_stats.py` |
| `grn/state_specific.py` | `state_specific/_grn.py` |
| `enrichment/sse.py` | `state_specific/_enrichment.py` |
| `enrichment/slide_grn.py` | `state_specific/_enrichment.py` |
| `enrichment/lf_bar_plots.py` | `state_specific/_enrichment.py` (plotters) + `io/_readers.py` (the 3 `load_*` fns) |
| `managers/enrichment_manager.py` | **split**: `enrich_episodic`/`enrich_all_episodes` → `temporal/manager.py`; `enrich_state_specific` → `state_specific/manager.py`; `batch_from_config` → `io/_paths.py` |
| `utils/custom.py` | `utils/genes.py` + `utils/states.py` + `utils/curves.py` |
| `utils/plots.py` | dissolved into the owning modules — see §8 |

**`core/` disappears.** Everything in it today is Temporal.

## 3. Manager design

One manager per module, all deriving from `base/manager.py`. They are *composition* layers
(architecture.md §3) — a manager holds problem objects, never inherits from them.

```python
class TemporalManager(BaseManager, TemporalAnalysisMixin):
    """Entry point for FIREFateTemporal."""
    def __init__(self, net, *, trajectory_range=(1, 3), num_points=40,
                 dist=1e-3, sparsity=0.01, output_dir=None): ...

    # --- construction (cap. 3 & 4) ---
    def prepare_curves(self, mode="expression") -> SmoothedCurvesGRN
    def build_transition_window(self, links=None)          # cap 3
    def build_episode(self, episode, time_slice)           # cap 4
    def build_all_episodes(self, n_episodes=8, points_per_episode=5)

    # --- enrichment (cap. 5-dynamic) ---
    def enrich_episode(self, episode, lf_genes)            # delegates to base.enrichment.ora
    def enrich_all_episodes(self, lf_genes)

    # --- analysis (from the mixin) ---
    def phases(self, switch_pseudotimes) -> ForceWavePhases
    def waves(self, links) -> TFForceWaves
    def validate(self, enriched_links, **kw) -> TFForceValidation
    def chromatin(self, tfs, base_path) -> SmoothedCurvesChromatin
```

`StateSpecificManager` (programs → GRN → enrichment → perturbation) and
`CrossPredictionManager` (fit programs on reference → transfer → stratify query) get the same
shape. The shared `base/enrichment.py:ora` is what makes the two enrichment paths one primitive
instead of two copies — the whole reason `EnrichmentManager` exists today.

## 4. Tests

```
tests/
├── conftest.py                 # mock dictys network moves here — shared by all modules
├── base/
│   ├── test_ora.py             # the hypergeometric primitive, vs scipy.stats.hypergeom
│   └── test_manager_contract.py
├── temporal/                   # ← today's tests/episodic/, resplit
│   ├── conftest.py             # curve fixtures, mock binding files
│   ├── test_align.py
│   ├── test_curves.py          # ← test_smoothed_curves_grn.py
│   ├── test_chromatin.py       # ← test_chromatin_curves.py
│   ├── test_curve_math.py      # ← unchanged
│   ├── test_forces.py          # ← the force/chunk half of test_episodic_units.py
│   ├── test_episodes.py        # ← test_episode_workflow.py
│   ├── test_phases.py          # NEW — state_dynamics.py has no tests today
│   ├── test_validation.py      # NEW — validation.py has no tests today
│   └── test_manager.py         # NEW
├── state_specific/             # NEW
└── cross_prediction/           # NEW
```

`tests/episodic/{ISSUES.md, README.md}` and the two diagnostic notebooks move to `tests/temporal/`
unchanged. The `test_currently_*` / `xfail(strict=True)` convention in the existing README carries
over to the new suites.

**Coverage gap this exposes:** `state_dynamics.py` (1 200 lines) and `validation.py` (520 lines)
are the two largest untested files, and `validation.py` is what the paper's enriched-vs-random
figure rests on. Those get tests first.

## 5. Notebooks

14 notebooks stay in `py_scripts/`. Each gets its import block replaced by:

```python
import firefate as ff
from firefate.temporal import EpisodeDynamics, SmoothedCurvesGRN, TFForceWaves
from firefate.io import DatasetPaths

paths = DatasetPaths.from_yaml("datasets.yaml")   # was: from config import *
```

The `from X import *` + `importlib.reload(X)` idiom used in `chromatin_dynamics.ipynb`,
`dynamic_validation.ipynb`, `episodic_enrichment.ipynb` and `phase_clustered_links.ipynb` is
replaced by `importlib.reload(firefate.temporal._waves)` on the real module.

## 6. Also needs updating (found during the survey)

- `docs/reference.rst` — every `automodule` path changes.
- `bash_scripts/{network/1_network_reconstruct,preproc/0_get_expression}.sbatch` — currently point
  at `/ocean/projects/cis240075p/...` paths that **do not exist on this cluster**; become
  `python -m firefate.cli.reconstruct_networks`.
- `bash_scripts/preproc/1b_traj_windows_parallel.sbatch` references
  `py_scripts/trajectory/process_single_edge.py`, **which is not in the repo**.
- `references/architecture.md` §6 and §10 are superseded by this document.
- `py_scripts/base_grn/{atac_rna_cell_pairing,motif_scan,peak_to_gene}.ipynb` are deleted in the
  working tree but not committed — confirm intentional before the restructure commit.

---

## 7. What was actually built (Temporal pass)

Decisions taken: **top-level module namespaces**, **plotting split by subject**
(later revised to full colocation -- see §8),
**`backends/` isolation**, **paste-ready notebook cells**. Verified by
`pytest tests/temporal`: **306 passed, 6 xfailed** -- byte-identical to the
pre-refactor baseline -- plus every one of the 46 modules importing cleanly and all
**77** previously-public names still reachable.

### Deviations from §1, and why

1. **`base/` holds two files, not seven.** `problem.py`, `curves.py`, `network.py` and
   `mixins.py` were not created. Each would have had exactly one implementer today, which
   is the speculative abstraction `CLAUDE.md` §2 forbids. What went in is what is genuinely
   shared: `manager.py` (output dir, result registry, save/load -- used by both managers)
   and `enrichment.py` (the ORA primitive -- the reason the old `EnrichmentManager`
   existed). Add the rest when a second implementer actually appears.

2. **No `_mixins.py` in any module.** Same reason: the mixin split earns its keep when two
   classes share analysis methods. Right now `TemporalManager` is the only consumer, so
   its analysis methods live on the class. Split them out when StateSpecific needs them.

3. **`backends/celloracle/` and `backends/slide/` were not created.** There is no Python
   to put in them yet -- the celloracle and SLIDE code is still `.R` files and notebooks
   outside the package. Empty directories are not tracked by git anyway. They land with
   the StateSpecific port. `backends/dictys/` is real and holds three modules.

4. **`cross_prediction/` is a documented empty package.** Its logic lives in
   `Cross_prediction/SLIDE/Crossprediction.R` and `TF_Dynamic_Activity/`, neither
   of which is Python. `__init__.py` says so and exports nothing.

5. **`state_specific/_slide.py` stayed separate** rather than merging into
   `_enrichment.py`. Two private modules under one public `__init__` is the moscot
   convention and keeps `git log --follow` working on both.

6. **`plot_force_heatmap_by_phase` lives in `temporal/_phases.py`, not `plotting/`.**
   Putting it in `plotting/` inverted the layering: it calls `order_links`, so
   `plotting` would have imported `temporal` while `temporal` already imports `plotting`
   -- a circular import. *This deviation is what §8 generalises to every figure.*

7. **Test files were renamed, not split.** `test_episodic_units.py` now spans three
   modules (`base`, `temporal/_forces`, `utils`) but was left whole; splitting test files
   adds risk without adding coverage. The real gap is the *missing* suites, now listed in
   `tests/temporal/README.md`.

8. **`utils/parallel.py` and `io/_qc.py`** were added beyond §1: `create_balanced_chunks`
   is used by both the force and filter paths, and `qc_reads` had no home in the plan.

### Also changed

* `pyproject.toml`: `requires-python` relaxed to `>=3.9` (the `dictys` env is 3.9, so
  `>=3.10` made `pip install -e .` refuse), and `pyyaml` added for `io/_paths.py`.
* `docs/reference.rst`: rewritten against the new module tree.
* `bash_scripts/{network/1_network_reconstruct,preproc/0_get_expression}.sbatch`: the dead
  `/ocean/...` script paths became `python -m firefate.cli.*`.
* `multiome_dynamic_regulation/py_scripts/NOTEBOOK_MIGRATION.md`: per-notebook cells.
* `multiome_dynamic_regulation/py_scripts/datasets.yaml`: the paths from `config.py`.

### Still open

* `bash_scripts/preproc/1b_traj_windows_parallel.sbatch` calls
  `py_scripts/trajectory/process_single_edge.py`, which is not in the repo at all.
* `py_scripts/base_grn/{atac_rna_cell_pairing,motif_scan,peak_to_gene}.ipynb` are deleted
  in the working tree but not committed -- pre-existing, untouched by this refactor.
* The four analysis notebooks still need their cells pasted (Step 0 of the migration doc
  is `pip install -e .`, which has **not** been run -- it would modify the `dictys` env).

---

## 8. Follow-up: `plotting/` dissolved into the modules

The first pass split `utils/plots.py` into a `plotting/` package of six subject
modules. That was then reversed in favour of full colocation: **a figure lives in
the file that owns its subject**, the same rule that already put
`plot_force_heatmap_by_phase` next to the phase classes. There is no
`firefate.plotting` any more.

### What made this safe

A call-site analysis (AST, ignoring import statements) over all 42 plotting symbols:

| | count |
|---|---|
| owned by exactly one module | 17 |
| **used by more than one module** | **0** |
| no internal caller — public API for notebooks | 25 |

Zero shared symbols means colocation needed no shared plotting namespace at all —
even `plotting/_utils.py` dissolved, because each of its three helpers had exactly
one consumer (`_get_qualitative_colors` → `_waves`, `create_pathway_color_scheme` and
`sort_tfs_by_gene_similarity` → `_episodes`). The 25 with no internal caller were
placed by subject.

### Where everything went

| figures | destination |
|---|---|
| expression / regulation curves and their heatmaps | `temporal/_curves.py` |
| chromatin binding dynamics | `temporal/_chromatin.py` |
| trajectory nodes | `temporal/_align.py` |
| force landscapes, force heatmaps, clustering | `temporal/_waves.py` |
| cell-state composition | `temporal/_states.py` |
| phase binding boxes, phase-ordered heatmaps | `temporal/_phases.py` |
| enriched-vs-random validation boxes | `temporal/_validation.py` |
| episodic enrichment dotplots and heatmaps | `temporal/_episodes.py` |
| SLIDE latent-factor enrichment bars | `state_specific/_enrichment.py` |

### Consequences

* **The layering problem disappears.** Deviation 6 above described a
  `plotting → temporal` cycle that forced `plot_force_heatmap_by_phase` out of
  `plotting/`. With no plotting package there is no edge to invert, and the special
  case is now just the general rule.
* **Module sizes.** `temporal/_episodes.py` is 1 430 lines and `temporal/_waves.py`
  1 217; the rest are under 1 000. Compare the 3 213-line `utils/plots.py` this
  replaced. If `_episodes.py` keeps growing, the split to reach for is
  `EpisodeDynamics` vs. the enrichment-result figures, not a new plotting package.
* **One import instead of two.** `from firefate.temporal import *` now yields the
  classes *and* their figures, so the notebook cells got shorter, not longer.
* **A latent bug surfaced.** `plot_main_trajectory_nodes` had been missing its
  `import networkx as nx` since it was moved out of `dynamic_grn/utils.py` — it had no
  test and no caller, so nothing had raised. Now fixed in `temporal/_align.py`.

Re-verified after the change: **306 passed, 6 xfailed**, 39 modules import cleanly,
all 77 previously-public names still reachable, no name collisions across
`temporal` / `state_specific` / `utils`.

