# Notebook migration — paste-ready cells

Every `.py` file that used to sit in `py_scripts/` now lives in `src/firefate/`. The
notebooks stayed put; only their import cells change. Cells are given as
**OLD → NEW**; paste the NEW block over the OLD one and re-run.

Nothing else in any notebook needs to change: the replacement for cell 1 deliberately
keeps the variable name `config`, so every downstream `config.OUTPUT_FOLDER`,
`config.PB['ep1']`, `config.CELL_LABELS` keeps working untouched. The one exception is
`config._BCELL_BASE` → `config.BCELL_BASE` (noted per notebook below).

**There is no `firefate.plotting`.** Every figure lives in the module that owns its
subject — force landscapes with `TFForceWaves`, phase heatmaps with the phase classes,
enrichment bars with the enrichment code. In practice that means `from firefate.temporal
import *` gives you the plots *and* the classes in one import, which is fewer lines than
before, not more.

---

## Step 0 — install the package once (required)

The `ensure_firefate_path.py` bootstrap is gone; `firefate` is now imported like any
other package. From a terminal:

```bash
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dictys
cd /projects/bhdw/asachan/methods/FIREFate
pip install -e .
```

`-e` means edits to `src/firefate/` are picked up without reinstalling. Restart any
running kernel afterwards.

> If you would rather not install, put this at the top of cell 1 instead — but the
> install is the supported path, and the `.sbatch` scripts now need it too:
> ```python
> import sys; sys.path.insert(0, "/projects/bhdw/asachan/methods/FIREFate/src")
> ```

---

## Step 1 — where each name moved

| Was | Now |
|---|---|
| `from utils_custom import *` | `from firefate.utils import *` |
| `from pseudotime_curves import *` | `from firefate.temporal import *` |
| `from episodic_dynamics import *` | `from firefate.temporal import *` |
| `from state_dynamics import *` | `from firefate.temporal import *` |
| `from dynamic_validation import TFForceValidation` | `from firefate.temporal import TFForceValidation` |
| `from episode_plots import *` | `from firefate.temporal import *` (figures are colocated now) |
| `from config import *` / `Config()` | `from firefate.io import DatasetPaths` / `DatasetPaths.from_yaml(...)` |
| `from utils import *` (dynamic_grn) | `from firefate.backends.dictys import *`, `from firefate.io import qc_reads` |
| `firefate.core.*` | `firefate.temporal` |
| `firefate.utils.custom` | `firefate.utils` (split into `genes`, `states`, `curves`, `parallel`) |
| `firefate.utils.plots` | gone — see the figure table below |
| `firefate.enrichment.build_tf_color_bar_table` | `firefate.state_specific.build_tf_color_bar_table` |
| `firefate.enrichment.load_lf_gene_colors` | `firefate.io.load_lf_gene_colors` |

### Where each figure went

You rarely need this — `from firefate.temporal import *` re-exports all of the Temporal
ones — but it is the map for `importlib.reload` and for direct module imports.

| figure | module |
|---|---|
| `plot_expression_for_multiple_genes`, `plot_gene_expression_subplots`, `fig_regulation_heatmap`, `fig_expression_gradient_heatmap`, `fig_expression_linear_heatmap` | `firefate.temporal._curves` |
| `plot_chromatin_tf_dynamics`, `plot_score_vs_count_subplots` | `firefate.temporal._chromatin` |
| `plot_main_trajectory_nodes` | `firefate.temporal._align` |
| `plot_force_landscape`, `plot_single_link_landscape`, `plot_force_by_tf`, `plot_gene_trajectories`, `plot_force_heatmap`, `plot_force_heatmap_with_clustering`, `cluster_heatmap` | `firefate.temporal._waves` |
| `plot_state_composition_bars`, `plot_state_composition_curves`, `plot_state_extrema` | `firefate.temporal._states` |
| `plot_phase_binding_boxes`, `plot_phase_ordered_force_heatmap`, `plot_force_heatmap_by_phase` | `firefate.temporal._phases` |
| `plot_force_validation_boxes`, `plot_force_validation_multi`, `plot_force_validation_by_phase`, `plot_force_validation_phase_cells` | `firefate.temporal._validation` |
| `plot_tf_episodic_enrichment_dotplot`, `plot_tf_target_episodic_heatmap`, `plot_tf_gene_coregulation_heatmap`, `sort_tfs_by_gene_similarity`, `create_pathway_color_scheme` | `firefate.temporal._episodes` |
| `plot_tf_enrichment_bars`, `plotly_tf_enrichment_bars`, `plot_strength_key_distribution`, `build_tf_color_bar_table`, `plot_episode_from_csvs` | `firefate.state_specific._enrichment` |

**One behavioural difference worth knowing.** The old modules had no `__all__`, so
`from pseudotime_curves import *` also dumped *their* imports (`np`, `pd`, `plt`,
`stats`, `hypergeom`, `tqdm`, …) into the notebook namespace. The new packages define
`__all__`, so a star import brings only FIREFate's own names. That is why the
replacement cells below import numpy/pandas/matplotlib explicitly — if a later cell
raises `NameError: name 'np' is not defined`, this is why.

---

## `analysis/chromatin_dynamics.ipynb`, `analysis/dynamic_validation.ipynb`, `analysis/episodic_enrichment.ipynb`, `analysis/phase_clustered_links.ipynb`

These four share the same cell 1.

**OLD — cell 1**
```python
import sys
# Ensure this analysis directory is importable regardless of kernel CWD
_here = '/projects/bhdw/asachan/methods/FIREFate/multiome_dynamic_regulation/py_scripts/analysis'
if _here not in sys.path:
    sys.path.insert(0, _here)

import dictys
from utils_custom import *
from pseudotime_curves import *
from episodic_dynamics import *
from config import *
```

**NEW — cell 1**
```python
import ast
import math
import os
import pickle
import sys

import dictys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import firefate as ff
from firefate.temporal import *      # classes AND their figures
from firefate.utils import *         # gene / state helpers (was utils_custom)
from firefate.io import DatasetPaths

# Keeps the name `config`, so every downstream config.OUTPUT_FOLDER / config.PB works.
config = DatasetPaths.from_yaml(
    '/projects/bhdw/asachan/methods/FIREFate/multiome_dynamic_regulation/py_scripts/datasets.yaml'
)
```

### `chromatin_dynamics.ipynb` — cell 2

**OLD**
```python
import importlib
import firefate.utils.plots, episode_plots
importlib.reload(firefate.utils.plots)   # picks up the new functions
importlib.reload(episode_plots)          # re-runs `from firefate.utils.plots import *`
from episode_plots import *              # re-bind names into the notebook namespace
```

**NEW** — there is no shim to reload any more; reload the real modules.
```python
import importlib
import firefate.temporal._chromatin, firefate.temporal._curves, firefate.temporal._waves
for _m in (firefate.temporal._chromatin, firefate.temporal._curves,
           firefate.temporal._waves, firefate.temporal):
    importlib.reload(_m)
from firefate.temporal import *          # re-bind names into the notebook namespace
```

### `dynamic_validation.ipynb` — cells 2, 26, 31, 34

**OLD — cell 2**
```python
import importlib
import firefate.utils.plots, firefate.utils.custom
import state_dynamics
# re-bind
importlib.reload(firefate.utils.plots)
importlib.reload(firefate.utils.custom)
importlib.reload(state_dynamics)
from state_dynamics import TFForceWaves, StateFrequency, BindingPhases
from dynamic_validation import TFForceValidation
```

**NEW — cell 2**
```python
import importlib
import firefate.utils
import firefate.temporal._phases, firefate.temporal._states
import firefate.temporal._waves, firefate.temporal._validation
for _m in (firefate.utils, firefate.temporal._waves, firefate.temporal._states,
           firefate.temporal._phases, firefate.temporal._validation, firefate.temporal):
    importlib.reload(_m)
from firefate.temporal import TFForceWaves, StateFrequency, BindingPhases, TFForceValidation
```

> Reload order matters: `_phases` imports from `_states`/`_align`, and `_validation`
> imports from `_phases` and `_waves`, so reload the dependencies first.

**OLD — cell 26** (first three lines only; leave the rest of the cell alone)
```python
import importlib, dynamic_validation
importlib.reload(dynamic_validation)
from dynamic_validation import TFForceValidation
```

**NEW — cell 26**
```python
import importlib, firefate.temporal._validation
importlib.reload(firefate.temporal._validation)
from firefate.temporal._validation import TFForceValidation
```

**OLD — cell 31**
```python
from firefate.core.pseudotime_curves import SmoothedCurvesChromatin
```

**NEW — cell 31**
```python
from firefate.temporal import SmoothedCurvesChromatin
```

**OLD — cell 34** (first three lines only)
```python
import importlib, state_dynamics
importlib.reload(state_dynamics)
from state_dynamics import BindingPhases
```

**NEW — cell 34**
```python
import importlib, firefate.temporal._phases
importlib.reload(firefate.temporal._phases)
from firefate.temporal._phases import BindingPhases
```

### `episodic_enrichment.ipynb` — cells 2, 12

**OLD — cell 2**
```python
import importlib
from firefate.utils import plots
importlib.reload(plots)
from episode_plots import *              # re-bind names into the notebook namespace
```

**NEW — cell 2**
```python
import importlib
import firefate.temporal._episodes
importlib.reload(firefate.temporal._episodes)
importlib.reload(firefate.temporal)
from firefate.temporal import *          # re-bind names into the notebook namespace
```

**OLD — cell 12** (first line, and the `plots.` call further down)
```python
from firefate.enrichment import build_tf_color_bar_table, load_lf_gene_colors
...
    fig, _ = plots.plot_tf_enrichment_bars(
```

**NEW — cell 12**
```python
from firefate.io import load_lf_gene_colors
from firefate.state_specific import build_tf_color_bar_table, plot_tf_enrichment_bars
...
    fig, _ = plot_tf_enrichment_bars(          # was plots.plot_tf_enrichment_bars
```

> The LF colour-bar plots belong to StateSpecific (they split a TF's targets by SLIDE
> latent-factor sign), which is why they are imported from there even in an episodic
> notebook.

### `phase_clustered_links.ipynb` — cells 2, 11

**OLD — cell 2**
```python
import importlib
import firefate.utils.plots, firefate.utils.custom
import state_dynamics
# re-bind
importlib.reload(firefate.utils.plots)
importlib.reload(firefate.utils.custom)
importlib.reload(state_dynamics)
```

**NEW — cell 2**
```python
import importlib
import firefate.utils, firefate.temporal._waves, firefate.temporal._phases
for _m in (firefate.utils, firefate.temporal._waves,
           firefate.temporal._phases, firefate.temporal):
    importlib.reload(_m)
```

**OLD — cell 11**
```python
import state_dynamics as sd
```

**NEW — cell 11**
```python
import firefate.temporal._phases as sd
```

---

## `analysis/LF_global_dynamics.ipynb`

**OLD — cell 1**
```python
from utils_custom import *
from pseudotime_curves import *
from episode_plots import *
from episodic_dynamics import *
from config import *
```

**NEW — cell 1**
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import firefate as ff
from firefate.temporal import *
from firefate.utils import *
from firefate.io import DatasetPaths

config = DatasetPaths.from_yaml(
    '/projects/bhdw/asachan/methods/FIREFate/multiome_dynamic_regulation/py_scripts/datasets.yaml'
)
```

**OLD — cell 45**
```python
from pseudotime_curves import *
```

**NEW — cell 45**
```python
from firefate.temporal import *
```

⚠️ This notebook also uses `config._BCELL_BASE`. Change that one reference to
`config.BCELL_BASE` (no leading underscore).

---

## `analysis/LF_local_dynamics.ipynb`

**OLD — cell 1**
```python
from utils_custom import *
from episodic_dynamics import *
from pseudotime_curves import *
```

**NEW — cell 1**
```python
import math
import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from firefate.temporal import *
from firefate.utils import *
```

---

## `analysis/t_cell_analysis.ipynb`

**OLD — cell 1**
```python
from utils_custom import *
from pseudotime_curves import *
from episodic_dynamics import *
```

**NEW — cell 1**
```python
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from firefate.temporal import *
from firefate.utils import *
```

---

## `dynamic_grn/dynamic_grn_b_cell.ipynb`

**OLD — cell 14** (first four lines only)
```python
import matplotlib.pyplot as plt
import h5py
import numpy as np
from utils import *
```

**NEW — cell 14**
```python
import matplotlib.pyplot as plt
import h5py
import numpy as np
from firefate.backends.dictys import read_h5_file, read_adata_from_pkl
from firefate.io import qc_reads
from firefate.temporal import plot_main_trajectory_nodes
```

The motif helpers that also lived in that `utils.py` (`parse_cisBP_motifs`,
`process_motif_file_in_homer_format`) are now a command:
`python -m firefate.cli.motifs_to_homer IN.meme OUT.motif`.

---

## Notebooks that need no change

`trajectory/*.ipynb` (7 notebooks) and `dynamic_grn/{data_prep_and_checks,t_cell_dictys_data}.ipynb`
import only third-party packages (`stream`, `scvelo`, `palantir`, `dictys`, …), so they
are unaffected.

---

## Optional: the new manager API

Everything above preserves the existing call sites. When you next touch a notebook,
the manager is the shorter path — it holds the trajectory parameters once instead of
threading them through every constructor:

```python
from firefate.temporal import TemporalManager

mgr = TemporalManager(
    dictys_dynamic_object=dyn_obj,
    trajectory_range=(1, 3), num_points=40, dist=0.001, sparsity=0.01,
    output_dir=config.OUTPUT_FOLDER,
)

grn        = mgr.build_episode(1, slice(0, 5))
enrichment = mgr.enrich_episode(1, slice(0, 5), lf_genes=lf_blimp1)
all_eps    = mgr.enrich_all_episodes(lf_blimp1, total_episodes=8, write=True)

waves = mgr.waves()                      # TFForceWaves on this trajectory
waves.compute_forces(links, varname='w_in')
waves.plot_force_heatmap(links)          # the figure is a method, not a separate import
mgr.results                              # every result computed so far
```

For pre-computed enrichment CSVs (the paper workflow, formerly
`EnrichmentManager.batch_from_config`):

```python
dfs = config.load_enrichment('BLIMP1', p_max=0.05)   # list, one frame per episode
```
