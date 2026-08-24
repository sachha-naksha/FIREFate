"""FIREFate: Functional and Interpretable Regulatory Encoding of cell Fate decisions.

FIREFate is three modules in one package, each answering a different question about
regulation:

``firefate.temporal``
    **FIREFateTemporal** -- how does TF regulation change *along* a trajectory?
    Transition-window and episodic GRNs, force waves, regulatory phases
    (capabilities 3, 4, dynamic 5).

``firefate.state_specific``
    **FIREFateStateSpecific** -- what separates two *fixed* states, and what happens
    if we perturb it? State and cross-state GRNs, enrichment against them
    (capabilities 2, static 5; 1 and 6 to follow).

``firefate.cross_prediction``
    **FIREFateCrossPrediction** -- do programs learned on one dataset *transfer* to
    stratify another? (capability 7; scaffolded, port in progress).

Each module's entry point is its manager. Supporting namespaces:
:mod:`firefate.base` (shared contracts and the ORA primitive),
:mod:`firefate.backends` (dictys and friends), :mod:`firefate.io`,
:mod:`firefate.utils`, :mod:`firefate.cli`.

There is no separate plotting namespace: every figure lives in the module that
owns its subject, so ``firefate.temporal`` carries the force landscapes and phase
heatmaps and ``firefate.state_specific`` carries the enrichment bars.

Examples
--------
>>> import firefate as ff
>>> mgr = ff.TemporalManager(dyn_obj, output_dir="./episodes")
>>> enrichment = mgr.enrich_episode(1, slice(0, 5), lf_genes=lf_blimp1)
"""

__version__ = "0.1.0"
__author__ = "Akanksha Sachan"

from firefate import (
    base,
    cross_prediction,
    io,
    state_specific,
    temporal,
    utils,
)
from firefate.base import BaseManager, ora
from firefate.state_specific import StateSpecificManager
from firefate.temporal import (
    AlignTimeScales,
    EpisodeDynamics,
    SmoothedCurvesChromatin,
    SmoothedCurvesGRN,
    StateFrequency,
    TemporalManager,
    TFForceValidation,
    TFForceWaves,
)

__all__ = [
    "AlignTimeScales",
    "BaseManager",
    "EpisodeDynamics",
    "SmoothedCurvesChromatin",
    "SmoothedCurvesGRN",
    "StateFrequency",
    "StateSpecificManager",
    "TFForceValidation",
    "TFForceWaves",
    "TemporalManager",
    "base",
    "cross_prediction",
    "io",
    "ora",
    "state_specific",
    "temporal",
    "utils",
]
