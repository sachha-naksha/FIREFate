"""FocalFire: Functional and Interpretable Regulatory Encoding of cell Fate decisions.

FocalFire is three modules in one package, each answering a different question about
regulation:

``focalfire.temporal``
    **FocalFireTemporal** -- how does TF regulation change *along* a trajectory?
    Transition-window and episodic GRNs, force waves, regulatory phases
    (capabilities 3, 4, dynamic 5).

``focalfire.state_specific``
    **FocalFireStateSpecific** -- what separates two *fixed* states, and what happens
    if we perturb it? State and cross-state GRNs, enrichment against them
    (capabilities 2, static 5; 1 and 6 to follow).

``focalfire.cross_prediction``
    **FocalFireCrossPrediction** -- do programs learned on one dataset *transfer* to
    stratify another? (capability 7; scaffolded, port in progress).

Each module's entry point is its manager. Supporting namespaces:
:mod:`focalfire.base` (shared contracts and the ORA primitive),
:mod:`focalfire.backends` (dictys and friends), :mod:`focalfire.io`,
:mod:`focalfire.utils`, :mod:`focalfire.cli`.

There is no separate plotting namespace: every figure lives in the module that
owns its subject, so ``focalfire.temporal`` carries the force landscapes and phase
heatmaps and ``focalfire.state_specific`` carries the enrichment bars.

Examples
--------
>>> import focalfire as ff
>>> mgr = ff.TemporalManager(dyn_obj, output_dir="./episodes")
>>> enrichment = mgr.enrich_episode(1, slice(0, 5), lf_genes=lf_blimp1)
"""

__version__ = "0.1.0"
__author__ = "Akanksha Sachan"

from focalfire import (
    base,
    cross_prediction,
    io,
    state_specific,
    temporal,
    utils,
)
from focalfire.base import BaseManager, ora
from focalfire.state_specific import StateSpecificManager
from focalfire.temporal import (
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
