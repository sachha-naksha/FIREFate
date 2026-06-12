"""Backward-compatible shim; implementation lives in ``firefate.core.pseudotime_curves``."""

from ensure_firefate_path import ensure

ensure()
from firefate.core.pseudotime_curves import *  # noqa: F403
