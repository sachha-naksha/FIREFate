"""Backward-compatible shim; implementation lives in ``firefate.core.episodic_dynamics``."""

from ensure_firefate_path import ensure

ensure()
from firefate.core.episodic_dynamics import *  # noqa: F403
