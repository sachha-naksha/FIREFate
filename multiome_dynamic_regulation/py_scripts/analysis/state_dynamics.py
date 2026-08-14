"""Backward-compatible shim; implementation lives in ``firefate.core.state_dynamics``."""

from ensure_firefate_path import ensure

ensure()
from firefate.core.state_dynamics import *  # noqa: F403
