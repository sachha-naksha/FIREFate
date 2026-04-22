"""Backward-compatible shim; implementation lives in ``firefate.utils.plots``."""

from ensure_firefate_path import ensure

ensure()
from firefate.utils.plots import *  # noqa: F403
