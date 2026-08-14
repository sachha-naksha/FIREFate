"""Backward-compatible shim; implementation lives in ``firefate.core.validation``."""

from ensure_firefate_path import ensure

ensure()
from firefate.core.validation import *  # noqa: F403
