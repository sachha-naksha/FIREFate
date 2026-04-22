"""Backward-compatible shim; implementation lives in ``firefate.utils.custom``."""

from ensure_firefate_path import ensure

ensure()
from firefate.utils.custom import *  # noqa: F403
