"""Put ``src/`` on ``sys.path`` when ``firefate`` is not installed (e.g. notebooks)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def ensure() -> None:
    if importlib.util.find_spec("firefate") is not None:
        return
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        src = parent / "src"
        if (src / "firefate" / "__init__.py").is_file():
            path = str(src)
            if path not in sys.path:
                sys.path.insert(0, path)
            return
    raise ModuleNotFoundError(
        "Cannot import firefate. From the FIREFate repo root, run: pip install -e ."
    )
