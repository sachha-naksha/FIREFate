"""The behaviour every FocalFire module manager shares.

A manager is the entry point of a FocalFire module. It *composes* domain objects
(``EpisodeDynamics``, ``StateSpecificEnrichment``, ...) rather than inheriting from
them -- see ``references/architecture.md`` §3. What the three modules genuinely
have in common is bookkeeping, and only that lives here:

* an optional output directory, created on first write;
* a registry of named results, so a notebook can run a batch and pick results out
  of it afterwards;
* one save/load pair that picks its format from the file extension.

Anything domain-specific belongs in the subclass.
"""
from __future__ import annotations

import os
from typing import Any, Iterator

import pandas as pd


class BaseManager:
    """Shared bookkeeping for :class:`~focalfire.temporal.TemporalManager` and friends.

    Parameters
    ----------
    output_dir
        Directory results are written to. ``None`` keeps everything in memory;
        :meth:`save` then raises rather than guessing a location.
    """

    def __init__(self, output_dir: str | None = None):
        self._output_dir = output_dir
        self._results: dict[Any, Any] = {}

    # ------------------------------------------------------------------ #
    # result registry                                                      #
    # ------------------------------------------------------------------ #

    @property
    def results(self) -> dict[Any, Any]:
        """Everything computed so far, keyed by whatever the subclass used."""
        return self._results

    def __getitem__(self, key: Any) -> Any:
        if key not in self._results:
            raise KeyError(
                f"No result stored under {key!r}. Available keys: {sorted(self._results)}"
            )
        return self._results[key]

    def __contains__(self, key: Any) -> bool:
        return key in self._results

    def __iter__(self) -> Iterator[Any]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def _store(self, key: Any, value: Any) -> Any:
        self._results[key] = value
        return value

    # ------------------------------------------------------------------ #
    # filesystem                                                           #
    # ------------------------------------------------------------------ #

    @property
    def output_dir(self) -> str | None:
        return self._output_dir

    def path_for(self, filename: str) -> str:
        """Absolute path for ``filename`` inside the output directory, creating it."""
        if self._output_dir is None:
            raise ValueError(
                f"Cannot place {filename!r}: this manager was built without an "
                f"`output_dir`. Pass one to the constructor, or keep results in memory."
            )
        os.makedirs(self._output_dir, exist_ok=True)
        return os.path.join(self._output_dir, filename)

    def save(self, obj: pd.DataFrame, filename: str) -> str:
        """Write ``obj`` into the output directory; format follows the extension.

        ``.csv`` writes without the index, ``.parquet`` keeps it -- matching how
        enrichment tables and GRN edge tables are respectively consumed.
        """
        path = self.path_for(filename)
        if filename.endswith(".csv"):
            obj.to_csv(path, index=False)
        elif filename.endswith(".parquet"):
            obj.to_parquet(path)
        else:
            raise ValueError(
                f"Unsupported extension for {filename!r}; expected '.csv' or '.parquet'."
            )
        return path

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        """Read a table written by :meth:`save`."""
        if str(path).endswith(".csv"):
            return pd.read_csv(path)
        if str(path).endswith(".parquet"):
            return pd.read_parquet(path)
        raise ValueError(
            f"Unsupported extension for {path!r}; expected '.csv' or '.parquet'."
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(output_dir={self._output_dir!r}, "
            f"results={sorted(self._results)!r})"
        )
