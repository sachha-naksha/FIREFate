"""Dataset locations, kept out of the package.

The old ``py_scripts/analysis/config.py`` hard-coded absolute HPC paths inside a
Python class, which meant the package could not be installed anywhere else and the
paths silently rotted when a cluster changed (several ``/ocean/...`` entries no
longer resolve). ``DatasetPaths`` keeps the same access pattern -- ``paths.PB['ep1']``,
``paths.OUTPUT_FOLDER`` -- but reads the values from a YAML file that lives next to
the notebooks, not in ``src/``.

The YAML has three sections::

    roots:                       # reusable prefixes, referenced as {name}
      bcell: /work/nvme/.../bcell
    scalars:                     # plain paths
      OUTPUT_FOLDER: "{figures}"
    groups:                      # episode -> path families
      PB:
        template: "{bcell}/outs/.../enrichment_ep{i}_pb.csv"
        episodes: [1, 2, 3, 4]

Every ``{name}`` is substituted from ``roots``; ``{i}`` is the episode number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DatasetPaths:
    """Named paths for one analysis, loaded from YAML.

    Attributes resolve in two ways, matching how the old ``Config`` class was used::

        paths.OUTPUT_FOLDER        # a scalar path
        paths.PB['ep1']            # a group, keyed 'ep1'..'epN'
    """

    scalars: dict[str, str] = field(default_factory=dict)
    groups: dict[str, dict[str, str]] = field(default_factory=dict)
    source: str | None = None

    # ------------------------------------------------------------------ #
    # construction                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetPaths":
        """Read a dataset YAML and expand every template."""
        import yaml

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"No dataset config at {path}. Copy `datasets.yaml` from the "
                f"notebook directory and point it at your own data."
            )
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        roots: dict[str, str] = raw.get("roots", {}) or {}

        # roots may reference each other, so expand them until stable
        for _ in range(len(roots) + 1):
            roots = {k: v.format(**roots) if "{" in v else v for k, v in roots.items()}

        scalars = {k: str(v).format(**roots) for k, v in (raw.get("scalars") or {}).items()}

        groups: dict[str, dict[str, str]] = {}
        for name, spec in (raw.get("groups") or {}).items():
            template = spec["template"]
            episodes = spec.get("episodes")
            if episodes is None:
                n = spec.get("n_episodes")
                if n is None:
                    raise KeyError(
                        f"Group {name!r} needs either 'episodes' or 'n_episodes'."
                    )
                episodes = range(1, int(n) + 1)
            groups[name] = {f"ep{i}": template.format(i=i, **roots) for i in episodes}

        return cls(scalars=scalars, groups=groups, source=str(path))

    # ------------------------------------------------------------------ #
    # access                                                               #
    # ------------------------------------------------------------------ #

    def __getattr__(self, name: str) -> Any:
        # only reached for names not found normally, so dataclass fields are safe
        scalars = self.__dict__.get("scalars", {})
        groups = self.__dict__.get("groups", {})
        if name in scalars:
            return scalars[name]
        if name in groups:
            return groups[name]
        raise AttributeError(
            f"{name!r} is not defined in {self.__dict__.get('source') or 'this DatasetPaths'}. "
            f"Known scalars: {sorted(scalars)}; known groups: {sorted(groups)}."
        )

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *self.scalars, *self.groups})

    # ------------------------------------------------------------------ #
    # loading what the paths point at                                      #
    # ------------------------------------------------------------------ #

    def load_enrichment(
        self,
        group: str,
        *,
        p_max: float | None = 0.05,
        missing_ok: bool = True,
    ) -> list[pd.DataFrame]:
        """Read a group's pre-computed enrichment CSVs, in episode order.

        Replaces ``EnrichmentManager.batch_from_config``. Episodes whose file is
        absent come back as empty frames when ``missing_ok``, so a partial run still
        lines up positionally with the episode index.
        """
        if group not in self.groups:
            raise KeyError(
                f"No path group {group!r}. Known groups: {sorted(self.groups)}."
            )
        paths = self.groups[group]
        out: list[pd.DataFrame] = []
        for key in sorted(paths, key=lambda k: int(k.removeprefix("ep"))):
            try:
                df = pd.read_csv(paths[key])
            except FileNotFoundError:
                if not missing_ok:
                    raise
                out.append(pd.DataFrame())
                continue
            if p_max is not None and "p_value" in df.columns:
                df = df[df["p_value"] < p_max]
            out.append(df)
        return out
