"""Entry point for **FIREFateStateSpecific** -- regulation of a fixed cell state.

``StateSpecificManager`` runs the same over-representation primitive as the
Temporal module (:func:`firefate.base.enrichment.calculate_tf_episodic_enrichment`)
but against a pre-computed state GRN edge table rather than an episode.

Examples
--------
>>> from firefate.state_specific import StateSpecificManager
>>> mgr = StateSpecificManager(lf_genes=lf_blimp1, output_dir="./state")
>>> enr = mgr.enrich(grn_edges)
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from firefate.base.enrichment import calculate_tf_episodic_enrichment
from firefate.base.manager import BaseManager


class StateSpecificManager(BaseManager):
    """Unified API for the StateSpecific module (capabilities 2 and static 5).

    Parameters
    ----------
    lf_genes
        The cellular-program gene set (a SLIDE latent factor) to enrich for.
    output_dir
        Where :meth:`enrich` writes when asked to. ``None`` keeps results in memory.
    """

    def __init__(self, lf_genes: Sequence[str], *, output_dir: str | None = None):
        super().__init__(output_dir=output_dir)
        self.lf_genes = list(lf_genes)

    def enrich(
        self,
        grn_edges: pd.DataFrame,
        *,
        force_col: str = "avg_force",
        key: str = "state",
        write: bool = False,
    ) -> pd.DataFrame:
        """Enrich :attr:`lf_genes` against a state-specific GRN edge table.

        Parameters
        ----------
        grn_edges
            MultiIndex ``(TF, Target)``, carrying at least ``force_col``.
        force_col
            Column holding the edge weight. Renamed to ``avg_force`` internally,
            which is the column name the shared ORA primitive reads.
        key
            Name this result is registered under.
        write
            Also save as ``enrichment_<key>.csv`` in :attr:`output_dir`.

        Returns
        -------
        pd.DataFrame
            Same schema as the Temporal module's episodic enrichment, sorted by
            ``enrichment_score`` with zero-score rows dropped.
        """
        if force_col not in grn_edges.columns:
            raise KeyError(
                f"Column {force_col!r} not found in grn_edges. "
                f"Available: {list(grn_edges.columns)}"
            )

        edges = grn_edges.copy()
        if force_col != "avg_force":
            edges["avg_force"] = edges[force_col]
        edges["is_in_lf"] = edges.index.get_level_values(1).isin(self.lf_genes)

        lf_active = edges[edges["is_in_lf"]].index.get_level_values(1).unique()
        all_targets = edges.index.get_level_values(1).unique()

        enrichment_df = calculate_tf_episodic_enrichment(
            edges,
            total_lf_genes=len(lf_active),
            total_genes_in_grn=len(all_targets),
        )
        enrichment_df = enrichment_df.sort_values(by="enrichment_score", ascending=False)
        enrichment_df = enrichment_df[enrichment_df["enrichment_score"] != 0]

        self._store(key, enrichment_df)
        if write:
            self.save(enrichment_df, f"enrichment_{key}.csv")
        return enrichment_df
