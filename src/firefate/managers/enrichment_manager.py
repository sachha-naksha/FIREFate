"""Unified API for TF-activity enrichment across GRN types.

Composes EpisodeDynamics and ORA primitives to provide a single entry point
for episodic, state-specific, and batch enrichment workflows.

Examples
--------
>>> from firefate.managers import EnrichmentManager
>>> mgr = EnrichmentManager(lf_genes=lf_blimp1, dictys_dynamic_object=dyn_obj)
>>> result = mgr.enrich_episodic(episode_idx=1, time_slice=slice(0, 5))
>>> results = mgr.enrich_all_episodes(total_episodes=8, output_folder="./results")
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from dictys.net import dynamic_network


class EnrichmentManager:
    """Unified API for TF-activity enrichment across GRN types.

    Composes EpisodeDynamics and ORA primitives. Does NOT inherit from either.

    Parameters
    ----------
    lf_genes : list[str]
        Cellular program gene set (from SLIDE latent factor).
    dictys_dynamic_object : dynamic_network, optional
        Required for episodic enrichment. Not needed for pre-computed edge tables.
    trajectory_range : tuple[float, float]
        Trajectory endpoints for episodic analysis.
    num_points : int
        Number of sampled pseudotime points.
    dist : float
        Smoothing bandwidth.
    sparsity : float
        Network sparsity threshold.
    """

    def __init__(
        self,
        lf_genes: list[str],
        dictys_dynamic_object: Optional["dynamic_network"] = None,
        trajectory_range: tuple[float, float] = (1, 3),
        num_points: int = 40,
        dist: float = 0.001,
        sparsity: float = 0.01,
    ):
        self.lf_genes = lf_genes
        self._dictys_obj = dictys_dynamic_object
        self._traj_range = trajectory_range
        self._num_points = num_points
        self._dist = dist
        self._sparsity = sparsity

    # ------------------------------------------------------------------ #
    # Episodic enrichment (capability 5 on episodic GRN)                   #
    # ------------------------------------------------------------------ #

    def enrich_episodic(
        self,
        episode_idx: int,
        time_slice: slice,
        percentile: float = 98,
        pval_threshold: float = 0.001,
        n_processes: int = 16,
    ) -> pd.DataFrame:
        """Run full episodic enrichment pipeline for one episode.

        Delegates to EpisodeDynamics internally.

        Parameters
        ----------
        episode_idx : int
            1-based episode index (for logging / file naming only).
        time_slice : slice
            Slice into the sampled pseudotime points that defines the episode.
        percentile : float
            Force-magnitude percentile for edge selection.
        pval_threshold : float
            Edge significance threshold (one-sample t-test).
        n_processes : int
            Parallel workers for edge filtering and force computation.

        Returns
        -------
        pd.DataFrame
            Columns: TF, p_value, enrichment_score, genes_in_lf, genes_dwnstrm, weights.
            Sorted by enrichment_score descending; zero-score rows dropped.
        """
        from firefate.core.episodic_dynamics import EpisodeDynamics

        if self._dictys_obj is None:
            raise ValueError(
                "dictys_dynamic_object required for episodic enrichment. "
                "Pass it to EnrichmentManager() or use enrich_state_specific() instead."
            )

        _ = episode_idx  # reserved for logging / filenames in callers

        epi = EpisodeDynamics(
            dictys_dynamic_object=self._dictys_obj,
            output_folder="",  # not saving intermediate files
            trajectory_range=self._traj_range,
            num_points=self._num_points,
            dist=self._dist,
            sparsity=self._sparsity,
        )
        epi.compute_expression_curves()
        epi.set_lf_genes(self.lf_genes)
        epi.build_episode_grn(time_slice=time_slice)
        epi.filter_edges(pval_threshold=pval_threshold, n_processes=n_processes)
        epi.compute_tf_expression()
        epi.calculate_forces()
        epi.select_top_edges(percentile)
        epi.annotate_lf_in_grn()
        return epi.calculate_enrichment()

    # ------------------------------------------------------------------ #
    # State-specific enrichment (capability 5 on state GRN)                #
    # ------------------------------------------------------------------ #

    def enrich_state_specific(
        self,
        grn_edges: pd.DataFrame,
        force_col: str = "avg_force",
    ) -> pd.DataFrame:
        """Enrich TF activity against a pre-computed state-specific GRN edge table.

        The GRN edge table should have a MultiIndex (TF, Target) and at least
        the column specified by ``force_col``.

        Parameters
        ----------
        grn_edges : pd.DataFrame
            MultiIndex (TF, Target), must have ``force_col`` column.
        force_col : str
            Column name containing edge weights / forces.

        Returns
        -------
        pd.DataFrame
            Same schema as ``enrich_episodic`` output.
        """
        from firefate.core.episodic_dynamics import calculate_tf_episodic_enrichment

        grn_edges = grn_edges.copy()
        if force_col not in grn_edges.columns:
            raise KeyError(
                f"Column {force_col!r} not found in grn_edges. "
                f"Available: {list(grn_edges.columns)}"
            )
        if force_col != "avg_force":
            grn_edges["avg_force"] = grn_edges[force_col]

        grn_edges["is_in_lf"] = grn_edges.index.get_level_values(1).isin(self.lf_genes)

        lf_active = grn_edges[grn_edges["is_in_lf"]].index.get_level_values(1).unique()
        all_targets = grn_edges.index.get_level_values(1).unique()

        enrichment_df = calculate_tf_episodic_enrichment(
            grn_edges,
            total_lf_genes=len(lf_active),
            total_genes_in_grn=len(all_targets),
        )
        enrichment_df = enrichment_df.sort_values(by="enrichment_score", ascending=False)
        enrichment_df = enrichment_df[enrichment_df["enrichment_score"] != 0]
        return enrichment_df

    # ------------------------------------------------------------------ #
    # Batch across all episodes                                            #
    # ------------------------------------------------------------------ #

    def enrich_all_episodes(
        self,
        total_episodes: int = 8,
        points_per_episode: int = 5,
        percentile: float = 98,
        pval_threshold: float = 0.001,
        output_folder: Optional[str] = None,
        n_processes: int = 16,
    ) -> dict[int, pd.DataFrame]:
        """Run enrichment for all episodes sequentially.

        Parameters
        ----------
        total_episodes : int
            Number of episodes to partition pseudotime into.
        points_per_episode : int
            Number of sampled points per episode window.
        percentile : float
            Force-magnitude percentile for edge selection.
        pval_threshold : float
            Edge significance threshold.
        output_folder : str, optional
            If provided, saves each episode result as CSV.
        n_processes : int
            Parallel workers per episode.

        Returns
        -------
        dict[int, pd.DataFrame]
            Mapping episode_idx → enrichment DataFrame.
        """
        results: dict[int, pd.DataFrame] = {}

        for ep_idx in range(1, total_episodes + 1):
            start = (ep_idx - 1) * points_per_episode
            end = ep_idx * points_per_episode

            enrichment_df = self.enrich_episodic(
                episode_idx=ep_idx,
                time_slice=slice(start, end),
                percentile=percentile,
                pval_threshold=pval_threshold,
                n_processes=n_processes,
            )
            results[ep_idx] = enrichment_df

            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                enrichment_df.to_csv(
                    os.path.join(output_folder, f"enrichment_episode_{ep_idx}.csv"),
                    index=False,
                )

        return results

    # ------------------------------------------------------------------ #
    # Load from config paths (paper workflow)                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def batch_from_config(
        config_paths: dict[str, str],
        p_value_threshold: float = 0.05,
    ) -> list[pd.DataFrame]:
        """Load pre-computed enrichment CSVs from a Config-style path dict.

        Useful for the paper analysis workflow where enrichment was pre-computed
        on HPC and saved to CSV.

        Parameters
        ----------
        config_paths : dict
            Keys like ``'ep1'``, ``'ep2'``, etc. Values are file paths to CSVs.
            Example: ``config.BLIMP1`` from ``Config`` class.
        p_value_threshold : float
            Filter loaded DataFrames to this significance level.

        Returns
        -------
        list[pd.DataFrame]
            One DataFrame per episode, ordered by key. Empty DataFrames for
            missing files.
        """
        dfs: list[pd.DataFrame] = []
        for key in sorted(config_paths.keys()):
            path = config_paths[key]
            try:
                df = pd.read_csv(path)
                if "p_value" in df.columns:
                    df = df[df["p_value"] < p_value_threshold]
                dfs.append(df)
            except FileNotFoundError:
                dfs.append(pd.DataFrame())
        return dfs
