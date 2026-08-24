"""Entry point for **FIREFateTemporal** -- regulation along a trajectory.

``TemporalManager`` composes the module's domain objects and exposes the three
things a caller actually wants: build a GRN at some temporal resolution, enrich a
cellular program against it, and interrogate the result.

Examples
--------
>>> from firefate.temporal import TemporalManager
>>> mgr = TemporalManager(dyn_obj, output_dir="./episodes")
>>> grn = mgr.build_episode(1, slice(0, 5))
>>> enr = mgr.enrich_episode(1, slice(0, 5), lf_genes=lf_blimp1)
>>> waves = mgr.waves(links=enr_links)
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional, Sequence

import pandas as pd

from firefate.base.manager import BaseManager
from firefate.temporal._align import AlignTimeScales
from firefate.temporal._chromatin import SmoothedCurvesChromatin
from firefate.temporal._curves import SmoothedCurvesGRN
from firefate.temporal._episodes import EpisodeDynamics
from firefate.temporal._phases import ForceWavePhases
from firefate.temporal._validation import TFForceValidation
from firefate.temporal._waves import TFForceWaves

if TYPE_CHECKING:  # pragma: no cover
    from dictys.net import dynamic_network


class TemporalManager(BaseManager):
    """Unified API for the Temporal module (capabilities 3, 4 and dynamic 5).

    Every method that builds something records it in :attr:`~BaseManager.results`
    so a batch run stays inspectable afterwards.

    Parameters
    ----------
    dictys_dynamic_object
        The loaded dictys dynamic network the whole module reads from.
    trajectory_range
        Trajectory endpoints delimiting the branch under analysis.
    num_points
        Number of sampled pseudotime points across ``trajectory_range``.
    dist
        Gaussian smoothing bandwidth.
    sparsity
        Network sparsity threshold.
    output_dir
        Where :meth:`build_all_episodes` and :meth:`enrich_all_episodes` write.
        ``None`` keeps results in memory only.
    """

    def __init__(
        self,
        dictys_dynamic_object: Optional["dynamic_network"] = None,
        *,
        trajectory_range: tuple[float, float] = (1, 3),
        num_points: int = 40,
        dist: float = 0.001,
        sparsity: float = 0.01,
        output_dir: str | None = None,
    ):
        super().__init__(output_dir=output_dir)
        self._net = dictys_dynamic_object
        self._traj_range = trajectory_range
        self._num_points = num_points
        self._dist = dist
        self._sparsity = sparsity

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #

    def _require_net(self, what: str) -> "dynamic_network":
        if self._net is None:
            raise ValueError(
                f"{what} needs a dictys dynamic network. Pass "
                f"`dictys_dynamic_object=` to TemporalManager()."
            )
        return self._net

    @property
    def _kwargs(self) -> dict[str, Any]:
        return {
            "trajectory_range": self._traj_range,
            "num_points": self._num_points,
            "dist": self._dist,
            "sparsity": self._sparsity,
        }

    def _episode_dynamics(self) -> EpisodeDynamics:
        return EpisodeDynamics(
            dictys_dynamic_object=self._require_net("Episode construction"),
            output_folder=self._output_dir or "",
            **self._kwargs,
        )

    @staticmethod
    def _episode_slice(episode: int, points_per_episode: int) -> slice:
        """1-based episode index to its slice of the sampled pseudotime points."""
        if episode < 1:
            raise ValueError(f"Episodes are 1-based; got episode={episode}.")
        return slice((episode - 1) * points_per_episode, episode * points_per_episode)

    # ------------------------------------------------------------------ #
    # curves and time alignment                                            #
    # ------------------------------------------------------------------ #

    def curves(self, mode: str = "expression") -> SmoothedCurvesGRN:
        """A :class:`SmoothedCurvesGRN` configured from this manager's settings."""
        return SmoothedCurvesGRN(
            dictys_dynamic_object=self._require_net("Curve smoothing"),
            mode=mode,
            **self._kwargs,
        )

    def time_scales(self) -> AlignTimeScales:
        """Window and sampled-point pseudotimes on a common scale."""
        return AlignTimeScales(
            dictys_dynamic_object=self._require_net("Time alignment"), **self._kwargs
        )

    def chromatin(self, tfs: Sequence[str] | None, base_path: str) -> SmoothedCurvesChromatin:
        """TF binding-score / OCR-count dynamics read from the per-window subsets."""
        return SmoothedCurvesChromatin(tfs=tfs, base_path=base_path)

    # ------------------------------------------------------------------ #
    # capability 3 -- transition-window GRN                                #
    # ------------------------------------------------------------------ #

    def build_transition_window(
        self,
        links: Sequence[tuple[str, str]] | None = None,
        mode: str = "expression",
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.Series]:
        """Beta curves (and forces, when ``links`` is given) over the whole window.

        Returns
        -------
        beta_curves, force_curves, dtime
            ``force_curves`` is ``None`` when ``links`` is omitted, because there is
            no TF-target beta slice to pair with TF expression.
        """
        curves = self.curves(mode=mode)
        if links is None:
            beta_curves, dtime = curves.get_smoothed_curves(mode="regulation")
            return self._store("transition_window", (beta_curves, None, dtime))

        beta_curves, dtime = curves.get_beta_curves(list(links))
        tf_expression_df, _ = curves.get_smoothed_curves(mode="tf_expression")
        tfs = beta_curves.index.get_level_values(0).unique()
        tf_expression = tf_expression_df.loc[tfs].iloc[:, -1]
        force_curves = SmoothedCurvesGRN.calculate_force_curves(beta_curves, tf_expression)
        return self._store("transition_window", (beta_curves, force_curves, dtime))

    # ------------------------------------------------------------------ #
    # capability 4 -- episodic GRN                                         #
    # ------------------------------------------------------------------ #

    def build_episode(
        self,
        episode: int,
        time_slice: slice,
        *,
        percentile: float = 98,
        pval_threshold: float = 0.001,
        n_processes: int = 16,
    ) -> pd.DataFrame:
        """Build the episodic GRN for one episode and return its top edges."""
        epi = self._episode_dynamics()
        epi.compute_expression_curves()
        epi.build_episode_grn(time_slice=time_slice)
        epi.filter_edges(pval_threshold=pval_threshold, n_processes=n_processes)
        epi.compute_tf_expression()
        epi.calculate_forces()
        return self._store(("grn", episode), epi.select_top_edges(percentile))

    def build_all_episodes(
        self,
        total_episodes: int = 8,
        points_per_episode: int = 5,
        *,
        percentile: float = 98,
        pval_threshold: float = 0.001,
        n_processes: int = 16,
        write: bool = False,
    ) -> dict[int, pd.DataFrame]:
        """Build every episodic GRN in sequence.

        With ``write=True`` each episode is also saved as
        ``episode_<i>.parquet`` in :attr:`output_dir`.
        """
        out: dict[int, pd.DataFrame] = {}
        for episode in range(1, total_episodes + 1):
            grn = self.build_episode(
                episode,
                self._episode_slice(episode, points_per_episode),
                percentile=percentile,
                pval_threshold=pval_threshold,
                n_processes=n_processes,
            )
            out[episode] = grn
            if write:
                self.save(grn, f"episode_{episode}.parquet")
        return out

    # ------------------------------------------------------------------ #
    # capability 5 (dynamic) -- episodic enrichment                        #
    # ------------------------------------------------------------------ #

    def enrich_episode(
        self,
        episode: int,
        time_slice: slice,
        lf_genes: Sequence[str],
        *,
        percentile: float = 98,
        pval_threshold: float = 0.001,
        n_processes: int = 16,
    ) -> pd.DataFrame:
        """Enrich a cellular program against one episode's GRN.

        Returns
        -------
        pd.DataFrame
            Columns ``TF, p_value, enrichment_score, genes_in_lf, genes_dwnstrm, weights``.
        """
        epi = self._episode_dynamics()
        epi.compute_expression_curves()
        epi.set_lf_genes(list(lf_genes))
        epi.build_episode_grn(time_slice=time_slice)
        epi.filter_edges(pval_threshold=pval_threshold, n_processes=n_processes)
        epi.compute_tf_expression()
        epi.calculate_forces()
        epi.select_top_edges(percentile)
        epi.annotate_lf_in_grn()
        return self._store(("enrichment", episode), epi.calculate_enrichment())

    def enrich_all_episodes(
        self,
        lf_genes: Sequence[str],
        total_episodes: int = 8,
        points_per_episode: int = 5,
        *,
        percentile: float = 98,
        pval_threshold: float = 0.001,
        n_processes: int = 16,
        write: bool = False,
    ) -> dict[int, pd.DataFrame]:
        """Enrich ``lf_genes`` against every episode in sequence.

        With ``write=True`` each result is saved as
        ``enrichment_episode_<i>.csv`` in :attr:`output_dir`.
        """
        out: dict[int, pd.DataFrame] = {}
        for episode in range(1, total_episodes + 1):
            enr = self.enrich_episode(
                episode,
                self._episode_slice(episode, points_per_episode),
                lf_genes,
                percentile=percentile,
                pval_threshold=pval_threshold,
                n_processes=n_processes,
            )
            out[episode] = enr
            if write:
                self.save(enr, f"enrichment_episode_{episode}.csv")
        return out

    # ------------------------------------------------------------------ #
    # analysis of what was built                                           #
    # ------------------------------------------------------------------ #

    def waves(self, **kwargs: Any) -> TFForceWaves:
        """A :class:`TFForceWaves` on this manager's trajectory.

        Call :meth:`TFForceWaves.compute_forces` on the result to score a link set.
        """
        return TFForceWaves(
            self._require_net("Force waves"), **{**self._kwargs, **kwargs}
        )

    def phases(
        self,
        switch_pseudotimes: Sequence[float],
        waves: TFForceWaves,
        **kwargs: Any,
    ) -> ForceWavePhases:
        """Bin links into regulatory phases by their softmax force-wave peak."""
        return ForceWavePhases(switch_pseudotimes=switch_pseudotimes, waves=waves, **kwargs)

    def validate(
        self,
        waves: TFForceWaves,
        enriched_links: Sequence[tuple[str, str]],
        **kwargs: Any,
    ) -> TFForceValidation:
        """Compare enriched links against a size-matched random null by abs-max force."""
        return TFForceValidation(waves=waves, enriched_links=enriched_links, **kwargs)


# --------------------------------------------------------------------------- #
# Process-level runners                                                        #
#                                                                              #
# These load the dictys object *inside* the worker process, so they stay        #
# module-level functions rather than methods -- a bound method would drag the   #
# whole manager (and its network) through the pickle.                           #
# --------------------------------------------------------------------------- #


def run_episodic_enrichment(
    episode_idx,
    dictys_dynamic_object_path,
    output_folder,
    trajectory_range,
    num_points,
    time_slice_start,
    time_slice_end,
    lf_genes,
    dist=0.001,
    sparsity=0.01,
    percentile=98,
):
    """Build and enrich one episode in a fresh process; returns the CSV path."""
    import dictys

    net = dictys.net.dynamic_network.from_file(dictys_dynamic_object_path)
    mgr = TemporalManager(
        net,
        trajectory_range=trajectory_range,
        num_points=num_points,
        dist=dist,
        sparsity=sparsity,
        output_dir=output_folder,
    )
    enrichment_df = mgr.enrich_episode(
        episode_idx,
        slice(time_slice_start, time_slice_end),
        lf_genes,
        percentile=percentile,
    )
    out_path = os.path.join(output_folder, f"enrichment_episode_{episode_idx}.csv")
    enrichment_df.to_csv(out_path, index=False)
    return out_path


def run_episodic_construction(
    episode_idx,
    dictys_dynamic_object_path,
    output_folder,
    trajectory_range,
    num_points,
    time_slice_start,
    time_slice_end,
    dist=0.001,
    sparsity=0.01,
    percentile=98,
):
    """Build one episodic GRN in a fresh process; returns the parquet path."""
    import dictys

    net = dictys.net.dynamic_network.from_file(dictys_dynamic_object_path)
    epi = EpisodeDynamics(
        dictys_dynamic_object=net,
        output_folder=output_folder,
        mode="expression",
        trajectory_range=trajectory_range,
        num_points=num_points,
        dist=dist,
        sparsity=sparsity,
    )
    epi.compute_expression_curves()
    epi.build_episode_grn(time_slice=slice(time_slice_start, time_slice_end))
    epi.filter_edges()
    epi.compute_tf_expression()
    avg_force = epi.calculate_forces()
    episodic_grn_edges = epi.select_top_edges(percentile)

    out_path = os.path.join(output_folder, f"episode_{episode_idx}.parquet")
    episodic_grn_edges.to_parquet(out_path)
    avg_force.to_parquet(os.path.join(output_folder, f"avg_force_episode_{episode_idx}.parquet"))
    return out_path
