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

A dataset's trajectory topology is declared once with :class:`TrajectorySegments`
(one manager per named segment, a linear trajectory being a single segment):

>>> traj = TrajectorySegments(dyn_obj, {"PB": (0, 2), "GC": (0, 3)}, num_points=100)
>>> traj["PB"].build_episode(1, slice(0, 5))
>>> traj.compare_sets({"State-specific": ss_links, "Episodic": ep_links})
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
from firefate.temporal._source import TFForceSource
from firefate.temporal._states import StateFrequency
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
        self._source: TFForceSource | None = None

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

    def force_source(self) -> TFForceSource:
        """The one :class:`TFForceSource` for this manager's trajectory segment.

        Built on first use from the manager's smoothing settings and shared by
        every episode and every :meth:`waves` object the manager creates, so they
        draw beta, regulator expression and forces from the same cache.
        """
        if self._source is None:
            self._source = TFForceSource(
                self._require_net("Force source"), **self._kwargs
            )
        return self._source

    def _episode_dynamics(self, network_type: str) -> EpisodeDynamics:
        return EpisodeDynamics(
            dictys_dynamic_object=None,
            output_folder=self._output_dir or "",
            network_type=network_type,
            source=self.force_source(),
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
        network_type: str = "w_in",
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.Series]:
        """Beta curves (and forces, when ``links`` is given) over the whole window.

        ``network_type`` selects the dictys network variable for the beta curves
        (default ``"w_in"``, total effect, as for force waves).

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

        beta_curves, dtime = curves.get_beta_curves(list(links), network_type=network_type)
        tf_expression_df, _ = curves.get_smoothed_curves(mode="tf_expression")
        # Regulator expression over the SAME sampled points as the beta curves (one
        # column per point), exactly as TFForceWaves.compute_forces pairs them. The
        # previous `.iloc[:, -1]` handed the kernel a single-column Series, which it
        # rejects (ISSUES.md #24).
        tfs = beta_curves.index.get_level_values(0).unique()
        tf_expression = tf_expression_df.loc[tfs]
        force_curves = SmoothedCurvesGRN.calculate_force_curves(beta_curves, tf_expression)
        force_curves.attrs["network_type"] = network_type
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
        network_type: str = "w",
    ) -> pd.DataFrame:
        """Build the episodic GRN for one episode and return its top edges.

        ``network_type`` selects the dictys network variable for the episodic beta
        curves (default ``"w"``, direct effect; see :class:`EpisodeDynamics`). It is
        stamped on the result as ``.attrs["network_type"]``.
        """
        epi = self._episode_dynamics(network_type)
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
        network_type: str = "w",
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
                network_type=network_type,
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
        network_type: str = "w",
    ) -> pd.DataFrame:
        """Enrich a cellular program against one episode's GRN.

        ``network_type`` is passed to the episodic construction (default ``"w"``,
        direct effect; see :meth:`build_episode`).

        Returns
        -------
        pd.DataFrame
            Columns ``TF, p_value, enrichment_score, genes_in_lf, genes_dwnstrm, weights``.
        """
        epi = self._episode_dynamics(network_type)
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
        network_type: str = "w",
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
                network_type=network_type,
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

        Without overrides it shares the manager's :meth:`force_source`; smoothing
        overrides in ``kwargs`` build a standalone branch instead. Call
        :meth:`TFForceWaves.compute_forces` on the result to score a link set.
        """
        if kwargs:
            return TFForceWaves(
                self._require_net("Force waves"), **{**self._kwargs, **kwargs}
            )
        return TFForceWaves(source=self.force_source())

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


class TrajectorySegments:
    """A dataset's trajectory topology, declared once as named segments.

    Each segment is a ``trajectory_range`` on the shared dictys network and gets
    its own :class:`TemporalManager` (hence its own :class:`TFForceSource`); the
    smoothing settings are shared.  A linear trajectory is one segment; a
    branched one is one segment per branch.  Everything that spans segments --
    the force selector, validation, the multi-set comparisons -- is derived here,
    so segment names and ranges are typed exactly once.

    Segments are topology only.  *Phases* are orthogonal: they are cell-state
    composition switches along a segment (:meth:`state_frequency` ->
    ``termination_pseudotime``) that the softmax peaks of prioritised links are
    binned into (:meth:`phases`), on a linear trajectory just as on a branch.

    Parameters
    ----------
    dictys_dynamic_object
        The loaded dictys dynamic network every segment reads from.
    segments
        ``{name: (start_node, end_node)}`` trajectory endpoints per segment, e.g.
        ``{"PB": (0, 2), "GC": (0, 3)}`` or ``{"linear": (0, 1)}``.
    num_points, dist, sparsity
        Smoothing settings shared by every segment.
    output_dir
        Root for per-segment outputs; segment ``name`` writes under
        ``output_dir/name``.  ``None`` keeps results in memory only.
    """

    def __init__(
        self,
        dictys_dynamic_object: "dynamic_network",
        segments: dict[str, tuple[float, float]],
        *,
        num_points: int = 40,
        dist: float = 0.001,
        sparsity: float = 0.01,
        output_dir: str | None = None,
    ):
        if not segments:
            raise ValueError("TrajectorySegments needs at least one segment.")
        self.dictys_dynamic_object = dictys_dynamic_object
        self.segments = {name: tuple(rng) for name, rng in segments.items()}
        self._managers = {
            name: TemporalManager(
                dictys_dynamic_object,
                trajectory_range=rng,
                num_points=num_points,
                dist=dist,
                sparsity=sparsity,
                output_dir=os.path.join(output_dir, name) if output_dir else None,
            )
            for name, rng in self.segments.items()
        }
        self._waves: dict[str, TFForceWaves] | None = None

    # ------------------------------------------------------------------ #
    # per-segment access                                                   #
    # ------------------------------------------------------------------ #

    @property
    def names(self) -> list[str]:
        return list(self.segments)

    def __getitem__(self, name: str) -> TemporalManager:
        """The segment's manager: episodes, curves, phases and validation on it."""
        if name not in self._managers:
            raise KeyError(f"No segment {name!r}. Segments: {self.names}")
        return self._managers[name]

    def __len__(self) -> int:
        return len(self.segments)

    def sources(self) -> dict[str, TFForceSource]:
        """``{name: TFForceSource}``, one per segment."""
        return {name: mgr.force_source() for name, mgr in self._managers.items()}

    def waves(self) -> dict[str, TFForceWaves]:
        """``{name: TFForceWaves}``, one per segment on its source (built once, so
        ``compute_forces`` results stay cached across the cross-segment methods)."""
        if self._waves is None:
            self._waves = {name: mgr.waves() for name, mgr in self._managers.items()}
        return self._waves

    # ------------------------------------------------------------------ #
    # cross-segment: forces, validation                                    #
    # ------------------------------------------------------------------ #

    def selector(self, network_type: str = "w_in") -> TFForceWaves.ForceSelector:
        """A :class:`TFForceWaves.ForceSelector` over every segment."""
        return TFForceWaves.ForceSelector(self.waves(), network_type=network_type)

    def validate(
        self,
        enriched_links: Sequence[tuple[str, str]],
        network_type: str = "w_in",
        mode: str | None = None,
        **kwargs: Any,
    ) -> TFForceValidation:
        """Enriched-vs-random validation by abs-max force across the segments.

        ``mode`` defaults to ``'combined'`` (each link on its stronger segment)
        with several segments and ``'lineage'`` with one.
        """
        if mode is None:
            mode = "combined" if len(self) > 1 else "lineage"
        return TFForceValidation(
            self.selector(network_type), enriched_links,
            network_type=network_type, mode=mode, **kwargs,
        )

    def compare_sets(
        self,
        enriched_sets: dict[str, Sequence[tuple[str, str]]],
        network_type: str = "w_in",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Several enriched sets vs one shared random null across the segments
        (:meth:`TFForceValidation.compare_sets`)."""
        return TFForceValidation.compare_sets(
            self.selector(network_type), enriched_sets,
            network_type=network_type, **kwargs,
        )

    def compare_sets_by_phase(
        self,
        enriched_sets: dict[str, Sequence[tuple[str, str]]],
        switch_pseudotimes: dict[str, Sequence[float]],
        network_type: str = "w_in",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """The per-(segment, phase) form of :meth:`compare_sets`
        (:meth:`TFForceValidation.compare_sets_by_phase`); ``switch_pseudotimes``
        is ``{segment name: phase boundaries}``."""
        unknown = set(switch_pseudotimes) - set(self.segments)
        if unknown:
            raise KeyError(f"switch_pseudotimes names unknown segment(s) {sorted(unknown)}. "
                           f"Segments: {self.names}")
        return TFForceValidation.compare_sets_by_phase(
            self.selector(network_type), enriched_sets, switch_pseudotimes,
            network_type=network_type, **kwargs,
        )

    # ------------------------------------------------------------------ #
    # phases along one segment                                             #
    # ------------------------------------------------------------------ #

    def state_frequency(self, name: str, cell_labels: Any, **kwargs: Any) -> StateFrequency:
        """Cell-state composition along segment ``name`` (its range is filled in);
        its ``termination_pseudotime`` values are the phase boundaries."""
        return StateFrequency(
            self.dictys_dynamic_object, cell_labels,
            trajectory_range=self[name]._traj_range, **kwargs,
        )

    def phases(
        self,
        name: str,
        switch_pseudotimes: Sequence[float],
        **kwargs: Any,
    ) -> ForceWavePhases:
        """Bin links into phases of segment ``name`` by their softmax force-wave
        peak, on that segment's :meth:`waves` object."""
        return self[name].phases(switch_pseudotimes, self.waves()[name], **kwargs)


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
    network_type="w",
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
        network_type=network_type,
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
    network_type="w",
):
    """Build one episodic GRN in a fresh process; returns the parquet path.

    ``network_type`` (default ``"w"``) is stamped on both parquet files as
    ``.attrs["network_type"]`` (read back by ``pandas.read_parquet``).
    """
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
        network_type=network_type,
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
