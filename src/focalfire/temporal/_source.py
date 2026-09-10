"""One trajectory segment's source of TF forces (ISSUES.md #23, stage 4)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import pandas as pd
from dictys.net import stat

from focalfire.temporal._curves import SmoothedCurvesGRN
from focalfire.temporal._forces import (
    calculate_force_curves,
    calculate_force_curves_parallel,
)


class TFForceSource:
    """The single place a trajectory segment's TF forces come from.

    A *segment* is one ``trajectory_range`` of a dictys dynamic network sampled
    at ``num_points`` equidistant pseudotime points with Gaussian bandwidth
    ``dist``.  A linear trajectory is one segment; a branched one is one segment
    per branch, combined downstream by :class:`TFForceWaves.ForceSelector`.

    Segments are about trajectory *topology* only.  *Phases* are orthogonal:
    they are set by cell-state composition switches along a segment (a state
    terminating or being depleted, see :meth:`RegulatoryPhases.from_states`),
    and the softmax peak of every prioritised link's force wave is binned into
    one of them.  A linear trajectory with ``N`` such switches therefore has
    ``N + 1`` phases exactly as a branch does.

    Every consumer -- episodic construction (:class:`EpisodeDynamics`), force
    waves / phases (:class:`TFForceWaves`) and validation -- draws from one of
    these, so they share by construction:

    * the sampled points and their pseudotimes (:meth:`dtime`),
    * the regulator expression at those points (:meth:`tf_expression`, cached),
    * the smoothed beta network per ``network_type`` (:meth:`full_beta_curves`,
      cached, and :meth:`beta_curves` for a link subset),
    * the force kernel and its name-based TF alignment
      (:meth:`force_curves_from_beta`, :meth:`force_curves`), and the
      ``attrs["network_type"]`` stamp on every force frame.

    The source has **no default** ``network_type``: each consumer passes its own
    (``"w"`` for episodes, ``"w_in"`` for waves / phases / validation).  What the
    consumers keep for themselves is only their *reduction* over time
    (``_reductions``) and, for episodes, the beta-level invariance filter.
    """

    def __init__(self, dictys_dynamic_object, trajectory_range, num_points=40,
                 dist=0.001, sparsity=0.01):
        self.dictys_dynamic_object = dictys_dynamic_object
        self.trajectory_range = trajectory_range
        self.num_points = num_points
        self.dist = dist
        self.sparsity = sparsity
        self.curves = SmoothedCurvesGRN(
            dictys_dynamic_object,
            trajectory_range=trajectory_range,
            num_points=num_points,
            dist=dist,
            sparsity=sparsity,
        )
        self._tf_expression: Optional[pd.DataFrame] = None
        self._dtime: Optional[pd.Series] = None
        self._full_beta: dict = {}

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _time_columns(n: int) -> list:
        return [f"time_{i}" for i in range(n)]

    def _window(self, frame: pd.DataFrame, time_slice) -> pd.DataFrame:
        """``frame`` restricted to ``time_slice`` with columns relabelled from
        ``time_0`` (the episode convention); the frame itself when no slice."""
        if time_slice is None:
            return frame
        out = frame.iloc[:, time_slice].copy()
        out.columns = self._time_columns(out.shape[1])
        return out

    # ------------------------------------------------------- sampled points

    def dtime(self, time_slice=None) -> pd.Series:
        """Pseudotime of every sampled point (or of the ``time_slice`` of them)."""
        self.tf_expression()
        if time_slice is None:
            return self._dtime
        return self._dtime.iloc[time_slice].reset_index(drop=True)

    def tf_expression(self, time_slice=None) -> pd.DataFrame:
        """Regulator log-CPM at the sampled points: one row per TF, columns
        ``time_0 ...``.  Computed once and cached; ``time_slice`` returns a copy
        restricted to those points."""
        if self._tf_expression is None:
            dy, dx = self.curves.get_smoothed_curves(mode="tf_expression")
            dy = dy.copy()
            dy.columns = self._time_columns(dy.shape[1])
            self._tf_expression, self._dtime = dy, dx
        return self._window(self._tf_expression, time_slice)

    # ------------------------------------------------------------- beta

    def full_beta_curves(self, network_type: str, time_slice=None) -> pd.DataFrame:
        """Every TF -> target beta of the ``network_type`` network at every sampled
        point, rows ``(TF, Target)`` in dictys regulator / gene order, including
        all-zero rows.  Smoothed once per ``network_type`` and cached, so
        consecutive episodes slice one smoothing instead of repeating it."""
        if network_type not in self._full_beta:
            d = self.dictys_dynamic_object
            pts, fsmooth = d.linspace(
                self.trajectory_range[0], self.trajectory_range[1], self.num_points, self.dist
            )
            dnet = fsmooth(stat.net(d, varname=network_type)).compute(pts)  # (n_tf, n_target, n_pts)
            index_to_gene = {idx: name for name, idx in d.ndict.items()}
            target_names = [index_to_gene[idx] for idx in range(dnet.shape[1])]
            tf_names = [index_to_gene[d.nids[0][tf_idx]] for tf_idx in range(dnet.shape[0])]
            index = pd.MultiIndex.from_tuples(
                [(tf, target) for tf in tf_names for target in target_names],
                names=["TF", "Target"],
            )
            frame = pd.DataFrame(
                dnet.reshape(-1, dnet.shape[2]), index=index,
                columns=self._time_columns(dnet.shape[2]),
            )
            frame.attrs["network_type"] = network_type
            self._full_beta[network_type] = frame
        out = self._window(self._full_beta[network_type], time_slice)
        out.attrs["network_type"] = network_type
        return out

    def beta_curves(self, links: Sequence[Tuple[str, str]], network_type: str,
                    time_slice=None) -> Tuple[pd.DataFrame, pd.Series]:
        """``(beta_curves, dtime)`` for ``links`` only (sub-network smoothing via
        :meth:`SmoothedCurvesGRN.get_beta_curves`), optionally restricted to
        ``time_slice``."""
        beta, dtime = self.curves.get_beta_curves(list(links), network_type=network_type)
        beta = self._window(beta, time_slice)
        beta.attrs["network_type"] = network_type
        if time_slice is not None:
            dtime = dtime.iloc[time_slice].reset_index(drop=True)
        return beta, dtime

    # ------------------------------------------------------------ forces

    def force_curves_from_beta(self, beta_curves: pd.DataFrame, network_type: str,
                               time_slice=None, tf_expression: Optional[pd.DataFrame] = None,
                               n_processes: Optional[int] = None, chunk_size: int = 50000,
                               epsilon: float = 1e-10) -> pd.DataFrame:
        """Forces for an already-built beta frame (e.g. the filtered episodic edges).

        Regulator expression defaults to this source's cached curves over
        ``time_slice`` (which must span as many points as ``beta_curves`` has
        columns); pass ``tf_expression`` to supply it explicitly.  With
        ``n_processes`` the chunked parallel kernel is used, otherwise the kernel
        is called directly.  The result is stamped ``attrs["network_type"]``.
        """
        if tf_expression is None:
            tf_expression = self.tf_expression(time_slice)
        if tf_expression.shape[1] != beta_curves.shape[1]:
            raise ValueError(
                f"beta_curves has {beta_curves.shape[1]} time point(s) but the TF "
                f"expression has {tf_expression.shape[1]} (time_slice={time_slice})."
            )
        if list(tf_expression.columns) != list(beta_curves.columns):
            tf_expression = tf_expression.copy()
            tf_expression.columns = beta_curves.columns
        if n_processes is None:
            forces = calculate_force_curves(beta_curves, tf_expression, epsilon=epsilon)
        else:
            forces = calculate_force_curves_parallel(
                beta_curves=beta_curves, tf_expression=tf_expression,
                n_processes=n_processes, chunk_size=chunk_size, epsilon=epsilon,
                save_intermediate=False,
            )
        forces.attrs["network_type"] = network_type
        return forces

    def force_curves(self, links: Sequence[Tuple[str, str]], network_type: str,
                     time_slice=None) -> Tuple[pd.DataFrame, pd.Series]:
        """``(force_curves, dtime)`` for ``links`` on the ``network_type`` network."""
        beta, dtime = self.beta_curves(links, network_type, time_slice)
        return self.force_curves_from_beta(beta, network_type, time_slice=time_slice), dtime
