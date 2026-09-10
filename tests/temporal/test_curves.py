"""``SmoothedCurvesGRN`` against a real (miniature) dictys dynamic network.

The mock network in ``conftest.py`` is built so that several quantities are
known exactly:

* genes with a constant CPM over all windows must keep their exact
  ``log2(CPM + 1)`` value at every pseudotime point, because the Gaussian
  smoothing weights are normalised to sum to one;
* the same holds for GRN edges with a constant per-window weight;
* pseudotime along a branch is exactly ``linspace(0, path_length, num_points)``.

The hand-rolled fast paths (``_regulation_curves_parallel``,
``_subnetwork_curves``) are additionally checked bit-for-bit against the stock
dictys stat chain they claim to reproduce.
"""

import numpy as np
import pandas as pd
import pytest
from dictys.net import stat

from firefate.temporal import SmoothedCurvesGRN
from conftest import (
    CONSTANT_LCPM,
    GENES,
    MOCK_EDGES,
    REGULATORS,
    VARNAME_SCALE,
)

NUM_POINTS = 10
SPARSITY = 0.1
DIST = 0.3


@pytest.fixture
def curves(mock_network):
    return SmoothedCurvesGRN(
        mock_network,
        trajectory_range=(0, 2),
        num_points=NUM_POINTS,
        dist=DIST,
        sparsity=SPARSITY,
    )


def reference_smoothed_net(network, trajectory_range=(0, 2), varname="w",
                           num_points=NUM_POINTS, dist=DIST):
    """Smoothed GRN straight from dictys, used as the ground truth."""
    pts, fsmooth = network.linspace(*trajectory_range, num_points, dist)
    return fsmooth(stat.net(network, varname=varname)).compute(pts)


# --------------------------------------------------------------------------- #
# construction                                                                  #
# --------------------------------------------------------------------------- #

class TestConstruction:
    def test_stores_parameters(self, mock_network):
        c = SmoothedCurvesGRN(mock_network, (1, 3), num_points=7, dist=0.5,
                              sparsity=0.02, mode="regulation")
        assert c.trajectory_range == (1, 3)
        assert c.num_points == 7
        assert c.dist == 0.5
        assert c.sparsity == 0.02
        assert c.mode == "regulation"

    def test_defaults(self, mock_network):
        c = SmoothedCurvesGRN(mock_network, (0, 2))
        assert (c.num_points, c.dist, c.sparsity, c.mode) == (40, 0.001, 0.01, "expression")


# --------------------------------------------------------------------------- #
# get_smoothed_curves                                                           #
# --------------------------------------------------------------------------- #

class TestSmoothedCurves:
    def test_expression_shape_and_labels(self, curves):
        dy, dx = curves.get_smoothed_curves(mode="expression")
        assert dy.shape == (len(GENES), NUM_POINTS)
        assert list(dy.index) == list(GENES)
        assert len(dx) == NUM_POINTS

    def test_pseudotime_is_evenly_spaced_along_the_branch(self, curves):
        _, dx = curves.get_smoothed_curves(mode="expression")
        # branch (0, 2) spans two unit-length trajectory edges
        assert dx.values == pytest.approx(np.linspace(0.0, 2.0, NUM_POINTS))

    @pytest.mark.parametrize("traj_range,length", [((0, 2), 2.0), ((0, 3), 2.0), ((1, 3), 1.0)])
    def test_pseudotime_for_each_branch(self, mock_network, traj_range, length):
        c = SmoothedCurvesGRN(mock_network, traj_range, num_points=6, dist=DIST)
        _, dx = c.get_smoothed_curves(mode="expression")
        assert dx.values == pytest.approx(np.linspace(0.0, length, 6))

    @pytest.mark.parametrize("gene,expected", sorted(CONSTANT_LCPM.items()))
    def test_constant_expression_genes_are_exact_at_every_point(self, curves, gene, expected):
        dy, _ = curves.get_smoothed_curves(mode="expression")
        assert dy.loc[gene].values == pytest.approx(np.full(NUM_POINTS, expected), abs=1e-12)

    def test_expression_matches_dictys_lcpm(self, curves, mock_network):
        dy, _ = curves.get_smoothed_curves(mode="expression")
        pts, fsmooth = mock_network.linspace(0, 2, NUM_POINTS, DIST)
        ref = fsmooth(stat.lcpm(mock_network, cut=0)).compute(pts)
        assert dy.values == pytest.approx(ref)

    def test_monotone_gene_curve_is_monotone(self, curves):
        # G2's CPM rises monotonically over the windows; smoothing with positive
        # weights cannot turn that into a non-monotone curve.
        dy, _ = curves.get_smoothed_curves(mode="expression")
        g2 = dy.loc["G2"].values
        assert np.all(np.diff(g2) > 0)
        assert np.all(np.diff(dy.loc["G3"].values) < 0)

    def test_tf_expression_mode_restricts_rows_to_regulators(self, curves):
        dy, _ = curves.get_smoothed_curves(mode="tf_expression")
        assert list(dy.index) == REGULATORS
        for tf in set(REGULATORS) & set(CONSTANT_LCPM):
            assert dy.loc[tf].values == pytest.approx(
                np.full(NUM_POINTS, CONSTANT_LCPM[tf]), abs=1e-12
            )

    def test_tf_expression_agrees_with_full_expression(self, curves):
        full, _ = curves.get_smoothed_curves(mode="expression")
        tfs, _ = curves.get_smoothed_curves(mode="tf_expression")
        assert tfs.values == pytest.approx(full.loc[REGULATORS].values)

    def test_weighted_regulation_matches_dictys(self, curves, mock_network):
        dy, _ = curves.get_smoothed_curves(mode="weighted_regulation")
        pts, fsmooth = mock_network.linspace(0, 2, NUM_POINTS, DIST)
        ref = stat.flnneighbor(
            fsmooth(stat.net(mock_network)), weighted_sparsity=SPARSITY
        ).compute(pts)
        assert dy.values == pytest.approx(ref)

    def test_mode_argument_overrides_instance_mode(self, mock_network):
        c = SmoothedCurvesGRN(mock_network, (0, 2), num_points=NUM_POINTS, dist=DIST,
                              sparsity=SPARSITY, mode="expression")
        dy, _ = c.get_smoothed_curves(mode="tf_expression")
        assert list(dy.index) == REGULATORS

    def test_instance_mode_is_used_when_no_argument_given(self, mock_network):
        c = SmoothedCurvesGRN(mock_network, (0, 2), num_points=NUM_POINTS, dist=DIST,
                              sparsity=SPARSITY, mode="tf_expression")
        dy, _ = c.get_smoothed_curves()
        assert list(dy.index) == REGULATORS

    def test_unknown_mode_raises(self, curves):
        with pytest.raises(ValueError, match="Unknown mode"):
            curves.get_smoothed_curves(mode="not-a-mode")

    def test_returns_curves_first_pseudotime_second(self, curves):
        dy, dx = curves.get_smoothed_curves(mode="expression")
        assert isinstance(dy, pd.DataFrame) and isinstance(dx, pd.Series)


# --------------------------------------------------------------------------- #
# _regulation_curves_parallel                                                   #
# --------------------------------------------------------------------------- #

class TestRegulationCurves:
    def reference_chain(self, network, sparsity=SPARSITY, traj_range=(0, 2)):
        """The stock dictys chain the fast path claims to reproduce."""
        pts, fsmooth = network.linspace(*traj_range, NUM_POINTS, DIST)
        return stat.flnneighbor(
            stat.fbinarize(fsmooth(stat.net(network)), sparsity=sparsity)
        ).compute(pts)

    def test_matches_the_dictys_chain_exactly(self, curves, mock_network):
        dy, _ = curves.get_smoothed_curves(mode="regulation")
        assert dy.values == pytest.approx(self.reference_chain(mock_network), abs=0.0, rel=0.0)

    @pytest.mark.parametrize("sparsity", [0.05, 0.1, 0.25, 0.5])
    def test_matches_the_dictys_chain_for_several_sparsities(self, mock_network, sparsity):
        c = SmoothedCurvesGRN(mock_network, (0, 2), num_points=NUM_POINTS, dist=DIST,
                              sparsity=sparsity)
        dy, _ = c.get_smoothed_curves(mode="regulation")
        assert dy.values == pytest.approx(self.reference_chain(mock_network, sparsity))

    @pytest.mark.parametrize("traj_range", [(0, 2), (0, 3), (1, 3)])
    def test_matches_the_dictys_chain_on_every_branch(self, mock_network, traj_range):
        c = SmoothedCurvesGRN(mock_network, traj_range, num_points=NUM_POINTS, dist=DIST,
                              sparsity=SPARSITY)
        dy, _ = c.get_smoothed_curves(mode="regulation")
        assert dy.values == pytest.approx(
            self.reference_chain(mock_network, traj_range=traj_range)
        )

    def test_nan_aware_path_matches_dictys(self, mock_network_nan):
        # The NaN branch of the hand-written smoothing must agree with dictys'
        # nan='ignore' semantics.
        c = SmoothedCurvesGRN(mock_network_nan, (0, 2), num_points=NUM_POINTS, dist=DIST,
                              sparsity=SPARSITY)
        dy, _ = c.get_smoothed_curves(mode="regulation")
        assert dy.values == pytest.approx(self.reference_chain(mock_network_nan))

    @pytest.mark.parametrize("n_jobs", [1, 2, 3, 8])
    def test_result_is_independent_of_n_jobs(self, curves, n_jobs):
        pts, fsmooth = curves.dictys_dynamic_object.linspace(0, 2, NUM_POINTS, DIST)
        base = curves._regulation_curves_parallel(pts, fsmooth, n_jobs=1)
        got = curves._regulation_curves_parallel(pts, fsmooth, n_jobs=n_jobs)
        assert got.values == pytest.approx(base.values)
        assert list(got.index) == list(base.index)

    def test_more_jobs_than_points_is_safe(self, curves):
        pts, fsmooth = curves.dictys_dynamic_object.linspace(0, 2, NUM_POINTS, DIST)
        got = curves._regulation_curves_parallel(pts, fsmooth, n_jobs=NUM_POINTS * 4)
        assert got.shape == (len(REGULATORS), NUM_POINTS)

    def test_rows_are_regulators_and_values_are_log2_of_a_count(self, curves):
        dy, _ = curves.get_smoothed_curves(mode="regulation")
        assert list(dy.index) == REGULATORS
        counts = 2.0 ** dy.values - 1.0
        assert counts == pytest.approx(np.round(counts))
        assert (counts >= 0).all()
        assert (counts <= len(GENES)).all()

    def test_number_of_retained_edges_matches_the_sparsity(self, curves):
        # k = int(sparsity * n_reg * n_target) edges are kept per point (more
        # only when there are ties at the cutoff).
        dy, _ = curves.get_smoothed_curves(mode="regulation")
        k = int(SPARSITY * len(REGULATORS) * len(GENES))
        kept = (2.0 ** dy.values - 1.0).sum(axis=0)
        assert (kept >= k).all()

    def test_strongest_regulator_is_ranked_top(self, curves):
        # ZNF1 -> G1 has |w| = 5, the largest edge anywhere in the mock network,
        # so ZNF1 must have a non-zero outdegree at every point.
        dy, _ = curves.get_smoothed_curves(mode="regulation")
        assert (dy.loc["ZNF1"].values > 0).all()

    def test_sparsity_below_one_edge_degenerates_like_dictys(self, mock_network):
        # k == 0 makes np.partition(...)[-0] pick an arbitrary element; the fast
        # path reproduces dictys' behaviour rather than guarding against it.
        c = SmoothedCurvesGRN(mock_network, (0, 2), num_points=NUM_POINTS, dist=DIST,
                              sparsity=1e-6)
        dy, _ = c.get_smoothed_curves(mode="regulation")
        assert dy.values == pytest.approx(self.reference_chain(mock_network, sparsity=1e-6))


# --------------------------------------------------------------------------- #
# _subnetwork_curves                                                            #
# --------------------------------------------------------------------------- #

class TestSubnetworkCurves:
    @pytest.mark.parametrize("varname", sorted(VARNAME_SCALE))
    def test_matches_slicing_the_fully_smoothed_network(self, curves, mock_network, varname):
        tf_idx = [0, 1]                     # TFA, TFB
        target_idx = [3, 4, 6]              # G1, G2, G4
        sub, dtime = curves._subnetwork_curves(tf_idx, target_idx, varname)
        ref = reference_smoothed_net(mock_network, varname=varname)
        assert sub == pytest.approx(ref[np.ix_(tf_idx, target_idx)])
        assert sub.shape == (2, 3, NUM_POINTS)
        assert dtime.values == pytest.approx(np.linspace(0.0, 2.0, NUM_POINTS))

    def test_nan_aware_path_matches_dictys(self, mock_network_nan):
        c = SmoothedCurvesGRN(mock_network_nan, (0, 2), num_points=NUM_POINTS, dist=DIST,
                              sparsity=SPARSITY)
        sub, _ = c._subnetwork_curves([0], [3], "w")
        ref = reference_smoothed_net(mock_network_nan)
        assert sub[0, 0] == pytest.approx(ref[0, 3])

    def test_single_tf_single_target(self, curves, mock_network):
        sub, _ = curves._subnetwork_curves([0], [3], "w")
        assert sub.shape == (1, 1, NUM_POINTS)
        # TFA -> G1 is constant 2.0 in every window
        assert sub[0, 0] == pytest.approx(np.full(NUM_POINTS, 2.0))

    def test_order_of_requested_indices_is_preserved(self, curves, mock_network):
        ref = reference_smoothed_net(mock_network)
        sub, _ = curves._subnetwork_curves([1, 0], [4, 3], "w")
        assert sub[0, 0] == pytest.approx(ref[1, 4])
        assert sub[1, 1] == pytest.approx(ref[0, 3])


# --------------------------------------------------------------------------- #
# get_beta_curves                                                               #
# --------------------------------------------------------------------------- #

class TestBetaCurves:
    def test_values_match_the_smoothed_network(self, curves, mock_network):
        beta, dtime = curves.get_beta_curves([("TFA", "G1"), ("TFB", "G4")], varname="w")
        ref = reference_smoothed_net(mock_network, varname="w")
        tf_pos = {name: i for i, name in enumerate(REGULATORS)}
        gene_pos = {name: i for i, name in enumerate(GENES)}
        for tf, target in beta.index:
            assert beta.loc[(tf, target)].values == pytest.approx(
                ref[tf_pos[tf], gene_pos[target]]
            ), f"{tf}->{target}"
        assert dtime.values == pytest.approx(np.linspace(0.0, 2.0, NUM_POINTS))

    def test_constant_edge_is_exactly_constant(self, curves):
        beta, _ = curves.get_beta_curves([("TFA", "G1")], varname="w")
        assert beta.loc[("TFA", "G1")].values == pytest.approx(
            np.full(NUM_POINTS, MOCK_EDGES[("TFA", "G1")][0])
        )

    @pytest.mark.parametrize("varname,scale", sorted(VARNAME_SCALE.items()))
    def test_varname_selects_the_network_variant(self, curves, varname, scale):
        beta, _ = curves.get_beta_curves([("TFA", "G1")], varname=varname)
        assert beta.loc[("TFA", "G1")].values == pytest.approx(np.full(NUM_POINTS, 2.0 * scale))

    def test_default_varname_is_w_in(self, curves):
        default, _ = curves.get_beta_curves([("TFA", "G1")])
        explicit, _ = curves.get_beta_curves([("TFA", "G1")], varname="w_in")
        assert default.values == pytest.approx(explicit.values)

    def test_columns_are_time_labels(self, curves):
        beta, _ = curves.get_beta_curves([("TFA", "G1")])
        assert list(beta.columns) == [f"time_{i}" for i in range(NUM_POINTS)]
        assert beta.index.names == ["TF", "Target"]

    def test_returns_the_full_tf_by_target_cross_product(self, curves):
        # Documented as "beta curves for specified links", but the TFs and the
        # targets are de-duplicated separately and then crossed, so asking for
        # two links returns four rows.
        beta, _ = curves.get_beta_curves([("TFA", "G1"), ("TFB", "G4")])
        assert set(beta.index) == {("TFA", "G1"), ("TFA", "G4"), ("TFB", "G1"), ("TFB", "G4")}

    def test_missing_tfs_and_targets_are_skipped(self, curves, capsys):
        beta, _ = curves.get_beta_curves(
            [("TFA", "G1"), ("NOT_A_TF", "G1"), ("TFA", "NOT_A_GENE")]
        )
        assert set(beta.index) == {("TFA", "G1")}
        assert "skipping 1 TF(s) and 1 target(s)" in capsys.readouterr().out

    def test_non_regulator_gene_counts_as_a_missing_tf(self, curves, capsys):
        # G1 is in the network but is not in nids[0], so it cannot act as a TF.
        beta, _ = curves.get_beta_curves([("G1", "G2"), ("TFA", "G2")])
        assert set(beta.index) == {("TFA", "G2")}
        assert "skipping 1 TF(s)" in capsys.readouterr().out

    def test_all_tfs_missing_yields_empty_frame(self, curves):
        beta, _ = curves.get_beta_curves([("NOPE", "G1")])
        assert len(beta) == 0

    def test_index_stays_aligned_with_the_data_when_entries_are_missing(self, curves,
                                                                       mock_network):
        # Regression guard: the row labels must still describe the rows once
        # missing TFs/targets have been dropped.
        ref = reference_smoothed_net(mock_network, varname="w")
        tf_pos = {name: i for i, name in enumerate(REGULATORS)}
        gene_pos = {name: i for i, name in enumerate(GENES)}
        beta, _ = curves.get_beta_curves(
            [("TFA", "G1"), ("NOT_A_TF", "G2"), ("TFB", "NOT_A_GENE"), ("TFB", "G4")],
            varname="w",
        )
        for tf, target in beta.index:
            assert beta.loc[(tf, target)].values == pytest.approx(
                ref[tf_pos[tf], gene_pos[target]]
            ), f"{tf}->{target}"

    def test_duplicate_links_are_collapsed(self, curves):
        beta, _ = curves.get_beta_curves([("TFA", "G1"), ("TFA", "G1")])
        assert len(beta) == 1

    def test_unknown_varname_raises(self, curves):
        with pytest.raises(AssertionError):
            curves.get_beta_curves([("TFA", "G1")], varname="not_a_variable")


# --------------------------------------------------------------------------- #
# get_beta_curves -> calculate_force_curves                                     #
# --------------------------------------------------------------------------- #

class TestBetaCurvesFeedingForceCurves:
    """The reachable half of ISSUES.md #2 -- FIXED.

    ``calculate_force_curves`` used to pair beta rows with expression rows purely
    by position, so a ``get_beta_curves`` frame was scored correctly only when the
    caller supplied the expression rows in the beta frame's TF-group order (which
    comes from ``list(set(...))``, see #13, so is not knowable in advance).  It now
    reindexes the expression frame by TF name, so any row order gives the same
    forces.
    """

    @staticmethod
    def _expected(beta, expr, tf, target, epsilon=1e-10):
        b = beta.loc[(tf, target)].values
        t = expr.loc[tf].values
        return np.sign(b) * ((np.abs(b) + epsilon) * (t + epsilon)) ** (1 / np.log(10))

    @pytest.fixture
    def beta_and_expression(self, curves):
        beta, _ = curves.get_beta_curves([("TFA", "G1"), ("TFB", "G4")], varname="w")
        expr, _ = curves.get_smoothed_curves(mode="tf_expression")
        expr.columns = beta.columns
        return beta, expr

    def test_every_tf_has_the_same_number_of_targets(self, beta_and_expression):
        # the property that makes the positional pairing work at all
        beta, _ = beta_and_expression
        assert beta.index.get_level_values(0).value_counts().nunique() == 1

    def test_correct_when_expression_follows_the_beta_group_order(self, beta_and_expression):
        beta, expr = beta_and_expression
        groups = list(dict.fromkeys(beta.index.get_level_values(0)))
        out = SmoothedCurvesGRN.calculate_force_curves(beta, expr.loc[groups])
        for tf, target in beta.index:
            assert out.loc[(tf, target)].values == pytest.approx(
                self._expected(beta, expr, tf, target)
            ), f"{tf}->{target}"

    def test_correct_when_expression_is_in_any_other_order(self, beta_and_expression):
        # ISSUES.md #2 (fixed): before the name-based reindex, reversing the
        # expression rows silently mis-scaled every non-zero edge.
        beta, expr = beta_and_expression
        groups = list(dict.fromkeys(beta.index.get_level_values(0)))
        out = SmoothedCurvesGRN.calculate_force_curves(beta, expr.loc[groups[::-1]])
        for tf, target in beta.index:
            assert out.loc[(tf, target)].values == pytest.approx(
                self._expected(beta, expr, tf, target)
            ), f"{tf}->{target}"

    def test_the_safe_idiom_derives_the_order_from_the_beta_frame(self, beta_and_expression):
        # what a caller should write, given the order is not knowable in advance
        beta, expr = beta_and_expression
        safe = expr.loc[beta.index.get_level_values(0).unique()]
        out = SmoothedCurvesGRN.calculate_force_curves(beta, safe)
        for tf, target in beta.index:
            assert out.loc[(tf, target)].values == pytest.approx(
                self._expected(beta, expr, tf, target)
            )
