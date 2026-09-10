"""One force source per trajectory segment (ISSUES.md #23, stage 4).

``TFForceSource`` is where a segment's sampled points, regulator expression,
smoothed beta network and forces come from.  Episodic construction, force
waves, phases and validation all draw from it; a linear trajectory is one
source, a branched one is one per branch.  These tests pin what the source
computes, that it caches, that the consumers really go through it, that the
manager shares one source, and that a switch-less (linear) trajectory works.
"""
import numpy as np
import pandas as pd
import pytest
from dictys.net import stat

from firefate.temporal import (
    EpisodeDynamics,
    ForceWavePhases,
    RegulatoryPhases,
    TemporalManager,
    TFForceSource,
    TFForceWaves,
    calculate_force_curves,
)
from conftest import GENES, REGULATORS, VARNAME_SCALE

TRAJ = (0, 2)
NUM_POINTS = 10
DIST = 0.3
SPARSITY = 0.1
KW = dict(trajectory_range=TRAJ, num_points=NUM_POINTS, dist=DIST, sparsity=SPARSITY)
LINKS = [("TFA", "G1"), ("TFB", "G4")]
EPISODE = slice(2, 7)
TIME = [f"time_{i}" for i in range(NUM_POINTS)]


def reference_net(network, network_type):
    pts, fsmooth = network.linspace(TRAJ[0], TRAJ[1], NUM_POINTS, DIST)
    return fsmooth(stat.net(network, varname=network_type)).compute(pts)


@pytest.fixture
def source(mock_network):
    return TFForceSource(mock_network, **KW)


class TestSampledPoints:
    def test_tf_expression_rows_are_the_regulators_with_time_columns(self, source):
        expr = source.tf_expression()
        assert list(expr.index) == REGULATORS
        assert list(expr.columns) == TIME

    def test_tf_expression_matches_the_curves_helper(self, source):
        direct, dtime = source.curves.get_smoothed_curves(mode="tf_expression")
        assert source.tf_expression().values == pytest.approx(direct.values)
        assert source.dtime().values == pytest.approx(dtime.values)

    def test_tf_expression_is_cached(self, source):
        assert source.tf_expression() is source.tf_expression()

    def test_slice_restricts_and_relabels(self, source):
        sub = source.tf_expression(EPISODE)
        assert list(sub.columns) == ["time_0", "time_1", "time_2", "time_3", "time_4"]
        assert sub.values == pytest.approx(source.tf_expression().values[:, 2:7])
        assert source.dtime(EPISODE).values == pytest.approx(source.dtime().values[2:7])
        assert list(source.dtime(EPISODE).index) == [0, 1, 2, 3, 4]


class TestBeta:
    @pytest.mark.parametrize("network_type", sorted(VARNAME_SCALE))
    def test_full_beta_matches_the_smoothed_network(self, source, mock_network, network_type):
        full = source.full_beta_curves(network_type)
        ref = reference_net(mock_network, network_type)
        assert full.shape == (len(REGULATORS) * len(GENES), NUM_POINTS)
        assert full.index.names == ["TF", "Target"]
        assert full.values == pytest.approx(ref.reshape(-1, NUM_POINTS))
        assert full.attrs["network_type"] == network_type

    def test_full_beta_is_cached_per_network_type(self, source):
        assert source.full_beta_curves("w") is source.full_beta_curves("w")
        assert source.full_beta_curves("w") is not source.full_beta_curves("w_in")

    def test_full_beta_slice_is_a_relabelled_copy(self, source):
        sub = source.full_beta_curves("w", EPISODE)
        full = source.full_beta_curves("w")
        assert sub.values == pytest.approx(full.values[:, 2:7])
        assert list(sub.columns) == ["time_0", "time_1", "time_2", "time_3", "time_4"]
        assert sub.attrs["network_type"] == "w"
        sub.iloc[0, 0] = 123.0
        assert full.iloc[0, 0] != 123.0

    def test_link_beta_matches_get_beta_curves(self, source):
        beta, dtime = source.beta_curves(LINKS, "w_in")
        direct, ddtime = source.curves.get_beta_curves(LINKS, network_type="w_in")
        pd.testing.assert_frame_equal(beta, direct)
        assert dtime.values == pytest.approx(ddtime.values)
        assert beta.attrs["network_type"] == "w_in"

    def test_link_beta_slice(self, source):
        beta, dtime = source.beta_curves(LINKS, "w", EPISODE)
        full, _ = source.beta_curves(LINKS, "w")
        assert beta.values == pytest.approx(full.values[:, 2:7])
        assert len(dtime) == 5

    def test_network_type_is_required(self, source):
        with pytest.raises(TypeError):
            source.full_beta_curves()
        with pytest.raises(TypeError):
            source.beta_curves(LINKS)
        with pytest.raises(TypeError):
            source.force_curves(LINKS)


class TestForces:
    def test_force_curves_is_the_kernel_on_the_source_beta_and_expression(self, source):
        forces, dtime = source.force_curves(LINKS, "w_in")
        beta, _ = source.beta_curves(LINKS, "w_in")
        expected = calculate_force_curves(beta, source.tf_expression())
        pd.testing.assert_frame_equal(forces, expected)
        assert forces.attrs["network_type"] == "w_in"
        assert len(dtime) == NUM_POINTS

    def test_sliced_forces_equal_the_slice_of_full_forces(self, source):
        full, _ = source.force_curves(LINKS, "w")
        sub, _ = source.force_curves(LINKS, "w", EPISODE)
        assert sub.values == pytest.approx(full.values[:, 2:7])

    def test_from_beta_rejects_a_mismatched_slice(self, source):
        beta, _ = source.beta_curves(LINKS, "w", EPISODE)
        with pytest.raises(ValueError, match="time point"):
            source.force_curves_from_beta(beta, "w")            # no slice: 10 vs 5

    def test_from_beta_accepts_explicit_expression_and_relabels_its_columns(self, source):
        beta, _ = source.beta_curves(LINKS, "w", EPISODE)
        expr = source.tf_expression().iloc[:, 2:7]               # columns time_2..time_6
        out = source.force_curves_from_beta(beta, "w", tf_expression=expr)
        expected = source.force_curves_from_beta(beta, "w", time_slice=EPISODE)
        pd.testing.assert_frame_equal(out, expected)

    @pytest.mark.slow
    def test_parallel_path_equals_the_direct_path(self, source):
        beta, _ = source.beta_curves(LINKS, "w")
        direct = source.force_curves_from_beta(beta, "w")
        parallel = source.force_curves_from_beta(beta, "w", n_processes=2, chunk_size=1)
        pd.testing.assert_frame_equal(direct, parallel)
        assert parallel.attrs["network_type"] == "w"


class TestConsumersDrawFromTheSource:
    def test_waves_built_on_a_source_use_it(self, source):
        waves = TFForceWaves(source=source)
        assert waves.source is source
        assert waves.curves is source.curves
        assert waves.dictys_dynamic_object is source.dictys_dynamic_object
        assert waves.trajectory_range == TRAJ
        waves.compute_forces(LINKS)
        expected, _ = source.force_curves(LINKS, "w_in")
        pd.testing.assert_frame_equal(waves.force_curves, expected)

    def test_waves_built_standalone_make_their_own_source(self, mock_network):
        waves = TFForceWaves(mock_network, **KW)
        assert isinstance(waves.source, TFForceSource)
        assert waves.source.num_points == NUM_POINTS

    def test_selector_scores_on_demand_through_the_source(self, source):
        waves = TFForceWaves(source=source)
        fc, _ = TFForceWaves.ForceSelector(waves, network_type="w").force_curves(LINKS)
        expected, _ = source.force_curves(LINKS, "w")
        pd.testing.assert_frame_equal(fc, expected)

    def test_episode_built_on_a_source_uses_it(self, source, tmp_path):
        epi = EpisodeDynamics(None, str(tmp_path), source=source)
        assert epi.source is source
        assert epi.curves is source.curves
        assert (epi.trajectory_range, epi.num_points, epi.dist, epi.sparsity) == (TRAJ, NUM_POINTS, DIST, SPARSITY)
        epi.compute_expression_curves()
        beta = epi.build_episode_grn(time_slice=EPISODE)
        full = source.full_beta_curves("w", EPISODE)
        kept = full[(full.sum(axis=1) != 0)
                    & ~full.index.get_level_values(0).str.startswith("ZNF")]
        pd.testing.assert_frame_equal(beta, kept)

    def test_episode_forces_go_through_the_source_kernel(self, source, tmp_path):
        epi = EpisodeDynamics(None, str(tmp_path), source=source)
        epi.compute_expression_curves()
        epi.build_episode_grn(time_slice=EPISODE)
        epi.filter_edges(n_processes=1)
        epi.compute_tf_expression()
        epi.calculate_forces(n_processes=None)
        expected = source.force_curves_from_beta(
            epi.filtered_edges_p001.drop("p_value", axis=1), "w", time_slice=EPISODE)
        pd.testing.assert_frame_equal(epi.force_curves, expected)

    def test_two_episodes_slice_one_smoothing(self, source, tmp_path):
        a = EpisodeDynamics(None, str(tmp_path / "a"), source=source)
        b = EpisodeDynamics(None, str(tmp_path / "b"), source=source)
        for epi, sl in ((a, slice(0, 5)), (b, slice(5, 10))):
            epi.compute_expression_curves()
            epi.build_episode_grn(time_slice=sl)
        full = source.full_beta_curves("w")
        assert a.episode_beta_dcurve.loc[LINKS[0]].values == pytest.approx(full.loc[LINKS[0]].values[0:5])
        assert b.episode_beta_dcurve.loc[LINKS[0]].values == pytest.approx(full.loc[LINKS[0]].values[5:10])
        assert list(source._full_beta) == ["w"]


class TestManagerSharesOneSource:
    @pytest.fixture
    def mgr(self, mock_network, tmp_path):
        return TemporalManager(mock_network, output_dir=str(tmp_path), **KW)

    def test_force_source_is_built_once(self, mgr):
        src = mgr.force_source()
        assert isinstance(src, TFForceSource)
        assert mgr.force_source() is src
        assert (src.trajectory_range, src.num_points, src.dist, src.sparsity) == (TRAJ, NUM_POINTS, DIST, SPARSITY)

    def test_waves_and_episodes_share_it(self, mgr):
        src = mgr.force_source()
        assert mgr.waves().source is src
        assert mgr._episode_dynamics("w").source is src

    def test_smoothing_overrides_make_a_standalone_branch(self, mgr):
        waves = mgr.waves(num_points=5)
        assert waves.source is not mgr.force_source()
        assert waves.source.num_points == 5

    def test_episode_results_are_unchanged(self, mgr, mock_network, tmp_path):
        via_manager = mgr.build_episode(1, EPISODE, percentile=0, n_processes=1)
        epi = EpisodeDynamics(mock_network, str(tmp_path / "solo"), **KW)
        epi.compute_expression_curves()
        epi.build_episode_grn(time_slice=EPISODE)
        epi.filter_edges(n_processes=1)
        epi.compute_tf_expression()
        epi.calculate_forces(n_processes=1)
        pd.testing.assert_frame_equal(via_manager, epi.select_top_edges(0))

    def test_no_network_raises_on_first_use(self, tmp_path):
        with pytest.raises(ValueError, match="dictys dynamic network"):
            TemporalManager(output_dir=str(tmp_path)).force_source()


class TestLinearTrajectoryIsOneSegment:
    def test_no_switches_means_a_single_phase(self):
        phases = RegulatoryPhases([])
        assert phases.n_phases == 1
        assert phases.phase_of(0.0) == 1 and phases.phase_of(5.0) == 1

    def test_force_wave_phases_without_switches(self, source):
        waves = TFForceWaves(source=source)
        waves.compute_forces(LINKS)
        df = ForceWavePhases([], waves=waves).classify_phases()
        assert set(df["phase"]) == {1}
        assert len(df) == len(waves.force_curves)

    def test_single_branch_selector_combined_mode(self, source):
        waves = TFForceWaves(source=source)
        waves.compute_forces(LINKS)
        combined = TFForceWaves.ForceSelector(waves).combined_abs_max_force(LINKS)
        assert set(combined) == set(LINKS)
        assert {d["branch"] for d in combined.values()} == {"lineage"}
