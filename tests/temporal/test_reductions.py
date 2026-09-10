"""The three time reductions of a force curve (ISSUES.md #23, stage 3).

``mean_force`` (episodic), ``softmax_peak`` (phases) and ``abs_max_force``
(validation) live side by side in ``_reductions`` and each consumer calls its
own.  These tests pin what each one computes, that the consumers really route
through them, and the documented fact that they give different numbers for the
same curve.
"""
import numpy as np
import pandas as pd
import pytest

from focalfire.temporal import (
    EpisodeDynamics,
    RegulatoryPhases,
    TFForceWaves,
    abs_max_force,
    aggregate_max_points,
    get_max_points,
    mean_force,
    softmax_peak,
)
from focalfire.temporal import _phases, _reductions

COLS = [f"time_{i}" for i in range(6)]
DTIME = np.linspace(0.0, 1.0, 6)


@pytest.fixture
def forces():
    index = pd.MultiIndex.from_tuples(
        [("TFA", "G1"), ("TFA", "G2"), ("TFB", "G3")], names=["TF", "Target"]
    )
    return pd.DataFrame(
        [[0.0, 0.0, 4.0, 0.0, 0.0, 0.0],       # single spike at t=2
         [1.0, -5.0, 2.0, 0.5, 0.0, 0.0],      # sign change, |max| at t=1
         [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]],      # flat
        index=index, columns=COLS,
    )


class TestMeanForce:
    def test_is_the_plain_mean_over_time_columns(self, forces):
        pd.testing.assert_series_equal(mean_force(forces), forces.mean(axis=1))

    def test_zero_time_points_count_in_the_denominator(self, forces):
        assert mean_force(forces).loc[("TFA", "G1")] == pytest.approx(4.0 / 6)

    def test_a_sign_change_can_cancel(self, forces):
        assert mean_force(forces).loc[("TFA", "G2")] == pytest.approx(-1.5 / 6)


class TestAbsMaxForce:
    def test_is_max_abs_over_time(self, forces):
        out = abs_max_force(forces)
        assert out == {("TFA", "G1"): 4.0, ("TFA", "G2"): 5.0, ("TFB", "G3"): 3.0}

    def test_links_restrict_order_and_skip_missing(self, forces):
        out = abs_max_force(forces, [("TFB", "G3"), ("NOPE", "X"), ("TFA", "G1")])
        assert list(out) == [("TFB", "G3"), ("TFA", "G1")]

    def test_selector_abs_max_is_this_function(self):
        assert TFForceWaves.ForceSelector.abs_max is abs_max_force


class TestSoftmaxPeak:
    def test_is_the_two_step_softmax_helper_chain(self, forces):
        expected = aggregate_max_points(get_max_points(forces, DTIME, top_k=3, temperature=0.5),
                                        method="mean")
        assert softmax_peak(forces, DTIME, top_k=3, temperature=0.5, method="mean") == expected

    def test_top1_of_a_single_spike_is_the_spike(self, forces):
        peak = softmax_peak(forces, DTIME, method="top1")[("TFA", "G1")]
        assert peak["pseudotime"] == pytest.approx(DTIME[2])
        assert peak["window_idx"] == 2
        assert peak["force"] == pytest.approx(4.0)

    def test_peak_follows_abs_force_not_signed_force(self, forces):
        peak = softmax_peak(forces, DTIME, method="top1")[("TFA", "G2")]
        assert peak["window_idx"] == 1
        assert peak["force"] == pytest.approx(-5.0)

    def test_weighted_mean_of_a_flat_curve_is_the_mean_of_its_top_k(self, forces):
        peak = softmax_peak(forces, DTIME, top_k=6)[("TFB", "G3")]
        assert peak["pseudotime"] == pytest.approx(DTIME.mean())
        assert peak["force"] == pytest.approx(3.0)

    def test_phases_module_reexports_the_helpers(self):
        assert _phases.get_max_points is _reductions.get_max_points
        assert _phases.aggregate_max_points is _reductions.aggregate_max_points
        assert _phases.softmax_peak is _reductions.softmax_peak

    def test_assign_phases_bins_the_softmax_peak(self, forces):
        boundaries = [0.3, 0.7]
        phases = RegulatoryPhases.assign_phases(forces, DTIME, boundaries, method="top1")
        peaks = softmax_peak(forces, DTIME, method="top1")
        for link, info in peaks.items():
            assert phases[link] == int(np.digitize(info["pseudotime"], boundaries, right=True)) + 1
        assert phases[("TFA", "G1")] == 2          # spike at t=0.4
        assert phases[("TFA", "G2")] == 1          # |max| at t=0.2


class TestOneCurveThreeNumbers:
    def test_the_reductions_disagree_by_design(self, forces):
        link = ("TFA", "G1")
        assert mean_force(forces).loc[link] == pytest.approx(4.0 / 6)
        assert abs_max_force(forces)[link] == pytest.approx(4.0)
        assert softmax_peak(forces, DTIME, method="top1")[link]["force"] == pytest.approx(4.0)
        assert softmax_peak(forces, DTIME, method="top1")[link]["pseudotime"] == pytest.approx(0.4)


class TestEpisodicUsesMeanForce:
    def test_avg_force_is_mean_force_of_the_stored_curves(self, mock_network, tmp_path):
        epi = EpisodeDynamics(mock_network, output_folder=str(tmp_path), trajectory_range=(0, 2),
                              num_points=10, dist=0.3, sparsity=0.1)
        epi.compute_expression_curves()
        epi.build_episode_grn(time_slice=slice(0, 5))
        epi.filter_edges(n_processes=1)
        epi.compute_tf_expression()
        avg = epi.calculate_forces(n_processes=1)
        pd.testing.assert_series_equal(avg["avg_force"], mean_force(epi.force_curves),
                                       check_names=False)
