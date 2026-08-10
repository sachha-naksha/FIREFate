"""``AlignTimeScales`` / ``EpisodeDynamics`` and the process-level runners.

Driven by the real (miniature) dictys network from ``conftest.py``.  The mock
GRN has exactly seven non-zero edges, so every stage of the workflow has a
countable, hand-checkable expected result:

===============  ==============================================================
edge             per-window weight
===============  ==============================================================
TFA -> G1        constant +2.0
TFA -> G2        ramp +1.0 -> +2.0
TFA -> G3        ramp +3.0 -> -3.0
TFA -> TFB       constant +1.0
TFB -> G4        constant -1.5
TFB -> G5        ramp -0.5 -> -1.5
ZNF1 -> G1       constant +5.0  (dropped by the hard-coded ZNF/ZBTB filter)
===============  ==============================================================
"""

import os

import numpy as np
import pandas as pd
import pytest

from firefate.core.episodic_dynamics import (
    AlignTimeScales,
    EpisodeDynamics,
    run_episodic_construction,
    run_episodic_enrichment,
)
from firefate.core.pseudotime_curves import SmoothedCurvesGRN

from conftest import (
    CONSTANT_LCPM,
    MOCK_EDGES,
    N_WINDOWS,
    WINDOW_PSEUDOTIME_FROM_N0,
    WINDOW_PSEUDOTIME_FROM_N1,
)

NUM_POINTS = 10
SPARSITY = 0.1
DIST = 0.3
LOG10E = 1.0 / np.log(10.0)

#: edges of the mock GRN that survive build_episode_grn (ZNF1 is filtered out)
EPISODE_EDGES = [("TFA", "TFB"), ("TFA", "G1"), ("TFA", "G2"), ("TFA", "G3"),
                 ("TFB", "G4"), ("TFB", "G5")]


def as_tuples(index):
    return [tuple(str(x) for x in row) for row in index]


# --------------------------------------------------------------------------- #
# AlignTimeScales                                                               #
# --------------------------------------------------------------------------- #

class TestAlignTimeScales:
    @pytest.mark.parametrize(
        "traj_range,expected",
        [
            ((0, 2), WINDOW_PSEUDOTIME_FROM_N0),
            ((0, 3), WINDOW_PSEUDOTIME_FROM_N0),
            ((1, 3), WINDOW_PSEUDOTIME_FROM_N1),
            ((1, 2), WINDOW_PSEUDOTIME_FROM_N1),
        ],
    )
    def test_window_pseudotime_is_the_distance_to_the_start_node(self, mock_network,
                                                                 traj_range, expected):
        aligner = AlignTimeScales(mock_network, trajectory_range=traj_range)
        assert aligner.pseudotime_of_windows() == pytest.approx(expected)

    def test_window_pseudotime_covers_every_window_including_other_branches(self,
                                                                           mock_network):
        # The returned vector is indexed by window ID over the *whole* network,
        # so callers must subset it to the branch they are working on.
        aligner = AlignTimeScales(mock_network, trajectory_range=(0, 2))
        assert len(aligner.pseudotime_of_windows()) == N_WINDOWS

    @pytest.mark.parametrize("traj_range,length", [((0, 2), 2.0), ((0, 3), 2.0), ((1, 3), 1.0)])
    def test_sampled_points_are_evenly_spaced(self, mock_network, traj_range, length):
        aligner = AlignTimeScales(mock_network, trajectory_range=traj_range,
                                  num_points=NUM_POINTS, dist=DIST)
        assert aligner.pseudotime_of_sampled_points() == pytest.approx(
            np.linspace(0.0, length, NUM_POINTS)
        )

    def test_sampled_points_honour_num_points(self, mock_network):
        aligner = AlignTimeScales(mock_network, trajectory_range=(0, 2), num_points=25,
                                  dist=DIST)
        assert len(aligner.pseudotime_of_sampled_points()) == 25

    def test_sampled_points_span_the_branch_windows(self, mock_network):
        aligner = AlignTimeScales(mock_network, trajectory_range=(0, 2),
                                  num_points=NUM_POINTS, dist=DIST)
        sampled = aligner.pseudotime_of_sampled_points()
        branch_windows = aligner.pseudotime_of_windows()[[0, 1, 2, 3, 4]]
        assert sampled.min() == pytest.approx(branch_windows.min())
        assert sampled.max() == pytest.approx(branch_windows.max())

    def test_returns_plain_numpy_arrays(self, mock_network):
        aligner = AlignTimeScales(mock_network, trajectory_range=(0, 2), dist=DIST)
        assert isinstance(aligner.pseudotime_of_windows(), np.ndarray)
        assert isinstance(aligner.pseudotime_of_sampled_points(), np.ndarray)


# --------------------------------------------------------------------------- #
# EpisodeDynamics: construction & state                                         #
# --------------------------------------------------------------------------- #

@pytest.fixture
def episode(mock_network, tmp_path):
    return EpisodeDynamics(
        mock_network,
        output_folder=str(tmp_path),
        trajectory_range=(0, 2),
        num_points=NUM_POINTS,
        dist=DIST,
        sparsity=SPARSITY,
    )


class TestEpisodeConstruction:
    def test_composes_the_curve_and_aligner_helpers_with_shared_parameters(self, episode):
        assert isinstance(episode.curves, SmoothedCurvesGRN)
        assert isinstance(episode.time_aligner, AlignTimeScales)
        for helper in (episode.curves, episode.time_aligner):
            assert helper.trajectory_range == (0, 2)
            assert helper.num_points == NUM_POINTS
            assert helper.dist == DIST
            assert helper.sparsity == SPARSITY
            assert helper.dictys_dynamic_object is episode.dictys_dynamic_object

    def test_state_starts_empty(self, episode):
        for attr in ["lcpm_dcurve", "dtime", "episode_beta_dcurve", "filtered_edges",
                     "filtered_edges_p001", "tf_lcpm_episode", "force_curves",
                     "avg_force_df", "episodic_grn_edges", "lf_genes", "lf_in_object",
                     "episodic_enrichment_df"]:
            assert getattr(episode, attr) is None

    def test_episodic_composition_is_an_unimplemented_stub(self, episode):
        assert episode.episodic_composition() is None


class TestExpressionCurves:
    def test_stores_and_returns_the_curves(self, episode):
        lcpm, dtime = episode.compute_expression_curves()
        assert episode.lcpm_dcurve is lcpm
        assert episode.dtime is dtime
        assert lcpm.shape[1] == NUM_POINTS

    def test_values_match_the_curves_helper(self, episode):
        lcpm, _ = episode.compute_expression_curves()
        direct, _ = episode.curves.get_smoothed_curves(mode="expression")
        assert lcpm.values == pytest.approx(direct.values)

    def test_constant_genes_are_exact(self, episode):
        lcpm, _ = episode.compute_expression_curves()
        for gene, value in CONSTANT_LCPM.items():
            assert lcpm.loc[gene].values == pytest.approx(np.full(NUM_POINTS, value), abs=1e-12)

    def test_mode_argument_switches_the_helper_mode(self, episode):
        episode.compute_expression_curves(mode="tf_expression")
        assert episode.curves.mode == "tf_expression"
        assert len(episode.lcpm_dcurve) == 3


# --------------------------------------------------------------------------- #
# build_episode_grn                                                             #
# --------------------------------------------------------------------------- #

class TestBuildEpisodeGrn:
    def test_keeps_exactly_the_non_zero_non_znf_edges(self, episode):
        grn = episode.build_episode_grn(time_slice=slice(0, 5))
        assert as_tuples(grn.index) == EPISODE_EDGES

    def test_znf_and_zbtb_regulators_are_dropped(self, episode):
        # ISSUES.md #12 -- BY DESIGN, not a defect: ZNF*/ZBTB* factors are not
        # relevant to FIREFate's biology and are dropped deliberately.  This is a
        # regression guard so the filter is not removed by accident.
        grn = episode.build_episode_grn(time_slice=slice(0, 5))
        # ZNF1 -> G1 is the strongest edge in the network and is still removed
        assert "ZNF1" not in set(grn.index.get_level_values(0))

    def test_all_zero_edges_are_dropped(self, episode):
        grn = episode.build_episode_grn(time_slice=slice(0, 5))
        assert len(grn) == len(MOCK_EDGES) - 1        # minus the ZNF1 edge
        assert (grn.abs().sum(axis=1) > 0).all()

    def test_values_match_the_smoothed_network(self, episode, mock_network):
        from dictys.net import stat
        grn = episode.build_episode_grn(time_slice=slice(0, 5))
        pts, fsmooth = mock_network.linspace(0, 2, NUM_POINTS, DIST)
        ref = fsmooth(stat.net(mock_network)).compute(pts)
        names = list(mock_network.nname)
        regs = [names[i] for i in mock_network.nids[0]]
        for tf, target in as_tuples(grn.index):
            assert grn.loc[(tf, target)].values == pytest.approx(
                ref[regs.index(tf), names.index(target), 0:5]
            ), f"{tf}->{target}"

    def test_constant_edges_are_exactly_constant(self, episode):
        grn = episode.build_episode_grn(time_slice=slice(0, 5))
        assert grn.loc[("TFA", "G1")].values == pytest.approx(np.full(5, 2.0))
        assert grn.loc[("TFB", "G4")].values == pytest.approx(np.full(5, -1.5))

    @pytest.mark.parametrize("time_slice,width", [
        (slice(0, 5), 5), (slice(3, 8), 5), (slice(0, 10), 10), (slice(2, 4), 2),
    ])
    def test_time_slice_selects_the_episode_window(self, episode, time_slice, width):
        grn = episode.build_episode_grn(time_slice=time_slice)
        assert list(grn.columns) == [f"time_{i}" for i in range(width)]

    def test_column_labels_always_restart_at_time_0(self, episode):
        # The episode's columns are renamed 0..n-1, so the frame no longer
        # records which absolute pseudotime points it came from.
        early = episode.build_episode_grn(time_slice=slice(0, 3))
        late = episode.build_episode_grn(time_slice=slice(7, 10))
        assert list(early.columns) == list(late.columns)
        assert not np.allclose(early.loc[("TFA", "G2")].values,
                               late.loc[("TFA", "G2")].values)

    def test_result_is_cached_on_the_instance(self, episode):
        grn = episode.build_episode_grn(time_slice=slice(0, 5))
        assert episode.episode_beta_dcurve is grn

    def test_index_names_are_tf_and_target(self, episode):
        grn = episode.build_episode_grn(time_slice=slice(0, 5))
        assert list(grn.index.names) == ["TF", "Target"]

    def test_edges_are_retained_by_row_sum_not_by_activity(self, episode):
        # ISSUES.md #14 -- accepted, rare.  Presence is decided by
        # ``sum(axis=1) != 0`` rather than "has any non-zero value".  TFA -> G3
        # swings from strongly positive to negative inside this episode and
        # survives only because the two halves do not cancel exactly; an edge
        # whose values did cancel would be dropped despite being active at every
        # time point.  Suggested change: ``(df != 0).any(axis=1)`` -- this test
        # passes unchanged either way.
        grn = episode.build_episode_grn(time_slice=slice(0, NUM_POINTS))
        row = grn.loc[("TFA", "G3")]
        assert (row > 0).any() and (row < 0).any()
        assert row.sum() != 0


# --------------------------------------------------------------------------- #
# filter_edges                                                                  #
# --------------------------------------------------------------------------- #

@pytest.fixture
def built(episode):
    episode.compute_expression_curves()
    episode.build_episode_grn(time_slice=slice(0, 5))
    return episode


@pytest.mark.slow
class TestFilterEdges:
    def test_keeps_the_consistent_edges(self, built):
        kept = built.filter_edges(n_processes=2, chunk_size=100)
        assert set(as_tuples(kept.index)) == {
            ("TFA", "TFB"), ("TFA", "G1"), ("TFA", "G2"), ("TFB", "G4"), ("TFB", "G5"),
        }

    def test_a_steeply_varying_edge_fails_the_strict_pvalue_threshold(self, built):
        # TFA -> G3 falls from +2.79 to +1.22 across the episode: it is direction
        # invariant and significant at alpha=0.05, but its p-value (~2e-3) does
        # not clear the p < 0.001 cut applied after the parallel filter.
        built.filter_edges(n_processes=2, chunk_size=100)
        assert ("TFA", "G3") in [tuple(map(str, i)) for i in built.filtered_edges.index]
        assert ("TFA", "G3") not in as_tuples(built.filtered_edges_p001.index)
        assert 0.001 < built.filtered_edges.loc[("TFA", "G3"), "p_value"] < 0.05

    def test_a_looser_threshold_keeps_it(self, built):
        kept = built.filter_edges(n_processes=2, chunk_size=100, pval_threshold=0.05)
        assert ("TFA", "G3") in as_tuples(kept.index)

    def test_both_result_frames_are_stored(self, built):
        kept = built.filter_edges(n_processes=2, chunk_size=100)
        assert built.filtered_edges_p001 is kept
        assert len(built.filtered_edges) >= len(kept)

    def test_pvalue_column_is_added(self, built):
        kept = built.filter_edges(n_processes=2, chunk_size=100)
        assert "p_value" in kept.columns
        assert (kept["p_value"] < 0.001).all()

    def test_constant_edges_get_a_degenerate_zero_pvalue(self, built):
        kept = built.filter_edges(n_processes=2, chunk_size=100)
        assert kept.loc[("TFA", "G1"), "p_value"] == 0.0


# --------------------------------------------------------------------------- #
# compute_tf_expression                                                         #
# --------------------------------------------------------------------------- #

@pytest.mark.slow
class TestComputeTfExpression:
    @pytest.fixture
    def filtered(self, built):
        built.filter_edges(n_processes=2, chunk_size=100)
        return built

    def test_one_row_per_regulator_in_the_filtered_grn(self, filtered):
        tf_expr = filtered.compute_tf_expression()
        assert list(tf_expr.index) == ["TFA", "TFB"]

    def test_columns_match_the_episode_time_columns(self, filtered):
        tf_expr = filtered.compute_tf_expression()
        assert list(tf_expr.columns) == [
            c for c in filtered.filtered_edges_p001.columns if c.startswith("time_")
        ]

    def test_values_come_from_the_expression_curves(self, filtered):
        tf_expr = filtered.compute_tf_expression()
        assert tf_expr.loc["TFA"].values == pytest.approx(
            filtered.lcpm_dcurve.loc["TFA"].values[:5]
        )

    def test_result_is_stored(self, filtered):
        assert filtered.compute_tf_expression() is filtered.tf_lcpm_episode

    @pytest.mark.xfail(
        strict=True,
        reason="ISSUES.md #3 (CONFIRMED, high; fix drafted in ISSUES.md) -- "
               "compute_tf_expression always takes the FIRST n columns of the "
               "pseudotime expression curve (``iloc[:, 0:n_time_cols]``) and relabels "
               "them time_0..time_n, regardless of which time_slice build_episode_grn "
               "used.  For any episode other than the first, the regulator expression "
               "is taken from the wrong stretch of pseudotime; the episode's time_slice "
               "is not even stored on the object.  When the fix lands, drop this marker "
               "and delete test_currently_reuses_the_first_time_points_for_every_episode.",
    )
    def test_expression_is_taken_from_the_episode_s_own_time_window(self, episode):
        episode.compute_expression_curves()
        episode.build_episode_grn(time_slice=slice(5, 10))
        episode.filter_edges(n_processes=2, chunk_size=100, pval_threshold=0.05)
        tf_expr = episode.compute_tf_expression()
        assert tf_expr.loc["TFB"].values == pytest.approx(
            episode.lcpm_dcurve.loc["TFB"].values[5:10]
        )

    def test_currently_reuses_the_first_time_points_for_every_episode(self, episode):
        episode.compute_expression_curves()
        episode.build_episode_grn(time_slice=slice(5, 10))
        episode.filter_edges(n_processes=2, chunk_size=100, pval_threshold=0.05)
        tf_expr = episode.compute_tf_expression()
        assert tf_expr.loc["TFB"].values == pytest.approx(
            episode.lcpm_dcurve.loc["TFB"].values[0:5]
        )


# --------------------------------------------------------------------------- #
# calculate_forces / edge selection                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.slow
class TestCalculateForces:
    @pytest.fixture
    def prepared(self, built):
        built.filter_edges(n_processes=2, chunk_size=100)
        built.compute_tf_expression()
        return built

    def test_average_force_matches_the_documented_log_formula(self, prepared):
        avg = prepared.calculate_forces(n_processes=2, chunk_size=100)
        beta = prepared.filtered_edges_p001.drop("p_value", axis=1)
        expr = prepared.tf_lcpm_episode
        for tf, target in as_tuples(beta.index):
            b = beta.loc[(tf, target)].values
            t = expr.loc[tf].values
            expected = np.sign(b) * ((np.abs(b) + 1e-10) * (t + 1e-10)) ** LOG10E
            assert avg.loc[(tf, target), "avg_force"] == pytest.approx(expected.mean())

    def test_constant_edge_force_is_exact(self, prepared):
        # TFA -> G1: beta = 2 and TFA's lcpm = 2 at every point
        avg = prepared.calculate_forces(n_processes=2, chunk_size=100)
        assert avg.loc[("TFA", "G1"), "avg_force"] == pytest.approx(4.0 ** LOG10E)

    def test_sign_of_the_force_follows_the_sign_of_beta(self, prepared):
        avg = prepared.calculate_forces(n_processes=2, chunk_size=100)
        assert avg.loc[("TFA", "G1"), "avg_force"] > 0
        assert avg.loc[("TFB", "G4"), "avg_force"] < 0

    def test_force_curves_and_average_are_both_stored(self, prepared):
        avg = prepared.calculate_forces(n_processes=2, chunk_size=100)
        assert prepared.avg_force_df is avg
        assert prepared.force_curves.shape == prepared.filtered_edges_p001.shape[0:1] + (5,)
        assert list(avg.columns) == ["avg_force"]

    @pytest.mark.xfail(
        strict=True,
        reason="ISSUES.md #1 (CONFIRMED, high) -- calculate_forces inherits the "
               "calculate_force_curves_chunk misalignment: the TF expression is repeated "
               "in value_counts() order (descending target count) onto rows that are "
               "grouped in network order, so edges get another TF's expression whenever "
               "the two orders differ.  NB the mock fixture happens to make the two "
               "orders agree (TFA has more targets AND a lower nids[0] index), which is "
               "why the end-to-end workflow tests above still pass -- that coincidence "
               "will not hold on a real network with hundreds of TFs.",
    )
    def test_expression_is_matched_to_the_right_regulator(self, episode):
        # TFA has one edge, TFB has two: the row order (TFA first) and the
        # value_counts order (TFB first) disagree.
        index = pd.MultiIndex.from_tuples(
            [("TFA", "G1"), ("TFB", "G2"), ("TFB", "G3")], names=["TF", "Target"]
        )
        episode.filtered_edges_p001 = pd.DataFrame(
            {"time_0": [1.0, 1.0, 1.0], "p_value": [0.0, 0.0, 0.0]}, index=index
        )
        episode.tf_lcpm_episode = pd.DataFrame(
            {"time_0": [10.0, 1000.0]}, index=["TFA", "TFB"]
        )
        avg = episode.calculate_forces(n_processes=2, chunk_size=100)
        assert avg.loc[("TFA", "G1"), "avg_force"] == pytest.approx(10.0 ** LOG10E)


class TestEdgeSelection:
    @pytest.fixture
    def with_forces(self, episode):
        index = pd.MultiIndex.from_tuples(
            [("TFA", f"G{i}") for i in range(10)], names=["TF", "Target"]
        )
        episode.avg_force_df = pd.DataFrame(
            {"avg_force": [5.0, -6.0, 0.1, -0.2, 3.0, -4.0, 0.5, -0.5, 1.0, -1.0]},
            index=index,
        )
        return episode

    def test_select_top_edges_uses_the_absolute_force_percentile(self, with_forces):
        top = with_forces.select_top_edges(percentile=80)
        threshold = np.percentile(np.abs(with_forces.avg_force_df["avg_force"]), 80)
        assert (top["avg_force"].abs() >= threshold).all()
        assert len(top) == 2                       # |−6| and |5|

    def test_select_top_edges_keeps_both_signs(self, with_forces):
        top = with_forces.select_top_edges(percentile=50)
        assert (top["avg_force"] > 0).any() and (top["avg_force"] < 0).any()

    def test_select_top_edges_stores_the_result(self, with_forces):
        assert with_forces.select_top_edges(90) is with_forces.episodic_grn_edges

    def test_percentile_zero_keeps_everything(self, with_forces):
        assert len(with_forces.select_top_edges(0)) == 10

    def test_activating_and_repressing_selection_is_two_sided(self, with_forces):
        top = with_forces.select_top_activating_and_repressing_edges(
            percentile_positive=80, percentile_negative=20
        )
        assert (top["avg_force"] > 0).any() and (top["avg_force"] < 0).any()
        assert top["avg_force"].is_monotonic_decreasing

    def test_activating_selection_survives_an_all_positive_input(self, episode):
        index = pd.MultiIndex.from_tuples([("TFA", "G1"), ("TFA", "G2")],
                                          names=["TF", "Target"])
        episode.avg_force_df = pd.DataFrame({"avg_force": [1.0, 2.0]}, index=index)
        top = episode.select_top_activating_and_repressing_edges(50, 50)
        assert (top["avg_force"] > 0).all()

    def test_activating_selection_survives_an_all_negative_input(self, episode):
        index = pd.MultiIndex.from_tuples([("TFA", "G1"), ("TFA", "G2")],
                                          names=["TF", "Target"])
        episode.avg_force_df = pd.DataFrame({"avg_force": [-1.0, -2.0]}, index=index)
        top = episode.select_top_activating_and_repressing_edges(50, 50)
        assert (top["avg_force"] < 0).all()


# --------------------------------------------------------------------------- #
# LF genes & enrichment                                                         #
# --------------------------------------------------------------------------- #

class TestLfGenesAndEnrichment:
    @pytest.fixture
    def with_grn(self, episode):
        index = pd.MultiIndex.from_tuples(
            [("TFA", "G1"), ("TFA", "G2"), ("TFA", "G3"),
             ("TFB", "G4"), ("TFB", "G5")],
            names=["TF", "Target"],
        )
        episode.episodic_grn_edges = pd.DataFrame(
            {"avg_force": [1.0, 2.0, 3.0, -4.0, -5.0]}, index=index
        )
        return episode

    def test_set_lf_genes_reports_which_are_in_the_network(self, episode):
        out = episode.set_lf_genes(["G1", "G2", "NOT_A_GENE"])
        assert out["present"] == ["G1", "G2"]
        assert out["missing"] == ["NOT_A_GENE"]
        assert out["stats"]["found"] == 2
        assert episode.lf_genes == ["G1", "G2", "NOT_A_GENE"]

    def test_annotate_requires_lf_genes(self, with_grn):
        with pytest.raises(ValueError, match="LF genes not set"):
            with_grn.annotate_lf_in_grn()

    def test_annotate_flags_lf_targets(self, with_grn):
        with_grn.set_lf_genes(["G1", "G4"])
        out = with_grn.annotate_lf_in_grn()
        assert out["is_in_lf"].tolist() == [True, False, False, True, False]

    def test_annotate_ignores_lf_genes_that_are_only_regulators(self, with_grn):
        # membership is tested on the target level only
        with_grn.set_lf_genes(["TFA"])
        out = with_grn.annotate_lf_in_grn()
        assert not out["is_in_lf"].any()

    def test_enrichment_is_computed_over_the_selected_edges(self, with_grn):
        from scipy.stats import hypergeom
        with_grn.set_lf_genes(["G1", "G4"])
        with_grn.annotate_lf_in_grn()
        out = with_grn.calculate_enrichment()
        # population: 5 distinct targets, 2 of them LF; TFA has 1/3, TFB has 1/2
        by_tf = out.set_index("TF")
        assert by_tf.loc["TFB", "enrichment_score"] == pytest.approx(1 / (2 * 2 / 5))
        assert by_tf.loc["TFB", "p_value"] == pytest.approx(hypergeom.sf(0, 5, 2, 2))

    def test_enrichment_is_sorted_and_zero_scores_dropped(self, with_grn):
        with_grn.set_lf_genes(["G4"])
        with_grn.annotate_lf_in_grn()
        out = with_grn.calculate_enrichment()
        assert list(out["TF"]) == ["TFB"]            # TFA has no LF target -> score 0
        assert out["enrichment_score"].is_monotonic_decreasing

    def test_enrichment_result_is_stored(self, with_grn):
        with_grn.set_lf_genes(["G1"])
        with_grn.annotate_lf_in_grn()
        assert with_grn.calculate_enrichment() is with_grn.episodic_enrichment_df

    def test_enrichment_without_annotation_raises(self, with_grn):
        with pytest.raises(KeyError):
            with_grn.calculate_enrichment()


# --------------------------------------------------------------------------- #
# end-to-end runners                                                            #
# --------------------------------------------------------------------------- #

@pytest.mark.slow
class TestRunners:
    def test_construction_writes_the_episode_and_force_parquets(self, mock_network_path,
                                                                tmp_path):
        out_folder = str(tmp_path / "construction")
        os.makedirs(out_folder)
        path = run_episodic_construction(
            0, mock_network_path, out_folder, trajectory_range=(0, 2),
            num_points=NUM_POINTS, time_slice_start=0, time_slice_end=5,
            dist=DIST, sparsity=SPARSITY, percentile=50,
        )
        assert path == os.path.join(out_folder, "episode_0.parquet")
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(out_folder, "avg_force_episode_0.parquet"))

        edges = pd.read_parquet(path)
        assert list(edges.columns) == ["avg_force"]
        assert set(as_tuples(edges.index)) <= set(EPISODE_EDGES)

    def test_construction_matches_the_step_by_step_workflow(self, mock_network,
                                                            mock_network_path, tmp_path):
        out_folder = str(tmp_path / "compare")
        os.makedirs(out_folder)
        path = run_episodic_construction(
            0, mock_network_path, out_folder, trajectory_range=(0, 2),
            num_points=NUM_POINTS, time_slice_start=0, time_slice_end=5,
            dist=DIST, sparsity=SPARSITY, percentile=50,
        )
        epi = EpisodeDynamics(mock_network, out_folder, trajectory_range=(0, 2),
                              num_points=NUM_POINTS, dist=DIST, sparsity=SPARSITY)
        epi.compute_expression_curves()
        epi.build_episode_grn(time_slice=slice(0, 5))
        epi.filter_edges()
        epi.compute_tf_expression()
        epi.calculate_forces()
        expected = epi.select_top_edges(50)
        assert pd.read_parquet(path)["avg_force"].values == pytest.approx(
            expected["avg_force"].values
        )

    def test_enrichment_writes_a_csv(self, mock_network_path, tmp_path):
        out_folder = str(tmp_path / "enrichment")
        os.makedirs(out_folder)
        path = run_episodic_enrichment(
            3, mock_network_path, out_folder, trajectory_range=(0, 2),
            num_points=NUM_POINTS, time_slice_start=0, time_slice_end=5,
            lf_genes=["G1", "G4"], dist=DIST, sparsity=SPARSITY, percentile=50,
        )
        assert path == os.path.join(out_folder, "enrichment_episode_3.csv")
        table = pd.read_csv(path)
        assert list(table.columns) == [
            "TF", "p_value", "enrichment_score", "genes_in_lf", "genes_dwnstrm", "weights",
        ]
        assert len(table) > 0

    def test_the_episode_index_only_names_the_output_file(self, mock_network_path, tmp_path):
        out_folder = str(tmp_path / "index")
        os.makedirs(out_folder)
        first = run_episodic_construction(
            0, mock_network_path, out_folder, (0, 2), NUM_POINTS, 0, 5,
            dist=DIST, sparsity=SPARSITY, percentile=50,
        )
        second = run_episodic_construction(
            7, mock_network_path, out_folder, (0, 2), NUM_POINTS, 0, 5,
            dist=DIST, sparsity=SPARSITY, percentile=50,
        )
        assert os.path.basename(second) == "episode_7.parquet"
        pd.testing.assert_frame_equal(pd.read_parquet(first), pd.read_parquet(second))
