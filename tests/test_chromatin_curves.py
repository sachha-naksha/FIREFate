"""``SmoothedCurvesChromatin`` -- extraction, trajectory mapping and plotting.

The binding files written by ``conftest.write_binding_windows`` follow the real
dictys ``binding.tsv.gz`` layout (``TF``/``loc``/``score`` with
``loc = chr:start:end``).  Scores are chosen so that the *mean of per-chromosome
means* differs from a plain global mean, which pins down the aggregation the
class actually performs.
"""

import gzip

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.ndimage import gaussian_filter1d

from firefate.core.pseudotime_curves import SmoothedCurvesChromatin

from conftest import BINDING_ROWS, EXPECTED_BINDING, write_binding_windows

ALL_TFS = ["TFA", "TFB", "TFC"]
N_BINDING_WINDOWS = len(BINDING_ROWS)


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #

def expected_series(tf, key, n_windows=N_BINDING_WINDOWS):
    """Per-window score (key=0) or count (key=1) with the class' sentinels."""
    out = []
    for w in range(1, n_windows + 1):
        if w in EXPECTED_BINDING.get(tf, {}):
            out.append(EXPECTED_BINDING[tf][w][key])
        else:
            out.append(np.nan if key == 0 else 0)
    return out


#: gaussian_filter1d rejects sigma=0, so this stands in for "no smoothing":
#: the kernel radius rounds to 0 and the filter is the identity.
NO_SMOOTHING = 0.01


@pytest.fixture
def dense_binding_dir(tmp_path):
    """Six windows in which every TF is present, so nothing is NaN.

    Window ``w`` holds ``w`` rows for TF ``X`` (score ``w``, so both its score
    and its count vary), and for TF ``Y`` one chr1 row of score ``10w`` plus
    ``7 - w`` chr2 rows of score 1 (score and count both vary, in opposite
    directions to X).
    """
    base = tmp_path / "dense"
    base.mkdir()
    rows = {
        w: [("X", "chr1", float(w))] * w
           + [("Y", "chr1", float(10 * w))]
           + [("Y", "chr2", 1.0)] * (7 - w)
        for w in range(1, 7)
    }
    write_binding_windows(str(base), rows)
    return str(base)


@pytest.fixture
def dense_analyzer(dense_binding_dir):
    a = SmoothedCurvesChromatin(["X", "Y"], dense_binding_dir)
    a.extract_data(n_windows=6, n_processes=2)
    return a


# --------------------------------------------------------------------------- #
# small static helpers                                                          #
# --------------------------------------------------------------------------- #

class TestToFloat:
    def test_plain_number(self):
        assert SmoothedCurvesChromatin._to_float(3) == 3.0
        assert SmoothedCurvesChromatin._to_float(2.5) == 2.5

    def test_numpy_scalar(self):
        assert SmoothedCurvesChromatin._to_float(np.float64(7.5)) == 7.5

    def test_series_returns_first_element(self):
        assert SmoothedCurvesChromatin._to_float(pd.Series([4.0, 9.0])) == 4.0

    def test_empty_series_returns_nan(self):
        assert np.isnan(SmoothedCurvesChromatin._to_float(pd.Series([], dtype=float)))

    def test_nan_passes_through(self):
        assert np.isnan(SmoothedCurvesChromatin._to_float(float("nan")))

    def test_multi_element_array_is_not_handled(self):
        # numpy arrays have no ``.values`` so they fall through to float()
        with pytest.raises(TypeError):
            SmoothedCurvesChromatin._to_float(np.array([1.0, 2.0]))


class TestSmooth:
    def test_matches_scipy(self):
        series = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
        assert SmoothedCurvesChromatin._smooth(series, 1.5) == pytest.approx(
            gaussian_filter1d(series, sigma=1.5)
        )

    def test_constant_series_is_unchanged(self):
        series = np.full(10, 4.0)
        assert SmoothedCurvesChromatin._smooth(series, 2.0) == pytest.approx(series)

    def test_length_is_preserved(self):
        series = np.arange(7.0)
        assert len(SmoothedCurvesChromatin._smooth(series, 3.0)) == 7

    def test_smoothing_reduces_variance(self):
        series = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0])
        assert SmoothedCurvesChromatin._smooth(series, 2.0).std() < series.std()

    def test_a_single_nan_poisons_the_whole_series(self):
        # ISSUE: gaussian_filter1d has no NaN handling, and extract_data uses
        # NaN as the "TF absent from this window" sentinel, so one missing
        # window silently turns a TF's entire smoothed trajectory into NaN.
        series = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        assert np.isnan(SmoothedCurvesChromatin._smooth(series, 1.0)).all()


# --------------------------------------------------------------------------- #
# _process_single_window                                                        #
# --------------------------------------------------------------------------- #

class TestProcessSingleWindow:
    @pytest.mark.parametrize("window", sorted(BINDING_ROWS))
    def test_scores_are_the_mean_of_per_chromosome_means(self, binding_dir, window):
        idx, scores, counts = SmoothedCurvesChromatin._process_single_window(
            window, ALL_TFS, binding_dir
        )
        assert idx == window
        for tf, per_window in EXPECTED_BINDING.items():
            if window in per_window:
                assert scores[tf] == pytest.approx(per_window[window][0])
                assert counts[tf] == pytest.approx(per_window[window][1])

    def test_score_is_not_a_plain_global_mean(self, binding_dir):
        # window 1: chr1 = [1, 3], chr2 = [10].  Per-chromosome means -> 6.0;
        # a global mean would be 14/3 = 4.67.
        _, scores, _ = SmoothedCurvesChromatin._process_single_window(1, ["TFA"], binding_dir)
        assert scores["TFA"] == pytest.approx(6.0)
        assert scores["TFA"] != pytest.approx(14.0 / 3.0)

    def test_count_is_the_mean_per_chromosome_not_the_total(self, binding_dir):
        _, _, counts = SmoothedCurvesChromatin._process_single_window(1, ["TFA"], binding_dir)
        assert counts["TFA"] == pytest.approx(1.5)     # mean(2, 1), not 3

    def test_absent_tf_gets_nan_score_and_zero_count(self, binding_dir):
        _, scores, counts = SmoothedCurvesChromatin._process_single_window(
            2, ["TFA", "TFB"], binding_dir
        )
        assert np.isnan(scores["TFB"])
        assert counts["TFB"] == 0

    def test_none_discovers_the_tfs_present_in_the_file(self, binding_dir):
        _, scores, counts = SmoothedCurvesChromatin._process_single_window(2, None, binding_dir)
        assert set(scores) == {"TFA", "TFC"}
        assert set(counts) == {"TFA", "TFC"}

    def test_missing_file_with_explicit_tfs_returns_sentinels(self, binding_dir):
        _, scores, counts = SmoothedCurvesChromatin._process_single_window(
            99, ["TFA", "TFB"], binding_dir
        )
        assert all(np.isnan(v) for v in scores.values())
        assert set(counts.values()) == {0}

    def test_missing_file_with_tf_discovery_returns_empty_dicts(self, binding_dir):
        assert SmoothedCurvesChromatin._process_single_window(99, None, binding_dir) == (
            99, {}, {}
        )

    def test_malformed_loc_column_is_swallowed_silently(self, tmp_path):
        # ISSUE: the bare ``except Exception`` in the worker turns any file
        # problem -- here a two-field ``loc`` -- into an all-NaN window that is
        # indistinguishable from "TF genuinely absent".
        base = tmp_path / "bad"
        folder = base / "Subset1"
        folder.mkdir(parents=True)
        df = pd.DataFrame({"TF": ["TFA"], "loc": ["chr1:100-200"], "score": [1.0]})
        with gzip.open(folder / "binding.tsv.gz", "wt") as f:
            df.to_csv(f, sep="\t", index=False)
        _, scores, counts = SmoothedCurvesChromatin._process_single_window(
            1, ["TFA"], str(base)
        )
        assert np.isnan(scores["TFA"])
        assert counts["TFA"] == 0

    def test_file_without_loc_column_is_swallowed_silently(self, tmp_path):
        base = tmp_path / "noloc"
        folder = base / "Subset1"
        folder.mkdir(parents=True)
        df = pd.DataFrame({"TF": ["TFA"], "score": [1.0]})
        with gzip.open(folder / "binding.tsv.gz", "wt") as f:
            df.to_csv(f, sep="\t", index=False)
        _, scores, _ = SmoothedCurvesChromatin._process_single_window(1, ["TFA"], str(base))
        assert np.isnan(scores["TFA"])


# --------------------------------------------------------------------------- #
# extract_data                                                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.slow
class TestExtractData:
    def test_explicit_tf_list(self, binding_dir):
        a = SmoothedCurvesChromatin(["TFA", "TFB"], binding_dir)
        a.extract_data(n_windows=N_BINDING_WINDOWS, n_processes=2)
        assert set(a.raw_scores) == {"TFA", "TFB"}
        assert a.raw_scores["TFA"] == pytest.approx(expected_series("TFA", 0))
        assert a.raw_counts["TFA"] == pytest.approx(expected_series("TFA", 1))

    def test_missing_windows_keep_the_nan_and_zero_sentinels(self, binding_dir):
        a = SmoothedCurvesChromatin(["TFB"], binding_dir)
        a.extract_data(n_windows=N_BINDING_WINDOWS, n_processes=2)
        assert np.isnan(a.raw_scores["TFB"][1])       # TFB absent from window 2
        assert a.raw_counts["TFB"][1] == 0

    def test_tf_discovery_takes_the_sorted_union(self, binding_dir):
        a = SmoothedCurvesChromatin(None, binding_dir)
        a.extract_data(n_windows=N_BINDING_WINDOWS, n_processes=2)
        assert a.tfs == sorted(ALL_TFS)
        assert a.raw_scores["TFC"] == pytest.approx(expected_series("TFC", 0), nan_ok=True)

    def test_discovered_tfs_absent_from_a_window_stay_nan(self, binding_dir):
        a = SmoothedCurvesChromatin(None, binding_dir)
        a.extract_data(n_windows=N_BINDING_WINDOWS, n_processes=2)
        # TFC only appears in window 2
        assert np.isnan(a.raw_scores["TFC"][0]) and np.isnan(a.raw_scores["TFC"][2])
        assert a.raw_scores["TFC"][1] == pytest.approx(8.0)

    def test_requesting_more_windows_than_exist_pads_with_sentinels(self, binding_dir):
        a = SmoothedCurvesChromatin(["TFA"], binding_dir)
        a.extract_data(n_windows=5, n_processes=2)
        assert len(a.raw_scores["TFA"]) == 5
        assert np.isnan(a.raw_scores["TFA"][3]) and np.isnan(a.raw_scores["TFA"][4])

    def test_requesting_a_tf_that_is_nowhere_gives_all_nan(self, binding_dir):
        a = SmoothedCurvesChromatin(["NOT_A_TF", "TFA"], binding_dir)
        a.extract_data(n_windows=N_BINDING_WINDOWS, n_processes=2)
        assert np.isnan(a.raw_scores["NOT_A_TF"]).all()

    def test_no_data_at_all_raises_rather_than_returning_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        a = SmoothedCurvesChromatin(["TFA"], str(empty))
        with pytest.raises(FileNotFoundError, match="No binding data extracted"):
            a.extract_data(n_windows=3, n_processes=2)

    def test_no_data_at_all_raises_in_discovery_mode_too(self, tmp_path):
        empty = tmp_path / "empty2"
        empty.mkdir()
        a = SmoothedCurvesChromatin(None, str(empty))
        with pytest.raises(FileNotFoundError):
            a.extract_data(n_windows=3, n_processes=2)


# --------------------------------------------------------------------------- #
# set_trajectory_info                                                           #
# --------------------------------------------------------------------------- #

class TestSetTrajectoryInfo:
    def test_maps_indices_to_pseudotimes(self):
        a = SmoothedCurvesChromatin(["X"], "unused")
        a.set_trajectory_info([0, 2, 4], [1, 3], [0.0, 0.1, 0.2, 0.3, 0.4])
        assert a.pb_pseudotime == pytest.approx([0.0, 0.2, 0.4])
        assert a.gc_pseudotime == pytest.approx([0.1, 0.3])

    def test_gc_defaults_to_the_pb_pseudotime_frame(self):
        a = SmoothedCurvesChromatin(["X"], "unused")
        a.set_trajectory_info([0, 1], [1, 2], [5.0, 6.0, 7.0])
        assert a.gc_pseudotime == pytest.approx([6.0, 7.0])

    def test_separate_gc_frame_is_honoured(self):
        a = SmoothedCurvesChromatin(["X"], "unused")
        a.set_trajectory_info([0, 1], [1, 2], [5.0, 6.0, 7.0],
                              gc_window_pseudotimes=[50.0, 60.0, 70.0])
        assert a.pb_pseudotime == pytest.approx([5.0, 6.0])
        assert a.gc_pseudotime == pytest.approx([60.0, 70.0])

    def test_accepts_lists_and_arrays_alike(self):
        a = SmoothedCurvesChromatin(["X"], "unused")
        a.set_trajectory_info([0], [1], np.array([1.0, 2.0]))
        assert isinstance(a.window_pseudotimes, np.ndarray)

    def test_out_of_range_index_raises(self):
        a = SmoothedCurvesChromatin(["X"], "unused")
        with pytest.raises(IndexError):
            a.set_trajectory_info([0, 9], [1], [1.0, 2.0])


# --------------------------------------------------------------------------- #
# process_dynamics                                                              #
# --------------------------------------------------------------------------- #

class TestProcessDynamics:
    def test_requires_trajectory_info(self, dense_analyzer):
        with pytest.raises(ValueError, match="Trajectories not set"):
            dense_analyzer.process_dynamics()

    def test_orders_and_smooths_the_selected_windows(self, dense_analyzer):
        dense_analyzer.set_trajectory_info([0, 1, 2, 3], [0, 4, 5], np.arange(6.0))
        dense_analyzer.process_dynamics(metric="score", smooth_sigma=1.0)
        raw = np.array(dense_analyzer.raw_scores["X"])
        assert dense_analyzer.series_pb["X"] == pytest.approx(
            gaussian_filter1d(raw[[0, 1, 2, 3]], sigma=1.0)
        )
        assert dense_analyzer.series_gc["X"] == pytest.approx(
            gaussian_filter1d(raw[[0, 4, 5]], sigma=1.0)
        )

    def test_window_order_follows_the_index_list_not_the_file_order(self, dense_analyzer):
        dense_analyzer.set_trajectory_info([3, 2, 1, 0], [0, 1], np.arange(6.0))
        dense_analyzer.process_dynamics(smooth_sigma=NO_SMOOTHING)
        raw = np.array(dense_analyzer.raw_scores["X"])
        assert dense_analyzer.series_pb["X"] == pytest.approx(raw[[3, 2, 1, 0]])

    def test_count_metric_uses_the_count_table(self, dense_analyzer):
        dense_analyzer.set_trajectory_info([0, 1], [2, 3], np.arange(6.0))
        dense_analyzer.process_dynamics(metric="count", smooth_sigma=NO_SMOOTHING)
        assert dense_analyzer.series_pb["X"] == pytest.approx(
            np.array(dense_analyzer.raw_counts["X"])[[0, 1]]
        )

    def test_unknown_metric_silently_falls_back_to_counts(self, dense_analyzer):
        # ISSUE: ``metric`` is not validated; anything that is not the exact
        # string 'score' is treated as 'count'.
        dense_analyzer.set_trajectory_info([0, 1], [2, 3], np.arange(6.0))
        dense_analyzer.process_dynamics(metric="scores", smooth_sigma=NO_SMOOTHING)
        from_typo = dense_analyzer.series_pb["X"].copy()
        dense_analyzer.process_dynamics(metric="count", smooth_sigma=NO_SMOOTHING)
        assert from_typo == pytest.approx(dense_analyzer.series_pb["X"])

    def test_relative_normalises_both_branches_on_one_scale(self, dense_analyzer):
        dense_analyzer.set_trajectory_info([0, 1, 2, 3], [0, 4, 5], np.arange(6.0))
        dense_analyzer.process_dynamics(smooth_sigma=1.0, relative=True)
        both = np.concatenate([dense_analyzer.series_pb["X"], dense_analyzer.series_gc["X"]])
        assert both.min() == pytest.approx(0.0)
        assert both.max() == pytest.approx(1.0)
        assert (both >= 0).all() and (both <= 1).all()

    def test_relative_keeps_the_two_branches_comparable(self, dense_analyzer):
        # A branch that never reaches the global maximum must not be stretched
        # to 1 on its own.
        dense_analyzer.set_trajectory_info([0, 1], [0, 1, 2, 3, 4, 5], np.arange(6.0))
        dense_analyzer.process_dynamics(smooth_sigma=NO_SMOOTHING, relative=True)
        assert dense_analyzer.series_pb["X"].max() < 1.0
        assert dense_analyzer.series_gc["X"].max() == pytest.approx(1.0)

    def test_relative_handles_a_flat_signal_without_dividing_by_zero(self, tmp_path):
        base = tmp_path / "flat"
        base.mkdir()
        write_binding_windows(str(base), {w: [("X", "chr1", 5.0)] for w in range(1, 4)})
        a = SmoothedCurvesChromatin(["X"], str(base))
        a.extract_data(n_windows=3, n_processes=1)
        a.set_trajectory_info([0, 1], [1, 2], np.arange(3.0))
        a.process_dynamics(relative=True)
        assert np.isfinite(a.series_pb["X"]).all()
        assert a.series_pb["X"] == pytest.approx(np.zeros(2))

    def test_missing_window_poisons_the_whole_smoothed_series(self, binding_dir):
        # ISSUE (consequence of _smooth having no NaN handling): TFB is missing
        # from a single window and its entire trajectory becomes NaN.
        a = SmoothedCurvesChromatin(["TFB"], binding_dir)
        a.extract_data(n_windows=3, n_processes=1)
        a.set_trajectory_info([0, 1, 2], [0, 2], np.arange(3.0))
        a.process_dynamics()
        assert np.isnan(a.series_pb["TFB"]).all()

    def test_tf_without_raw_data_raises(self, dense_analyzer):
        dense_analyzer.tfs = list(dense_analyzer.tfs) + ["GHOST"]
        dense_analyzer.set_trajectory_info([0, 1], [2, 3], np.arange(6.0))
        with pytest.raises(IndexError):
            dense_analyzer.process_dynamics()


# --------------------------------------------------------------------------- #
# plotting                                                                      #
# --------------------------------------------------------------------------- #

class TestPlot:
    @pytest.fixture
    def ready(self, dense_analyzer):
        dense_analyzer.set_trajectory_info([0, 1, 2, 3, 4, 5], [0, 1, 2],
                                           np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
        dense_analyzer.process_dynamics(smooth_sigma=1.0)
        return dense_analyzer

    def test_requires_processed_data(self, dense_analyzer):
        with pytest.raises(ValueError, match="No processed data"):
            dense_analyzer.plot({"Static": {"X": "red"}})

    def test_two_traces_per_tf(self, ready):
        fig = ready.plot({"Static": {"X": "red"}, "Episodic": {"Y": "blue"}})
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4

    def test_unknown_tf_is_warned_about_and_skipped(self, ready, capsys):
        fig = ready.plot({"Static": {"GHOST": "red", "X": "blue"}})
        assert "GHOST not found" in capsys.readouterr().out
        assert len(fig.data) == 2

    def test_pb_is_truncated_to_the_gc_range(self, ready):
        fig = ready.plot({"Static": {"X": "red"}}, truncate_pb=True)
        pb_trace = fig.data[1]
        assert max(pb_trace.x) <= max(ready.gc_pseudotime)
        assert len(pb_trace.x) == len(pb_trace.y)

    def test_without_truncation_the_full_pb_branch_is_drawn(self, ready):
        fig = ready.plot({"Static": {"X": "red"}}, truncate_pb=False)
        assert len(fig.data[1].x) == len(ready.pb_pseudotime)

    def test_labels_and_title_are_passed_through(self, ready):
        fig = ready.plot({"Static": {"X": "red"}}, y_label="Signal", title="Mock")
        assert fig.layout.yaxis.title.text == "Signal"
        assert fig.layout.title.text == "Mock"

    def test_gc_is_dashed_and_pb_is_solid(self, ready):
        fig = ready.plot({"Static": {"X": "red"}})
        assert fig.data[0].line.dash == "dash"
        assert fig.data[1].line.dash == "solid"


class TestScoreVsCountPlot:
    @pytest.fixture
    def ready(self, dense_analyzer):
        dense_analyzer.set_trajectory_info([0, 1, 2], [3, 4, 5], np.arange(6.0))
        return dense_analyzer

    def test_requires_raw_data(self, dense_binding_dir):
        a = SmoothedCurvesChromatin(["X"], dense_binding_dir)
        with pytest.raises(ValueError, match="No raw data"):
            a.plot_score_vs_count_comparison({"Static": {"X": "red"}})

    def test_requires_trajectory_info(self, dense_analyzer):
        with pytest.raises(ValueError, match="Trajectories not set"):
            dense_analyzer.plot_score_vs_count_comparison({"Static": {"X": "red"}})

    def test_three_traces_per_tf(self, ready):
        fig = ready.plot_score_vs_count_comparison({"Static": {"X": "red", "Y": "blue"}})
        assert len(fig.data) == 6

    def test_subplot_grid_size(self, ready):
        fig = ready.plot_score_vs_count_comparison(
            {"Static": {"X": "red", "Y": "blue"}}, subplot_cols=1
        )
        assert fig.layout.height == 320 * 2
        assert fig.layout.width == 420 * 1

    def test_curves_are_min_max_normalised(self, ready):
        fig = ready.plot_score_vs_count_comparison({"Static": {"X": "red"}})
        score_trace, count_trace = fig.data[0], fig.data[1]
        for trace in (score_trace, count_trace):
            vals = np.asarray(trace.y, dtype=float)
            assert np.nanmin(vals) == pytest.approx(0.0)
            assert np.nanmax(vals) == pytest.approx(1.0)

    def test_shaded_band_runs_between_score_and_max_of_the_two(self, ready):
        # The fill polygon's lower edge is always the score curve; the upper
        # edge is max(count, score), i.e. the band is non-empty exactly where
        # the count exceeds the score.  (The `np.where(cond, score, score)`
        # that builds the lower edge has two identical branches -- redundant,
        # though not wrong.)
        fig = ready.plot_score_vs_count_comparison({"Static": {"X": "red"}})
        score = np.asarray(fig.data[0].y, dtype=float)
        count = np.asarray(fig.data[1].y, dtype=float)
        band = np.asarray(fig.data[2].y, dtype=float)
        n = len(score)
        assert band[:n] == pytest.approx(np.where(count > score, count, score))
        assert band[n:] == pytest.approx(score[::-1])
