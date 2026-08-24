"""Module-level helper functions of ``firefate.temporal._episodes``.

Everything here works on small hand-written frames whose expected output can be
computed by hand (or from ``scipy`` directly), independently of dictys.
"""

import os
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import hypergeom

from firefate.base import calculate_tf_episodic_enrichment
from firefate.temporal import (
    calculate_force_curves_parallel,
    filter_edges_by_significance_and_direction,
    get_episodic_grn_subset,
    get_unique_regs_by_target,
)
from firefate.temporal._forces import (
    calculate_force_curves_chunk,
    filter_chunk_of_edges,
)
from firefate.utils import create_balanced_chunks

LOG10E = 1.0 / np.log(10.0)


def edge_frame(rows, columns):
    """Build a (TF, Target) indexed frame from ``{(tf, target): values}``."""
    index = pd.MultiIndex.from_tuples(list(rows), names=["TF", "Target"])
    return pd.DataFrame(list(rows.values()), index=index, columns=columns)


# --------------------------------------------------------------------------- #
# calculate_tf_episodic_enrichment                                              #
# --------------------------------------------------------------------------- #

@pytest.fixture
def enrichment_grn():
    """A 6-edge episodic GRN with 2 LF genes among 5 distinct targets.

    ==== ================= ===== =====
    TF   targets           k     n
    ==== ================= ===== =====
    TFA  G1*, G2*, G3      2     3
    TFB  G4, G5            0     2
    TFC  G1*               1     1
    ==== ================= ===== =====

    (* = LF gene).  Population: N = 5 targets, K = 2 LF genes.
    """
    index = pd.MultiIndex.from_tuples(
        [("TFA", "G1"), ("TFA", "G2"), ("TFA", "G3"),
         ("TFB", "G4"), ("TFB", "G5"),
         ("TFC", "G1")],
        names=["TF", "Target"],
    )
    return pd.DataFrame(
        {
            "avg_force": [1.0, -2.0, 3.0, 4.0, -5.0, 6.0],
            "is_in_lf": [True, True, False, False, False, True],
        },
        index=index,
    )


class TestEnrichment:
    def test_pvalues_match_the_hypergeometric_survival_function(self, enrichment_grn):
        out = calculate_tf_episodic_enrichment(enrichment_grn, total_lf_genes=2,
                                               total_genes_in_grn=5).set_index("TF")
        # P(X >= k) with N=5, K=2, n=<targets>
        assert out.loc["TFA", "p_value"] == pytest.approx(hypergeom.sf(1, 5, 2, 3))
        assert out.loc["TFB", "p_value"] == pytest.approx(hypergeom.sf(-1, 5, 2, 2))
        assert out.loc["TFC", "p_value"] == pytest.approx(hypergeom.sf(0, 5, 2, 1))

    def test_pvalues_against_hand_computed_values(self, enrichment_grn):
        out = calculate_tf_episodic_enrichment(enrichment_grn, 2, 5).set_index("TF")
        # TFA: P(X>=2) = C(2,2)C(3,1)/C(5,3) = 3/10
        assert out.loc["TFA", "p_value"] == pytest.approx(0.3)
        # TFB: k = 0, so P(X >= 0) = 1
        assert out.loc["TFB", "p_value"] == pytest.approx(1.0)
        # TFC: P(X>=1) = 1 - C(3,1)/C(5,1) = 0.4
        assert out.loc["TFC", "p_value"] == pytest.approx(0.4)

    def test_enrichment_score_is_observed_over_expected(self, enrichment_grn):
        out = calculate_tf_episodic_enrichment(enrichment_grn, 2, 5).set_index("TF")
        assert out.loc["TFA", "enrichment_score"] == pytest.approx(2 / (3 * 2 / 5))
        assert out.loc["TFB", "enrichment_score"] == pytest.approx(0.0)
        assert out.loc["TFC", "enrichment_score"] == pytest.approx(1 / (1 * 2 / 5))

    def test_reports_lf_and_non_lf_targets_with_weights(self, enrichment_grn):
        out = calculate_tf_episodic_enrichment(enrichment_grn, 2, 5).set_index("TF")
        assert out.loc["TFA", "genes_in_lf"] == ("G1", "G2")
        assert out.loc["TFA", "genes_dwnstrm"] == ("G3",)
        assert out.loc["TFA", "weights"] == (1.0, -2.0)
        assert out.loc["TFB", "genes_in_lf"] == ()

    def test_one_row_per_tf_with_the_expected_columns(self, enrichment_grn):
        out = calculate_tf_episodic_enrichment(enrichment_grn, 2, 5)
        assert list(out.columns) == [
            "TF", "p_value", "enrichment_score", "genes_in_lf", "genes_dwnstrm", "weights",
        ]
        assert sorted(out["TF"]) == ["TFA", "TFB", "TFC"]

    def test_no_lf_genes_in_the_population_disables_the_test(self, enrichment_grn):
        out = calculate_tf_episodic_enrichment(enrichment_grn, total_lf_genes=0,
                                               total_genes_in_grn=5).set_index("TF")
        assert (out["p_value"] == 1.0).all()
        assert (out["enrichment_score"] == 0).all()

    def test_a_tf_whose_targets_are_all_lf_is_maximally_enriched(self):
        index = pd.MultiIndex.from_tuples([("TFA", "G1"), ("TFA", "G2")],
                                          names=["TF", "Target"])
        df = pd.DataFrame({"avg_force": [1.0, 1.0], "is_in_lf": [True, True]}, index=index)
        out = calculate_tf_episodic_enrichment(df, total_lf_genes=2,
                                               total_genes_in_grn=10).set_index("TF")
        assert out.loc["TFA", "p_value"] == pytest.approx(hypergeom.sf(1, 10, 2, 2))
        assert out.loc["TFA", "enrichment_score"] == pytest.approx(2 / (2 * 2 / 10))

    def test_single_target_tf_is_handled(self, enrichment_grn):
        # df.loc[tf] on a MultiIndex still yields a frame for a single edge
        out = calculate_tf_episodic_enrichment(enrichment_grn, 2, 5).set_index("TF")
        assert out.loc["TFC", "genes_in_lf"] == ("G1",)

    def test_empty_grn_gives_an_empty_result(self):
        index = pd.MultiIndex.from_tuples([], names=["TF", "Target"])
        df = pd.DataFrame({"avg_force": [], "is_in_lf": []}, index=index)
        assert len(calculate_tf_episodic_enrichment(df, 0, 0)) == 0


# --------------------------------------------------------------------------- #
# get_unique_regs_by_target                                                     #
# --------------------------------------------------------------------------- #

class TestUniqueRegsByTarget:
    def test_groups_regulators_per_target(self):
        index = pd.MultiIndex.from_tuples(
            [("TFA", "G1"), ("TFB", "G1"), ("TFA", "G2")], names=["TF", "Target"]
        )
        df = pd.DataFrame({"avg_force": [1.0, 2.0, 3.0]}, index=index)
        out = get_unique_regs_by_target(df)
        assert out == {
            "G1": [("TFA", "G1"), ("TFB", "G1")],
            "G2": [("TFA", "G2")],
        }

    def test_duplicate_edges_are_collapsed(self):
        index = pd.MultiIndex.from_tuples(
            [("TFA", "G1"), ("TFA", "G1")], names=["TF", "Target"]
        )
        df = pd.DataFrame({"avg_force": [1.0, 2.0]}, index=index)
        assert get_unique_regs_by_target(df) == {"G1": [("TFA", "G1")]}

    def test_empty_input(self):
        index = pd.MultiIndex.from_tuples([], names=["TF", "Target"])
        assert get_unique_regs_by_target(pd.DataFrame({"avg_force": []}, index=index)) == {}


# --------------------------------------------------------------------------- #
# filter_chunk_of_edges                                                         #
# --------------------------------------------------------------------------- #

TIME_COLS = [f"time_{i}" for i in range(5)]

FILTER_ROWS = {
    ("TFA", "keep_positive"): [1.0, 1.1, 0.9, 1.05, 0.95],
    ("TFA", "keep_negative"): [-1.0, -1.1, -0.9, -1.05, -0.95],
    ("TFA", "too_few_nonzero"): [1.0, 1.0, 0.0, 0.0, 0.0],
    ("TFA", "not_significant"): [1.0, -1.0, 1.0, -1.0, 1.2],
    ("TFA", "direction_variant"): [5.0, 5.0, 5.0, 5.0, -0.1],
    ("TFA", "all_zero"): [0.0, 0.0, 0.0, 0.0, 0.0],
    ("TFA", "exactly_three_nonzero"): [2.0, 2.1, 1.9, 0.0, 0.0],
    ("TFA", "single_nonzero"): [3.0, 0.0, 0.0, 0.0, 0.0],
    ("TFA", "constant_nonzero"): [4.0, 4.0, 4.0, 4.0, 4.0],
}


@pytest.fixture
def filter_df():
    return edge_frame(FILTER_ROWS, TIME_COLS)


def run_chunk(df, **kwargs):
    return {idx: (keep, p) for idx, keep, p in
            filter_chunk_of_edges(df.index, df=df, time_cols=TIME_COLS, **kwargs)}


class TestFilterChunk:
    def test_consistent_edges_are_kept_with_their_pvalue(self, filter_df):
        out = run_chunk(filter_df)
        for name in ["keep_positive", "keep_negative", "exactly_three_nonzero"]:
            keep, p = out[("TFA", name)]
            values = np.array(FILTER_ROWS[("TFA", name)])
            expected_p = stats.ttest_1samp(values[values != 0], 0).pvalue
            assert keep is True or keep == np.True_, name
            assert p == pytest.approx(expected_p)

    def test_too_few_nonzero_timepoints_is_rejected_without_a_pvalue(self, filter_df):
        for name in ["too_few_nonzero", "all_zero"]:
            keep, p = run_chunk(filter_df)[("TFA", name)]
            assert not keep
            assert np.isnan(p)

    def test_non_significant_edge_is_rejected_but_reports_its_pvalue(self, filter_df):
        keep, p = run_chunk(filter_df)[("TFA", "not_significant")]
        values = np.array(FILTER_ROWS[("TFA", "not_significant")])
        assert not keep
        assert p == pytest.approx(stats.ttest_1samp(values, 0).pvalue)
        assert p > 0.05

    def test_sign_changing_edge_is_rejected_when_invariance_is_checked(self, filter_df):
        keep, p = run_chunk(filter_df)[("TFA", "direction_variant")]
        assert not keep
        assert p < 0.05          # it *was* significant, only the sign flip rejected it

    def test_sign_changing_edge_is_kept_when_invariance_is_off(self, filter_df):
        out = run_chunk(filter_df, check_direction_invariance=False)
        keep, _ = out[("TFA", "direction_variant")]
        assert keep

    def test_alpha_controls_significance(self, filter_df):
        strict = run_chunk(filter_df, alpha=1e-12)
        assert not strict[("TFA", "keep_positive")][0]
        lenient = run_chunk(filter_df, alpha=0.999, check_direction_invariance=False)
        assert lenient[("TFA", "not_significant")][0]

    def test_min_nonzero_timepoints_is_applied(self, filter_df):
        out = run_chunk(filter_df, min_nonzero_timepoints=5)
        assert not out[("TFA", "exactly_three_nonzero")][0]
        assert out[("TFA", "keep_positive")][0]

    def test_min_observations_is_applied_independently(self, filter_df):
        # both thresholds count the same non-zero values, so raising only
        # min_observations must reject the same rows
        out = run_chunk(filter_df, min_nonzero_timepoints=1, min_observations=5)
        assert not out[("TFA", "exactly_three_nonzero")][0]
        assert np.isnan(out[("TFA", "exactly_three_nonzero")][1])

    def test_a_single_observation_cannot_be_tested(self, filter_df):
        # ttest of one value yields nan, which fails ``p < alpha`` -> rejected
        out = run_chunk(filter_df, min_nonzero_timepoints=1, min_observations=1)
        keep, p = out[("TFA", "single_nonzero")]
        assert not keep
        assert np.isnan(p)

    def test_a_perfectly_constant_edge_gets_a_degenerate_pvalue_of_zero(self, filter_df):
        # Zero variance makes the one-sample t statistic infinite, so scipy
        # returns p == 0 exactly: such an edge always survives any alpha.
        keep, p = run_chunk(filter_df)[("TFA", "constant_nonzero")]
        assert keep
        assert p == 0.0

    def test_every_row_is_reported_exactly_once(self, filter_df):
        out = filter_chunk_of_edges(filter_df.index, df=filter_df, time_cols=TIME_COLS)
        assert [r[0] for r in out] == list(filter_df.index)

    def test_a_subset_of_indices_can_be_processed(self, filter_df):
        subset = filter_df.index[[0, 3]]
        out = filter_chunk_of_edges(subset, df=filter_df, time_cols=TIME_COLS)
        assert [r[0] for r in out] == list(subset)


# --------------------------------------------------------------------------- #
# filter_edges_by_significance_and_direction                                    #
# --------------------------------------------------------------------------- #

@pytest.mark.slow
class TestFilterEdges:
    def test_keeps_only_the_passing_edges(self, filter_df):
        out = filter_edges_by_significance_and_direction(filter_df, n_processes=2)
        assert set(out.index.get_level_values(1)) == {
            "keep_positive", "keep_negative", "exactly_three_nonzero", "constant_nonzero",
        }

    def test_adds_a_pvalue_column_and_keeps_the_time_columns(self, filter_df):
        out = filter_edges_by_significance_and_direction(filter_df, n_processes=2)
        assert list(out.columns) == TIME_COLS + ["p_value"]
        for idx in out.index:
            assert out.loc[idx, TIME_COLS].values == pytest.approx(filter_df.loc[idx].values)

    def test_pvalues_match_a_serial_recomputation(self, filter_df):
        out = filter_edges_by_significance_and_direction(filter_df, n_processes=2)
        for idx in out.index:
            values = filter_df.loc[idx].values
            expected = stats.ttest_1samp(values[values != 0], 0).pvalue
            assert out.loc[idx, "p_value"] == pytest.approx(expected)

    def test_original_row_order_is_preserved(self, filter_df):
        out = filter_edges_by_significance_and_direction(filter_df, n_processes=2)
        original = [i for i in filter_df.index if i in set(out.index)]
        assert list(out.index) == original

    def test_row_order_survives_out_of_order_chunk_completion(self):
        rows = {("TF%d" % i, "G%d" % i): [1.0 + 0.01 * i] * 5 for i in range(40)}
        df = edge_frame(rows, TIME_COLS)
        out = filter_edges_by_significance_and_direction(
            df, n_processes=4, chunk_size=3
        )
        assert list(out.index) == list(df.index)

    def test_chunking_does_not_change_the_result(self, filter_df):
        one = filter_edges_by_significance_and_direction(filter_df, n_processes=2,
                                                         chunk_size=1000)
        many = filter_edges_by_significance_and_direction(filter_df, n_processes=2,
                                                          chunk_size=2)
        pd.testing.assert_frame_equal(one, many)

    def test_an_existing_pvalue_column_is_replaced_not_tested(self, filter_df):
        df = filter_df.copy()
        df["p_value"] = 0.5
        out = filter_edges_by_significance_and_direction(df, n_processes=2)
        assert list(out.columns) == TIME_COLS + ["p_value"]
        assert (out["p_value"] != 0.5).all()

    def test_direction_invariance_can_be_disabled(self, filter_df):
        out = filter_edges_by_significance_and_direction(
            filter_df, n_processes=2, check_direction_invariance=False
        )
        assert ("TFA", "direction_variant") in out.index

    def test_nothing_passing_yields_an_empty_frame(self):
        df = edge_frame({("TFA", "G1"): [0.0] * 5}, TIME_COLS)
        out = filter_edges_by_significance_and_direction(df, n_processes=2)
        assert len(out) == 0
        assert list(out.columns) == TIME_COLS + ["p_value"]


# --------------------------------------------------------------------------- #
# create_balanced_chunks                                                        #
# --------------------------------------------------------------------------- #

class TestBalancedChunks:
    @pytest.mark.parametrize(
        "n_rows,n_chunks,expected",
        [
            (10, 3, [4, 3, 3]),
            (10, 1, [10]),
            (10, 10, [1] * 10),
            (9, 2, [5, 4]),
            (7, 4, [2, 2, 2, 1]),
        ],
    )
    def test_sizes_are_balanced_with_the_remainder_up_front(self, n_rows, n_chunks, expected):
        df = pd.DataFrame({"a": range(n_rows)})
        assert [len(c) for c in create_balanced_chunks(df, n_chunks)] == expected

    def test_chunks_concatenate_back_to_the_original(self):
        df = pd.DataFrame({"a": range(23)}, index=[f"r{i}" for i in range(23)])
        rebuilt = pd.concat(create_balanced_chunks(df, 5))
        pd.testing.assert_frame_equal(rebuilt, df)

    def test_more_chunks_than_rows_drops_the_empty_ones(self):
        df = pd.DataFrame({"a": range(3)})
        chunks = create_balanced_chunks(df, 10)
        assert [len(c) for c in chunks] == [1, 1, 1]

    def test_zero_chunks_raises(self):
        with pytest.raises(ZeroDivisionError):
            create_balanced_chunks(pd.DataFrame({"a": [1, 2]}), 0)

    def test_empty_frame_gives_no_chunks(self):
        assert create_balanced_chunks(pd.DataFrame({"a": []}), 3) == []


# --------------------------------------------------------------------------- #
# force curves                                                                  #
# --------------------------------------------------------------------------- #

def expected_force(beta, tf_expr, epsilon=1e-10):
    """``sign(b) * exp(log10(|b| + eps) + log10(t + eps))``."""
    return np.sign(beta) * ((np.abs(beta) + epsilon) * (tf_expr + epsilon)) ** LOG10E


@pytest.fixture
def force_inputs():
    """Beta curves grouped by TF, in *descending target count* order.

    TFB (2 targets) comes before TFA (1 target), which is the order
    ``value_counts()`` produces -- the only layout for which the positional
    ``np.repeat`` in ``calculate_force_curves_chunk`` lines up.
    """
    index = pd.MultiIndex.from_tuples(
        [("TFB", "G2"), ("TFB", "G3"), ("TFA", "G1")], names=["TF", "Target"]
    )
    cols = ["time_0", "time_1"]
    beta = pd.DataFrame([[2.0, -2.0], [0.5, 0.5], [-4.0, 1.0]], index=index, columns=cols)
    expr = pd.DataFrame([[10.0, 20.0], [100.0, 1.0]], index=["TFA", "TFB"], columns=cols)
    return beta, expr


class TestForceCurvesChunk:
    def test_matches_the_implemented_log_formula(self, force_inputs):
        beta, expr = force_inputs
        out = calculate_force_curves_chunk(beta, expr)
        tf_values = expr.loc[["TFB", "TFB", "TFA"]].values
        assert out.values == pytest.approx(expected_force(beta.values, tf_values))

    def test_preserves_index_columns_and_sign(self, force_inputs):
        beta, expr = force_inputs
        out = calculate_force_curves_chunk(beta, expr)
        assert out.index.equals(beta.index)
        assert list(out.columns) == list(beta.columns)
        assert np.array_equal(np.sign(out.values), np.sign(beta.values))

    def test_epsilon_is_configurable(self, force_inputs):
        beta, expr = force_inputs
        a = calculate_force_curves_chunk(beta, expr, epsilon=1e-10)
        b = calculate_force_curves_chunk(beta, expr, epsilon=1e-2)
        assert not np.allclose(a.values, b.values)
        assert b.values == pytest.approx(
            expected_force(beta.values, expr.loc[["TFB", "TFB", "TFA"]].values, 1e-2)
        )

    def test_zero_beta_stays_zero(self):
        index = pd.MultiIndex.from_tuples([("TFA", "G1")], names=["TF", "Target"])
        beta = pd.DataFrame([[0.0]], index=index, columns=["time_0"])
        expr = pd.DataFrame([[5.0]], index=["TFA"], columns=["time_0"])
        assert calculate_force_curves_chunk(beta, expr).values == pytest.approx(
            np.zeros((1, 1))
        )

    def test_missing_tf_expression_raises(self):
        index = pd.MultiIndex.from_tuples([("GHOST", "G1")], names=["TF", "Target"])
        beta = pd.DataFrame([[1.0]], index=index, columns=["time_0"])
        expr = pd.DataFrame([[5.0]], index=["TFA"], columns=["time_0"])
        with pytest.raises(KeyError):
            calculate_force_curves_chunk(beta, expr)

    @pytest.mark.xfail(
        strict=True,
        reason="ISSUES.md #1 (CONFIRMED, high) -- calculate_force_curves_chunk reindexes "
               "tf_expression to the value_counts() order (descending target count) and "
               "then repeats it positionally onto beta_chunk's rows.  Unless the beta "
               "rows happen to be grouped in that same order, every edge is multiplied "
               "by another TF's expression.  Rows coming out of build_episode_grn are "
               "grouped in network order, and filter_edges then drops rows so the target "
               "counts become unequal -- so the episodic force curves are mis-assigned. "
               "get_tf_indices is NOT in this code path (it is called only from "
               "get_beta_curves), so it does not account for the ordering here.",
    )
    def test_expression_is_matched_by_tf_name_not_by_row_order(self):
        index = pd.MultiIndex.from_tuples(
            [("TFA", "G1"), ("TFB", "G2"), ("TFB", "G3")], names=["TF", "Target"]
        )
        beta = pd.DataFrame([[1.0], [1.0], [1.0]], index=index, columns=["time_0"])
        expr = pd.DataFrame([[10.0], [1000.0]], index=["TFA", "TFB"], columns=["time_0"])
        out = calculate_force_curves_chunk(beta, expr)
        expected = expected_force(beta.values, np.array([[10.0], [1000.0], [1000.0]]))
        assert out.values == pytest.approx(expected)

    def test_currently_swaps_expression_between_tfs(self):
        # The concrete symptom of the issue above, pinned so the behaviour
        # change is caught either way.
        index = pd.MultiIndex.from_tuples(
            [("TFA", "G1"), ("TFB", "G2"), ("TFB", "G3")], names=["TF", "Target"]
        )
        beta = pd.DataFrame([[1.0], [1.0], [1.0]], index=index, columns=["time_0"])
        expr = pd.DataFrame([[10.0], [1000.0]], index=["TFA", "TFB"], columns=["time_0"])
        out = calculate_force_curves_chunk(beta, expr)
        assert out.loc[("TFA", "G1"), "time_0"] == pytest.approx(
            expected_force(1.0, 1000.0)          # TFB's expression!
        )


@pytest.mark.slow
class TestForceCurvesParallel:
    def test_matches_the_single_chunk_computation(self, force_inputs):
        beta, expr = force_inputs
        out = calculate_force_curves_parallel(beta, expr, n_processes=2, chunk_size=1000)
        pd.testing.assert_frame_equal(out, calculate_force_curves_chunk(beta, expr))

    def test_row_order_is_restored_after_chunking(self):
        rows = {("TF%d" % i, "G%d" % i): [1.0 + i] for i in range(25)}
        beta = edge_frame(rows, ["time_0"])
        expr = pd.DataFrame(
            {"time_0": [2.0] * 25}, index=[f"TF{i}" for i in range(25)]
        )
        out = calculate_force_curves_parallel(beta, expr, n_processes=2, chunk_size=4)
        assert list(out.index) == list(beta.index)
        assert out.values == pytest.approx(expected_force(beta.values, expr.values))

    def test_only_time_prefixed_columns_are_used(self, force_inputs):
        beta, expr = force_inputs
        beta = beta.copy()
        beta["p_value"] = 0.01
        expr = expr.copy()
        expr["p_value"] = 0.0
        out = calculate_force_curves_parallel(beta, expr, n_processes=2, chunk_size=1000)
        assert list(out.columns) == ["time_0", "time_1"]

    def test_columns_not_named_time_something_are_silently_dropped(self):
        # ISSUES.md #16 -- reviewed, not actioned.  The time columns are found
        # by a ``startswith("time_")`` prefix match, so a frame that names its
        # pseudotime columns anything else produces a result with no columns at
        # all instead of an error.  Pinned so the convention cannot drift.
        index = pd.MultiIndex.from_tuples([("TFA", "G1")], names=["TF", "Target"])
        beta = pd.DataFrame([[1.0]], index=index, columns=["t0"])
        expr = pd.DataFrame([[1.0]], index=["TFA"], columns=["t0"])
        out = calculate_force_curves_parallel(beta, expr, n_processes=2)
        assert out.shape == (1, 0)


# --------------------------------------------------------------------------- #
# get_episodic_grn_subset                                                       #
# --------------------------------------------------------------------------- #

def write_episode_pickle(folder, idx, rows):
    index = pd.MultiIndex.from_tuples(list(rows), names=["TF", "Target"])
    df = pd.DataFrame({"avg_force": list(rows.values())}, index=index)
    with open(os.path.join(folder, f"episode_{idx}.pkl"), "wb") as f:
        pickle.dump(df, f)


@pytest.fixture
def episode_folder(tmp_path):
    folder = tmp_path / "episodes"
    folder.mkdir()
    write_episode_pickle(str(folder), 0, {("TFA", "G1"): 1.0, ("TFA", "G2"): 2.0})
    write_episode_pickle(str(folder), 1, {("TFA", "G1"): 3.0, ("TFB", "G3"): 9.0})
    write_episode_pickle(str(folder), 2, {("TFB", "G1"): -4.0})
    return str(folder)


class TestEpisodicGrnSubset:
    def test_builds_an_edge_by_episode_matrix(self, episode_folder):
        out = get_episodic_grn_subset(episode_folder, ["TFA", "TFB"], ["G1", "G2"])
        assert list(out.columns) == [0, 1, 2]
        assert set(out.index) == {("TFA", "G1"), ("TFA", "G2"), ("TFB", "G1")}
        assert out.loc[("TFA", "G1")].tolist() == [1.0, 3.0, 0.0]
        assert out.loc[("TFB", "G1")].tolist() == [0.0, 0.0, -4.0]

    def test_absent_edges_are_filled_with_zero(self, episode_folder):
        out = get_episodic_grn_subset(episode_folder, ["TFA"], ["G2"])
        assert out.loc[("TFA", "G2")].tolist() == [2.0, 0.0, 0.0]

    def test_targets_outside_the_selection_are_dropped(self, episode_folder):
        out = get_episodic_grn_subset(episode_folder, ["TFA", "TFB"], ["G1"])
        assert set(out.index) == {("TFA", "G1"), ("TFB", "G1")}

    def test_index_and_column_names(self, episode_folder):
        out = get_episodic_grn_subset(episode_folder, ["TFA"], ["G1"])
        assert list(out.index.names) == ["TF", "target"]
        assert out.columns.name == "episode"

    def test_episode_columns_are_sorted_numerically(self, tmp_path):
        folder = tmp_path / "many"
        folder.mkdir()
        for idx in [10, 2, 1]:
            write_episode_pickle(str(folder), idx, {("TFA", "G1"): float(idx)})
        out = get_episodic_grn_subset(str(folder), ["TFA"], ["G1"])
        assert list(out.columns) == [1, 2, 10]

    def test_alternative_value_column(self, tmp_path):
        folder = tmp_path / "alt"
        folder.mkdir()
        index = pd.MultiIndex.from_tuples([("TFA", "G1")], names=["TF", "Target"])
        df = pd.DataFrame({"avg_force": [1.0], "other": [7.0]}, index=index)
        with open(folder / "episode_0.pkl", "wb") as f:
            pickle.dump(df, f)
        out = get_episodic_grn_subset(str(folder), ["TFA"], ["G1"], value_col="other")
        assert out.loc[("TFA", "G1")].tolist() == [7.0]

    def test_missing_folder_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No episode pkl files"):
            get_episodic_grn_subset(str(tmp_path / "nothing"), ["TFA"], ["G1"])

    def test_does_not_read_the_parquet_files_the_runners_write(self, tmp_path):
        # ISSUES.md #6 (CONFIRMED) -- run_episodic_construction saves
        # ``episode_<i>.parquet`` but this reader only globs ``episode_<i>.pkl``,
        # so the two halves of the workflow cannot be chained without a manual
        # conversion.  Whichever format wins, this test must be updated with it.
        folder = tmp_path / "parquet_only"
        folder.mkdir()
        index = pd.MultiIndex.from_tuples([("TFA", "G1")], names=["TF", "Target"])
        pd.DataFrame({"avg_force": [1.0]}, index=index).to_parquet(
            folder / "episode_0.parquet"
        )
        with pytest.raises(FileNotFoundError):
            get_episodic_grn_subset(str(folder), ["TFA"], ["G1"])
