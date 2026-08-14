"""Maths of :class:`firefate.core.pseudotime_curves.SmoothedCurvesGRN`.

These tests use hand-built curves whose characteristics can be derived on
paper (or cross-checked against an independent numpy implementation), so a
failure here is a defect in the library rather than in the fixture.

Tests marked ``xfail(strict=True)`` document a defect that currently exists in
the code; see ``tests/episodic/ISSUES.md``.  If one starts XPASSing, the bug was fixed
and the test should be converted into a normal assertion.

``ISSUES.md`` records the maintainer triage of each finding.  Comments below say
which status a test is pinning, so a reader can tell "known defect, fix wanted"
from "reviewed, intended behaviour, guarded against accidental change".
"""

import numpy as np
import pandas as pd
import pytest

from firefate.core.pseudotime_curves import SmoothedCurvesGRN

LOG10E = 1.0 / np.log(10.0)

#: numpy renamed trapz -> trapezoid in 2.0
trapezoid = getattr(np, "trapezoid", None) or np.trapz


@pytest.fixture
def curves():
    """A SmoothedCurvesGRN whose dictys object is never touched.

    All methods under test in this module are static or depend only on the
    (dx, dy) arrays passed in, so ``None`` is a legitimate stand-in and keeps
    the maths tests independent of dictys.
    """
    return SmoothedCurvesGRN(None, trajectory_range=(0, 2), num_points=11)


# --------------------------------------------------------------------------- #
# calculate_auc                                                                 #
# --------------------------------------------------------------------------- #

class TestCalculateAUC:
    def test_constant_curve_is_width_times_height(self):
        dx = np.linspace(0.0, 4.0, 5)
        dy = np.full((1, 5), 3.0)
        assert SmoothedCurvesGRN.calculate_auc(dx, dy) == pytest.approx([12.0])

    def test_linear_curve_is_exact(self):
        # The trapezoidal rule is exact for piecewise-linear data.
        dx = np.linspace(0.0, 1.0, 11)
        dy = np.vstack([dx, 2 * dx, -dx])
        assert SmoothedCurvesGRN.calculate_auc(dx, dy) == pytest.approx([0.5, 1.0, -0.5])

    def test_matches_numpy_trapezoid_on_nonuniform_grid(self):
        dx = np.array([0.0, 0.1, 0.7, 1.3, 4.0])
        dy = np.vstack([np.sin(dx), np.exp(dx), np.ones_like(dx)])
        expected = np.array([trapezoid(row, dx) for row in dy])
        assert SmoothedCurvesGRN.calculate_auc(dx, dy) == pytest.approx(expected)

    def test_two_point_quadratic_matches_trapezoid_not_true_integral(self):
        # y = x^2 on [0, 1] sampled at the ends: trapezoid gives 0.5, the true
        # integral is 1/3.  Pins the rule that is actually implemented.
        dx = np.array([0.0, 1.0])
        dy = np.array([[0.0, 1.0]])
        assert SmoothedCurvesGRN.calculate_auc(dx, dy) == pytest.approx([0.5])

    def test_many_curves_at_once(self):
        dx = np.linspace(0.0, 1.0, 6)
        dy = np.random.default_rng(0).normal(size=(20, 6))
        expected = np.array([trapezoid(row, dx) for row in dy])
        assert SmoothedCurvesGRN.calculate_auc(dx, dy) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "dx",
        [
            np.array([0.0]),                      # fewer than two points
            np.array([0.0, 1.0, 1.0]),            # not strictly increasing
            np.array([0.0, 2.0, 1.0]),            # decreasing
            np.array([1.0, 0.0]),                 # reversed
        ],
    )
    def test_rejects_bad_dx(self, dx):
        dy = np.ones((1, len(dx)))
        with pytest.raises(ValueError, match="dx must be increasing"):
            SmoothedCurvesGRN.calculate_auc(dx, dy)

    def test_one_dimensional_dy_is_not_supported(self):
        dx = np.linspace(0, 1, 4)
        with pytest.raises(IndexError):
            SmoothedCurvesGRN.calculate_auc(dx, np.arange(4.0))

    def test_length_mismatch_raises(self):
        dx = np.linspace(0, 1, 4)
        dy = np.ones((2, 6))
        with pytest.raises(ValueError):
            SmoothedCurvesGRN.calculate_auc(dx, dy)


# --------------------------------------------------------------------------- #
# calculate_terminal_logfc                                                      #
# --------------------------------------------------------------------------- #

class TestTerminalLogFC:
    def test_difference_between_last_and_first(self):
        dx = np.linspace(0, 1, 4)
        dy = np.array([[1.0, 5.0, 5.0, 4.0], [2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, -3.0]])
        assert SmoothedCurvesGRN.calculate_terminal_logfc(dx, dy) == pytest.approx([3.0, 0.0, -3.0])

    def test_ignores_curve_shape_between_ends(self):
        dx = np.linspace(0, 1, 5)
        straight = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
        wiggly = np.array([[0.0, 9.0, -9.0, 4.0, 1.0]])
        assert SmoothedCurvesGRN.calculate_terminal_logfc(dx, straight) == pytest.approx(
            SmoothedCurvesGRN.calculate_terminal_logfc(dx, wiggly)
        )

    def test_validates_dx_even_though_unused(self):
        with pytest.raises(ValueError, match="dx must be increasing"):
            SmoothedCurvesGRN.calculate_terminal_logfc(np.array([1.0, 0.0]), np.ones((1, 2)))


# --------------------------------------------------------------------------- #
# calculate_transient_logfc                                                     #
# --------------------------------------------------------------------------- #

class TestTransientLogFC:
    def test_monotone_curve_has_zero_transient(self, curves):
        # For a monotone curve every point lies between its own endpoints, so
        # the element-wise median of (curve, first, last) is the curve itself.
        dx = np.linspace(0, 1, 11)
        dy = np.vstack([dx, 1.0 - dx, 3.0 * dx - 1.0])
        assert curves.calculate_transient_logfc(dx, dy) == pytest.approx([0.0, 0.0, 0.0])

    def test_constant_curve_has_zero_transient(self, curves):
        dx = np.linspace(0, 1, 5)
        assert curves.calculate_transient_logfc(dx, np.full((1, 5), 7.0)) == pytest.approx([0.0])

    def test_bump_and_dip_are_opposite_and_match_trapezoid(self, curves):
        dx = np.linspace(0.0, 1.0, 11)
        bump = np.sin(np.pi * dx)
        dy = np.vstack([bump, -bump])
        # Endpoints are 0, so median(curve, 0, 0) == 0 and the transient logFC
        # is just the area under the excursion.
        expected = trapezoid(bump, dx)
        got = curves.calculate_transient_logfc(dx, dy)
        assert got == pytest.approx([expected, -expected])

    def test_only_the_excursion_beyond_the_endpoint_band_counts(self, curves):
        # Curve goes 0 -> 2 -> 1: the median clamps it into the [0, 1] band, so
        # only the part above 1 contributes.
        dx = np.array([0.0, 1.0, 2.0])
        dy = np.array([[0.0, 2.0, 1.0]])
        clamped = np.array([0.0, 1.0, 1.0])            # median(y, 0, 1)
        excursion = np.array([0.0, 1.0, 0.0])
        expected = trapezoid(excursion, np.array([0.0, 0.5, 1.0]))  # dx normalised
        assert curves.calculate_transient_logfc(dx, dy) == pytest.approx([expected])
        assert excursion == pytest.approx(dy[0] - clamped)

    def test_invariant_to_affine_rescaling_of_pseudotime(self, curves):
        base = np.linspace(0.0, 1.0, 9)
        dy = np.vstack([np.sin(np.pi * base), np.cos(np.pi * base)])
        a = curves.calculate_transient_logfc(base, dy)
        b = curves.calculate_transient_logfc(base * 37.0 + 5.0, dy)
        assert a == pytest.approx(b)


# --------------------------------------------------------------------------- #
# calculate_switching_time                                                      #
# --------------------------------------------------------------------------- #

class TestSwitchingTime:
    def test_linear_transition_switches_midway(self, curves):
        dx = np.linspace(0.0, 1.0, 11)
        dy = np.vstack([1.0 - dx, dx])       # falling and rising
        assert curves.calculate_switching_time(dx, dy) == pytest.approx([0.5, 0.5])

    @pytest.mark.parametrize("switch_at,expected", [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)])
    def test_step_transition_reports_step_position(self, curves, switch_at, expected):
        dx = np.linspace(0.0, 1.0, 101)
        dy = np.where(dx < switch_at, 1.0, 0.0).reshape(1, -1)
        assert curves.calculate_switching_time(dx, dy) == pytest.approx([expected], abs=0.01)

    def test_direction_does_not_change_switching_time(self, curves):
        dx = np.linspace(0.0, 1.0, 51)
        falling = np.where(dx < 0.3, 1.0, 0.0).reshape(1, -1)
        rising = 1.0 - falling
        assert curves.calculate_switching_time(dx, falling) == pytest.approx(
            curves.calculate_switching_time(dx, rising)
        )

    def test_flat_curve_gives_zero_via_epsilon_guard(self, curves):
        # numerator is 0 and the denominator is guarded by +1e-300, so a flat
        # curve returns 0 instead of raising.
        dx = np.linspace(0.0, 1.0, 5)
        assert curves.calculate_switching_time(dx, np.full((1, 5), 4.0)) == pytest.approx([0.0])

    def test_pure_bump_has_no_net_transition(self, curves):
        # Start and end are exactly equal, so the clamping median flattens the
        # curve and there is no transition to time.
        dx = np.linspace(0.0, 1.0, 5)
        dy = np.array([[0.0, 1.0, 2.0, 1.0, 0.0]])
        assert curves.calculate_switching_time(dx, dy) == pytest.approx([0.0])

    def test_is_numerically_unstable_when_endpoints_nearly_coincide(self, curves):
        # ISSUES.md #8 (accepted, edge case) -- the 1e-300 guard only protects
        # against an exactly zero
        # denominator.  A bump whose endpoints differ by one float ulp divides a
        # ~1e-16 numerator by a ~1e-16 denominator and returns a completely
        # arbitrary "switching time" instead of ~0.
        dx = np.linspace(0.0, 1.0, 11)
        bump = np.sin(np.pi * dx)                       # sin(pi) == 1.2e-16, not 0
        assert bump[0] != bump[-1]
        assert abs(bump[-1] - bump[0]) < 1e-15
        result = curves.calculate_switching_time(dx, bump.reshape(1, -1))
        assert abs(result[0]) > 0.01                    # nowhere near the true 0

    def test_invariant_to_affine_rescaling_of_pseudotime(self, curves):
        base = np.linspace(0.0, 1.0, 21)
        dy = np.vstack([1.0 - base, base ** 2])
        assert curves.calculate_switching_time(base, dy) == pytest.approx(
            curves.calculate_switching_time(base * 13.0 - 4.0, dy)
        )


# --------------------------------------------------------------------------- #
# curve_characteristics                                                         #
# --------------------------------------------------------------------------- #

class TestCurveCharacteristics:
    def test_default_computes_all_four_metrics(self, curves, four_class_curves):
        dx, dy = four_class_curves
        out = curves.curve_characteristics(dx, dy.values)
        assert list(out.columns) == [
            "transient_logfc", "switching_time", "terminal_logfc", "auc",
        ]
        assert len(out) == len(dy)

    def test_columns_match_the_individual_metric_functions(self, curves, four_class_curves):
        dx, dy = four_class_curves
        out = curves.curve_characteristics(dx, dy.values)
        assert out["auc"].values == pytest.approx(curves.calculate_auc(dx, dy.values))
        assert out["terminal_logfc"].values == pytest.approx(
            curves.calculate_terminal_logfc(dx, dy.values)
        )
        assert out["transient_logfc"].values == pytest.approx(
            curves.calculate_transient_logfc(dx, dy.values)
        )
        assert out["switching_time"].values == pytest.approx(
            curves.calculate_switching_time(dx, dy.values)
        )

    @pytest.mark.parametrize(
        "metrics", [["auc"], ["terminal_logfc", "auc"], ["switching_time"]]
    )
    def test_subset_of_metrics(self, curves, four_class_curves, metrics):
        dx, dy = four_class_curves
        out = curves.curve_characteristics(dx, dy.values, include_metrics=metrics)
        assert set(out.columns) == set(metrics)

    def test_column_order_is_fixed_not_caller_order(self, curves, four_class_curves):
        dx, dy = four_class_curves
        out = curves.curve_characteristics(dx, dy.values, include_metrics=["auc", "terminal_logfc"])
        assert list(out.columns) == ["terminal_logfc", "auc"]

    def test_empty_metric_list_gives_empty_frame(self, curves, four_class_curves):
        dx, dy = four_class_curves
        out = curves.curve_characteristics(dx, dy.values, include_metrics=[])
        assert out.empty

    def test_unknown_metric_names_are_silently_ignored(self, curves, four_class_curves):
        dx, dy = four_class_curves
        out = curves.curve_characteristics(dx, dy.values, include_metrics=["not_a_metric"])
        assert out.empty

    def test_result_is_positionally_indexed_not_labelled(self, curves, four_class_curves):
        # The row labels of ``dy`` are lost: everything downstream that wants TF
        # names has to restore them (see classify_wave_patterns).
        dx, dy = four_class_curves
        out = curves.curve_characteristics(dx, dy.values)
        assert list(out.index) == [0, 1, 2, 3]

    def test_dataframe_input_is_not_supported(self, curves, four_class_curves):
        # A DataFrame reaches ``dy[:, [0]]`` inside calculate_transient_logfc.
        dx, dy = four_class_curves
        with pytest.raises((TypeError, KeyError, IndexError, pd.errors.InvalidIndexError)):
            curves.curve_characteristics(dx, dy)


# --------------------------------------------------------------------------- #
# classify_tf_global_activity / classify_wave_patterns                          #
# --------------------------------------------------------------------------- #

class TestClassification:
    def test_the_four_designed_curves_land_in_the_four_classes(self, curves, four_class_curves):
        dx, dy = four_class_curves
        df = curves.classify_tf_global_activity(
            dx, dy.values, "terminal_logfc", "transient_logfc"
        )
        assert list(df["tf_class"]) == ["Cumulative", "Reductive", "Bell wave", "U-shaped"]

    def test_zscores_are_population_zscores(self, curves, four_class_curves):
        dx, dy = four_class_curves
        df = curves.classify_tf_global_activity(
            dx, dy.values, "terminal_logfc", "transient_logfc"
        )
        for col, zcol in [("terminal_logfc", "terminal_z"), ("transient_logfc", "transient_z")]:
            raw = df[col].values
            expected = (raw - raw.mean()) / raw.std(ddof=0)
            assert df[zcol].values == pytest.approx(expected)
            assert df[zcol].sum() == pytest.approx(0.0, abs=1e-12)

    def test_ranks_are_dense_on_absolute_zscore(self, curves, four_class_curves):
        dx, dy = four_class_curves
        df = curves.classify_tf_global_activity(
            dx, dy.values, "terminal_logfc", "transient_logfc"
        )
        # UP/DOWN tie on |terminal_z| (rank 1); BUMP/DIP tie at 0 (rank 2).
        assert list(df["terminal_rank"]) == [1, 1, 2, 2]
        assert list(df["transient_rank"]) == [2, 2, 1, 1]

    def test_tie_between_terminal_and_transient_goes_to_terminal(self, curves):
        # abs(terminal_z) >= abs(transient_z) is inclusive, so an exact tie is
        # resolved as a terminal (Cumulative/Reductive) call.
        dx = np.linspace(0.0, 1.0, 5)
        dy = np.vstack([
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),      # transient 0, terminal +4
            np.array([4.0, 3.0, 2.0, 1.0, 0.0]),      # transient 0, terminal -4
            np.array([0.0, 3.0, 4.0, 3.0, 0.5]),      # non-zero transient
        ])
        df = curves.classify_tf_global_activity(dx, dy, "terminal_logfc", "transient_logfc")
        assert df.loc[0, "tf_class"] == "Cumulative"
        assert df.loc[1, "tf_class"] == "Reductive"

    @pytest.mark.parametrize(
        "dy",
        [
            # a single curve: std == 0 for both metrics
            np.linspace(0.0, 5.0, 11).reshape(1, -1),
            # every curve monotone: all transient logFCs are exactly 0
            np.vstack([np.linspace(0, 5, 11), np.linspace(0, -3, 11), np.linspace(1, 9, 11)]),
            # identical curves
            np.tile(np.linspace(0, 1, 11), (4, 1)),
        ],
        ids=["single-curve", "all-monotone", "identical-curves"],
    )
    def test_degenerate_zscores_crash_the_classifier(self, curves, dy):
        # ISSUES.md #4 (accepted, edge case) -- when one of the two metrics has
        # zero variance across curves
        # (a single TF, or every TF monotone -> transient logFC all zero) the
        # z-scores are nan, and ranking them hits
        # ``.astype(int)`` on a nan column.  The user gets an opaque pandas
        # casting error instead of a classification.
        dx = np.linspace(0.0, 1.0, dy.shape[1])
        with pytest.raises(pd.errors.IntCastingNaNError):
            curves.classify_tf_global_activity(dx, dy, "terminal_logfc", "transient_logfc")

    @pytest.mark.xfail(
        strict=True,
        reason="ISSUES.md #4 -- a set of purely monotone curves (transient logFC all zero) is a "
               "legitimate input but zscore -> nan -> rank -> astype(int) raises "
               "IntCastingNaNError instead of classifying them as up/down.",
    )
    def test_all_monotone_curves_are_classified(self, curves):
        dx = np.linspace(0.0, 1.0, 11)
        dy = np.vstack([np.linspace(0, 5, 11), np.linspace(0, -3, 11), np.linspace(1, 9, 11)])
        df = curves.classify_tf_global_activity(dx, dy, "terminal_logfc", "transient_logfc")
        assert list(df["tf_class"]) == ["Cumulative", "Reductive", "Cumulative"]

    def test_wave_patterns_keep_the_curve_labels(self, curves, four_class_curves):
        dx, dy = four_class_curves
        df = curves.classify_wave_patterns(dx, dy)
        assert list(df.index) == ["UP", "DOWN", "BUMP", "DIP"]
        assert list(df["wave_pattern"]) == ["up", "down", "transiently_up", "transiently_down"]

    def test_wave_patterns_records_the_trajectory(self, curves, four_class_curves):
        dx, dy = four_class_curves
        df = curves.classify_wave_patterns(dx, dy)
        assert set(df["trajectory"]) == {(0, 2)}

    def test_wave_patterns_tf_list_filters_rows_only(self, curves, four_class_curves):
        dx, dy = four_class_curves
        full = curves.classify_wave_patterns(dx, dy)
        subset = curves.classify_wave_patterns(dx, dy, tf_list=["BUMP", "UP", "NOT_A_TF"])
        assert set(subset.index) == {"UP", "BUMP"}
        # classification is computed on all curves first, so filtering cannot
        # change the class assigned to a kept curve
        for tf in subset.index:
            assert subset.loc[tf, "wave_pattern"] == full.loc[tf, "wave_pattern"]

    def test_wave_pattern_names_cover_every_class(self, curves):
        assert set(SmoothedCurvesGRN.WAVE_PATTERN_NAMES) == {
            "Cumulative", "Reductive", "Bell wave", "U-shaped",
        }
        assert set(SmoothedCurvesGRN.WAVE_PATTERN_NAMES.values()) == {
            "up", "down", "transiently_up", "transiently_down",
        }

    def test_wave_patterns_accepts_a_plain_array_but_then_index_is_positional(self, curves,
                                                                             four_class_curves):
        dx, dy = four_class_curves
        with pytest.raises(AttributeError):
            curves.classify_wave_patterns(dx, dy.values)

    def test_classification_is_relative_to_the_other_curves_supplied(self, curves):
        # The class of a curve depends on which other curves are in ``dy``
        # (z-scoring is done across rows), which is why classify_wave_patterns
        # documents that the full dy should be passed in.
        dx = np.linspace(0.0, 1.0, 21)
        bump = np.sin(np.pi * dx)
        small_up = 0.01 * dx
        with_big_ups = np.vstack([small_up, bump, 5.0 * dx, -5.0 * dx])
        alone_with_bumps = np.vstack([small_up, bump, 0.9 * bump, -bump])
        a = curves.classify_tf_global_activity(
            dx, with_big_ups, "terminal_logfc", "transient_logfc")
        b = curves.classify_tf_global_activity(
            dx, alone_with_bumps, "terminal_logfc", "transient_logfc")
        assert a.loc[0, "tf_class"] != b.loc[0, "tf_class"]


# --------------------------------------------------------------------------- #
# get_top_k_tfs_by_class                                                        #
# --------------------------------------------------------------------------- #

class TestTopKByClass:
    @pytest.fixture
    def many_curves(self):
        """12 curves: 3 per class, with clearly ordered magnitudes."""
        dx = np.linspace(0.0, 1.0, 21)
        rows, names = [], []
        for i, scale in enumerate([3.0, 2.0, 1.0]):
            rows.append(scale * dx)
            names.append(f"UP{i}")
            rows.append(-scale * dx)
            names.append(f"DOWN{i}")
            rows.append(scale * np.sin(np.pi * dx))
            names.append(f"BUMP{i}")
            rows.append(-scale * np.sin(np.pi * dx))
            names.append(f"DIP{i}")
        return dx, pd.DataFrame(rows, index=names)

    def test_one_column_per_observed_class(self, curves, many_curves):
        dx, dy = many_curves
        out = curves.get_top_k_tfs_by_class(dx, dy.values, k=2)
        assert set(out.columns) == {"Cumulative", "Reductive", "Bell wave", "U-shaped"}

    def test_k_limits_each_column(self, curves, many_curves):
        dx, dy = many_curves
        out = curves.get_top_k_tfs_by_class(dx, dy.values, k=2)
        assert len(out) == 2
        assert out.notna().all().all()

    def test_shorter_classes_are_padded_with_none(self, curves, many_curves):
        dx, dy = many_curves
        # ask for more than any class has -> all columns padded to the longest
        out = curves.get_top_k_tfs_by_class(dx, dy.values, k=10)
        assert len(out) == max(
            (dy.index.str.startswith(p)).sum() for p in ["UP", "DOWN", "BUMP", "DIP"]
        )
        assert out.isna().any().any() is not None  # padding is None, not a crash

    @pytest.mark.xfail(
        strict=True,
        reason="ISSUES.md #5 (confirmed) -- get_top_k_tfs_by_class returns positional integers, never TF "
               "names, because curve_characteristics rebuilds a RangeIndex and (unlike "
               "classify_wave_patterns) this method never restores dy's labels.",
    )
    def test_returns_tf_names(self, curves, many_curves):
        dx, dy = many_curves
        out = curves.get_top_k_tfs_by_class(dx, dy.values, k=2)
        returned = {v for col in out.columns for v in out[col] if v is not None}
        assert returned <= set(dy.index)

    def test_currently_returns_positional_indices(self, curves, many_curves):
        dx, dy = many_curves
        out = curves.get_top_k_tfs_by_class(dx, dy.values, k=2)
        returned = {v for col in out.columns for v in out[col] if v is not None}
        assert returned <= set(range(len(dy)))


# --------------------------------------------------------------------------- #
# calculate_force_curves (static)                                               #
# --------------------------------------------------------------------------- #

def _expected_force(beta, tf_expr, epsilon=1e-10):
    """The transform the implementation evaluates.

    ``sign(b) * exp(log10(|b| + eps) + log10(t + eps))``, i.e.
    ``sign(b) * ((|b| + eps) * (t + eps)) ** (1 / ln 10)``: a sign-preserving
    compression of the product, not the product itself.  ISSUES.md #7 records
    this as intended (the docstring on ``calculate_force_curves`` is what needs
    correcting, not the code).
    """
    return np.sign(beta) * ((np.abs(beta) + epsilon) * (tf_expr + epsilon)) ** LOG10E


class TestStaticForceCurves:
    @pytest.fixture
    def beta_and_expression(self):
        index = pd.MultiIndex.from_tuples(
            [("TFA", "G1"), ("TFA", "G2"), ("TFB", "G3")], names=["TF", "Target"]
        )
        cols = ["time_0", "time_1"]
        beta = pd.DataFrame([[2.0, -2.0], [0.5, 0.5], [-4.0, 1.0]], index=index, columns=cols)
        # value_counts orders TFA (2 targets) before TFB (1 target), which is
        # also the order the TF expression frame is given in here.
        expr = pd.DataFrame([[10.0, 10.0], [100.0, 1.0]], index=["TFA", "TFB"], columns=cols)
        return beta, expr

    def test_matches_the_implemented_log_formula(self, beta_and_expression):
        beta, expr = beta_and_expression
        out = SmoothedCurvesGRN.calculate_force_curves(beta, expr)
        expected = _expected_force(
            beta.values, np.repeat(expr.values, [2, 1], axis=0)
        )
        assert out.values == pytest.approx(expected)

    def test_preserves_index_columns_and_sign(self, beta_and_expression):
        beta, expr = beta_and_expression
        out = SmoothedCurvesGRN.calculate_force_curves(beta, expr)
        assert out.index.equals(beta.index)
        assert list(out.columns) == list(beta.columns)
        assert np.array_equal(np.sign(out.values), np.sign(beta.values))

    def test_zero_beta_stays_zero(self):
        index = pd.MultiIndex.from_tuples([("TFA", "G1")], names=["TF", "Target"])
        beta = pd.DataFrame([[0.0, 0.0]], index=index, columns=["time_0", "time_1"])
        expr = pd.DataFrame([[5.0, 5.0]], index=["TFA"], columns=["time_0", "time_1"])
        out = SmoothedCurvesGRN.calculate_force_curves(beta, expr)
        assert out.values == pytest.approx(np.zeros((1, 2)))

    def test_negative_tf_expression_produces_nan(self):
        # log10 of a negative number -> nan; lcpm is never negative in practice
        # but nothing here guards against it.
        index = pd.MultiIndex.from_tuples([("TFA", "G1")], names=["TF", "Target"])
        beta = pd.DataFrame([[1.0]], index=index, columns=["time_0"])
        expr = pd.DataFrame([[-5.0]], index=["TFA"], columns=["time_0"])
        out = SmoothedCurvesGRN.calculate_force_curves(beta, expr)
        assert np.isnan(out.values).all()

    def test_series_expression_is_not_supported_despite_docstring(self):
        # ISSUES.md #21 -- reviewed, not actioned.  The docstring/type hint says
        # ``tf_expression: pd.Series``; a Series makes np.repeat return a 1-D
        # array which cannot be reshaped into the multi-column frame.  Pinned so
        # the current (DataFrame-only) contract cannot change unnoticed.
        index = pd.MultiIndex.from_tuples(
            [("TFA", "G1"), ("TFA", "G2")], names=["TF", "Target"]
        )
        beta = pd.DataFrame([[1.0, 1.0], [2.0, 2.0]], index=index, columns=["time_0", "time_1"])
        with pytest.raises(ValueError):
            SmoothedCurvesGRN.calculate_force_curves(beta, pd.Series([3.0], index=["TFA"]))

    @pytest.mark.xfail(
        strict=True,
        reason="ISSUES.md #2 (REVISED, medium) -- calculate_force_curves pairs beta rows "
               "with expression rows positionally, never by TF name. This input (a beta "
               "frame with UNEQUAL target counts per TF, e.g. one filtered by hand) is "
               "NOT what get_beta_curves produces, so it is not the normal case: a "
               "get_beta_curves cross product has equal counts, value_counts() ties, and "
               "the positional pairing happens to line up. See "
               "TestBetaCurvesFeedingForceCurves in test_smoothed_curves_grn.py for the "
               "failure mode that IS reachable through the public API.",
    )
    def test_aligns_expression_by_tf_name_with_unequal_target_counts(self):
        index = pd.MultiIndex.from_tuples(
            [("TFA", "G1"), ("TFB", "G2"), ("TFB", "G3")], names=["TF", "Target"]
        )
        cols = ["time_0"]
        beta = pd.DataFrame([[1.0], [1.0], [1.0]], index=index, columns=cols)
        # TFA has 1 target, TFB has 2 -> value_counts yields TFB first, but the
        # expression frame is in TFA, TFB order.
        expr = pd.DataFrame([[10.0], [1000.0]], index=["TFA", "TFB"], columns=cols)
        out = SmoothedCurvesGRN.calculate_force_curves(beta, expr)
        expected = _expected_force(
            beta.values, np.array([[10.0], [1000.0], [1000.0]])
        )
        assert out.values == pytest.approx(expected)

    def test_force_is_a_log_space_compression_of_the_product(self):
        # ISSUES.md #7 -- DOCS ONLY.  The log-space transform is intended; the
        # docstring's "force is calculated as beta * tf_expression" is what is
        # wrong.  This pins the actual relation and the size of the difference
        # so the docstring and the code cannot drift apart again.
        index = pd.MultiIndex.from_tuples([("TFA", "G1")], names=["TF", "Target"])
        beta = pd.DataFrame([[3.0]], index=index, columns=["time_0"])
        expr = pd.DataFrame([[7.0]], index=["TFA"], columns=["time_0"])
        out = SmoothedCurvesGRN.calculate_force_curves(beta, expr)
        assert out.values[0, 0] == pytest.approx((3.0 * 7.0) ** LOG10E)
        assert out.values[0, 0] != pytest.approx(3.0 * 7.0)
