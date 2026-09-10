"""One TF-force kernel for the whole package (ISSUES.md #23, stage 1).

``focalfire.temporal._forces.calculate_force_curves`` is the kernel.  The episodic
path reaches it through ``calculate_force_curves_parallel`` (one call per chunk)
and the phase / validation path through ``SmoothedCurvesGRN.calculate_force_curves``.
These tests pin that every entry point is the same function, so a change to the
formula or to the TF alignment cannot land in one path and not the other, which
is how ISSUES #1 and #2 came to be the same bug twice.
"""
import numpy as np
import pandas as pd
import pytest

from focalfire.temporal import calculate_force_curves as exported_kernel
from focalfire.temporal._curves import SmoothedCurvesGRN
from focalfire.temporal._forces import (
    calculate_force_curves,
    calculate_force_curves_chunk,
    calculate_force_curves_parallel,
)

LOG10E = 1 / np.log(10)

ENTRY_POINTS = [calculate_force_curves, SmoothedCurvesGRN.calculate_force_curves]


@pytest.fixture
def beta_and_expression():
    index = pd.MultiIndex.from_tuples(
        [("TFA", "G1"), ("TFB", "G2"), ("TFB", "G3")], names=["TF", "Target"]
    )
    cols = ["time_0", "time_1", "time_2"]
    beta = pd.DataFrame(
        [[2.0, -2.0, 0.0], [0.5, 0.5, -0.25], [-4.0, 1.0, 3.0]],
        index=index, columns=cols,
    )
    # expression rows deliberately NOT in the beta frame's TF order
    expr = pd.DataFrame(
        [[100.0, 1.0, 7.0], [10.0, 20.0, 30.0]], index=["TFB", "TFA"], columns=cols
    )
    return beta, expr


def test_chunk_name_is_an_alias_of_the_kernel():
    assert calculate_force_curves_chunk is calculate_force_curves


def test_package_exports_the_kernel():
    assert exported_kernel is calculate_force_curves


def test_kernel_formula_and_name_alignment(beta_and_expression):
    beta, expr = beta_and_expression
    out = calculate_force_curves(beta, expr)
    tf = expr.loc[beta.index.get_level_values(0)].values
    expected = np.sign(beta.values) * (
        (np.abs(beta.values) + 1e-10) * (tf + 1e-10)
    ) ** LOG10E
    assert out.values == pytest.approx(expected)
    assert out.index.equals(beta.index)
    assert list(out.columns) == list(beta.columns)


def test_static_method_delegates_to_the_kernel(beta_and_expression):
    beta, expr = beta_and_expression
    pd.testing.assert_frame_equal(
        SmoothedCurvesGRN.calculate_force_curves(beta, expr),
        calculate_force_curves(beta, expr),
    )


def test_static_method_forwards_epsilon(beta_and_expression):
    beta, expr = beta_and_expression
    via_static = SmoothedCurvesGRN.calculate_force_curves(beta, expr, epsilon=1e-2)
    pd.testing.assert_frame_equal(via_static, calculate_force_curves(beta, expr, epsilon=1e-2))
    assert not np.allclose(via_static.values, calculate_force_curves(beta, expr).values)


@pytest.mark.slow
def test_parallel_driver_matches_the_kernel(beta_and_expression):
    beta, expr = beta_and_expression
    pd.testing.assert_frame_equal(
        calculate_force_curves_parallel(beta, expr, n_processes=2, chunk_size=1),
        calculate_force_curves(beta, expr),
    )


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_every_entry_point_rejects_a_series(entry, beta_and_expression):
    beta, _ = beta_and_expression
    with pytest.raises(ValueError):
        entry(beta, pd.Series([3.0, 4.0], index=["TFA", "TFB"]))


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_every_entry_point_raises_on_a_missing_regulator(entry, beta_and_expression):
    beta, expr = beta_and_expression
    with pytest.raises(KeyError, match="TFB"):
        entry(beta, expr.drop("TFB"))
