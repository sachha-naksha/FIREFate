"""Small numeric helpers on pseudotime-indexed curves."""
from __future__ import annotations

import numpy as np
import pandas as pd


def curvature_of_expression(dcurve: pd.DataFrame, dtime: pd.Series):
    """
    Calculate the curvature of expression curves.
    """
    # First derivative (dx/dt)
    dx_dt = pd.DataFrame(
        np.gradient(dcurve, dtime, axis=1), index=dcurve.index, columns=dcurve.columns
    )
    # Second derivative (d2x/dt2)
    d2x_dt2 = pd.DataFrame(
        np.gradient(dx_dt, dtime, axis=1), index=dcurve.index, columns=dcurve.columns
    )
    return d2x_dt2
