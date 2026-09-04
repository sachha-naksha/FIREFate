"""Recover the hypergeometric population parameters from the committed enrichment CSVs.

The committed ``enrichment_ep{i}_{pb,gc}.csv`` files do not record how large the
episodic GRN was, but the two ORA parameters are **exactly invertible** from what they
do record.

For each TF row, with :math:`k = |{\\tt genes\\_in\\_lf}|` and
:math:`n = k + |{\\tt genes\\_dwnstrm}|`:

.. math::

    \\mathrm{ES} = \\frac{k}{nK/N} \\;\\Longrightarrow\\; \\frac{N}{K} = \\frac{\\mathrm{ES}\\, n}{k}

so every row of a file independently determines the **ratio** :math:`N/K`. The absolute
scale then follows from the p-value, :math:`p = \\mathrm{hypergeom.sf}(k-1, N, K, n)`,
by scanning integer :math:`K` — and :math:`K` is bounded above by the size of the
cellular program (57 genes), so the scan covers the whole feasible space and the
solution is unique.

Recovered per episode:

* ``N`` -- number of distinct target genes in the episodic GRN after percentile selection
* ``K`` -- number of program genes active as targets in that GRN

These are the numbers a rerun must reproduce for its parameter reconstruction
(trajectory range, ``num_points``, ``points_per_episode``, ``percentile``) to be
credible. ``run_validation.py`` reports ``fixed_n_targets_in_grn`` and
``fixed_n_lf_active``, which are directly comparable.

Usage
-----
    python recover_published_params.py [--csv-dir DIR] [--out published_params.csv]
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

DEFAULT_CSV_DIR = (
    "/work/nvme/bhdw/asachan/data_files/firefate/bcell/outs/"
    "intermediate_tmp_files/direct_effect_enrichment"
)
#: Upper bound on K: the cellular program is Z11 u Z3 (GC_PB), HLA- dropped.
MAX_PROGRAM_GENES = 57


def _parse_tuple(s) -> tuple:
    try:
        if not isinstance(s, str) or s in ("", "()"):
            return ()
        v = ast.literal_eval(s)
        return tuple(v) if isinstance(v, tuple) else (v,)
    except Exception:
        return ()


def recover(path: str | Path, k_max: int = MAX_PROGRAM_GENES) -> dict:
    """Solve for ``(N, K)`` from one enrichment CSV."""
    df = pd.read_csv(path)
    k = df["genes_in_lf"].fillna("()").apply(lambda s: len(_parse_tuple(s))).values
    n_dwn = df["genes_dwnstrm"].fillna("()").apply(lambda s: len(_parse_tuple(s))).values
    n = k + n_dwn
    es = df["enrichment_score"].values
    p = df["p_value"].values

    m = (k > 0) & (n > 0) & (es > 0)
    if m.sum() < 2:
        return {"file": Path(path).name, "n_rows": len(df), "N": np.nan, "K": np.nan}

    ratio = es[m] * n[m] / k[m]
    r = float(np.median(ratio))
    cv = float(ratio.std() / ratio.mean()) if len(ratio) > 1 else 0.0

    best_N = best_K = None
    best_err = np.inf
    for K in range(2, k_max + 1):
        N = int(round(r * K))
        if N <= K:
            continue
        p_expect = hypergeom.sf(k[m] - 1, N, K, n[m])
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.abs(
                np.log10(np.clip(p_expect, 1e-300, None))
                - np.log10(np.clip(p[m], 1e-300, None))
            )
        err = float(np.nanmax(d))
        if err < best_err:
            best_N, best_K, best_err = N, K, err

    return {
        "file": Path(path).name,
        "n_rows": int(len(df)),
        "n_rows_used": int(m.sum()),
        "ratio_N_over_K": r,
        "ratio_cv": cv,
        "N_targets_in_grn": best_N,
        "K_lf_active": best_K,
        "max_abs_dlog10p": best_err,
        "median_tf_outdegree": float(np.median(n[m])),
        "max_tf_outdegree": int(n[m].max()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)
    ap.add_argument(
        "--out", default=str(Path(__file__).resolve().parent / "published_params.csv")
    )
    args = ap.parse_args(argv)

    rows = []
    for branch in ("pb", "gc"):
        for i in (1, 2, 3, 4):
            f = Path(args.csv_dir) / f"enrichment_ep{i}_{branch}.csv"
            if not f.exists():
                print(f"missing: {f}")
                continue
            rec = recover(f)
            rec["branch"], rec["episode"] = branch, i
            rows.append(rec)

    out = pd.DataFrame(rows)
    cols = [
        "branch", "episode", "n_rows", "N_targets_in_grn", "K_lf_active",
        "ratio_N_over_K", "ratio_cv", "max_abs_dlog10p",
        "median_tf_outdegree", "max_tf_outdegree",
    ]
    out = out[cols]
    out.to_csv(args.out, index=False)

    print(out.to_string(index=False))
    print(f"\nwritten: {args.out}")
    print(
        "\nratio_cv ~ 0 and max_abs_dlog10p ~ 1e-14 mean the inversion is exact, "
        "not a fit.\nA rerun should reproduce N_targets_in_grn and K_lf_active to "
        "within a few percent\nif its trajectory range / num_points / percentile are right."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
