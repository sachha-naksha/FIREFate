# Findings from the `episodic_dynamics` / `pseudotime_curves` test suite

Every item below was reproduced by a test in this directory.  Behaviour that is
wrong is pinned twice: an `xfail(strict=True)` test that asserts what the code
*should* do (it will start failing as an `XPASS` once the bug is fixed, which is
the signal to promote it to a normal assertion), plus a plain test that pins the
current behaviour so a change cannot slip through unnoticed.

Nothing in the library was modified.

---

## High severity

### 1. Force curves are multiplied by the wrong TF's expression

`episodic_dynamics.calculate_force_curves_chunk` (and therefore
`calculate_force_curves_parallel` and `EpisodeDynamics.calculate_forces`):

```python
targets_per_tf = beta_chunk.index.get_level_values(0).value_counts()   # count-sorted
tf_expr_subset = tf_expression.loc[targets_per_tf.index]               # count order
expanded_tf_expr = pd.DataFrame(
    np.repeat(tf_expr_subset.values, targets_per_tf.values, axis=0),
    index=beta_chunk.index,                                            # row order
    ...)
```

`value_counts()` orders TFs by *descending target count*; the repeated blocks are
then attached to `beta_chunk.index` in *row* order.  The two agree only when the
beta rows happen to be grouped by descending target count.  Rows coming out of
`build_episode_grn` are grouped in regulator (network) order, so as soon as two
TFs have different target counts every edge of at least one of them is scaled by
another TF's expression.

Reproduction (`test_episodic_units.py::TestForceCurvesChunk`): with rows
`[(TFA,G1), (TFB,G2), (TFB,G3)]` and expression `TFA=10, TFB=1000`, the edge
`TFA -> G1` comes out with TFB's expression of 1000.

Suggested fix: index by name instead of by position, e.g.

```python
expanded = tf_expression.reindex(beta_chunk.index.get_level_values(0)).values
```

Tests: `TestForceCurvesChunk::test_expression_is_matched_by_tf_name_not_by_row_order`
(xfail), `::test_currently_swaps_expression_between_tfs`,
`test_episode_workflow.py::TestCalculateForces::test_expression_is_matched_to_the_right_regulator`
(xfail).

### 2. Same defect, unguarded, in `SmoothedCurvesGRN.calculate_force_curves`

The static method in `pseudotime_curves.py` does the same `np.repeat` but never
even reindexes `tf_expression`, so it uses the caller's row order for the
*values* and the count-sorted order for the *repeat counts*.  It is correct only
if the expression frame is already sorted by descending target count and the beta
rows are grouped the same way.

Test: `test_curve_math.py::TestStaticForceCurves::test_aligns_expression_by_tf_name` (xfail).

### 3. `compute_tf_expression` ignores the episode's time slice

```python
tf_lcpm_episode = tf_lcpm_values.iloc[:, 0:n_time_cols]
```

`n_time_cols` is only the *width* of the episode; the columns taken are always
the first `n` pseudotime points, and they are then relabelled `time_0 ...` so the
mismatch is invisible downstream.  Episode 3 of a trajectory is therefore built
from episode-3 betas and episode-0 regulator expression.  `EpisodeDynamics` never
stores the `time_slice` passed to `build_episode_grn`, so the method has no way
to do the right thing as written.  Both `run_episodic_construction` and
`run_episodic_enrichment` are affected for every episode after the first.

Tests: `test_episode_workflow.py::TestComputeTfExpression::test_expression_is_taken_from_the_episode_s_own_time_window`
(xfail), `::test_currently_reuses_the_first_time_points_for_every_episode`.

### 4. `classify_tf_global_activity` crashes on legitimate inputs

`stats.zscore` returns all-NaN when a metric has zero variance across curves, and
the next statement is

```python
df["terminal_z"].abs().rank(method="dense", ascending=False).astype(int)
```

which raises `pandas.errors.IntCastingNaNError`.  This happens for a single TF,
for a set of curves that are all monotone (every transient logFC is exactly
zero), and for identical curves.  `classify_wave_patterns` and
`get_top_k_tfs_by_class` inherit the crash.

Tests: `test_curve_math.py::TestClassification::test_degenerate_zscores_crash_the_classifier`,
`::test_all_monotone_curves_are_classified` (xfail).

---

## Medium severity

### 5. `get_top_k_tfs_by_class` never returns TF names

`curve_characteristics` builds a fresh frame from a dict of arrays, so its index
is a `RangeIndex`.  `classify_wave_patterns` repairs this (`df.index = dy.index`)
but `get_top_k_tfs_by_class` does not, so despite the comment
`# extract tf names (assuming index contains tf names)` it returns positional
integers.  It also raises `ValueError` on an empty classification
(`max()` over an empty sequence).

Tests: `test_curve_math.py::TestTopKByClass::test_returns_tf_names` (xfail),
`::test_currently_returns_positional_indices`.

### 6. The episode writer and the episode reader use different file formats

`run_episodic_construction` writes `episode_<i>.parquet`;
`get_episodic_grn_subset` globs `episode_<i>.pkl` and `pickle.load`s it, raising
`FileNotFoundError: No episode pkl files found` on a folder the runner just
filled.  The two halves of the workflow cannot be chained without a manual
conversion.

Test: `test_episodic_units.py::TestEpisodicGrnSubset::test_does_not_read_the_parquet_files_the_runners_write`.

### 7. "Force" is not beta x expression

Both force implementations compute

```python
sign(beta) * np.exp(np.log10(|beta| + eps) + np.log10(tf + eps))
```

`exp` of a base-10 logarithm: the result is `sign(b) * (|b| * t) ** (1/ln 10)`,
i.e. the product raised to the power 0.4343, not the product.  The docstring of
`SmoothedCurvesGRN.calculate_force_curves` says "force is calculated as beta *
tf_expression".  The transform is monotone so rankings are largely preserved, but
magnitudes are heavily compressed (beta=3, expr=7 gives 3.7, not 21) and the
values have no interpretable unit.  If the log-space form is intentional,
`np.exp` should be `10 **` and the docstring should say so.

Test: `test_curve_math.py::TestStaticForceCurves::test_force_is_not_beta_times_expression`.

### 8. `calculate_switching_time` is unstable when a curve starts and ends at (nearly) the same value

The denominator is guarded with `+ 1e-300`, which only rescues an *exactly* zero
difference.  A bump whose endpoints differ by one float ulp (`sin(pi) = 1.2e-16`)
divides a ~1e-16 numerator by a ~1e-16 denominator and returns an arbitrary
number (0.05 in the test) where the answer should be ~0.  Real smoothed curves
with near-equal endpoints hit this.  A relative guard
(`if |dy[:,0]-dy[:,-1]| < tol: 0`) would be safer.

Test: `test_curve_math.py::TestSwitchingTime::test_is_numerically_unstable_when_endpoints_nearly_coincide`.

### 9. One missing window destroys a whole chromatin trajectory

`extract_data` stores `NaN` for "TF absent from this window", and
`_smooth` is `scipy.ndimage.gaussian_filter1d`, which has no NaN handling: a
single NaN propagates to every point of the smoothed series.  The TF then plots
as nothing at all, and `relative=True` additionally emits an all-NaN slice
warning.  A `np.nan_to_num` / masked convolution (or interpolation over the gap)
is needed.

Tests: `test_chromatin_curves.py::TestSmooth::test_a_single_nan_poisons_the_whole_series`,
`TestProcessDynamics::test_missing_window_poisons_the_whole_smoothed_series`.

### 10. `_process_single_window` swallows every error

The worker wraps its whole body in `except Exception:` and returns NaN/0, so a
missing file, a malformed `loc` field, a missing column and a genuinely absent TF
are all indistinguishable.  The `got_data` guard in `extract_data` only fires
when *every* window failed; a run where 90% of the windows are unreadable
completes "successfully".

Tests: `test_chromatin_curves.py::TestProcessSingleWindow::test_malformed_loc_column_is_swallowed_silently`,
`::test_file_without_loc_column_is_swallowed_silently`.

---

## Low severity / code quality

11. **Dead expensive computation.** `build_episode_grn` builds
    `stat1_netbin = stat.fbinarize(...)` and computes `dnetbin` over all points,
    but `dnetbin_episode` is only ever read for `.shape[0]` / `.shape[1]`, which
    are identical to `dnet_episode`'s.  The binarisation is the most expensive
    step in the function and its result is discarded.

12. **Undocumented hard-coded biology.** `build_episode_grn` silently drops every
    regulator whose name starts with `ZNF` or `ZBTB`.  It is not a parameter and
    not mentioned in the docstring.
    (`test_episode_workflow.py::TestBuildEpisodeGrn::test_znf_and_zbtb_regulators_are_dropped`)

13. **`get_beta_curves` does not return the links it was asked for.** TFs and
    targets are de-duplicated separately with `set()` and then crossed, so two
    requested links come back as four rows.  Because `list(set(...))` is used,
    the row order also varies between processes (string hash randomisation), so
    results are not byte-reproducible across runs.
    (`test_smoothed_curves_grn.py::TestBetaCurves::test_returns_the_full_tf_by_target_cross_product`)

14. **Edge retention uses a row sum.**
    `episode_beta_dcurve[episode_beta_dcurve.sum(axis=1) != 0]` drops an edge
    whose values cancel over the episode rather than one that is zero
    everywhere.  Floating point makes exact cancellation unlikely, so this is
    latent rather than active.
    (`test_episode_workflow.py::TestBuildEpisodeGrn::test_edges_are_retained_by_row_sum_not_by_activity`)

15. **Duplicated threshold.** In `filter_chunk_of_edges`,
    `min_nonzero_timepoints` and `min_observations` are applied to the same count
    of non-zero values, so one of them is redundant.  Separately,
    `filter_edges_by_significance_and_direction` ships the *entire* DataFrame to
    every worker via `partial(..., df=df)`, which defeats the memory saving the
    index-chunking was written for.

16. **Column selection by prefix.** `calculate_force_curves_parallel` picks
    columns with `startswith("time_")`; a frame that names its pseudotime columns
    anything else silently produces a result with zero columns instead of an
    error.
    (`test_episodic_units.py::TestForceCurvesParallel::test_columns_not_named_time_something_are_silently_dropped`)

17. **`process_dynamics` does not validate `metric`.** Anything other than the
    exact string `'score'` (including the typo `'scores'`) is treated as
    `'count'`.
    (`test_chromatin_curves.py::TestProcessDynamics::test_unknown_metric_silently_falls_back_to_counts`)

18. **Constant edges get `p_value == 0.0`.** A zero-variance one-sample t-test
    has an infinite statistic, so a perfectly flat edge passes every significance
    threshold (and emits a `RuntimeWarning` from scipy in the worker).
    (`test_episodic_units.py::TestFilterChunk::test_a_perfectly_constant_edge_gets_a_degenerate_pvalue_of_zero`)

19. **`EpisodeDynamics.episodic_composition()` is an unimplemented stub** that
    returns `None` with a docstring describing behaviour it does not have.

20. **Redundant `np.where`.** In `plot_score_vs_count_comparison` the fill
    polygon's lower edge is `np.where(norm_count > norm_score, norm_score,
    norm_score)` -- both branches are the same expression.  The resulting
    geometry happens to be the intended one (shade where count exceeds score),
    but the line reads like a copy-paste slip.

21. **Docstring/type-hint mismatch.** `calculate_force_curves` annotates
    `tf_expression: pd.Series` and describes it as "Series with tf expression
    values", but a Series makes `np.repeat` return a 1-D array and the
    `pd.DataFrame(...)` construction raises.  Only a DataFrame works.

22. **`pyproject.toml` declares `requires-python = ">=3.10"`** while the `dictys`
    environment these modules run in is Python 3.9.

---

## Checked and found correct

Worth recording, since these are the parts most likely to be suspected:

* `_regulation_curves_parallel` reproduces
  `flnneighbor(fbinarize(fsmooth(net), sparsity))` **bit-for-bit** -- verified
  against the stock dictys chain for several sparsities, all three branches of
  the trajectory, the NaN-aware path and every `n_jobs` setting.  Its degenerate
  behaviour at `int(sparsity * n_reg * n_target) == 0` matches dictys' own.
* `_subnetwork_curves` equals slicing the fully smoothed network, for `w`,
  `w_n` and `w_in`, including the NaN path.
* Both fast paths index the target axis by gene index, which assumes
  `nids[1] == arange(n_gene)`.  That holds by construction in dictys
  (`network.from_folders` sets `nname = nids[1]`), so it is safe.
* `calculate_auc` is the trapezoidal rule and matches `numpy.trapezoid` on
  uniform and non-uniform grids.
* `calculate_transient_logfc` / `calculate_switching_time` are invariant to
  affine rescaling of pseudotime and give the textbook answers on monotone,
  constant and step curves.
* `calculate_tf_episodic_enrichment` matches `scipy.stats.hypergeom.sf(k-1, N,
  K, n)` and hand-computed fold enrichments.
* `filter_edges_by_significance_and_direction` preserves the input row order
  regardless of chunk size or the order in which workers finish.
* `create_balanced_chunks` distributes the remainder over the leading chunks and
  round-trips to the original frame.
