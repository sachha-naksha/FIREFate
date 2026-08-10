# Findings from the `episodic_dynamics` / `pseudotime_curves` test suite

Every item below was reproduced by a test in this directory.  Behaviour that is
wrong is pinned twice: an `xfail(strict=True)` test that asserts what the code
*should* do (it will start failing as an `XPASS` once the bug is fixed, which is
the signal to promote it to a normal assertion), plus a plain test that pins the
current behaviour so a change cannot slip through unnoticed.

Nothing in the library was modified.

## Triage status


| status | meaning |
| --- | --- |
| `CONFIRMED` | real defect, fix wanted |
| `REVISED` | the finding as originally written overstated the problem; see the note |
| `ACCEPTED (edge case)` | real but only reachable on unusual input; recorded, not urgent |
| `DOCS ONLY` | the code is intended as written, the documentation is what is wrong |
| `BY DESIGN` | intentional behaviour, not a defect |
| `NOT ACTIONED` | reviewed and judged acceptable as-is |
| `UNTRIAGED` | not yet reviewed |

---

## High severity

### 1. Force curves are multiplied by the wrong TF's expression

**Status: CONFIRMED.**  Re-checked after the observation that
`utils.custom.get_tf_indices` might account for the ordering.  It does not:
`get_tf_indices` is called from `get_beta_curves` only (`grep` over `src/`
returns exactly one call site) and is nowhere in the episodic force path.  The
equal-target-count argument that resolves issue 2 does not apply here ->
`build_episode_grn` produces a cross product, but `filter_edges` then *drops*
rows, so target counts become unequal and `value_counts()` genuinely re-sorts
while the rows stay grouped in `nids[0]` order.

One caveat on the evidence: in the test fixture the two orders happen to
coincide (TFA has both more targets *and* a lower `nids[0]` index), which is why
the end-to-end workflow tests pass.  With a real network of hundreds of TFs,
network order coinciding with descending-target-count order is
unlikely, so the mis-assignment should be assumed active in production runs.

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

Tests: `test_curve_math.py::TestStaticForceCurves::test_aligns_expression_by_tf_name_with_unequal_target_counts` (xfail), and -- for the
half that *is* reachable through the public API -- `test_smoothed_curves_grn.py::TestBetaCurvesFeedingForceCurves`.

**Status: REVISED -- downgrade from High to Medium.  The original wording
overstated this.**

The claim "each edge is multiplied by the wrong TF's expression whenever the
expression frame is not sorted by target count" is untrue for the input this
function is actually given.  `get_beta_curves` -- its only producer in the
codebase -- returns the full TF x target **cross product**, so every TF has
exactly the same number of targets.  `value_counts()` on tied counts preserves
first-appearance order (verified on pandas 2.2.2, including 20 shuffled TFs with
7 targets each), so `targets_per_tf.index` equals the beta frame's TF-group order
and the `np.repeat` blocks land correctly.  Running the real
`get_beta_curves -> calculate_force_curves` flow gives the right answer for every
edge.

What remains is narrower, but still real:

* **Positional coupling.**  The caller must supply `tf_expression` rows in the
  same order as the beta frame's TF groups.  That order comes from
  `list(set(tf_list))` inside `get_beta_curves`, so it is neither the caller's
  link order nor stable between runs (see issue 13).  Supplying any other order
  silently corrupts every non-zero edge -- verified by swapping two TFs'
  expression rows; only the zero-beta rows survive, because `sign(0) == 0`.
  The safe idiom is
  `tf_expr.loc[beta.index.get_level_values(0).unique()]`, and the robust fix is
  the same name-based reindex proposed for issue 1.
* **Reliance on undocumented behaviour.**  pandas specifies that
  `value_counts()` sorts by count; the ordering of *ties* is not part of the
  contract.  The correctness above holds by accident of the current
  implementation.
* The `xfail` test written for this uses TFA with 1 target and TFB with 2 -- an
  input that cannot come out of `get_beta_curves`.  It is a valid demonstration
  of a hand-filtered beta frame, but it should not be read as the normal case,
  and has been renamed
  `test_aligns_expression_by_tf_name_with_unequal_target_counts` to say so.

**Test-suite follow-up (done).**  `TestBetaCurvesFeedingForceCurves` in
`test_smoothed_curves_grn.py` now covers the reachable failure: a real
`get_beta_curves` frame is correct when the expression rows follow the beta
frame's TF-group order, silently wrong for every non-zero edge in any other
order (zero-beta rows survive because `sign(0) == 0`), and correct again with
the `beta.index.get_level_values(0).unique()` idiom.

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

**Status: CONFIRMED.**

Suggested fix -- record the slice when the episode is built and use it, since
the runners already know it:

```python
def __init__(self, ...):
    ...
    self.time_slice = slice(0, num_points)      # whole branch until an episode is built

def build_episode_grn(self, time_slice=slice(0, 5)):
    ...
    self.time_slice = time_slice                # <-- remember it
    ...

def compute_tf_expression(self):
    if self.lcpm_dcurve is None or self.filtered_edges_p001 is None:
        raise ValueError("Run compute_expression_curves() and filter_edges() first.")
    tf_names = self.filtered_edges_p001.index.get_level_values(0).unique()
    time_cols = [c for c in self.filtered_edges_p001.columns if c.startswith("time_")]

    tf_lcpm_episode = self.lcpm_dcurve.loc[tf_names].iloc[:, self.time_slice].copy()
    if tf_lcpm_episode.shape[1] != len(time_cols):
        raise ValueError(
            f"Episode has {len(time_cols)} time points but the expression slice "
            f"{self.time_slice} selects {tf_lcpm_episode.shape[1]}."
        )
    tf_lcpm_episode.columns = time_cols
    self.tf_lcpm_episode = tf_lcpm_episode
    return tf_lcpm_episode
```

Notes on the fix:

* `.copy()` also removes the `SettingWithCopyWarning` risk from assigning to
  `.columns` on a view of `lcpm_dcurve`.
* The width check turns a silent misalignment into an error if the two stages
  ever disagree.
* No change is needed in `run_episodic_construction` /
  `run_episodic_enrichment`; they already pass `time_slice_start`/`_end` through
  to `build_episode_grn`.
* Flipping the `xfail` marker on
  `test_expression_is_taken_from_the_episode_s_own_time_window` verifies the
  fix, and `test_currently_reuses_the_first_time_points_for_every_episode`
  should be deleted at the same time.

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

**Status: ACCEPTED (edge case).**  Agreed that a real run classifies many TFs at
once, so a zero-variance metric column is unusual.  Worth keeping on the list
because the failure mode is an opaque pandas casting error rather than a
diagnosable message, and because "every TF monotone" is reachable on a short
trajectory or a heavily filtered TF list.

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

**Status: CONFIRMED.**  `classify_wave_patterns` already does the right thing
(`df.index = dy.index`); applying the same two lines here, plus an early return
of an empty frame when there is nothing to rank, resolves both halves.

### 6. The episode writer and the episode reader use different file formats

`run_episodic_construction` writes `episode_<i>.parquet`;
`get_episodic_grn_subset` globs `episode_<i>.pkl` and `pickle.load`s it, raising
`FileNotFoundError: No episode pkl files found` on a folder the runner just
filled.  The two halves of the workflow cannot be chained without a manual
conversion.

Test: `test_episodic_units.py::TestEpisodicGrnSubset::test_does_not_read_the_parquet_files_the_runners_write`.

**Status: CONFIRMED.**  Either format works as long as the two sides agree;
parquet on both sides keeps the existing writer and only changes the glob
pattern and the load call in `get_episodic_grn_subset`.

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

Test: `test_curve_math.py::TestStaticForceCurves::test_force_is_a_log_space_compression_of_the_product` (renamed from `test_force_is_not_beta_times_expression`, which read as an accusation rather than a description).

**Status: DOCS ONLY.**  The log-space transform is intended; the docstring is
what is wrong.  Replacing "force is calculated as beta * tf_expression" with the
actual relation -- `sign(beta) * (|beta| * tf_expression) ** (1 / ln 10)`, i.e.
a sign-preserving compression of the product -- and saying why the compression
is wanted, closes this.  The same sentence is worth adding to
`calculate_force_curves_chunk`, whose comment gives the code but not the
interpretation.  No code change.

### 8. `calculate_switching_time` is unstable when a curve starts and ends at (nearly) the same value

The denominator is guarded with `+ 1e-300`, which only rescues an *exactly* zero
difference.  A bump whose endpoints differ by one float ulp (`sin(pi) = 1.2e-16`)
divides a ~1e-16 numerator by a ~1e-16 denominator and returns an arbitrary
number (0.05 in the test) where the answer should be ~0.  Real smoothed curves
with near-equal endpoints hit this.  A relative guard
(`if |dy[:,0]-dy[:,-1]| < tol: 0`) would be safer.

Test: `test_curve_math.py::TestSwitchingTime::test_is_numerically_unstable_when_endpoints_nearly_coincide`.

**Status: ACCEPTED (edge case).**  Recorded, no fix planned.  If it is ever
revisited, a relative guard is a two-line change: compute
`denom = dy[:, 0] - dy[:, -1]` and return 0 where
`abs(denom) < 1e-12 * max(abs(dy).max(axis=1), 1)` instead of adding `1e-300`.

### 9. One missing window destroys a whole chromatin trajectory

`extract_data` stores `NaN` for "TF absent from this window", and
`_smooth` is `scipy.ndimage.gaussian_filter1d`, which has no NaN handling: a
single NaN propagates to every point of the smoothed series.  The TF then plots
as nothing at all, and `relative=True` additionally emits an all-NaN slice
warning.  A `np.nan_to_num` / masked convolution (or interpolation over the gap)
is needed.

Tests: `test_chromatin_curves.py::TestSmooth::test_a_single_nan_poisons_the_whole_series`,
`TestProcessDynamics::test_missing_window_poisons_the_whole_smoothed_series`.

**Status: UNTRIAGED.**  Not covered in the review.

### 10. `_process_single_window` swallows every error

The worker wraps its whole body in `except Exception:` and returns NaN/0, so a
missing file, a malformed `loc` field, a missing column and a genuinely absent TF
are all indistinguishable.  The `got_data` guard in `extract_data` only fires
when *every* window failed; a run where 90% of the windows are unreadable
completes "successfully".

Tests: `test_chromatin_curves.py::TestProcessSingleWindow::test_malformed_loc_column_is_swallowed_silently`,
`::test_file_without_loc_column_is_swallowed_silently`.

**Status: ACCEPTED.**  Related to issue 9: these two together decide whether a
partly unreadable input directory fails loudly or produces a quietly empty
result.

### Distinguishing the failure modes

There are four distinct conditions that currently all arrive as "NaN score,
zero count", and they want different responses:

| condition | what it means | wanted response |
| --- | --- | --- |
| TF not in this window's file | a real observation: no binding called here | NaN, carry on |
| `binding.tsv.gz` missing | this window was never produced | flag the *window*, not the TF |
| file unreadable (truncated/not gzip) | corrupt output | flag the window, name the file |
| file readable but malformed (`loc`, missing column) | upstream format change | fail loudly -- every window will be affected |

The measured exception types, so the handler can be specific:

```text
missing file    -> FileNotFoundError
corrupt gzip    -> gzip.BadGzipFile
loc 'chr1:100-200'   -> split gives 2 columns   (uniform, caught by a width check)
loc 'chr1_100_200'   -> split gives 1 column
loc mixed 3- and 2-field -> 3 columns WITH NaN  (width check alone misses this)
```

**Suggestion 1 -- return the reason instead of swallowing it.**  A worker in a
`Pool` cannot usefully print or warn (output interleaves and is often lost), so
the reason has to travel back to the parent as data.  Add a fourth element to
the worker's return tuple:

```python
@staticmethod
def _process_single_window(i, tfs, base_path):
    """Returns (i, scores, counts, error); error is None on success, else a
    short string naming why this window could not be used."""
    file_path = f"{base_path}/Subset{i}/binding.tsv.gz"
    try:
        df = pd.read_csv(file_path, sep="\t", compression="gzip")
    except FileNotFoundError:
        return (i, {}, {}, "missing file")
    except Exception as e:                       # corrupt/truncated/not gzip
        return (i, {}, {}, f"unreadable: {type(e).__name__}: {e}")

    missing = {"TF", "loc", "score"} - set(df.columns)
    if missing:
        return (i, {}, {}, f"missing column(s): {sorted(missing)}")

    parts = df["loc"].str.split(":", expand=True)
    if parts.shape[1] != 3 or parts.isna().any().any():
        return (i, {}, {}, f"malformed 'loc' (expected chr:start:end, "
                           f"e.g. {df['loc'].iloc[0]!r})")
    df[["chr", "start", "end"]] = parts

    # ... aggregation exactly as now, but OUTSIDE the try block ...
    return (i, window_scores, window_counts, None)
```

The important structural change is that the `try` now wraps only the I/O and
parse.  A `KeyError` or `IndexError` in the aggregation loop is a *bug*, not a
data condition, and should be allowed to propagate instead of being disguised as
an empty window.

**Suggestion 2 -- aggregate and report in the parent.**  `extract_data` already
has the "did anything work at all" check; extend it to summarise by reason
rather than only detecting the total wipe-out:

```python
errors = {i: err for i, _, _, err in results if err is not None}
if errors:
    by_reason = defaultdict(list)
    for i, err in errors.items():
        by_reason[err.split(":")[0]].append(i)
    print(f"WARNING: {len(errors)}/{n_windows} windows could not be read:")
    for reason, windows in sorted(by_reason.items()):
        print(f"  {len(windows):>4} windows -- {reason} "
              f"(e.g. Subset{min(windows)}: {errors[min(windows)]})")

self.failed_windows = sorted(errors)          # 1-based window IDs

if len(errors) == n_windows:
    raise FileNotFoundError(...)              # as now
if len(errors) > max_failed_windows:          # new keyword, default 0
    raise RuntimeError(
        f"{len(errors)}/{n_windows} windows unreadable; pass "
        f"max_failed_windows to continue anyway.")
```

Defaulting `max_failed_windows=0` makes a partly-broken input directory fail
fast, which is almost always what is wanted for a batch job; a caller who knows
some windows are legitimately absent opts in explicitly.

**Suggestion 3 -- keep the two kinds of NaN apart.**  Once `self.failed_windows`
exists, `NaN` in `raw_scores` means only "TF not bound in this window", and a
failed window is a separate, queryable fact.  That is also what makes issue 9
tractable: `process_dynamics` can drop failed windows from `pb_indices` /
`gc_indices` before smoothing, instead of letting `gaussian_filter1d` propagate
their NaNs across the whole trajectory.

**Tests to add with the change** -- deliberately *not* written up front, since
they would have to assume an API shape that has not been agreed:

* a readable window reports `error is None`;
* "missing file" is distinguishable from "TF absent from a perfectly good
  window";
* each failure mode produces a different message -- including the mixed-width
  `loc` case, where some rows have three fields and some two, which a
  column-count check alone does not catch;
* an exception raised in the *aggregation* propagates instead of being disguised
  as a bad window;
* `failed_windows` is populated, and `extract_data` fails fast on a partly
  broken directory;
* `process_dynamics` can skip failed windows (the issue 9 payoff).

The two existing silent-swallow tests
(`test_malformed_loc_column_is_swallowed_silently`,
`test_file_without_loc_column_is_swallowed_silently`) stay as pins on current
behaviour and should be rewritten as assertions on the reported error when this
lands.

---

## Low severity / code quality

11. **Dead expensive computation.** `build_episode_grn` builds
    `stat1_netbin = stat.fbinarize(...)` and computes `dnetbin` over all points,
    but `dnetbin_episode` is only ever read for `.shape[0]` / `.shape[1]`, which
    are identical to `dnet_episode`'s.  The binarisation is the most expensive
    step in the function and its result is discarded.

    **Status: CONFIRMED** -- and worth more than the "low severity" label
    suggests, given the function is currently too slow.  Two independent wins:

    ```python
    # 1. drop the binarisation entirely: dnetbin_episode is only ever read for
    #    .shape[0] / .shape[1], which equal dnet_episode's.
    #    (delete stat1_netbin and dnetbin)

    # 2. smooth only the points the episode needs, instead of all of them:
    dnet_episode = stat1_net.compute(pts[time_slice])      # was: compute(pts)[:, :, time_slice]
    n_tfs, n_targets, n_times = dnet_episode.shape
    ```

    `stat1_net.compute(pts[time_slice])` is bit-identical to computing every
    point and slicing (verified: `np.array_equal` is True), because Gaussian
    smoothing is independent per output point.  Change 2 scales the smoothing
    matmul down by `episode_width / num_points` -- for 8 episodes over 40 points
    that is an 8x reduction.  Change 1 removes a step that re-runs the smoothing
    *and* does a per-point `np.partition` over `n_reg x n_target` elements; on
    the toy fixture it already costs ~4x the smoothing itself, and it grows with
    the full network size rather than the episode size.

12. **Undocumented hard-coded biology.** `build_episode_grn` silently drops every
    regulator whose name starts with `ZNF` or `ZBTB`.  It is not a parameter and
    not mentioned in the docstring.
    (`test_episode_workflow.py::TestBuildEpisodeGrn::test_znf_and_zbtb_regulators_are_dropped`)

    **Status: BY DESIGN.**  Confirmed intentional: `ZNF*`/`ZBTB*` factors are not
    relevant to the biology FIREFate targets and are dropped deliberately.  Not a
    defect.  The only thing left is that a reader of the code cannot tell this --
    a one-line comment or docstring sentence saying *why* would make the filter
    self-explanatory.  The test stays as a regression guard so the filter is not
    removed by accident.

13. **`get_beta_curves` does not return the links it was asked for.** TFs and
    targets are de-duplicated separately with `set()` and then crossed, so two
    requested links come back as four rows.  Because `list(set(...))` is used,
    the row order also varies between processes (string hash randomisation), so
    results are not byte-reproducible across runs.
    (`test_smoothed_curves_grn.py::TestBetaCurves::test_returns_the_full_tf_by_target_cross_product`)

    **Status: expanded below on request.**

    The function takes a list of `(TF, target)` links, but immediately throws
    away the pairing between them:

    ```python
    tf_list     = list(set([link[0] for link in specified_links]))   # TFs,     unordered
    target_list = list(set([link[1] for link in specified_links]))   # targets, unordered
    ...
    index_tuples = [(tf, target) for tf in found_tfs for target in found_targets]
    ```

    Two separate consequences.

    **(a) You get the cross product, not the links you asked for.**  Requesting
    `[(TFA, G1), (TFB, G4)]` returns four rows -- `TFA->G1`, `TFA->G4`,
    `TFB->G1`, `TFB->G4`.  The two extra rows are real smoothed curves for edges
    you never asked about (zeros, if those edges are absent from the network).
    The cost is quadratic: a query of *n* links spanning *T* distinct TFs and
    *G* distinct targets returns `T x G` rows, not *n*.  Asking about 50
    curated links that happen to span 40 TFs and 45 targets returns 1,800 rows
    -- 1,750 of them unrequested -- and anything applied downstream
    (`calculate_force_curves`, plotting, writing to disk) pays for all of them.
    Callers who want just their links have to re-filter with
    `beta.loc[beta.index.isin(specified_links)]`, which is easy to forget
    because the frame looks plausible either way.

    **(b) The row order is not reproducible between runs.**  Python randomises
    string hashing per process, so `list(set(...))` iterates in a different
    order every time the interpreter starts.  The same call, same data, four
    different `PYTHONHASHSEED` values:

    ```text
    seed=1  : TFB->G1, TFB->G4, TFB->G2, TFA->G1, TFA->G4, TFA->G2, ZNF1->G1, ...
    seed=7  : ZNF1->G1, ZNF1->G2, ZNF1->G4, TFB->G1, TFB->G2, TFB->G4, TFA->G1, ...
    seed=42 : TFA->G4, TFA->G2, TFA->G1, TFB->G4, TFB->G2, TFB->G1, ZNF1->G4, ...
    seed=99 : ZNF1->G2, ZNF1->G4, ZNF1->G1, TFB->G2, TFB->G4, TFB->G1, TFA->G2, ...
    ```

    Both the TF group order and the target order within each group move.  The
    *contents* are identical, so anything that selects by label (`.loc`) is
    unaffected -- which is why the rest of the test suite compares with `set(...)`
    and passes.  What breaks is anything positional:

    * `calculate_force_curves` (issue 2) is positionally coupled to exactly this
      order, so the caller cannot construct a matching `tf_expression` frame
      except by deriving it from the returned beta frame;
    * a beta frame written to parquet/CSV differs byte-for-byte between runs,
      so results cannot be diffed or checksummed across sessions;
    * `.iloc[:5]`, `.head()` and "the first N edges" mean something different
      each run, including inside plots.

    Both parts are fixed together by building the index from the requested links
    instead of from two sets:

    ```python
    links = list(dict.fromkeys(map(tuple, specified_links)))     # de-dup, keep order
    tf_list     = list(dict.fromkeys(tf for tf, _ in links))
    target_list = list(dict.fromkeys(t for _, t in links))
    ...
    # after computing the sub-network, select just the requested pairs:
    beta_dcurve = beta_dcurve.loc[[l for l in links if l in beta_dcurve.index]]
    ```

    `dict.fromkeys` de-duplicates while preserving first-appearance order, so
    the result is both deterministic and restricted to the requested links.  If
    the cross product is wanted somewhere, it is worth making that an explicit
    argument (`cross=True`) rather than the only behaviour.

14. **Edge retention uses a row sum.**
    `episode_beta_dcurve[episode_beta_dcurve.sum(axis=1) != 0]` drops an edge
    whose values cancel over the episode rather than one that is zero
    everywhere.  Floating point makes exact cancellation unlikely, so this is
    latent rather than active.
    (`test_episode_workflow.py::TestBuildEpisodeGrn::test_edges_are_retained_by_row_sum_not_by_activity`)

    **Status: ACCEPTED (rare).**  Agreed it is unlikely to fire.  The cheapest
    way to remove the risk entirely is a one-token change that also says what is
    meant -- "keep edges that are present in this episode":

    ```python
    episode_beta_dcurve = episode_beta_dcurve[(episode_beta_dcurve != 0).any(axis=1)]
    ```

    Same result on every input where the current test agrees, no cancellation
    hazard, and marginally cheaper than summing (it short-circuits on the first
    non-zero).  Absent edges are exactly `0.0` in every window and smooth to
    exactly `0.0`, so `.any()` keeps dropping them.

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

**Status for items 15-22: NOT ACTIONED.**  All eight were reviewed and judged
acceptable as-is; they are kept here as a record of what was examined, not as a
backlog.  The tests covering them assert *current* behaviour rather than desired
behaviour, so they will keep passing and will flag it if any of these change by
accident:

* 16 -- `test_episodic_units.py::TestForceCurvesParallel::test_columns_not_named_time_something_are_silently_dropped`
* 17 -- `test_chromatin_curves.py::TestProcessDynamics::test_unknown_metric_silently_falls_back_to_counts`
* 18 -- `test_episodic_units.py::TestFilterChunk::test_a_perfectly_constant_edge_gets_a_degenerate_pvalue_of_zero`
* 19 -- `test_episode_workflow.py::TestEpisodeConstruction::test_episodic_composition_is_an_unimplemented_stub`
* 20 -- `test_chromatin_curves.py::TestScoreVsCountPlot::test_shaded_band_runs_between_score_and_max_of_the_two`
* 21 -- `test_curve_math.py::TestStaticForceCurves::test_series_expression_is_not_supported_despite_docstring`

Items 15 and 22 have no behavioural test (they are about worker memory use and
packaging metadata respectively).

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
