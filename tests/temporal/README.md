# Tests for `firefate.temporal`

Coverage of `src/firefate/temporal/_episodes.py` and
`src/firefate/temporal/_curves.py`: every public class and function, the
private fast paths, and the two process-level runners.

## Running

```bash
conda activate dictys
cd FIREFate
pytest tests/temporal -q                 # everything (~20 s)
pytest tests/temporal -q -m "not slow"   # skip the tests that spawn worker processes
```

## Layout

| file | subject |
| --- | --- |
| `conftest.py` | the mock dictys network, mock binding files, curve fixtures |
| `test_curve_math.py` | AUC, transient/terminal logFC, switching time, curve characteristics, wave-pattern classification, static force curves |
| `test_curves.py` | `SmoothedCurvesGRN` against a real dictys network: smoothed expression/regulation curves, the parallel regulation fast path, sub-network curves, beta curves |
| `test_chromatin.py` | `SmoothedCurvesChromatin`: binding extraction, trajectory mapping, smoothing, both plots |
| `test_episodic_units.py` | module-level helpers, now spread across three modules: the ORA primitive (`firefate.base`), edge filtering and force curves (`temporal/_forces.py`), chunking (`firefate.utils`), episode subsetting |
| `test_episodes.py` | `AlignTimeScales`, the `EpisodeDynamics` workflow end to end, `run_episodic_construction` / `run_episodic_enrichment` |
| `ISSUES.md` | defects and inconsistencies the suite found, each with the test that reproduces it |

## Not yet covered

The Temporal module has two large files with no tests at all. They are the next
suites to write, in this order:

| module | lines | why it matters |
| --- | --- | --- |
| `temporal/_validation.py` | ~520 | `TFForceValidation` produces the enriched-vs-random figure the paper rests on. Nothing pins the null-sampling, the `exclude='tf_and_target'` logic, or `compare_sets_by_phase`. |
| `temporal/_phases.py` | ~700 | `RegulatoryPhases` / `ForceWavePhases` / `BindingPhases` and the softmax peak helpers. The phase boundaries feed both the heatmaps and the validation split. |
| `temporal/_states.py` | ~200 | `StateFrequency.termination_pseudotime` defines those phase boundaries. |
| `temporal/manager.py` | ~330 | `TemporalManager` -- episode slicing arithmetic, the result registry, `write=True` paths. |

The other two modules (`tests/state_specific/`, `tests/cross_prediction/`) do not
exist yet; they land with their respective ports.

## The mock data

`conftest.build_mock_network()` assembles a genuine
`dictys.net.dynamic_network` -- not a stub -- from a Y-shaped trajectory with
4 nodes, 7 windows, 21 cells, 8 genes and 3 regulators.  Because it is a real
dictys object, the tests exercise the same `linspace` / `stat` / Gaussian
smoothing machinery as production runs, and the hand-written fast paths in
`temporal/_curves.py` can be compared directly against the stock dictys stat
chains they replace.

The data is chosen so that key results are exact rather than approximate:

* genes with a constant CPM keep their exact `log2(CPM + 1)` at every
  pseudotime point (smoothing weights are normalised to sum to 1), so e.g. `G1`
  is exactly `4.0` everywhere;
* GRN edges with a constant per-window weight smooth to exactly that weight, so
  `TFA -> G1` is exactly `2.0` at every point;
* pseudotime along a branch is exactly `linspace(0, path_length, num_points)`;
* window pseudotimes are exactly `[0, .5, 1, 1.5, 2, 1.5, 2]` from node 0.

The seven non-zero mock edges cover a constant positive edge, a positive ramp, a
sign-flipping ramp, a TF -> TF edge, a constant negative edge, a negative ramp,
and a `ZNF*` edge that exists purely to pin down the hard-coded ZNF/ZBTB filter
in `build_episode_grn`.

The mock binding files (`Subset<i>/binding.tsv.gz`) use the real dictys layout
(`TF`, `loc = chr:start:end`, `score`) with scores picked so the *mean of
per-chromosome means* differs from a plain global mean -- otherwise the tests
could not tell which aggregation the code performs.

## Conventions

* A test named `test_currently_...` pins behaviour that is wrong but load-bearing;
  the matching `xfail(strict=True)` test asserts what the behaviour should be.
* `xfail(strict=True)` means a fix turns the test into an `XPASS` failure. That
  is the prompt to delete the `currently` test and drop the marker.
* Every expected value is either derived on paper (and stated in a comment) or
  computed by an independent implementation (`numpy.trapezoid`,
  `scipy.stats.hypergeom`, the stock dictys stat chain) -- never by calling the
  function under test.
