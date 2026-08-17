# Tests for `firefate.core`

Coverage of `src/firefate/core/episodic_dynamics.py` and
`src/firefate/core/pseudotime_curves.py`: every public class and function, the
private fast paths, and the two process-level runners.

## Running

```bash
conda activate dictys
cd FIREFate
pytest tests/episodic -q                 # everything (~20 s)
pytest tests/episodic -q -m "not slow"   # skip the tests that spawn worker processes
```

## Layout

| file | subject |
| --- | --- |
| `conftest.py` | the mock dictys network, mock binding files, curve fixtures |
| `test_curve_math.py` | AUC, transient/terminal logFC, switching time, curve characteristics, wave-pattern classification, static force curves |
| `test_smoothed_curves_grn.py` | `SmoothedCurvesGRN` against a real dictys network: smoothed expression/regulation curves, the parallel regulation fast path, sub-network curves, beta curves |
| `test_chromatin_curves.py` | `SmoothedCurvesChromatin`: binding extraction, trajectory mapping, smoothing, both plots |
| `test_episodic_units.py` | module-level helpers of `episodic_dynamics`: enrichment, edge filtering, chunking, force curves, episode subsetting |
| `test_episode_workflow.py` | `AlignTimeScales`, the `EpisodeDynamics` workflow end to end, `run_episodic_construction` / `run_episodic_enrichment` |
| `ISSUES.md` | defects and inconsistencies the suite found, each with the test that reproduces it |

## The mock data

`conftest.build_mock_network()` assembles a genuine
`dictys.net.dynamic_network` -- not a stub -- from a Y-shaped trajectory with
4 nodes, 7 windows, 21 cells, 8 genes and 3 regulators.  Because it is a real
dictys object, the tests exercise the same `linspace` / `stat` / Gaussian
smoothing machinery as production runs, and the hand-written fast paths in
`pseudotime_curves.py` can be compared directly against the stock dictys stat
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
