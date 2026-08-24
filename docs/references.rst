References
##########

Methods FIREFate builds on
==========================

**Dictys** — Wang, L. *et al.* (2023).
  Dictys: dynamic gene regulatory network dissects developmental continuum with
  single-cell multi-omics. *Nature Methods* 20, 1368-1378.
  Supplies the per-window network reconstruction and the smoothing machinery behind
  :class:`~firefate.temporal.SmoothedCurvesGRN`.

**SLIDE** — Rahimikollu, J. *et al.* (2024).
  SLIDE: significant latent factor interaction discovery and exploration across
  biological domains. *Nature Methods* 21, 835-845.
  Supplies the latent factors that FIREFate treats as cellular programs.

**CellOracle** — Kamimoto, K. *et al.* (2023).
  Dissecting cell identity via network inference and in silico gene perturbation.
  *Nature* 614, 742-751.
  The state-specific GRN and in-silico knockout path.

**STREAM** — Chen, H. *et al.* (2019).
  Single-cell trajectories reconstruction, exploration and mapping of omics data with
  STREAM. *Nature Communications* 10, 1903.
  Trajectory inference upstream of the temporal module.

**MultiVelo** — Li, C., Virgilio, M. C., Collins, K. L. & Welch, J. D. (2023).
  Multi-omic single-cell velocity models epigenome-transcriptome interactions and
  improves cell fate prediction. *Nature Biotechnology* 41, 387-398.

**moscot** — Klein, D. *et al.* (2025).
  Mapping cells through time and space with moscot. *Nature*.
  The optimal-transport work the cross-prediction module draws on, and the source of
  this package's layout conventions.
