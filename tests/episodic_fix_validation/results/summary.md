# Episodic enrichment: pre-fix vs post-fix

Both variants were computed from **identical** smoothed networks and identical
filtered edge sets; they differ only in ISSUES #1 (force/expression alignment)
and #3 (episode time-slice for regulator expression).

- `dynamic.h5`: `/work/nvme/bhdw/asachan/data_files/firefate/bcell/outs/dynamic.h5`
- LF gene set: union of `/work/nvme/bhdw/asachan/data_files/firefate/bcell/latent_factors/feature_list_Z11_GC_PB.txt,/work/nvme/bhdw/asachan/data_files/firefate/bcell/latent_factors/feature_list_Z3_GC_PB.txt` (57 genes after dropping `HLA-`)
- num_points=20, points_per_episode=5, dist=0.001, sparsity=0.01
- percentile=98.0, pval_threshold=0.001, significance alpha=0.05

## Per-episode agreement

| branch | ep | edges filtered | TFs fixed | TFs legacy | sig∩ | only fixed | only legacy | Jaccard | ES ρ | top20 ∩ | expr identical |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PB | 1 | 500,816 | 50 | 57 | 2 | 0 | 6 | 0.250 | 0.846 | 11/20 | True |
| PB | 2 | 458,924 | 43 | 52 | 6 | 2 | 10 | 0.333 | 0.893 | 8/20 | False |
| PB | 3 | 284,629 | 32 | 33 | 3 | 3 | 5 | 0.273 | 0.901 | 11/20 | False |
| PB | 4 | 230,471 | 20 | 25 | 0 | 4 | 7 | 0.000 | 0.881 | 9/20 | False |
| GC | 1 | 648,254 | 59 | 69 | 3 | 5 | 7 | 0.200 | 0.772 | 7/20 | True |
| GC | 2 | 586,532 | 49 | 52 | 2 | 4 | 2 | 0.250 | 0.742 | 6/20 | False |
| GC | 3 | 228,735 | 25 | 21 | 2 | 2 | 5 | 0.222 | 0.905 | 13/20 | False |
| GC | 4 | 578,961 | 50 | 54 | 6 | 2 | 7 | 0.400 | 0.905 | 10/20 | False |

## Verdict

- episodes compared: **8**
- episodes where the enriched-TF set changed: **8**
- mean significant-TF Jaccard: **0.241**
- mean ES Spearman (TFs present in both): **0.856**
- mean top-20 overlap: **9.4/20**
- episodes with no significant TF in either variant: **0**

TFs gained or lost at the significance threshold, per episode:

- **PB ep1**
  - only before fix: `ATF3;FOSL1;IKZF3;MAX;MYC;POU2F2`
- **PB ep2**
  - only after fix: `REL;RELA`
  - only before fix: `CREBL2;HES1;IKZF2;MYC;NFAT5;PBX2;POGK;POU2F2;TCF12;USF2`
- **PB ep3**
  - only after fix: `IKZF3;MEF2C;REL`
  - only before fix: `CREBL2;MLX;RUNX1;RUNX3;TFEB`
- **PB ep4**
  - only after fix: `CREB3L2;MEF2A;RUNX3;STAT3`
  - only before fix: `ARNTL;HNF1B;MTF2;MYNN;PLAG1;RELB;TEAD2`
- **GC ep1**
  - only after fix: `IKZF3;MEF2C;MYC;NFATC2;RELA`
  - only before fix: `ATF3;BPTF;CENPB;FOSL1;IKZF1;IRF2;RUNX3`
- **GC ep2**
  - only after fix: `IKZF3;NR3C1;RELA;RUNX3`
  - only before fix: `BHLHE40;CENPB`
- **GC ep3**
  - only after fix: `CDC5L;MAX`
  - only before fix: `MYC;PBX2;TCF12;TCF4;USF2`
- **GC ep4**
  - only after fix: `CREB3L2;NFATC2`
  - only before fix: `CEBPB;POGK;POU2F1;POU3F1;SOX4;TCF4;ZFP14`
