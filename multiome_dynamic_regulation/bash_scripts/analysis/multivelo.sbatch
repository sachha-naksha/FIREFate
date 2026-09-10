#!/bin/bash
#SBATCH --account=bgdb-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=multivelo
#SBATCH --output=multivelo_%j.log

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate multivelo

python3 << 'PYEOF'
import os
import numpy as np
import scanpy as sc
import scvelo as scv
import multivelo as mv
import anndata
import scipy.sparse

# Monkey-patch for scipy .A removal
scipy.sparse.csc_matrix.A = property(lambda self: self.toarray())
scipy.sparse.csr_matrix.A = property(lambda self: self.toarray())

# Allow nullable strings for any downstream writes
anndata.settings.allow_write_nullable_strings = True

CACHE = "/projects/bgdb/asachan/methods/FocalFire/multiome_dynamic_regulation/py_scripts/trajectory/.cache"
OUT_DIR = "/work/hdd/bgdb/asachan/datasets/B_cell_dictys_actb1_added_v2/multivelo_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════ LOAD CHECKPOINTS ═══════════════
print("Loading cached adata_rna...")
adata_rna = sc.read_h5ad(os.path.join(CACHE, "adata_rna_filtered.h5ad"))
print(f"  RNA: {adata_rna.shape}, layers: {list(adata_rna.layers.keys())}")

print("Loading cached adata_atac...")
adata_atac = sc.read_h5ad(os.path.join(CACHE, "adata_atac_filtered.h5ad"))
print(f"  ATAC: {adata_atac.shape}, layers: {list(adata_atac.layers.keys())}")

# ═══════════════ VERIFY ALIGNMENT ═══════════════
assert np.all(adata_rna.obs_names == adata_atac.obs_names), "Cell order mismatch!"
assert np.all(adata_rna.var_names == adata_atac.var_names), "Gene order mismatch!"
print(f"✓ Aligned: {adata_rna.n_obs} cells × {adata_rna.n_vars} genes")

# ═══════════════ GENERATE CELLTYPE COLORS ═══════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sc.pl.umap(adata_rna, color="celltype", show=False)
plt.close()
print(f"✓ celltype_colors generated: {adata_rna.uns.get('celltype_colors', 'MISSING')}")

# ═══════════════ RUN MULTIVELO ═══════════════
print("Starting recover_dynamics_chrom...")

adata_result = mv.recover_dynamics_chrom(
    adata_rna,
    adata_atac,
    max_iter=5,
    init_mode="invert",
    parallel=True,
    n_jobs=16,
    save_plot=False,
    rna_only=False,
    fit=True,
    n_anchors=500,
    extra_color_key="celltype",
)

# ═══════════════ SAVE RESULT ═══════════════
result_path = os.path.join(OUT_DIR, "multivelo_result.h5ad")
adata_result.write(result_path)
print(f"✓ Saved: {result_path}")

# ═══════════════ POST-PROCESSING ═══════════════
print("Computing velocity graph...")
mv.velocity_graph(adata_result)

if "X_umap" not in adata_result.obsm:
    adata_result.obsm["X_umap"] = adata_rna.obsm["X_umap"].copy()
if "celltype" not in adata_result.obs.columns:
    adata_result.obs["celltype"] = adata_rna.obs["celltype"].values

scv.tl.latent_time(adata_result)

final_path = os.path.join(OUT_DIR, "multivelo_result_final.h5ad")
adata_result.write(final_path)
print(f"✓ Saved final: {final_path}")
print("Done!")

PYEOF
