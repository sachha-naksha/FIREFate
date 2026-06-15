import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import os
import json
import celloracle as co
from celloracle.applications import Gradient_calculator, Oracle_development_module
from celloracle.visualizations.config import CONFIG

# -----------------------
# paths / settings
# -----------------------
wd = '/ocean/projects/cis240075p/heidarir/CellOracle_Tonsil_Bcells'
out_path = os.path.join(wd, 'out_data', 'TF_KO_viz_allTFs')
os.makedirs(out_path, exist_ok=True)

# global subfolders
os.makedirs(f"{out_path}/figures", exist_ok=True)
os.makedirs(f"{out_path}/out_files", exist_ok=True)

sc.settings.figdir = f"{out_path}/figures"
plt.rcParams['figure.figsize'] = [6, 4.5]
plt.rcParams["savefig.dpi"] = 300

# -----------------------
# load once
# -----------------------
oracle = co.load_hdf5(f"{wd}/out_data/sc_preproc_co/out_files/tonsil_Bcells.celloracle.oracle")
stream_xdr = pd.read_csv(f"{wd}/out_data/sc_preproc_co/out_files/stream_xdr.csv", index_col=0)
links = co.load_hdf5(f"{wd}/out_data/clusterGRNs/links.celloracle.links")

# subset and set embeddings/pseudotime
oracle.adata = oracle.adata[oracle.adata.obs_names.isin(stream_xdr.index)].copy()
oracle.adata.obs['S2_pseudotime'] = stream_xdr.loc[oracle.adata.obs_names, 'S2_pseudotime'].values
oracle.adata.obsm['X_dr'] = stream_xdr.loc[oracle.adata.obs_names, ['x', 'y']].values
oracle.adata.obsm['X_umap'] = oracle.adata.obsm['X_dr']
oracle.embedding = oracle.adata.obsm['X_dr']

# -----------------------
# helper: check gene existence
# -----------------------
def gene_in_adata(adata, gene):
    if gene in adata.var_names:
        return True
    if adata.raw is not None and gene in adata.raw.var_names:
        return True
    return False

# -----------------------
# helper: make per-cell colors from annotation palette
# -----------------------
def get_cell_colors(adata, key="annotation_figure_1"):
    palette_key = f"{key}_colors"
    if key not in adata.obs:
        raise KeyError(f"Missing adata.obs['{key}']")
    if palette_key not in adata.uns:
        raise KeyError(f"Missing adata.uns['{palette_key}']")

    adata.obs[key] = adata.obs[key].astype("category")
    cats = adata.obs[key].cat.categories
    palette = adata.uns[palette_key]
    lut = dict(zip(cats, palette))
    return adata.obs[key].map(lut).values

# -----------------------
# helper: check if gene is a regulator usable for simulation
# (CellOracle will error if TF/motif info isn't available in the base GRN)
# -----------------------
def gene_is_simulatable_regulator(oracle_obj, gene):
    # Try common attributes across CellOracle versions
    for attr in ["TFdict", "cluster_specific_TFdict", "TFdict_by_cluster"]:
        if hasattr(oracle_obj, attr):
            d = getattr(oracle_obj, attr)
            try:
                if isinstance(d, dict) and gene in d:
                    return True
            except Exception:
                pass
    # Fall back: if we can't tell, let simulate_shift decide
    return None

# -----------------------
# global log for missing TFs and errors
# -----------------------
log_path = os.path.join(out_path, "out_files", "TF_loop_log.txt")
with open(log_path, "w") as f:
    f.write("TF loop log\n")

# -----------------------
# your TF list
# -----------------------
#gois =   ['SP4', 'JUND', 'NFIA', 'THRB',
 #                   'SP3', 'ETV6', 'BHLHE40', 'SPI1', 'BCL6', 
  #                  'NR3C2', 'MYBL1', 'SPIB', 'NFATC1']

# gois =  ['SPIB', 'MEF2A', 'STAT1', 'SPI1', 'SP3', 'HIVEP3', 'EZH2', 'BCL6',
#          'BHLHE41', 'STAT1', 'NFKB1', 'THRB', 'NR6A1'] 

#gois =  ['TFDP2', 'NFATC1', 'IRF1', 'STAT1', 'STAT4', 'IRF8', 'HES1', 'AIRE', 
    #'BHLHE40', 'THRB', 'NR4A3', 'THRA', 'POU6F1']

# gois =[
#     'PAX5', 'MEF2A', 'MYC', 'NFATC1', 'EZH2', 'TCF12', 'IRF1', 'ESR1', 'IKZF2', 'MYBL1', 'CLOCK', 'KLF7', 'TFDP2'
# ]
TFs_all = pd.read_csv( "/ocean/projects/cis240075p/heidarir/CellOracle_Tonsil_Bcells/out_data/TF_KO_viz/out_files/TFs.csv")
gois = TFs_all['TF'].tolist()

# -----------------------
# shared constants
# -----------------------
key_annot = "annotation_figure_1"

# Ensure annotation exists (uncomment if you need to pull it from stream_xdr)
# if key_annot not in oracle.adata.obs and key_annot in stream_xdr.columns:
#     oracle.adata.obs[key_annot] = stream_xdr.loc[oracle.adata.obs_names, key_annot].values

cell_colors = get_cell_colors(oracle.adata, key=key_annot)
emb = oracle.embedding  # (n_cells, 2)

# CONFIG tweaks (once)
CONFIG["default_args_quiver"] = {"width": 0.002, "minlength": 2, "headwidth": 5, "headlength": 5, "minshaft": 1}
CONFIG["cmap_ps"] = "coolwarm"
CONFIG["default_args"] = {"lw": 0.3, "rasterized": False}

# p-mass params (for Gradient_calculator development module)
n_grid_main = 30
min_mass_gradient = 50000  # keep your original for Gradient_calculator (your choice)

# p-mass params (for ORACLE grid flow plotting; this must be small)
n_grid_oracle = 40
min_mass_oracle = 0.01  # typical scale in tutorials; adjust if you filter too much/too little

# -----------------------
# prep GRN simulation once
# -----------------------
links.filter_links()
oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True)

# -----------------------
# loop over TFs
# -----------------------
all_scores = []

for goi in gois:
    goi_dir = os.path.join(out_path, "out_files", goi)
    fig_dir = os.path.join(goi_dir, "figures")
    out_dir = os.path.join(goi_dir, "out_files")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # check TF presence in adata (and skip without stopping)
    if not gene_in_adata(oracle.adata, goi):
        msg = f"{goi} not found in adata.var_names (or adata.raw.var_names). Skipping.\n"
        with open(log_path, "a") as f:
            f.write(msg)
        with open(os.path.join(out_dir, f"{goi}_missing.txt"), "w") as f:
            f.write(msg)
        continue

    # optional: pre-check simulatable regulator (helps avoid AFF1-like errors)
    simulatable = gene_is_simulatable_regulator(oracle, goi)
    if simulatable is False:
        msg = f"{goi} appears not to be in Oracle TF dict/base GRN regulator set. Skipping.\n"
        with open(log_path, "a") as f:
            f.write(msg)
        with open(os.path.join(out_dir, f"{goi}_not_in_base_grn.txt"), "w") as f:
            f.write(msg)
        continue

    try:
        # 1) optional: UMAP expression plot (saved)
        fig = sc.pl.umap(
            oracle.adata,
            color=[goi, oracle.cluster_column_name],
            layer="counts",
            use_raw=False,
            cmap="viridis",
            show=False,
            return_fig=True
        )
        fig.savefig(os.path.join(fig_dir, f"{goi}_umap_expr.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 2) simulate perturbation + transition prob + embedding shift
        print(f"Simulating perturbation for {goi}", flush=True)
        oracle.simulate_shift(perturb_condition={goi: 0.0}, n_propagation=2)

        print(f"Estimating transition probability for {goi}", flush=True)
        oracle.estimate_transition_prob(n_neighbors=750, knn_random=True, sampled_fraction=1)

        print(f"Calculating embedding shift for {goi}", flush=True)
        oracle.calculate_embedding_shift(sigma_corr=0.05)

        # 2b) IMPORTANT: build grid flow field for plotting (creates oracle.flow)
        print(f"Calculating p-mass / mass filter for grid flow plotting: {goi}", flush=True)
        oracle.calculate_p_mass(smooth=0.8, n_grid=n_grid_oracle, n_neighbors=200)
        oracle.calculate_mass_filter(min_mass=min_mass_oracle, plot=False)

        # 3) flow plots on grid (simulation vs random vs scanpy)
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        ax[0].scatter(emb[:, 0], emb[:, 1], s=10, c=cell_colors, linewidths=0)
        ax[0].set_title(f"Simulated shift: {goi} KO")
        ax[0].axis("off")
        oracle.plot_simulation_flow_on_grid(scale=8, ax=ax[0], show_background=False)

        ax[1].scatter(emb[:, 0], emb[:, 1], s=10, c=cell_colors, linewidths=0)
        ax[1].set_title(f"Randomized shift: {goi} KO")
        ax[1].axis("off")
        oracle.plot_simulation_flow_random_on_grid(scale=8, ax=ax[1], show_background=False)

        sc.pl.umap(oracle.adata, color=[key_annot], ax=ax[2], show=False)
        ax[2].set_title("Clustered cell identity")

        fig.savefig(os.path.join(fig_dir, f"{goi}_flow_panels.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 4) gradient + development module (per-TF)
        gradient = Gradient_calculator(oracle_object=oracle, pseudotime_key="S2_pseudotime")
        gradient.calculate_p_mass(smooth=0.8, n_grid=n_grid_main, n_neighbors=500)
        gradient.calculate_mass_filter(min_mass=min_mass_gradient, plot=False)
        gradient.transfer_data_into_grid(args={"method": "knn", "n_knn": 200}, plot=False)
        gradient.calculate_gradient()

        # save gradient object in TF folder
        gradient_path = os.path.join(out_dir, f"{goi}.celloracle.gradient")
        gradient.to_hdf5(gradient_path)

        dev = Oracle_development_module()
        dev.load_differentiation_reference_data(gradient_object=gradient)
        dev.load_perturb_simulation_data(oracle_object=oracle, name=f"{goi}")
        dev.calculate_inner_product()
        dev.calculate_digitized_ip(n_bins=10)

        # development flow + inner product over pseudotime
        show_background = True
        s_grid = CONFIG['s_grid']
        s = CONFIG['s_scatter']

        fig, ax = plt.subplots(1, 2, figsize=[10, 5])
        dev.plot_reference_flow_on_grid(
            ax=ax[0], scale=40, show_background=show_background, s=s, args=CONFIG["default_args_quiver"]
        )
        ax[0].set_title("Development flow")

        dev.plot_inner_product_on_pseudotime(ax=ax[1], vm=1, s=s_grid)
        fig.savefig(os.path.join(fig_dir, f"{goi}_development.svg"), format="svg", bbox_inches="tight")
        plt.close(fig)

        # perturb simulation + inner product on grid
        fig, ax = plt.subplots(1, 2, figsize=[10, 5])
        dev.plot_simulation_flow_on_grid(
            ax=ax[0], scale=8, show_background=show_background, s=s, args=CONFIG["default_args_quiver"]
        )
        ax[0].set_title("Perturb simulation")

        dev.plot_inner_product_on_grid(ax=ax[1], vm=0.04, s=s_grid, show_background=show_background)
        ax[1].set_title("Inner product\n(Perturb * Development)")
        fig.savefig(os.path.join(fig_dir, f"{goi}_perturbation.svg"), format="svg", bbox_inches="tight")
        plt.close(fig)

        # 5) perturbation score (PC vs GCBC) + save f_dict
        if key_annot not in oracle.adata.obs:
            msg = f"{goi}: missing adata.obs['{key_annot}']; cannot compute PC/GCBC scores.\n"
            with open(log_path, "a") as f:
                f.write(msg)
            f_dict = {"goi": goi, "error": msg.strip()}
        else:
            cell_idx_pc = np.where(oracle.adata.obs[key_annot].isin(['PC']))[0]
            cell_idx_gc = np.where(oracle.adata.obs[key_annot].isin(['GCBC']))[0]

            # PC
            dev_pc = Oracle_development_module()
            dev_pc.load_differentiation_reference_data(gradient_object=gradient)
            dev_pc.load_perturb_simulation_data(oracle_object=oracle, cell_idx_use=cell_idx_pc, name=f"{goi}_pc")
            dev_pc.calculate_inner_product()
            dev_pc.calculate_digitized_ip(n_bins=10)
            pc_pos_ps = dev_pc.get_sum_of_positive_ips()['score'].sum()
            pc_neg_ps = dev_pc.get_sum_of_negative_ips()['score'].sum()
            pc_score = abs(pc_pos_ps) + abs(pc_neg_ps)

            # GCBC
            dev_gc = Oracle_development_module()
            dev_gc.load_differentiation_reference_data(gradient_object=gradient)
            dev_gc.load_perturb_simulation_data(oracle_object=oracle, cell_idx_use=cell_idx_gc, name=f"{goi}_gc")
            dev_gc.calculate_inner_product()
            dev_gc.calculate_digitized_ip(n_bins=10)
            gc_pos_ps = dev_gc.get_sum_of_positive_ips()['score'].sum()
            gc_neg_ps = dev_gc.get_sum_of_negative_ips()['score'].sum()
            gc_score = abs(gc_pos_ps) + abs(gc_neg_ps)

            f_dict = {
                "goi": goi,
                "pc_score": float(pc_score),
                "gc_score": float(gc_score),
                "pc_neg_ps": float(pc_neg_ps),
                "gc_neg_ps": float(gc_neg_ps),
                "pc_pos_ps": float(pc_pos_ps),
                "gc_pos_ps": float(gc_pos_ps),
                "n_pc_cells": int(len(cell_idx_pc)),
                "n_gcbc_cells": int(len(cell_idx_gc)),
            }

        # save f_dict per TF
        json_path = os.path.join(out_dir, f"{goi}_f_dict.json")
        with open(json_path, "w") as f:
            json.dump(f_dict, f, indent=2)

        all_scores.append(f_dict)

    except Exception as e:
        msg = f"{goi}: ERROR: {repr(e)}\n"
        with open(log_path, "a") as f:
            f.write(msg)
        with open(os.path.join(out_dir, f"{goi}_error.txt"), "w") as f:
            f.write(msg)
        continue

# save combined summary table
summary_df = pd.DataFrame(all_scores)
summary_df.to_csv(os.path.join(out_path, "out_files", "all_TF_scores.csv"), index=False)

print("Done. Summary saved to:", os.path.join(out_path, "out_files", "all_TF_scores.csv"))
print("Log saved to:", log_path)