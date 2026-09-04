import os
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings

import muon as mu
import anndata as ad
from scvi.model import MULTIVI

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

ad.settings.allow_write_nullable_strings = True

mdata = mu.read_h5mu('/ocean/projects/cis240075p/cliu57/dynamic_OT/datasets/B_cell/atac_rna.h5mu')

MULTIVI.setup_mudata(
    mdata,
    rna_layer="counts",
    atac_layer="counts",
    modalities={"rna_layer": "rna", "atac_layer": "atac"}
)

model = MULTIVI(mdata)
model.train(max_epochs=200)  # adjust depending on dataset size

# latent
mdata.obsm["X_MultiVI"] = model.get_latent_representation()
mdata.write_h5mu("/ocean/projects/cis240075p/cliu57/dynamic_OT/datasets/B_cell/rna_atac_multivi.h5mu")
model.save("/ocean/projects/cis240075p/cliu57/dynamic_OT/datasets/B_cell/multivi_model", overwrite=True)