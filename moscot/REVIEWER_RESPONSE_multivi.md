# Response to Reviewer — MultiVI / moscot overfitting

**Reviewer comment.** "The ChromBPNet neural network seems to be trained only on the data in
the paper, raising a concern on whether there was sufficient training data for the network to
capture the biology rather than overfitting its responses. Multiv (Methods section) might have
similar issues, but no details are available."

---

## Response

We thank the reviewer for raising this, and we can see why the two models were grouped
together — we agree that our original description of MultiVI was too brief, and we are glad of
the opportunity to clarify. Unlike ChromBPNet, however, MultiVI is not a predictive model in
our study, so the question of whether it has seen sufficient training data to generalise does
not really arise: it is an unsupervised autoencoder that is fit to, and applied only to, our
own paired cells, and its role is simply to compress the two modalities into a shared
low-dimensional embedding that we use to define cell–cell distances for the optimal-transport
analysis. It is never asked to make a prediction about a cell, sample or sequence it has not
seen, and so there is no held-out setting in which it could fail to generalise. We would add
that our dataset is comfortably large relative to the size of the model, and that we did check
directly for the behaviour the reviewer describes: training used a standard held-out validation
split, and the validation loss plateaued rather than rising, with the gap between training and
validation loss remaining flat from early in training through to the final epoch (new
Supplementary Fig. X). An overfitting model would show a progressively widening gap and a
rising validation loss, and we observe neither. Reassuringly, the transport maps and driver-TF
rankings are also reproduced when the MultiVI embedding is replaced by one built without any
neural network (new Supplementary Fig. `[X]`). The full architecture, training settings and
these diagnostics are now given in the Methods, reproduced below, and we hope this addresses
the reviewer's concern.

---

## Text added to the Methods

**Joint multiome embedding with MultiVI.** Cells with paired scRNA-seq and scATAC-seq profiles
(n = 28,494 passing QC in both modalities) were assembled into a single MuData object holding
raw counts for 3,018 highly variable genes and 191,255 ATAC peaks, and a joint representation
of the two modalities was learned with MultiVI (scvi-tools v1.4.1) using the raw count layer of
each modality as input and no batch or covariate keys. MultiVI is an unsupervised variational
autoencoder that is fit to, and applied only to, the cells of this dataset; it performs
dimensionality reduction and is not used to predict held-out cells, samples or sequences. We
used the scvi-tools default architecture — two-layer encoders and decoders with 128 hidden
units, an 11-dimensional latent space, dropout 0.1, a zero-inflated negative binomial
likelihood with gene-specific dispersion for the expression modality, learned region factors
for the accessibility modality, layer normalisation, equal modality weights and a Jeffreys
modality penalty — giving 75,950,365 parameters fit to 28,494 × 194,273 ≈ 5.5 × 10^9 observed
count values (~73 observations per parameter). Training followed the scvi-tools defaults, which
hold out 10% of cells as a validation set: the model was fit on 25,645 cells and evaluated on
2,849 held-out cells for 200 epochs, with the KL term linearly warmed to full weight over the
first 50 epochs. Validation reconstruction loss fell alongside the training loss, plateaued
from approximately epoch 25, reached its minimum at epoch 182 and ended within 0.3% of that
minimum at epoch 200; the train–validation gap was stable throughout (5.6% of the validation
loss at epoch 10 versus 6.1% at epoch 200; Supplementary Fig. X), indicating that the latent
space did not become fit to noise in the training cells. The resulting 11-dimensional joint
latent representation was used for all downstream distance and neighbourhood computations.

---

## Supplementary figure legend

**Supplementary Fig. X. MultiVI training and validation loss.** (a) Reconstruction loss on the
25,645 training cells and the 2,849 held-out validation cells over 200 training epochs; dashed
line marks the validation minimum at epoch 182. (b) Train–validation gap, expressed as a
percentage of the validation loss, which remains flat (5.6% → 6.1%) over epochs 10–199.

---

## Still needed before sending

- [ ] The PCA + LSI embedding-swap re-run, and its concordance statistics.
      **This experiment has not been run** — either run it or delete that sentence.
- [ ] ChromBPNet half of the response (training regions, held-out chromosomes, fold structure,
      held-out performance, bias model) — written by whoever ran it; no ChromBPNet code is in
      this repository.

## Verified from `multiVI_model.pt`

| Item | Value |
|---|---|
| scvi-tools version | 1.4.1 |
| Train / validation / test cells | 25,645 / 2,849 / 0 (exactly 90/10) |
| Epochs trained | 200 (ran to completion; **early stopping was not triggered**) |
| Architecture | 2-layer encoder/decoder, 128 hidden, 11 latent, dropout 0.1, ZINB, layer norm |
| Parameters | 75,950,365 |
| Validation loss minimum | epoch 182; final epoch within +0.29% |
| Train–validation gap | 5.6% (epoch 10) → 6.1% (epoch 199) |
