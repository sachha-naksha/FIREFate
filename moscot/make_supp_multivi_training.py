"""Supplementary figure: MultiVI train vs. validation loss.

Reads the training history extracted from multiVI_model.pt (multivi_training_history.csv)
and writes supp_multivi_training.{png,pdf}.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRAIN_C, VAL_C = "#2a78d6", "#eb6834"
INK, MUTED = "#1a1a19", "#6b6a63"

df = pd.read_csv("multivi_training_history.csv", index_col=0)
ep = df.index.to_numpy()
tr = df["reconstruction_loss_train"].to_numpy()
va = df["reconstruction_loss_validation"].to_numpy()
gap = (va - tr) / va * 100
best = int(np.argmin(va))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

# (a) loss curves
ax1.plot(ep, tr, color=TRAIN_C, lw=1.5, label="Train (n = 25,645 cells)")
ax1.plot(ep, va, color=VAL_C, lw=1.5, label="Validation (n = 2,849 cells)")
ax1.axvline(best, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
ax1.annotate(f"validation minimum\n(epoch {best})", xy=(best, va[best]),
             xytext=(best - 8, va[best] + 2600), ha="right", fontsize=6.5,
             color=MUTED, linespacing=1.3)
ax1.text(ep[-1] + 3, tr[-1], "Train", color=TRAIN_C, fontsize=7, va="center")
ax1.text(ep[-1] + 3, va[-1], "Validation", color=VAL_C, fontsize=7, va="center")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Reconstruction loss")
ax1.set_title("a", loc="left", fontweight="bold", fontsize=9)
ax1.set_xlim(-4, 232)
ax1.legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1.4)

# (b) generalisation gap
ax2.plot(ep[1:], gap[1:], color=INK, lw=1.5)
ax2.set_ylim(0, 8)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Train–validation gap (%)")
ax2.set_title("b", loc="left", fontweight="bold", fontsize=9)
ax2.set_xlim(-4, 204)
ax2.annotate(f"{gap[10]:.1f}% → {gap[-1]:.1f}%\nover epochs 10–199",
             xy=(100, gap[100]), xytext=(100, 2.0), ha="center",
             fontsize=6.5, color=MUTED, linespacing=1.3)

for ax in (ax1, ax2):
    ax.grid(axis="y", color="#e8e8e4", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(MUTED); ax.spines[s].set_lw(0.8)
    ax.tick_params(labelsize=7, colors=MUTED, length=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels(): lbl.set_color(INK)
    ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)

fig.tight_layout()
fig.savefig("supp_multivi_training.png", dpi=400, bbox_inches="tight")
fig.savefig("supp_multivi_training.pdf", bbox_inches="tight")
print("wrote supp_multivi_training.png / .pdf")
