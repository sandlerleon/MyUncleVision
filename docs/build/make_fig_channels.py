# -*- coding: utf-8 -*-
"""Figure: the two channels on one lattice that contains both kinds of anomaly (a test cohort of the mixed scenario, seed 121; first image).
    python docs/build/make_fig_channels.py      -> docs/figures/fig_channels.png"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.colors import ListedColormap                         # noqa: E402

from myuncle_vision import cohort as C, gaar_opt as GO, morphology as M, two_channel as T, tc_study as S     # noqa: E402

SEED = 121
slides, kinds = [], []
for i in range(10):
    X, k = M.make_lattice_mixed(SEED * 1000 + i, out_shape="block", out_block=3, shift=S.MIX_SHIFT)
    slides.append(M._slide("f%d" % i, X))
    kinds.append(k)
C.analyse(slides, GO.to_config(dict(GO.DEFAULT, smooth="none"), n_states=6, ref_per_slide=300, ref_max=3000))
n = 32
r = [s.extra["p_emb_ms"].reshape(n, n) for s in slides]
U = [s.extra["U"].reshape(n, n, -1) for s in slides]
valid = np.ones((n, n), bool)
b = [T.directional_disorder(u, valid, 1)[0] for u in U]
ra, rb = T.cohort_rank(r), T.cohort_rank(b)
k0 = kinds[0]
flag = np.zeros((n, n), int)
flag[ra[0] >= 0.97] = 1
flag[np.nan_to_num(rb[0], nan=0.0) >= 0.97] = 2
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.05))
cm3 = ListedColormap(["#f2f2f2", "#d95f02", "#1f78b4"])
ax[0].imshow(k0, cmap=cm3, vmin=0, vmax=2, interpolation="nearest")
ax[0].set_title("(a) anomalies: feature outliers\n(orange), rearranged (blue)", fontsize=7)
im1 = ax[1].imshow(ra[0], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
ax[1].set_title("(b) channel A: rarity\n(cohort rank)", fontsize=7)
im2 = ax[2].imshow(np.nan_to_num(rb[0], nan=0.0), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
ax[2].set_title("(c) channel B: directional\ndisorder (cohort rank)", fontsize=7)
ax[3].imshow(flag, cmap=cm3, vmin=0, vmax=2, interpolation="nearest")
ax[3].set_title("(d) top 3% of channel A (orange)\nand of channel B (blue)", fontsize=7)
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
fig.tight_layout(w_pad=0.6)
out = os.path.join(HERE, "..", "figures", "fig_channels.png")
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.04)
print("written", out, "| share of the outliers in the top 3%% of A: %.2f, of the rearranged sites in the top 3%% of B: %.2f" % (np.mean(ra[0][k0 == 1] >= 0.97), np.mean(np.nan_to_num(rb[0], nan=0.0)[k0 == 2] >= 0.97)))
