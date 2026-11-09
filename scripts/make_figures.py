# -*- coding: utf-8 -*-
"""Figures of the paper and the white paper:  python scripts/make_figures.py   (needs results/validation_*.json)"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from myuncle_vision import _core, lattice as lat      # noqa: E402

FIG = os.path.join(ROOT, "docs", "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)
C1, C2, C3, C4, GR = "#1f4e9c", "#d1495b", "#2a9d8f", "#e9a23b", "0.35"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.axis("off")
    boxes = [("whole-slide\nimages (SVS)", 0.02), ("tissue mask,\npatch grid, QC", 0.19), ("patch\nembeddings\n(DINOv2)", 0.36), ("sphere + tissue\nlattice", 0.53),
             ("order D, entropy\nH_chi, states,\ntransitions", 0.70), ("novelty, regions,\ncandidate states", 0.87)]
    for t, x in boxes:
        ax.add_patch(FancyBboxPatch((x - 0.07, 0.52), 0.14, 0.3, boxstyle="round,pad=0.01", fc="#eef3fb", ec=C1, lw=1.2))
        ax.text(x, 0.67, t, ha="center", va="center", fontsize=8)
    for a, b in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(b[1] - 0.075, 0.67), xytext=(a[1] + 0.075, 0.67), arrowprops=dict(arrowstyle="->", color=GR))
    ax.add_patch(FancyBboxPatch((0.45, 0.06), 0.3, 0.22, boxstyle="round,pad=0.01", fc="#fdf1e1", ec=C4, lw=1.2))
    ax.text(0.60, 0.17, "MyUncle core (unmodified)\nlocal_disorder, Bridge.S/H (d = 2),\nMeanField, MC", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(0.60, 0.50), xytext=(0.60, 0.30), arrowprops=dict(arrowstyle="<->", color=C4))
    ax.add_patch(FancyBboxPatch((0.80, 0.06), 0.18, 0.22, boxstyle="round,pad=0.01", fc="#e8f5f3", ec=C3, lw=1.2))
    ax.text(0.89, 0.17, "findings, heat maps,\nmasks, explanations,\nscenarios, images_used.csv", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(0.89, 0.30), xytext=(0.89, 0.50), arrowprops=dict(arrowstyle="->", color=GR))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(os.path.join(FIG, "fig_pipeline.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_bridge():
    core = _core.load()
    k = np.concatenate([np.linspace(1e-3, 2, 100), np.geomspace(2, 400, 200)])
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))
    ax[0].plot(core.Bridge.A(k), core.Bridge.H(k) / 2.0, color=C2, lw=3, label="MyUncle Bridge.H / 2 (d = 2)")
    for d, c, ls in ((2, "k", ":"), (16, C1, "-"), (64, C3, "-"), (384, C4, "-")):
        ax[0].plot(lat.A_d(k, d), lat.order_information(k, d) / d, color=c, ls=ls, lw=1.4, label="vMF, d = %d" % d)
    ax[0].set_xlabel("mean resultant length R (order)")
    ax[0].set_ylabel("H_chi / d  (nats per dimension)")
    ax[0].set_ylim(-0.1, 3.5)
    ax[0].set_title("(a) order information, d dimensions", loc="left", fontsize=9)
    ax[0].legend(frameon=False, fontsize=7)
    err = np.abs(lat.vmf_entropy(k, 2) - core.Bridge.S(k))
    ax[1].semilogy(k, np.maximum(err, 1e-18), ".", ms=3, color=C1)
    ax[1].axhline(1e-9, color=C2, ls="--", lw=1)
    ax[1].text(150, 2e-9, "test tolerance 1e-9", color=C2, fontsize=7)
    ax[1].set_ylim(1e-18, 1e-6)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("concentration kappa")
    ax[1].set_ylabel("|S_vmf(d=2) - MyUncle Bridge.S|")
    ax[1].set_title("(b) agreement with the vendored core", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_bridge.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_validation():
    res = {}
    for enc in ("handcrafted", "dinov2_vits14"):
        p = os.path.join(RES, "validation_%s.json" % enc)
        if os.path.exists(p):
            res[enc] = json.load(open(p))
    if not res:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    axes = ["novelty", "p_emb", "p_het", "p_adj", "D"]
    names = ["novelty\n(OR)", "appearance\nrarity", "state-wise\ndisorder", "contact\nrarity", "D alone\n(baseline)"]
    w = 0.38
    for i, (typ, title) in enumerate((("disorganized", "(a) disorganized patchwork"), ("novel_state", "(b) coherent novel state"))):
        for j, (enc, r) in enumerate(res.items()):
            v = [r["auroc_vs_normal"][typ][a] for a in axes]
            ax[i].bar(np.arange(len(axes)) + (j - 0.5) * w, v, w, label=enc, color=[C1, C3][j])
        ax[i].axhline(0.5, color="k", lw=0.8, ls="--")
        ax[i].set_xticks(range(len(axes)))
        ax[i].set_xticklabels(names, fontsize=7)
        ax[i].set_title(title, loc="left", fontsize=9)
        ax[i].set_ylim(0.4, 1.0)
    ax[0].set_ylabel("AUROC vs normal patches")
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_validation_auroc.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    for enc in res:
        src = os.path.join(RES, "validation_example_%s.png" % enc)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, os.path.join(FIG, "fig_validation_example_%s.png" % enc))


if __name__ == "__main__":
    fig_pipeline()
    fig_bridge()
    fig_validation()
    print("figures written to", FIG)
