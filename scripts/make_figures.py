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


BENCH = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "bench")
FROZEN = "MyUncleVision frozen: appearance, multi-scale, lattice-smoothed"
MAIN = [(FROZEN, "MyUncleVision (frozen)"), ("appearance, multi-scale", "appearance, multi-scale"), ("three-axis OR, 1% scale (original default)", "three-axis OR (v0.1)"),
        ("Mahalanobis distance", "Mahalanobis"), ("Mahalanobis distance, lattice-smoothed (rank)", "Mahalanobis + smoothing"), ("Isolation Forest", "Isolation Forest"), ("Local Outlier Factor", "LOF"), ("kNN distance, k = 10", "kNN k = 10")]


def fig_heldout():
    res = {}
    for enc in ("handcrafted", "dinov2_vits14"):
        p = os.path.join(RES, "bench_held_%s.json" % enc)
        if os.path.exists(p):
            res[enc] = json.load(open(p))
    if not res:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    w = 0.38
    for i, (key, title) in enumerate((("auroc", "(a) AUROC"), ("average_precision", "(b) average precision"))):
        for j, (enc, r) in enumerate(res.items()):
            ax[i].bar(np.arange(len(MAIN)) + (j - 0.5) * w, [r["methods"][m][key] for m, _ in MAIN], w, label=enc, color=[C1, C3][j])
        ax[i].set_xticks(range(len(MAIN)))
        ax[i].set_xticklabels([l for _, l in MAIN], rotation=55, ha="right", fontsize=6.5)
        ax[i].set_title(title + " on the held-out cohort", loc="left", fontsize=9)
        if key == "auroc":
            ax[i].axhline(0.5, color="k", lw=0.8, ls="--")
            ax[i].set_ylim(0.3, 1.0)
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_heldout.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_example(enc_name="dinov2", slide_index=3):
    """one held-out slide: planted truth against the unsmoothed appearance score, the frozen score and the smoothed Mahalanobis baseline"""
    import cv2
    from myuncle_vision.bench import load_cohort, all_scores, HELD_ANOM
    from myuncle_vision.encoder import get_encoder
    enc = get_encoder(enc_name, "cpu")
    slides, truth = load_cohort("held", BENCH, enc, log=lambda *a: None)
    M = all_scores(slides)
    s = slides[slide_index]
    from myuncle_vision import wsi
    sl = wsi.Slide(os.path.join(BENCH, "held", "slides", "held_%d.tif" % (301 + slide_index)))
    th = s.thumb
    z = np.load(os.path.join(BENCH, "held", "truth", "held_%d.npz" % (301 + slide_index)))
    ext = [0, s.shape[1] * s.side0 / s.scale, s.shape[0] * s.side0 / s.scale, 0]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = [("planted truth", None), ("appearance, multi-scale (no smoothing)", "appearance, multi-scale"), ("MyUncleVision frozen (smoothed)", FROZEN), ("Mahalanobis + smoothing", "Mahalanobis distance, lattice-smoothed (rank)")]
    for a, (t, key) in zip(ax, panels):
        a.imshow(th)
        if key:
            m = np.full(s.shape, np.nan)
            m[s.rows, s.cols] = M[key][slide_index]
            a.imshow(np.ma.masked_invalid(m), extent=ext, cmap="magma", alpha=0.8)
        for n in HELD_ANOM + ("blur", "pen", "fold"):
            if n in z.files and z[n].max() > 0:
                mk = cv2.resize(z[n].astype(np.uint8), (th.shape[1], th.shape[0]), interpolation=cv2.INTER_NEAREST)
                cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cs:
                    a.plot(c[:, 0, 0], c[:, 0, 1], color="lime" if n in HELD_ANOM else "red", lw=1.3)
        a.set_title(t, fontsize=9)
        a.axis("off")
    fig.suptitle("held-out synthetic slide held_%d (%s encoder): green = planted anomalies, red = planted artifacts" % (301 + slide_index, enc.name), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_heldout_example.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_bridge()
    fig_heldout()
    fig_example()
    print("figures written to", FIG)
