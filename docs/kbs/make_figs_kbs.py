# -*- coding: utf-8 -*-
"""Figures of the Knowledge-Based Systems manuscript, from results/kbs_*.json.   python docs/kbs/make_figs_kbs.py"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)
J = lambda n: json.load(open(os.path.join(RES, n)))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})
A, B, MX = "channel A (similarity-weighted rarity)", "channel B (directional disorder)", "fusion: max"


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, name + ".tif"), bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("saved", name)


def fig_phase():
    P = J("kbs_phase.json")
    S, D = P["s_levels"], P["delta_levels"]
    def grid(det, key):
        return np.array([[P["grid"]["s=%g,delta=%g" % (s, d)][det][key] for d in D] for s in S])
    panels = [(A, "feature", "A: feature AP"), (B, "arrangement", "B: arrangement AP"), (MX, "worst", "max(A, B): worst-type AP")]
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.1), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.2], wspace=0.35))
    for ax, (det, key, ttl) in zip(axs[:3], panels):
        g = grid(det, key)
        im = ax.imshow(g, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for i in range(len(S)):
            for j in range(len(D)):
                ax.text(j, i, "%.2f" % g[i, j], ha="center", va="center", fontsize=6.5, color="w" if g[i, j] < 0.6 else "k")
        ax.set_xticks(range(len(D)), ["%g" % d for d in D])
        ax.set_yticks(range(len(S)), ["%g" % s for s in S])
        ax.set_xlabel("arrangement disruption $\\delta$")
        if ax is axs[0]:
            ax.set_ylabel("feature strength $s$")
        ax.set_title(ttl, loc="left")
    pos = axs[2].get_position()
    cax = fig.add_axes([pos.x1 + 0.006, pos.y0, 0.007, pos.height])
    fig.colorbar(im, cax=cax)
    ax = axs[3]
    pos3 = ax.get_position()
    ax.set_position([pos3.x0 + 0.03, pos3.y0, pos3.width - 0.03, pos3.height])
    cs = sorted(P["coherence"], key=lambda k: float(k.split("=")[1]))
    x = [float(k.split("=")[1]) for k in cs]
    for det, key, lab, col, ls in ((A, "feature", "A, feature anomalies", "#1b4f8a", "-"), (B, "arrangement", "B, arrangement anomalies", "#d1495b", "-"),
                                   ("center-to-neighbor distance", "arrangement", "neighbor distance, arrangement", "#d1495b", ":"), (MX, "worst", "max fusion, worst type", "#2e8b57", "-")):
        ax.plot(x, [P["coherence"][k][det][key] for k in cs], marker="o", ms=3, color=col, ls=ls, label=lab)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(x, ["%g" % v for v in x])
    ax.set_xlabel("background correlation length (sites)")
    ax.set_ylabel("average precision")
    ax.set_ylim(0, 1)
    ax.set_title("coherence ($s$=2, $\\delta$=1)", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=6.8)
    for k, ax in enumerate(axs):
        ax.text(-0.12 if k else -0.2, 1.05, "abcd"[k], transform=ax.transAxes, fontweight="bold", fontsize=10)
    save(fig, "fig_phase")


def fig_maps():
    """one stage-2 confirmatory lattice (G3, shuffled regions): truth, isolation forest, Shekhar statistic, frozen stage-2 fusion; and one rotated-patch lattice"""
    from myuncle_vision import kbs_stage2 as S
    from myuncle_vision import kbs_study as K
    fz = J("kbs2_frozen.json")
    ch = fz["chosen"]
    rows = []
    for c, (nm, (g, kw)) in enumerate(list(fz["conditions"].items())[:2]):
        Us, kinds = [], []
        for i in range(K.N_PER_COHORT):
            U, kd = S.GEN[g](fz["seed0"] + 1000 * c + i, **kw)
            Us.append(U)
            kinds.append(kd)
        sc = S.scores2(Us, rng_seed=fz["seed0"] + 1000 * c)
        rows.append((nm, kinds[0], [K._crank(sc["isolation forest (embeddings)"])[0], K._crank(sc["spatial outlier statistic (Shekhar)"])[0], sc[ch][0]]))
    fig, axs = plt.subplots(2, 4, figsize=(9.5, 5.0))
    cm = matplotlib.colors.ListedColormap(["#f2f2f2", "#1b4f8a", "#d1495b"])
    for r, (nm, kd, maps) in enumerate(rows):
        axs[r, 0].imshow(kd, cmap=cm, vmin=0, vmax=2, interpolation="nearest")
        axs[r, 0].set_ylabel(nm.replace("G3 oriented texture, ", "G3, "), fontsize=8)
        for k, (f, ttl) in enumerate(zip(maps, ("isolation forest (rank)", "Shekhar statistic (rank)", "max fusion (frozen, stage 2)"))):
            ax = axs[r, k + 1]
            ax.imshow(f, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            ax.contour(kd > 0, levels=[0.5], colors="c", linewidths=0.6)
            if r == 0:
                ax.set_title(ttl, fontsize=8)
        if r == 0:
            axs[r, 0].set_title("truth (blue feature, red arrangement)", fontsize=8)
    for k, ax in enumerate(axs.ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.02, 0.98, "abcdefgh"[k], transform=ax.transAxes, va="top", fontweight="bold", fontsize=9, color="k",
                bbox=dict(boxstyle="square,pad=0.1", fc="w", ec="none", alpha=0.7))
    fig.tight_layout()
    save(fig, "fig_maps")


def fig_confirm():
    """per-type AP of key detectors in the two confirmatory stages"""
    C1, C2 = J("kbs_confirm.json"), J("kbs2_confirm.json")
    ch1, ch2 = C1["chosen"], C2["chosen"]
    dets = [("rarity", "cohort kNN rarity"), ("LOF (embeddings)", "LOF"), ("isolation forest (embeddings)", "isolation forest"), (B, "channel B (directional disorder)"),
            ("spatial outlier statistic (Shekhar)", "Shekhar statistic"), ("contextual LOF ([u, neighbor mean])", "contextual LOF"), ("OR(LOF, Shekhar)", "OR(LOF, Shekhar)"),
            ("fusion: cohort selection (tail index)", "cohort-level selection"), ("fusion: max (rarity, B)", "max(rarity, B) [frozen, stage 1]"),
            ("fusion: max (isolation forest, Shekhar)", "max(isolation forest, Shekhar) [frozen, stage 2]")]
    panels = [(C1["pooled_summary"], "Stage 1: G2, three conditions pooled (n = 90)")]
    for nm, v in C2["conditions"].items():
        if nm.startswith("G3"):
            panels.append((v["summary"], "Stage 2: " + nm.replace("oriented texture, ", "") + " (n = 30)"))
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    y = np.arange(len(dets))
    for ax, (sm, ttl) in zip(axs, panels):
        get = lambda k, t: sm[k][t]
        fv = [get(k, "feature") for k, _ in dets]
        av = [get(k, "arrangement") for k, _ in dets]
        ax.barh(y - 0.2, fv, 0.4, color="#1b4f8a", label="feature anomalies")
        ax.barh(y + 0.2, av, 0.4, color="#d1495b", label="arrangement anomalies")
        ax.set_title(ttl, loc="left", fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("average precision")
    axs[0].set_yticks(y, [lab for _, lab in dets], fontsize=7.2)
    axs[0].invert_yaxis()
    axs[0].legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_confirm")


if __name__ == "__main__":
    which = sys.argv[1:] or ["phase"]
    for w in which:
        {"phase": fig_phase, "maps": fig_maps, "confirm": fig_confirm}[w]()
