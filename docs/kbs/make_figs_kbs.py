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
    """one confirmatory-generator lattice (G2): truth, channel A, channel B, max fusion"""
    from myuncle_vision import kbs_study as K
    fz = J("kbs_frozen.json")
    Us, kinds = [], []
    s0 = fz["seed0"]
    for i in range(K.N_PER_COHORT):
        U, kd = K.GENERATORS["G2"](s0 + i, **list(fz["conditions"].values())[0])
        Us.append(U)
        kinds.append(kd)
    sc = K.cohort_scores(Us, with_baselines=False)
    i = 0
    rank = lambda f: K.T.cohort_rank(f)[i]
    fig, axs = plt.subplots(1, 4, figsize=(10, 2.8))
    cm = matplotlib.colors.ListedColormap(["#f2f2f2", "#1b4f8a", "#d1495b"])
    axs[0].imshow(kinds[i], cmap=cm, vmin=0, vmax=2, interpolation="nearest")
    axs[0].set_title("a  ground truth (blue: feature, red: arrangement)", loc="left", fontsize=8)
    for ax, (nm, ttl) in zip(axs[1:], ((A, "b  channel A (rank)"), (B, "c  channel B (rank)"), (fz["chosen_fusion"], "d  frozen fusion"))):
        f = sc[nm][i] if nm == fz["chosen_fusion"] else rank(sc[nm])
        ax.imshow(f, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.contour(kinds[i] > 0, levels=[0.5], colors="c", linewidths=0.6)
        ax.set_title(ttl, loc="left", fontsize=8)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, "fig_maps")


def fig_confirm():
    C = J("kbs_confirm.json")
    ch = C["chosen"]
    order = ["rarity", "rarity, fixed mean", A, B, "center-to-neighbor distance", "LOF (embeddings)", "isolation forest (embeddings)", "spatial outlier statistic (Shekhar)",
             "contextual LOF ([u, neighbor mean])", "OR(LOF, Shekhar)", "fusion: cohort selection (tail index)", "fusion: max (A, neighbor distance)", ch]
    seen, order2 = set(), []
    for o in order:
        if o not in seen:
            order2.append(o)
            seen.add(o)
    conds = list(C["conditions"])
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    y = np.arange(len(order2))
    for ax, cnd in zip(axs, conds):
        sm = C["conditions"][cnd]["summary"]
        ax.barh(y - 0.2, [sm[o]["feature"] for o in order2], 0.4, color="#1b4f8a", label="feature anomalies")
        ax.barh(y + 0.2, [sm[o]["arrangement"] for o in order2], 0.4, color="#d1495b", label="arrangement anomalies")
        ax.set_title(cnd.replace("G2 mixed, ", ""), loc="left", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("average precision")
    axs[0].set_yticks(y, [o.replace(" (similarity-weighted rarity)", "").replace(" (directional disorder)", "").replace("fusion: ", "") for o in order2], fontsize=7)
    axs[0].invert_yaxis()
    axs[0].legend(frameon=False, loc="lower right")
    save(fig, "fig_confirm")


if __name__ == "__main__":
    which = sys.argv[1:] or ["phase"]
    for w in which:
        {"phase": fig_phase, "maps": fig_maps, "confirm": fig_confirm}[w]()
