# -*- coding: utf-8 -*-
"""Spatial-arrangement anomalies: do MyUncle's local order quantities and a local-manifold statistic detect what rarity cannot?

FIRST ATTEMPT, kept for the record (results/og_geometry_first_attempt.json): the fixed blends studied here did not beat the similarity-weighted rarity on the test scenarios.
The study that followed, with the directional disorder and a label-free choice of the channel, is tc_study.py (docs/TWO_CHANNEL.md).

Scenarios: the ten of og_study (feature-outlier anomalies and the MyUncle / image scenarios) plus three REARRANGEMENT scenarios of the independent generator (regions whose embeddings are permuted among their
sites: the marginal distribution is unchanged, only the spatial arrangement is destroyed; blocks of 6 and 8 and irregular clusters of 24 sites, 7% of the sites, a spatially correlated background).
Detectors (all label-free): rarity r (none), similarity-weighted rarity (simw), the rank of the local disorder D (equivalently of the von Mises-Fisher information H_chi), the rank of the local manifold
inconsistency e_i (window 5, m = 2, 4, 8), and blends (1 - lambda) simw + lambda rank(D) or rank(e).  Selection on the development scenarios (F = 0.5 mean AP + 0.5 min AP over the 13 scenarios), one
evaluation on the independent test scenarios with paired bootstrap intervals.

    python -m myuncle_vision.og_geometry cache dev|test     # build the rearrangement scenarios
    python -m myuncle_vision.og_geometry study              # selection on dev, test evaluation -> results/og_geometry.json
"""
import json
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import average_precision_score

from . import cohort as C
from . import gaar_opt as GO
from . import gating as G
from . import local_geometry as LG
from . import morphology as M
from . import og_study as O

DEV_SEEDS, TEST_SEEDS = (11, 12, 13, 14, 15), (111, 112, 113, 114, 115)


def scenario_arr(name, shape, size, seeds):
    cohorts = []
    for sd in seeds:
        slides, masks = [], []
        for i in range(O.IMAGES_PER_COHORT):
            X, m = M.make_lattice_smooth(sd * 1000 + i, shape=shape, block=size)
            slides.append(M._slide("a%d_%d" % (sd, i), X))
            masks.append(m.reshape(-1).astype(int))
        for s in slides:
            s.extra = {}
        C.analyse(slides, GO.to_config(dict(GO.DEFAULT, smooth="none"), n_states=6, ref_per_slide=300, ref_max=3000))
        cohorts.append(O._lattice_items(slides, masks, [np.ones(len(mk), bool) for mk in masks]))
    return dict(name=name, kind="arr", group=1, cohorts=cohorts)


def build_cache(split):
    seeds = DEV_SEEDS if split == "dev" else TEST_SEEDS
    out = [scenario_arr("arr-%s-%s" % (split, nm.split("-", 1)[1]), sh, sz, seeds) for nm, sh, sz in M.ARR_SHAPES]
    pickle.dump(out, open(os.path.join(O.CACHE_DIR, "og_geo_cache_%s.pkl" % split), "wb"))
    return out


def load_all(split):
    return O.load_cache(split) + pickle.load(open(os.path.join(O.CACHE_DIR, "og_geo_cache_%s.pkl" % split), "rb"))


MS = (2, 4, 8)


def precompute(scn):
    """per cohort and image: rank fields of the local disorder and of the manifold inconsistency (m = 2, 4, 8) over the cohort"""
    for ci, items in enumerate(scn["cohorts"]):
        D = [LG.disorder_map(it["U"], it["valid"]) for it in items]
        allD = np.concatenate([x[np.isfinite(x)] for x in D])
        srt = np.sort(allD)
        scn.setdefault("pre", {})[ci] = dict(D=[np.where(np.isfinite(x), np.searchsorted(srt, x, side="right") / len(srt), np.nan) for x in D])
        for m in MS:
            E = [LG.manifold_inconsistency(it["U"], it["valid"], 5, m) for it in items]
            allE = np.concatenate([x[np.isfinite(x)] for x in E])
            srtE = np.sort(allE)
            scn["pre"][ci]["E%d" % m] = [np.where(np.isfinite(x), np.searchsorted(srtE, x, side="right") / len(srtE), np.nan) for x in E]
    return scn


def detector_scores(scn, cfg):
    out = []
    for ci, items in enumerate(scn["cohorts"]):
        tau = G.estimate_tau([it["U"] for it in items], [it["valid"] for it in items])
        kind = cfg["kind"]
        if kind == "none":
            sc = G.cohort_apply(items, "none", 1, 0, 0, 0, tau)
        elif kind == "simw":
            sc = G.cohort_apply(items, "simw", 3, 0, 0, 0, tau)
        elif kind == "disorder":
            sc = scn["pre"][ci]["D"]
        elif kind == "manifold":
            sc = scn["pre"][ci]["E%d" % cfg["m"]]
        elif kind == "blend":
            base = G.cohort_apply(items, "simw", 3, 0, 0, 0, tau)
            other = scn["pre"][ci]["D"] if cfg["with"] == "disorder" else scn["pre"][ci]["E%d" % cfg["m"]]
            lam = cfg["lam"]
            sc = [(1 - lam) * b + lam * o for b, o in zip(base, other)]
        elif kind == "max":
            base = G.cohort_apply(items, "simw", 3, 0, 0, 0, tau)
            other = scn["pre"][ci]["D"]
            sc = [np.fmax(b, o) for b, o in zip(base, other)]
        out.append(sc)
    return out


def configs():
    c = {"none (rarity)": dict(kind="none"), "similarity-weighted rarity": dict(kind="simw"), "local disorder D (rank; same as H_chi)": dict(kind="disorder")}
    for m in MS:
        c["local manifold inconsistency (m=%d)" % m] = dict(kind="manifold", m=m)
    for lam in (0.25, 0.5, 0.75):
        c["simw + disorder (lambda=%.2f)" % lam] = dict(kind="blend", with_="disorder", lam=lam)
        c["simw + manifold m=4 (lambda=%.2f)" % lam] = dict(kind="blend", m=4, lam=lam)
    c["max(simw, disorder)"] = dict(kind="max")
    for v in c.values():
        if "with_" in v:
            v["with"] = v.pop("with_")
        if v["kind"] == "blend" and "with" not in v:
            v["with"] = "manifold"
    return c


def fitness(scns, cfg):
    aps = []
    for sc in scns:
        ys, ss = O._flatten(sc, detector_scores(sc, cfg))
        aps.append(O.ap_of(ys, ss))
    a = np.asarray(aps)
    return aps, float(0.5 * np.nanmean(a) + 0.5 * np.nanmin(a))


def study(out, B=300):
    dev = [precompute(s) for s in load_all("dev")]
    test = [precompute(s) for s in load_all("test")]
    cfgs = configs()
    devtab = {}
    for nm, cfg in cfgs.items():
        aps, f = fitness(dev, cfg)
        devtab[nm] = dict(aps=aps, fitness=f)
        print("dev", nm, round(f, 3), flush=True)
    res = dict(dev_scenarios=[s["name"] for s in dev], test_scenarios=[s["name"] for s in test], dev=devtab, test={}, bootstrap={})
    per = {}
    for nm, cfg in cfgs.items():
        res["test"][nm] = {}
        per[nm] = []
        for sc in test:
            ys, ss = O._flatten(sc, detector_scores(sc, cfg))
            per[nm].append((ys, ss))
            res["test"][nm][sc["name"]] = O.ap_of(ys, ss)
        a = np.array([res["test"][nm][sc["name"]] for sc in test])
        res["test"][nm]["_fitness"] = float(0.5 * np.nanmean(a) + 0.5 * np.nanmin(a))
        print("test", nm, round(res["test"][nm]["_fitness"], 3), flush=True)
    rng = np.random.default_rng(0)
    n_u = [len(per["none (rarity)"][k][0]) // test[k]["group"] for k in range(len(test))]
    F = {nm: [] for nm in cfgs}
    for _ in range(B):
        draws = [rng.integers(0, n_u[k], n_u[k]) for k in range(len(test))]
        for nm in cfgs:
            aps = []
            for k, sc in enumerate(test):
                gp = sc["group"]
                idx = [u * gp + j for u in draws[k] for j in range(gp)]
                aps.append(O.ap_of(per[nm][k][0], per[nm][k][1], idx))
            a = np.asarray(aps)
            F[nm].append(0.5 * np.nanmean(a) + 0.5 * np.nanmin(a))
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    for ref in ("none (rarity)", "similarity-weighted rarity"):
        for nm in cfgs:
            d = np.array(F[nm]) - np.array(F[ref])
            res["bootstrap"].setdefault(nm, {})["fitness_ci"] = ci(F[nm])
            res["bootstrap"][nm]["diff_vs_" + ref.split()[0]] = dict(ci=ci(d), share_better=float(np.mean(d > 0)))
    best_dev = max(devtab, key=lambda k: devtab[k]["fitness"])
    res["best_dev_config"] = best_dev
    json.dump(res, open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    if sys.argv[1] == "cache":
        build_cache(sys.argv[2])
    elif sys.argv[1] == "study":
        study("results/og_geometry.json")
