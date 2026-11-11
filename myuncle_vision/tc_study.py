# -*- coding: utf-8 -*-
"""Two-channel study: rarity (what is there) and directional local order (how it is arranged), and their label-free fusion.

Scenarios (development and independent test versions of each):
    the ten of og_study            feature-outlier anomalies of six shapes in the controlled generator, two image scenarios, two MyUncle simulation scenarios
    three arrangement scenarios    regions whose embeddings are permuted among their sites (blocks of 6 and 8, irregular clusters of 24; 7% of the sites; spatially correlated background)
    two mixed scenarios            rearranged 6 x 6 blocks (3.5%) together with feature outliers (3 x 3 blocks or scattered sites, 1.5%) in the same lattices
Detectors: rarity, similarity-weighted rarity (channel A); the directional disorder (channel B), MyUncle's isotropic local disorder D, the center-to-neighbor distance and the local manifold
inconsistency as alternatives for channel B; the fusion rules of two_channel.py.  The fusion rule is selected on the development scenarios (F = 0.5 mean AP + 0.5 min AP) and evaluated once on the test
scenarios with paired bootstrap intervals over images (simulation runs for the MyUncle scenarios).

    python -m myuncle_vision.tc_study calibrate          # detectability of the feature outliers of the mixed scenarios (development seeds, unsmoothed rarity only)
    python -m myuncle_vision.tc_study cache dev|test
    python -m myuncle_vision.tc_study dev                # development table -> results/two_channel_dev.json
    python -m myuncle_vision.tc_study test               # selection on dev, one test evaluation -> results/two_channel.json
"""
import json
import os
import pickle
import sys

import numpy as np

from . import cohort as C
from . import gaar_opt as GO
from . import gating as G
from . import local_geometry as LG
from . import morphology as M
from . import og_study as O
from . import two_channel as T

ARR_SEEDS = {"dev": (11, 12, 13, 14, 15), "test": (111, 112, 113, 114, 115)}
MIX_SEEDS = {"dev": (21, 22, 23, 24, 25), "test": (121, 122, 123, 124, 125)}
MIX = [("mixed-block3", "block", 3), ("mixed-scattered", "block", 1)]
MIX_SHIFT = 2.0          # calibrated on the unsmoothed rarity of one development cohort: AP of the outliers against normal sites about 0.55 (tc_study.py calibrate)
FUSIONS = [("select", 0.0), ("select0", 0.0), ("tailw", 1.0), ("tailw", 2.0), ("tailw", 4.0), ("zmax", 0.0), ("rankmax", 0.0)]


def _items(slides, ys, sels, kinds=None):
    items = []
    for i, (s, y, sel) in enumerate(zip(slides, ys, sels)):
        H, W = s.shape

        def grid(v, fill=np.nan, dtype=float):
            g = np.full((H, W) + np.shape(v)[1:], fill, dtype=dtype)
            g[s.rows, s.cols] = v
            return g
        valid = np.zeros((H, W), bool)
        valid[s.rows, s.cols] = s.ok
        it = dict(r=grid(s.extra["p_emb_ms"]), U=grid(s.extra["U"], 0.0), valid=valid, y=grid(y, 0, int), sel=grid(sel & s.ok, False, bool), d=grid(s.extra["d_emb"]),
                  dms=grid(s.extra["d_emb_by_scale"]))
        if kinds is not None:
            it["kind"] = grid(kinds[i], 0, int)
        items.append(it)
    return items


def _from_gaar(scn, group):
    O._analyse_none(scn)
    ys, sels = [], []
    for i, s in enumerate(scn.slides):
        y, sel = scn.truth_fn(i, s)
        ys.append(np.zeros(len(s.rows), int) + y)
        sels.append(np.zeros(len(s.rows), bool) | sel)
    return dict(name=scn.name, kind="other", group=group, cohorts=[_items(scn.slides, ys, sels)])


def _generated(name, kind, seeds, make):
    cohorts = []
    for sd in seeds:
        slides, masks, kinds = [], [], []
        for i in range(O.IMAGES_PER_COHORT):
            X, m = make(sd * 1000 + i)
            slides.append(M._slide("%s%d_%d" % (kind[0], sd, i), X))
            kinds.append(m.reshape(-1).astype(int))
            masks.append((m.reshape(-1) > 0).astype(int))
        for s in slides:
            s.extra = {}
        C.analyse(slides, GO.to_config(dict(GO.DEFAULT, smooth="none"), n_states=6, ref_per_slide=300, ref_max=3000))
        cohorts.append(_items(slides, masks, [np.ones(len(mk), bool) for mk in masks], kinds))
    return dict(name=name, kind=kind, group=1, cohorts=cohorts)


def build_cache(split):
    out = []
    sim = GO.BB.DEV if split == "dev" else GO.TEST_SIM
    out += [_from_gaar(GO.image_scenario(split, "handcrafted"), 1), _from_gaar(GO.image_scenario(split, "dinov2_vits14"), 1),
            _from_gaar(GO.sim_scenario(sim, "rotor", "sim-%s-rotor" % split), 3), _from_gaar(GO.sim_scenario(sim, "rotor_D", "sim-%s-rotor_D" % split), 3)]
    for nm, sh, sz in O.SHAPES:
        out.append(_generated("morph-%s-%s" % (split, nm), "morph", O.DEV_SEEDS if split == "dev" else O.TEST_SEEDS, lambda s, sh=sh, sz=sz: M.make_lattice(s, shape=sh, block=sz, **M.GEN)))
    for nm, sh, sz in M.ARR_SHAPES:
        out.append(_generated("arr-%s-%s" % (split, nm.split("-", 1)[1]), "arr", ARR_SEEDS[split], lambda s, sh=sh, sz=sz: M.make_lattice_smooth(s, shape=sh, block=sz)))
    for nm, sh, sz in MIX:
        out.append(_generated("%s-%s" % (nm.replace("mixed-", "mixed-%s-" % split), ""), "mixed", MIX_SEEDS[split], lambda s, sh=sh, sz=sz: M.make_lattice_mixed(s, out_shape=sh, out_block=sz, shift=MIX_SHIFT)))
        out[-1]["name"] = out[-1]["name"].rstrip("-")
        print("cached", out[-1]["name"], flush=True)
    pickle.dump(out, open(os.path.join(O.CACHE_DIR, "tc_cache_%s.pkl" % split), "wb"))
    return out


def load_cache(split):
    return pickle.load(open(os.path.join(O.CACHE_DIR, "tc_cache_%s.pkl" % split), "rb"))


def calibrate(shifts=(0.75, 1.0, 1.5, 2.0, 3.0)):
    """detectability of the feature outliers of the mixed scenarios by the UNSMOOTHED rarity (development seed 21, feature outliers against normal sites)"""
    for sh in shifts:
        for nm, shape, sz in MIX:
            scn = _generated("cal", "mixed", (21,), lambda s: M.make_lattice_mixed(s, out_shape=shape, out_block=sz, shift=sh))
            y = np.concatenate([it["kind"].ravel() for it in scn["cohorts"][0]])
            r = np.concatenate([it["r"].ravel() for it in scn["cohorts"][0]])
            m = y != 2
            print("shift %.1f %s  AP(rarity, outliers vs normal) %.3f  prevalence %.3f" % (sh, nm, O.average_precision_score(y[m] == 1, r[m]), np.mean(y[m] == 1)), flush=True)


# ------------------------------------------------------------------------------------------------ channels and detectors
def channels(scn, manifold=False):
    """per cohort: dict of lists of score lattices"""
    out = []
    for items in scn["cohorts"]:
        tau = G.estimate_tau([it["U"] for it in items], [it["valid"] for it in items])
        Us, vs = [it["U"] for it in items], [it["valid"] for it in items]
        dd = [T.directional_disorder(U, v, 1) for U, v in zip(Us, vs)]
        c = dict(r=[it["r"] for it in items], simw=G.cohort_apply(items, "simw", 3, 0, 0, 0, tau), mean=G.cohort_apply(items, "mean", 3, 0, 0, 0, tau), d=[it["d"] for it in items],
                 dir=[x[0] for x in dd], dirmean=[x[1] for x in dd], dir2=[T.directional_disorder(U, v, 2)[0] for U, v in zip(Us, vs)], D=[LG.disorder_map(U, v) for U, v in zip(Us, vs)],
                 cnb=[T.center_neighbor_distance(U, v) for U, v in zip(Us, vs)])
        if manifold:
            c["manifold2"] = [LG.manifold_inconsistency(U, v, 5, 2) for U, v in zip(Us, vs)]
        out.append(c)
    return out


def _fill(fields, ref):
    """undefined sites of a channel-B score get the lowest score (the site is still evaluated)"""
    return [np.where(np.isfinite(r), np.nan_to_num(x, nan=-1e9), np.nan) for x, r in zip(fields, ref)]


def detectors(with_manifold=False):
    d = {"rarity": lambda c: c["r"], "rarity, fixed 3x3 mean": lambda c: c["mean"], "rarity, similarity-weighted (channel A)": lambda c: c["simw"],
         "directional disorder (channel B)": lambda c: _fill(c["dir"], c["r"]), "directional disorder, 3x3 quadrants": lambda c: _fill(c["dir2"], c["r"]),
         "mean quadrant disorder": lambda c: _fill(c["dirmean"], c["r"]), "isotropic local disorder D": lambda c: _fill(c["D"], c["r"]), "center-to-neighbor distance": lambda c: _fill(c["cnb"], c["r"])}
    if with_manifold:
        d["local manifold inconsistency (m=2)"] = lambda c: _fill(c["manifold2"], c["r"])
    for b, lab in (("dir", "directional disorder"), ("cnb", "center-to-neighbor distance"), ("D", "isotropic disorder D")):
        for rule, p in FUSIONS:
            nm = "fusion %s%s: A + %s" % (rule, " p=%g" % p if rule == "tailw" else "", lab)
            d[nm] = lambda c, b=b, rule=rule, p=p: T.fuse(c["simw"], c["d"], c[b], rule, p)[0]
    return d


def _flat(scn, scores, which=None):
    """per image: labels and scores.  which = 1 or 2: positives of that kind against the normal sites only (mixed scenarios)"""
    ys, ss = [], []
    for items, sc in zip(scn["cohorts"], scores):
        for it, s in zip(items, sc):
            m = it["sel"] & np.isfinite(s)
            y = it["y"]
            if which is not None:
                m = m & ((it["kind"] == which) | (it["kind"] == 0))
                y = (it["kind"] == which).astype(int)
            ys.append(y[m])
            ss.append(s[m])
    return ys, ss


def F_of(aps):
    a = np.asarray(aps, float)
    return float(0.5 * np.nanmean(a) + 0.5 * np.nanmin(a))


def table(scns, with_manifold=False):
    CH = [channels(s, with_manifold) for s in scns]
    dets = detectors(with_manifold)
    per = {nm: [_flat(s, [fn(c) for c in ch]) for s, ch in zip(scns, CH)] for nm, fn in dets.items()}
    aps = {nm: [O.ap_of(*x) for x in per[nm]] for nm in dets}
    typed = {}
    for k, s in enumerate(scns):
        if s["kind"] == "mixed":
            for nm, fn in dets.items():
                sc = [fn(c) for c in CH[k]]
                typed.setdefault(s["name"], {})[nm] = dict(outliers=O.ap_of(*_flat(s, sc, 1)), rearranged=O.ap_of(*_flat(s, sc, 2)))
    tails = [[dict(A=T.tail_index(c["d"]), B=T.tail_index(c["dir"]), B_cnb=T.tail_index(c["cnb"]), B_D=T.tail_index(c["D"]), channel=T.fuse(c["simw"], c["d"], c["dir"], "select")[1]["channel"]) for c in ch]
             for ch in CH]
    return per, aps, typed, tails


def run_dev(out):
    scns = load_cache("dev")
    per, aps, typed, tails = table(scns)
    res = dict(scenarios=[s["name"] for s in scns], ap=aps, fitness={k: F_of(v) for k, v in aps.items()}, typed=typed, tails=tails)
    json.dump(res, open(out, "w"), indent=1)
    _print(res)
    return res


def _print(res):
    names = [n.replace("morph-", "").replace("dev-", "").replace("test-", "")[:9] for n in res["scenarios"]]
    print(" " * 52 + " ".join(n.rjust(9) for n in names) + "       F    mean")
    for nm, a in res["ap"].items():
        print(nm[:52].ljust(52) + " ".join(("%.3f" % x).rjust(9) for x in a) + "   %.3f   %.3f" % (F_of(a), float(np.nanmean(a))), flush=True)


def run_test(dev_json, out, B=300, seed=0):
    dev = json.load(open(dev_json))
    fus = [k for k in dev["fitness"] if k.startswith("fusion") and k.endswith("A + directional disorder")]
    best = max(fus, key=lambda k: round(dev["fitness"][k], 3))          # ties go to the first listed rule (the guarded selection, declared primary before the test)
    scns = load_cache("test")
    per, aps, typed, tails = table(scns, with_manifold=True)
    res = dict(scenarios=[s["name"] for s in scns], selected_on_dev=best, dev_fitness=dev["fitness"], ap=aps, fitness={k: F_of(v) for k, v in aps.items()}, typed=typed, tails=tails)
    rng = np.random.default_rng(seed)
    A, R0 = "rarity, similarity-weighted (channel A)", "rarity"
    n_u = [len(per[A][k][0]) // scns[k]["group"] for k in range(len(scns))]
    names_b = [nm for nm in aps if not nm.startswith("fusion") or nm.endswith("A + directional disorder") or nm.startswith("fusion select:")]     # single detectors, fusions with channel B, selections
    boot = {nm: [] for nm in names_b}
    arr_idx = [k for k, s in enumerate(scns) if s["kind"] == "arr"]
    old_idx = [k for k, s in enumerate(scns) if s["kind"] in ("other", "morph")]
    part = {nm: dict(arr=[], old=[]) for nm in names_b}
    for _ in range(B):
        draws = [rng.integers(0, n_u[k], n_u[k]) for k in range(len(scns))]
        for nm in names_b:
            a = []
            for k, sc in enumerate(scns):
                gp = sc["group"]
                idx = [u * gp + j for u in draws[k] for j in range(gp)]
                a.append(O.ap_of(per[nm][k][0], per[nm][k][1], idx))
            boot[nm].append(F_of(a))
            part[nm]["arr"].append(float(np.nanmean([a[k] for k in arr_idx])))
            part[nm]["old"].append(float(np.nanmean([a[k] for k in old_idx])))
    ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
    res["bootstrap"] = {}
    for nm in names_b:
        b = dict(fitness_ci=ci(boot[nm]))
        for ref, key in ((A, "A"), (R0, "rarity"), ("fusion select: A + center-to-neighbor distance", "select_cnb"), ("fusion select: A + isotropic disorder D", "select_D")):
            dlt = np.array(boot[nm]) - np.array(boot[ref])
            b["diff_vs_" + key] = dict(ci=ci(dlt), share_better=float(np.mean(dlt > 0)))
            for grp in ("arr", "old"):
                dg = np.array(part[nm][grp]) - np.array(part[ref][grp])
                b["diff_%s_vs_%s" % (grp, key)] = dict(ci=ci(dg), share_better=float(np.mean(dg > 0)))
        b["mean_arr_ci"], b["mean_old_ci"] = ci(part[nm]["arr"]), ci(part[nm]["old"])
        res["bootstrap"][nm] = b
    res["mean_arr"] = {nm: float(np.nanmean([aps[nm][k] for k in arr_idx])) for nm in aps}
    res["mean_old"] = {nm: float(np.nanmean([aps[nm][k] for k in old_idx])) for nm in aps}
    json.dump(res, open(out, "w"), indent=1)
    _print(res)
    print("selected on dev:", best)
    return res


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cmd = sys.argv[1]
    if cmd == "calibrate":
        calibrate()
    elif cmd == "cache":
        build_cache(sys.argv[2])
    elif cmd == "dev":
        run_dev("results/two_channel_dev.json")
    elif cmd == "test":
        run_test("results/two_channel_dev.json", "results/two_channel.json")
