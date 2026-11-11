# -*- coding: utf-8 -*-
"""Order-gated spatial aggregation: development, search over rules with GaaR, and an independent test (see gating.py for the rules).

    python -m myuncle_vision.og_study cache dev      # analyse the development scenarios once (rarity scores, embeddings, truth) -> cache
    python -m myuncle_vision.og_study cache test     # the same for the independent test scenarios (new seeds)
    python -m myuncle_vision.og_study grid           # every rule configuration on the development scenarios -> results/og_grid_dev.json
    python -m myuncle_vision.og_study search         # GaaR strategies and random search on that table -> results/og_search.json
    python -m myuncle_vision.og_study test           # champions on the test scenarios, paired bootstrap intervals -> results/og_test.json

Everything after the rarity score is label-free except the selection of the rule parameters, which uses the development labels (supervised selection, as in gaar_opt).
"""
import itertools
import json
import os
import pickle
import sys

import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score, roc_auc_score

from . import cohort as C
from . import gaar_opt as GO
from . import gating as G
from . import morphology as M

CACHE_DIR = GO.BENCH_ROOT
SHAPES = [("scattered", "block", 1), ("block3", "block", 3), ("block6", "block", 6), ("bar8", "bar", 8), ("irregular12", "irregular", 12), ("boundary3", "boundary", 3)]
DEV_SEEDS, TEST_SEEDS = (1, 2, 3, 4, 5), (101, 102, 103, 104, 105)
IMAGES_PER_COHORT = 10
RULE_GRID = {"rule": list(G.RULES), "window": [3, 5, 7], "theta": [0.2, 0.4, 0.6, 0.8], "alpha": [2.0, 5.0, 10.0], "lam": [0.0, 0.25, 0.5]}
DEFAULT_RULE = {"rule": "mean", "window": 3, "theta": 0.6, "alpha": 5.0, "lam": 0.0}            # the frozen configuration (theta, alpha unused)
Q = 0.03


def canonical(g):
    """configurations that differ only in unused parameters are the same configuration"""
    g = dict(g)
    if g["rule"] == "none":
        return dict(rule="none", window=1, theta=0.0, alpha=0.0, lam=0.0)
    if g["rule"] in ("mean", "simw", "onesided"):
        g["theta"], g["alpha"] = 0.0, 0.0
    return g


# ------------------------------------------------------------------------------------------------ cache of rarity scores
def _lattice_items(slides, ys, sels):
    items = []
    for s, y, sel in zip(slides, ys, sels):
        H, W = s.shape
        r = np.full((H, W), np.nan)
        r[s.rows, s.cols] = s.extra["p_emb_ms"]
        U = np.zeros((H, W, s.extra["U"].shape[1]))
        U[s.rows, s.cols] = s.extra["U"]
        valid = np.zeros((H, W), bool)
        valid[s.rows, s.cols] = s.ok
        yy = np.zeros((H, W), int)
        yy[s.rows, s.cols] = y
        ss = np.zeros((H, W), bool)
        ss[s.rows, s.cols] = sel & s.ok
        items.append(dict(r=r, U=U, valid=valid, y=yy, sel=ss))
    return items


def _analyse_none(scn):
    for s in scn.slides:
        s.extra = {k: v for k, v in s.extra.items() if k in ("path", "backend", "dimensions", "mpp_assumed", "mpp_eff")}
    C.analyse(scn.slides, GO.to_config(dict(GO.DEFAULT, smooth="none"), scn.n_states, **scn.cfg_extra))


def scenario_from_gaar(scn, group):
    _analyse_none(scn)
    ys, sels = [], []
    for i, s in enumerate(scn.slides):
        y, sel = scn.truth_fn(i, s)
        yy = np.zeros(len(s.rows), int)
        ss = np.zeros(len(s.rows), bool)
        yy[:] = y
        ss[:] = sel
        ys.append(yy)
        sels.append(ss)
    return dict(name=scn.name, kind="other", group=group, cohorts=[_lattice_items(scn.slides, ys, sels)])


def scenario_morph(name, shape, size, seeds):
    cohorts = []
    for sd in seeds:
        slides, masks = [], []
        for i in range(IMAGES_PER_COHORT):
            X, m = M.make_lattice(sd * 1000 + i, shape=shape, block=size, **M.GEN)
            slides.append(M._slide("m%d_%d" % (sd, i), X))
            masks.append(m.reshape(-1).astype(int))
        for s in slides:
            s.extra = {}
        C.analyse(slides, GO.to_config(dict(GO.DEFAULT, smooth="none"), n_states=6, ref_per_slide=300, ref_max=3000))
        cohorts.append(_lattice_items(slides, masks, [np.ones(len(mk), bool) for mk in masks]))
    return dict(name=name, kind="morph", group=1, cohorts=cohorts)


def build_cache(split):
    out = []
    if split == "dev":
        out += [scenario_from_gaar(GO.image_scenario("dev", "handcrafted"), 1), scenario_from_gaar(GO.image_scenario("dev", "dinov2_vits14"), 1),
                scenario_from_gaar(GO.sim_scenario(GO.BB.DEV, "rotor", "sim-dev-rotor"), 3), scenario_from_gaar(GO.sim_scenario(GO.BB.DEV, "rotor_D", "sim-dev-rotor_D"), 3)]
        seeds = DEV_SEEDS
    else:
        out += [scenario_from_gaar(GO.image_scenario("test", "handcrafted"), 1), scenario_from_gaar(GO.image_scenario("test", "dinov2_vits14"), 1),
                scenario_from_gaar(GO.sim_scenario(GO.TEST_SIM, "rotor", "sim-test-rotor"), 3), scenario_from_gaar(GO.sim_scenario(GO.TEST_SIM, "rotor_D", "sim-test-rotor_D"), 3)]
        seeds = TEST_SEEDS
    for nm, sh, sz in SHAPES:
        out.append(scenario_morph("morph-%s-%s" % (split, nm), sh, sz, seeds))
        print("cached", out[-1]["name"], flush=True)
    pickle.dump(out, open(os.path.join(CACHE_DIR, "og_cache_%s.pkl" % split), "wb"))
    return out


def load_cache(split):
    return pickle.load(open(os.path.join(CACHE_DIR, "og_cache_%s.pkl" % split), "rb"))


# ------------------------------------------------------------------------------------------------ evaluation
def _scores(scn, g):
    """per cohort: score lattices after applying the rule configuration g"""
    out = []
    for items in scn["cohorts"]:
        tau = G.estimate_tau([it["U"] for it in items], [it["valid"] for it in items])
        out.append(G.cohort_apply(items, g["rule"], g["window"], g["theta"], g["alpha"], g["lam"], tau))
    return out


def _flatten(scn, scores):
    ys, ss = [], []
    for items, sc in zip(scn["cohorts"], scores):
        for it, s in zip(items, sc):
            m = it["sel"] & np.isfinite(s)
            ys.append(it["y"][m])
            ss.append(s[m])
    return ys, ss        # per image


def ap_of(ys, ss, idx=None):
    idx = range(len(ys)) if idx is None else idx
    y = np.concatenate([ys[i] for i in idx])
    s = np.concatenate([ss[i] for i in idx])
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def evaluate(scn, g, with_regions=False):
    sc = _scores(scn, g)
    ys, ss = _flatten(scn, sc)
    y, s = np.concatenate(ys), np.concatenate(ss)
    out = dict(ap=ap_of(ys, ss), auroc=float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan"))
    if with_regions and scn["kind"] == "morph":
        rec, fr, n_img = [], 0, 0
        for items, scores in zip(scn["cohorts"], sc):
            allv = np.concatenate([x[np.isfinite(x)] for x in scores])
            thr = np.quantile(allv, 1.0 - Q)
            for it, x in zip(items, scores):
                cand = np.isfinite(x) & (x >= thr)
                lab, nl = ndi.label(it["y"].astype(bool), structure=np.ones((3, 3)))
                rec.extend(bool((cand & (lab == k)).any()) for k in range(1, nl + 1))
                clab, nc = ndi.label(cand, structure=np.ones((3, 3)))
                dil = ndi.binary_dilation(it["y"].astype(bool), iterations=1)
                fr += sum(1 for k in range(1, nc + 1) if not (dil & (clab == k)).any())
                n_img += 1
        out["region_recall"] = float(np.mean(rec)) if rec else float("nan")
        out["false_regions_per_image"] = fr / max(n_img, 1)
    return out


def fitness_over(scenarios, g):
    aps = [evaluate(sc, g)["ap"] for sc in scenarios]
    a = np.asarray(aps)
    return aps, float(0.5 * np.nanmean(a) + 0.5 * np.nanmin(a))


# ------------------------------------------------------------------------------------------------ grid, search, test
_DEV = None


def _init():
    import warnings
    warnings.filterwarnings("ignore")
    global _DEV
    _DEV = load_cache("dev")


def _eval_key(k):
    g = json.loads(k)
    aps, f = fitness_over(_DEV, g)
    return k, aps, f


def all_configs():
    keys = list(RULE_GRID)
    return [dict(zip(keys, c)) for c in itertools.product(*[RULE_GRID[k] for k in keys])]


def run_grid(out, workers=5):
    import multiprocessing as mp
    done = json.load(open(out)) if os.path.exists(out) else {}
    uniq = sorted({json.dumps(canonical(g), sort_keys=True) for g in all_configs()})
    todo = [k for k in uniq if k not in done]
    print("grid: %d unique configurations (of %d), %d to do" % (len(uniq), len(all_configs()), len(todo)), flush=True)
    with mp.Pool(workers, initializer=_init) as pool:
        for i, (k, aps, f) in enumerate(pool.imap_unordered(_eval_key, todo), 1):
            done[k] = dict(aps=aps, fitness=f)
            if i % 20 == 0 or i == len(todo):
                json.dump(done, open(out, "w"))
                print("  %d/%d" % (i, len(todo)), flush=True)
    return done


def run_search(grid_json, out, budget=40, repeats=100):
    Gm = GO.load_gaar()
    table = json.load(open(grid_json))
    fit = {k: v["fitness"] for k, v in table.items()}
    lookup = lambda g: fit[json.dumps(canonical(g), sort_keys=True)]
    vals = np.array(sorted(fit.values(), reverse=True))
    thr1, thr5 = float(np.quantile(vals, 0.99)), float(np.quantile(vals, 0.95))
    res = dict(n_configs=len(all_configs()), n_unique=len(fit), budget=budget, repeats=repeats, optimum=float(vals[0]), strategies={})
    best_key = max(fit, key=fit.get)
    res["optimum_config"] = json.loads(best_key)
    res["frozen_fitness"] = lookup(DEFAULT_RULE)
    res["frozen_rank"] = int((vals > lookup(DEFAULT_RULE)).sum()) + 1

    class Dom(Gm.GridDomain):
        def __init__(self):
            super().__init__(RULE_GRID, lookup, name="og_rules")
            self._rng = np.random.default_rng(0)

        def reseed(self, s):
            import random
            self._r = random.Random(s)

        def seed_genome(self):
            return {k: self._r.choice(v) for k, v in self.grid.items()}
    for strat in ("greedy", "restart", "population", "random"):
        bests, hit1, hit5, champs = [], 0, 0, []
        for r in range(repeats):
            dom = Dom()
            dom.reseed(2000 + r)
            o = Gm.GaaR(dom, strategy=strat, seed=r, db_path=os.devnull, log=False).run(budget=budget, verbose=False)
            bests.append(o["best_score"])
            champs.append(o["best_genome"])
            hit1 += o["best_score"] >= thr1
            hit5 += o["best_score"] >= thr5
        res["strategies"][strat] = dict(median_best=float(np.median(bests)), mean_best=float(np.mean(bests)), hit_top1pct=hit1 / repeats, hit_top5pct=hit5 / repeats, champion_seed0=champs[0], best_score_seed0=float(bests[0]))
        print(strat, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["strategies"][strat].items() if k != "champion_seed0"}, flush=True)
    # best development configuration of every rule family
    fam = {}
    for k, v in table.items():
        g = json.loads(k)
        if g["rule"] not in fam or v["fitness"] > fam[g["rule"]][1]:
            fam[g["rule"]] = (g, v["fitness"])
    res["family_champions"] = {r: dict(config=g, dev_fitness=f) for r, (g, f) in fam.items()}
    json.dump(res, open(out, "w"), indent=1)
    return res


def run_test(search_json, grid_json, out, B=300, seed=0):
    sr = json.load(open(search_json))
    test = load_cache("test")
    cands = {"none": canonical(dict(DEFAULT_RULE, rule="none")), "frozen (mean 3)": canonical(DEFAULT_RULE)}
    for r, d in sr["family_champions"].items():
        if r not in ("none", "mean"):
            cands["best %s (dev)" % r] = d["config"]
    cands["best mean (dev)"] = sr["family_champions"]["mean"]["config"]
    cands["similarity-weighted (3), untuned"] = canonical(dict(DEFAULT_RULE, rule="simw"))
    cands["GaaR restart champion"] = canonical(sr["strategies"]["restart"]["champion_seed0"])
    cands["GaaR random-search champion"] = canonical(sr["strategies"]["random"]["champion_seed0"])
    cands["exhaustive development optimum"] = sr["optimum_config"]
    res = dict(candidates=cands, scenarios=[s["name"] for s in test], results={}, bootstrap={})
    per_unit = {}
    for name, g in cands.items():
        res["results"][name] = {}
        per_unit[name] = []
        for sc in test:
            m = evaluate(sc, g, with_regions=True)
            res["results"][name][sc["name"]] = m
            ys, ss = _flatten(sc, _scores(sc, g))
            per_unit[name].append((ys, ss))
        aps = [res["results"][name][sc["name"]]["ap"] for sc in test]
        res["results"][name]["_fitness"] = float(0.5 * np.nanmean(aps) + 0.5 * np.nanmin(aps))
        print(name, round(res["results"][name]["_fitness"], 3), flush=True)
    # paired bootstrap over independent units (images; simulation runs of three snapshots)
    rng = np.random.default_rng(seed)
    n_u = [len(per_unit["none"][k][0]) // test[k]["group"] for k in range(len(test))]
    F = {nm: [] for nm in cands}
    AP = {nm: [[] for _ in test] for nm in cands}
    for _ in range(B):
        draws = [rng.integers(0, n_u[k], n_u[k]) for k in range(len(test))]
        for nm in cands:
            aps = []
            for k, sc in enumerate(test):
                gp = sc["group"]
                idx = [u * gp + j for u in draws[k] for j in range(gp)]
                aps.append(ap_of(per_unit[nm][k][0], per_unit[nm][k][1], idx))
                AP[nm][k].append(aps[-1])
            a = np.asarray(aps)
            F[nm].append(0.5 * np.nanmean(a) + 0.5 * np.nanmin(a))
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    for ref in ("frozen (mean 3)", "none", "similarity-weighted (3), untuned"):
        for nm in cands:
            d = np.array(F[nm]) - np.array(F[ref])
            res["bootstrap"].setdefault(nm, {})["fitness_ci"] = ci(F[nm])
            res["bootstrap"][nm]["diff_vs_" + ref.split()[0]] = dict(ci=ci(d), share_better=float(np.mean(d > 0)))
            res["bootstrap"][nm]["ap_ci"] = [ci(AP[nm][k]) for k in range(len(test))]
    json.dump(res, open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "grid", "search", "test"])
    ap.add_argument("split", nargs="?", default="dev")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "cache":
        build_cache(a.split)
    elif a.cmd == "grid":
        run_grid("results/og_grid_dev.json", a.workers)
    elif a.cmd == "search":
        run_search("results/og_grid_dev.json", "results/og_search.json")
    elif a.cmd == "test":
        run_test("results/og_search.json", "results/og_grid_dev.json", "results/og_test.json")
