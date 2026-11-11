# -*- coding: utf-8 -*-
"""Two further experiments on the aggregation rules, borrowed from physics of the author's earlier lattice work (no references: unpublished ideas used as design principles):

 1. nucleation and growth with hysteresis (`ngh`): a soft nucleation threshold on the rarity (a site must be rare enough to seed) followed by growth into compatible neighbors that never lowers a score;
    parameters selected on the development scenarios, evaluated once on the independent test scenarios with paired bootstrap intervals (same protocol as og_study).
 2. the closure gap: a label-free statistic (the retention of the top-3% sites under a fixed mean) that measures how steep-feedback-like the score field is, tested as a predictor of whether
    smoothing helps, over all cohorts of the development and test scenarios; a retention-based choice between smoothing and no smoothing is fitted on the development cohorts and tested on the test cohorts.

    python -m myuncle_vision.og_extra ngh        -> results/og_ngh.json
    python -m myuncle_vision.og_extra closure    -> results/og_closure.json
"""
import itertools
import json
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from . import gating as G
from . import og_study as O

NGH_GRID = {"window": [3, 5, 7], "theta": [0.90, 0.95, 0.98, 0.99], "alpha": [0.6, 0.8, 1.0]}


def ngh_configs():
    return [dict(rule="ngh", window=w, theta=t, alpha=a, lam=0.0) for w, t, a in itertools.product(NGH_GRID["window"], NGH_GRID["theta"], NGH_GRID["alpha"])]


def run_ngh(out, B=300):
    dev, test = O.load_cache("dev"), O.load_cache("test")
    table = {}
    for g in ngh_configs():
        aps, f = O.fitness_over(dev, g)
        table[json.dumps(g, sort_keys=True)] = dict(aps=aps, fitness=f)
        print("ngh", g["window"], g["theta"], g["alpha"], round(f, 3), flush=True)
    best = json.loads(max(table, key=lambda k: table[k]["fitness"]))
    cands = {"none": O.canonical(dict(O.DEFAULT_RULE, rule="none")), "frozen (mean 3)": O.canonical(O.DEFAULT_RULE), "similarity-weighted (3), untuned": O.canonical(dict(O.DEFAULT_RULE, rule="simw")),
             "nucleation-growth (dev-selected)": best, "nucleation-growth (3, 0.95, 0.8 untuned)": dict(rule="ngh", window=3, theta=0.95, alpha=0.8, lam=0.0)}
    res = O.evaluate_candidates(test, cands, B)
    res["ngh_dev_table"] = table
    res["ngh_best_dev_fitness"] = table[json.dumps(best, sort_keys=True)]["fitness"]
    json.dump(res, open(out, "w"), indent=1)
    for nm in cands:
        print(nm, round(res["results"][nm]["_fitness"], 3), res["bootstrap"][nm]["diff_vs_similarity-weighted"]["ci"], flush=True)
    return res


def _cohort_rows(scenarios, split):
    rows = []
    for sc in scenarios:
        for ci, items in enumerate(sc["cohorts"]):
            tau = G.estimate_tau([it["U"] for it in items], [it["valid"] for it in items])
            vals = {}
            for nm, g in (("none", dict(rule="none", window=1, theta=0, alpha=0, lam=0)), ("mean", dict(rule="mean", window=3, theta=0, alpha=0, lam=0)), ("simw", dict(rule="simw", window=3, theta=0, alpha=0, lam=0))):
                sc_l = G.cohort_apply(items, g["rule"], g["window"], g["theta"], g["alpha"], g["lam"], tau)
                y = np.concatenate([it["y"][it["sel"] & np.isfinite(s)] for it, s in zip(items, sc_l)])
                f = np.concatenate([s[it["sel"] & np.isfinite(s)] for it, s in zip(items, sc_l)])
                vals[nm] = float(average_precision_score(y, f)) if 0 < y.sum() < len(y) else float("nan")
            ret = float(np.nanmean([G.closure_gap(it["r"], it["valid"]) for it in items]))
            rows.append(dict(split=split, scenario=sc["name"], cohort=ci, retention=ret, **vals))
    return rows


def run_closure(out):
    dev, test = O.load_cache("dev"), O.load_cache("test")
    rows = _cohort_rows(dev, "dev") + _cohort_rows(test, "test")
    ok = [r for r in rows if np.isfinite(r["none"]) and np.isfinite(r["mean"])]
    rho = np.array([r["retention"] for r in ok])
    eff = np.array([r["mean"] - r["none"] for r in ok])
    sp = spearmanr(rho, eff)
    res = dict(rows=rows, spearman_retention_vs_mean_effect=dict(rho=float(sp.correlation), p=float(sp.pvalue), n=len(ok)))
    devr = [r for r in ok if r["split"] == "dev"]
    tstr = [r for r in ok if r["split"] == "test"]
    best_t, best_m = None, -1
    for t in sorted({round(r["retention"], 3) for r in devr}):
        m = np.mean([r["mean"] if r["retention"] >= t else r["none"] for r in devr])
        if m > best_m:
            best_t, best_m = t, m
    res["threshold_from_dev"] = best_t
    pick = lambda rs, t: np.mean([r["mean"] if r["retention"] >= t else r["none"] for r in rs])
    res["test_mean_ap"] = dict(always_none=float(np.mean([r["none"] for r in tstr])), always_mean=float(np.mean([r["mean"] for r in tstr])), always_simw=float(np.mean([r["simw"] for r in tstr])), retention_choice=float(pick(tstr, best_t)),
                                oracle=float(np.mean([max(r["none"], r["mean"]) for r in tstr])), n_cohorts=len(tstr))
    res["dev_mean_ap"] = dict(always_none=float(np.mean([r["none"] for r in devr])), always_mean=float(np.mean([r["mean"] for r in devr])), retention_choice=float(best_m), n_cohorts=len(devr))
    res["share_chose_mean_test"] = float(np.mean([r["retention"] >= best_t for r in tstr]))
    json.dump(res, open(out, "w"), indent=1)
    print("spearman", res["spearman_retention_vs_mean_effect"], "threshold", best_t)
    print("test mean AP", {k: round(v, 3) for k, v in res["test_mean_ap"].items()})
    return res


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    if sys.argv[1] == "ngh":
        run_ngh("results/og_ngh.json")
    elif sys.argv[1] == "closure":
        run_closure("results/og_closure.json")
