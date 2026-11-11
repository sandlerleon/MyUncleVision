# -*- coding: utf-8 -*-
"""Two further experiments on the aggregation rules, borrowed from physics of the author's earlier lattice work (no references: unpublished ideas used as design principles):

 1. nucleation and growth with hysteresis (`ngh`): a soft nucleation threshold on the rarity (a site must be rare enough to seed) followed by growth into compatible neighbors that never lowers a score;
    parameters selected on the development scenarios, evaluated once on the independent test scenarios with paired bootstrap intervals (same protocol as og_study).
 2. the closure gap: a label-free statistic (the retention of the top-3% sites under a fixed mean) that measures how steep-feedback-like the score field is, tested as a predictor of whether
    smoothing helps, over all cohorts of the development and test scenarios; a retention-based choice between smoothing and no smoothing is fitted on the development cohorts and tested on the test cohorts.

    python -m myuncle_vision.og_extra ngh        -> results/og_ngh.json
    python -m myuncle_vision.og_extra closure    -> results/og_closure.json
    python -m myuncle_vision.og_extra survival   -> results/og_survival.json
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


def survival_fraction(mask, valid, r_a, r_n, theta, w=3):
    """share of the anomalous sites that stay above `theta` after the fixed w x w mean when anomalous sites have rank r_a and normal sites r_n (the contrast-survival model);
    a site with k anomalous among n valid window sites scores (k r_a + (n - k) r_n)/n"""
    from scipy.ndimage import uniform_filter
    ok = valid
    k = uniform_filter((mask & ok).astype(float), w, mode="constant") * w * w
    n = uniform_filter(ok.astype(float), w, mode="constant") * w * w
    pred = (k * r_a + (n - k) * r_n) / np.maximum(n, 1)
    sel = mask & ok
    return float(np.mean(pred[sel] > theta)) if sel.any() else float("nan")


def run_survival(out, q=0.03, w=3):
    """contrast-survival model of the fixed w x w mean.  Normal sites have rarity ranks with mean r_n, anomalous sites r_a (calibrated on the development cohorts, with labels).  After the mean a site with k anomalous
    sites among its n valid window sites scores (k r_a + (n - k) r_n)/n; it stays detectable when that exceeds the cohort's own top-q threshold theta of the smoothed scores (label-free).  The model's prediction is the share
    phi of anomalous sites that survive; it is compared with the measured average precision of the fixed mean on every controlled-generator cohort of the development and test scenarios."""
    from scipy.ndimage import uniform_filter
    dev, test = O.load_cache("dev"), O.load_cache("test")
    morph = lambda scs: [sc for sc in scs if sc["kind"] == "morph"]
    ra, rn = [], []
    for sc in morph(dev):
        for items in sc["cohorts"]:
            for it in items:
                ok = it["valid"] & np.isfinite(it["r"])
                ra.append(it["r"][ok & (it["y"] == 1)])
                rn.append(it["r"][ok & (it["y"] == 0)])
    r_a, r_n = float(np.mean(np.concatenate(ra))), float(np.mean(np.concatenate(rn)))
    rows = []
    for split, scs in (("dev", morph(dev)), ("test", morph(test))):
        for sc in scs:
            for ci, items in enumerate(sc["cohorts"]):
                sm_all, phis = [], []
                # the cohort's own top-q threshold of the (measured) smoothed score
                sc_l = G.cohort_apply(items, "mean", w, 0, 0, 0, 0.5)
                theta = float(np.quantile(np.concatenate([x[np.isfinite(x)] for x in sc_l]), 1.0 - q))
                ph = [survival_fraction(it["y"] == 1, it["valid"] & np.isfinite(it["r"]), r_a, r_n, theta, w) for it in items]
                wts = [int(((it["y"] == 1) & it["valid"]).sum()) for it in items]
                phi = float(np.average(np.nan_to_num(ph), weights=np.maximum(wts, 1)))
                none = G.cohort_apply(items, "none", 1, 0, 0, 0, 0.5)
                ap = {}
                for nm, scores in (("none", none), ("mean", sc_l)):
                    y = np.concatenate([it["y"][it["sel"] & np.isfinite(s)] for it, s in zip(items, scores)])
                    f = np.concatenate([s[it["sel"] & np.isfinite(s)] for it, s in zip(items, scores)])
                    ap[nm] = float(average_precision_score(y, f))
                rows.append(dict(split=split, scenario=sc["name"], cohort=ci, phi=phi, theta=theta, ap_none=ap["none"], ap_mean=ap["mean"]))
    phi = np.array([r["phi"] for r in rows])
    apm = np.array([r["ap_mean"] for r in rows])
    sp_all = spearmanr(phi, apm)
    te = [r for r in rows if r["split"] == "test"]
    sp_test = spearmanr([r["phi"] for r in te], [r["ap_mean"] for r in te])
    by = {}
    for r in rows:
        by.setdefault(r["scenario"].split("-")[-1], []).append((r["phi"], r["ap_mean"]))
    res = dict(r_a=r_a, r_n=r_n, q=q, window=w, k_star_formula="ceil(n (theta - r_n) / (r_a - r_n))", rows=rows, spearman_all=dict(rho=float(sp_all.correlation), p=float(sp_all.pvalue), n=len(rows)),
               spearman_test=dict(rho=float(sp_test.correlation), p=float(sp_test.pvalue), n=len(te)), by_shape={k: dict(phi=float(np.mean([a for a, _ in v])), ap_mean=float(np.mean([b for _, b in v]))) for k, v in by.items()})
    json.dump(res, open(out, "w"), indent=1)
    print("r_a %.3f r_n %.3f" % (r_a, r_n), "spearman all", res["spearman_all"], "test", res["spearman_test"])
    for k, v in res["by_shape"].items():
        print("  %-12s predicted survival %.2f   measured AP of the fixed mean %.2f" % (k, v["phi"], v["ap_mean"]))
    return res


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    if sys.argv[1] == "ngh":
        run_ngh("results/og_ngh.json")
    elif sys.argv[1] == "closure":
        run_closure("results/og_closure.json")
    elif sys.argv[1] == "survival":
        w_ = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        run_survival("results/og_survival.json" if w_ == 3 else "results/og_survival_w%d.json" % w_, w=w_)
