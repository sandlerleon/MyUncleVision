# -*- coding: utf-8 -*-
"""Experiment D of the Knowledge-Based Systems study (descriptive, run after both freezes): robustness of the stage-2 rule and its comparators to one-at-a-time changes of
embedding dimension, lattice size, feature-anomaly prevalence, window and background coherence, with runtime per lattice.

    python -m myuncle_vision.kbs_robust      -> results/kbs_robust.json
"""
import json
import os
import time

import numpy as np

from . import kbs_stage2 as S
from . import kbs_study as K


def _job(args):
    gen, seeds, kw, window = args
    Us, kinds = [], []
    for sd in seeds:
        U, kd = S.GEN[gen](sd, **kw)
        Us.append(U)
        kinds.append(kd)
    t0 = time.perf_counter()
    tm = {}
    sc = K.cohort_scores(Us, window=window, rng_seed=int(seeds[0]), with_baselines=True, timing=tm)
    t1 = time.perf_counter()
    sc["within-lattice kNN rarity"] = [S.within_rarity(U) for U in Us]
    sc["cohort kNN rarity"] = sc["rarity"]
    t2 = time.perf_counter()
    for m in S.MARGINAL:
        for s in S.SPATIAL:
            rm = [np.nan_to_num(x, nan=0.0) for x in K._crank(sc[m])]
            rs = [np.nan_to_num(x, nan=0.0) for x in K._crank(sc[s])]
            sc["max(%s | %s)" % (m, s)] = [np.fmax(x, y) for x, y in zip(rm, rs)]
            sc["OR(%s | %s)" % (m, s)] = [1 - (1 - x) * (1 - y) for x, y in zip(rm, rs)]
    n = len(Us)
    tm.update(all_scores=(t1 - t0) / n, within_knn=(t2 - t1) / n)
    return {nm: [K.lattice_metrics(kinds[i], v[i]) for i in range(n)] for nm, v in sc.items()}, tm


def run(gen, kw, n_lat, seed0, window=3, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    seeds = [seed0 + i for i in range(n_lat)]
    jobs = [(gen, seeds[c:c + K.N_PER_COHORT], kw, window) for c in range(0, n_lat, K.N_PER_COHORT)]
    res, tms = {}, []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for per, tm in ex.map(_job, jobs):
            tms.append(tm)
            for nm, lst in per.items():
                d = res.setdefault(nm, {"feature": [], "arrangement": [], "all": []})
                for m in lst:
                    for kk in d:
                        d[kk].append(m[kk])
    return res, {k: float(np.mean([t[k] for t in tms])) for k in tms[0]}


def main():
    fz = json.load(open(os.path.join(K.RES, "kbs2_frozen.json")))
    ch = fz["chosen"]
    keep = [ch, "fusion: max (rarity, B)", "OR(LOF, Shekhar)", "contextual LOF ([u, neighbor mean])", "fusion: max (isolation forest, Shekhar)", "LOF (embeddings)",
            "spatial outlier statistic (Shekhar)", "channel B (directional disorder)", "fusion: cohort selection (tail index)"]
    settings = [("G3", dict(arr="shuffle"), "dimension", "d", (2, 16, 64, 384)),
                ("G3", dict(arr="shuffle"), "lattice size", "n", (32, 64, 128)),
                ("G3", dict(arr="shuffle"), "feature prevalence", "prev_f", (0.01, 0.03, 0.05)),
                ("G3", dict(arr="shuffle"), "window", "window", (3, 5, 7)),
                ("G2", dict(phi=40.0, delta=1.0), "coherence (G2 kappa scale)", "kappa_scale", (0.25, 0.5, 1.0, 2.0)),
                ("G3", dict(arr="shuffle"), "noise (G3)", "noise", (0.04, 0.08, 0.16, 0.32))]
    out, seed = {}, 1200000
    for gen, base, label, key, vals in settings:
        out[label] = {}
        for v in vals:
            kw, win = dict(base), 3
            if key == "window":
                win = v
            else:
                kw[key] = v
            heavy = (key == "n" and v >= 128) or (key == "d" and v >= 384)
            res, tm = run(gen, kw, 10, seed, window=win, workers=4 if heavy else 8)
            seed += 1000
            sm = K.summary(res)
            out[label][str(v)] = dict(generator=gen, summary={k: sm[k] for k in keep if k in sm}, timing=tm)
            print(label, v, {k[:18]: round(sm[k]["worst"], 3) for k in keep if k in sm}, "t/lattice %.2fs" % tm["all_scores"], flush=True)
    K._save(dict(chosen=ch, settings=out, lattices_per_setting=10), "kbs_robust.json")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
