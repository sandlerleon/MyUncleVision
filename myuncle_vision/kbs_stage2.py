# -*- coding: utf-8 -*-
"""Stage 2 of the Knowledge-Based Systems study: post-hoc diagnosis of the failed confirmatory test, a revised rule frozen a second time, and a new confirmatory test.

History (see results/kbs_frozen.json, kbs_confirm.json and the git log): the rule frozen in stage 1, max(cohort rarity, B), failed on generator G2 because its rarity is referenced to the
cohort while G2's lattices do not share one normal distribution (every lattice has its own domain directions).  kbs_study.py is left unchanged (its sha256 is in the stage-1 freeze file).

    explore   (POST HOC, labelled as such)  fresh seeds of the four G1 development conditions and of the three G2 conditions, 20 lattices each; every pair of a marginal channel
              {cohort kNN rarity (k=5), within-lattice kNN rarity (k=20), LOF (k=20), isolation forest} and a spatial channel {B, Shekhar statistic, center-to-neighbor distance}, fused by
              max and OR of cohort ranks -> results/kbs2_explore.json
    freeze    selected pair and fusion (criterion below) + G3 and replication conditions, seeds, endpoints, sha256 of this file -> results/kbs2_frozen.json
    confirm   G3 (new generator, written before the second freeze and never scored before it) and fresh seeds of G1 and G2 -> results/kbs2_confirm.json

Selection criterion (declared here before explore is run): the pair and fusion with the highest mean over the seven exploration conditions of the worst-type AP; ties within 0.005 go to the
simpler/earlier entry in the order max before OR, marginal order as listed, spatial order as listed.
"""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage as ndi
from sklearn.neighbors import NearestNeighbors

from . import kbs_study as K

RES = K.RES
MARGINAL = ["cohort kNN rarity", "within-lattice kNN rarity", "LOF (embeddings)", "isolation forest (embeddings)"]
SPATIAL = ["channel B (directional disorder)", "spatial outlier statistic (Shekhar)", "center-to-neighbor distance"]
FUSE = ["max", "OR"]


def within_rarity(U, k=20):
    X = U.reshape(-1, U.shape[-1])
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    return ((dist[:, 1:] ** 2) / 2.0).mean(1).reshape(U.shape[:2])


def scores2(Us, rng_seed=0):
    sc = K.cohort_scores(Us, rng_seed=rng_seed, with_baselines=True)
    sc["cohort kNN rarity"] = sc["rarity"]
    sc["within-lattice kNN rarity"] = [within_rarity(U) for U in Us]
    ranks = {nm: [np.nan_to_num(x, nan=0.0) for x in K._crank(sc[nm])] for nm in MARGINAL + SPATIAL}
    for m in MARGINAL:
        for s in SPATIAL:
            sc["max(%s | %s)" % (m, s)] = [np.fmax(x, y) for x, y in zip(ranks[m], ranks[s])]
            sc["OR(%s | %s)" % (m, s)] = [1 - (1 - x) * (1 - y) for x, y in zip(ranks[m], ranks[s])]
    return sc


PAIRS = ["%s(%s | %s)" % (f, m, s) for f in FUSE for m in MARGINAL for s in SPATIAL]


# ---------------------------------------------------------------------------------------------------------------- generator G3
G3_MAP_SEED = 4242


def gen3(seed, n=32, d=16, wavelength=8.0, noise=0.08, psi_range=(0.0, 0.6), dpsi=1.2, prev_f=0.015, prev_a=0.04, arr="shuffle", f_size=(2, 10), a_size=(16, 30), a_block=6):
    """oriented-texture lattice.  A latent feature f = [cos phi, sin phi, 0.6 cos psi, 0.6 sin psi] (phi: stripe phase, psi: a slowly varying texture code inside psi_range) is mapped into d
    dimensions by a random linear map SHARED by all G3 lattices (a fixed 'encoder', so normal embeddings lie on one common 2-torus), plus iid noise; then normalized.
    Feature anomalies: random-walk blobs whose texture code psi lies dpsi outside the normal range (on the common manifold, but in a region normal lattices never visit).
    Arrangement anomalies: arr = 'shuffle' (two distant blobs exchange and reshuffle their embeddings) or 'rotate' (a 6 x 6 patch is rotated by 90 degrees: every embedding and every local
    neighborhood stays ordinary, only the stripe orientation of the patch is wrong)."""
    rng = np.random.default_rng(seed)
    P = np.random.default_rng(G3_MAP_SEED).normal(size=(4, d))
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    alpha = rng.uniform(0, np.pi) + 0.6 * ndi.gaussian_filter(rng.normal(size=(n, n)), 6, mode="wrap") * 6
    phi = 2 * np.pi / wavelength * (xx * np.cos(alpha) + yy * np.sin(alpha)) + rng.uniform(0, 2 * np.pi)
    g = ndi.gaussian_filter(rng.normal(size=(n, n)), 5, mode="wrap")
    psi = psi_range[0] + (psi_range[1] - psi_range[0]) * (g - g.min()) / (g.max() - g.min() + 1e-12)
    kind = np.zeros((n, n), int)
    occ = np.zeros((n, n), bool)
    F = None

    def feat(ph, ps):
        return np.stack([np.cos(ph), np.sin(ph), 0.6 * np.cos(ps), 0.6 * np.sin(ps)], -1)
    # feature anomalies first on the latent code
    target, got, guard = prev_f * n * n, 0, 0
    while got < target and guard < 200:
        guard += 1
        sz = int(rng.integers(f_size[0], f_size[1] + 1))
        px = K._place(occ, rng, lambda: K._blob(rng, n, sz))
        if px is None:
            continue
        occ[px[:, 0], px[:, 1]] = True
        psi[px[:, 0], px[:, 1]] = psi_range[1] + dpsi + rng.uniform(-0.1, 0.1)
        kind[px[:, 0], px[:, 1]] = 1
        got += len(px)
    X = feat(phi, psi) @ P + rng.normal(0, noise, (n, n, d))
    U = X / np.linalg.norm(X, axis=-1, keepdims=True)
    if arr == "shuffle":
        target, got, guard = prev_a * n * n, 0, 0
        while got < target and guard < 50:
            guard += 1
            sz = int(rng.integers(a_size[0], a_size[1] + 1))
            r1 = K._place(occ, rng, lambda: K._blob(rng, n, sz))
            if r1 is None:
                continue
            occ[r1[:, 0], r1[:, 1]] = True
            r2 = None
            for _ in range(200):
                c = K._place(occ, rng, lambda: K._blob(rng, n, sz))
                if c is not None and np.hypot(*(c.mean(0) - r1.mean(0))) > n / 3:
                    r2 = c
                    break
            if r2 is None:
                occ[r1[:, 0], r1[:, 1]] = False
                continue
            occ[r2[:, 0], r2[:, 1]] = True
            v1, v2 = U[r1[:, 0], r1[:, 1]].copy(), U[r2[:, 0], r2[:, 1]].copy()
            U[r1[:, 0], r1[:, 1]] = v2[rng.permutation(sz)]
            U[r2[:, 0], r2[:, 1]] = v1[rng.permutation(sz)]
            kind[r1[:, 0], r1[:, 1]] = 2
            kind[r2[:, 0], r2[:, 1]] = 2
            got += 2 * sz
    elif arr == "rotate":
        k = max(1, int(round(prev_a * n * n / (a_block * a_block))))
        for _ in range(k):
            px = K._place(occ, rng, lambda: K._block(rng, n, a_block))
            if px is None:
                break
            occ[px[:, 0], px[:, 1]] = True
            y0, x0 = px[:, 0].min(), px[:, 1].min()
            U[y0:y0 + a_block, x0:x0 + a_block] = np.rot90(U[y0:y0 + a_block, x0:x0 + a_block].copy())
            kind[y0:y0 + a_block, x0:x0 + a_block] = 2
    return U, kind


GEN = {"G1": K.gen1, "G2": lambda seed, **kw: K.gen2(seed, **kw)[:2], "G3": gen3}


def _job(args):
    gen, seeds, kw = args
    Us, kinds = [], []
    for sd in seeds:
        U, kd = GEN[gen](sd, **kw)
        Us.append(U)
        kinds.append(kd)
    sc = scores2(Us, rng_seed=int(seeds[0]))
    return {nm: [K.lattice_metrics(kinds[i], v[i]) for i in range(len(Us))] for nm, v in sc.items()}


def run(gen, kw, n_lat, seed0, workers=8):
    seeds = [seed0 + i for i in range(n_lat)]
    jobs = [(gen, seeds[c:c + K.N_PER_COHORT], kw) for c in range(0, n_lat, K.N_PER_COHORT)]
    res = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for per in ex.map(_job, jobs):
            for nm, lst in per.items():
                d = res.setdefault(nm, {"feature": [], "arrangement": [], "all": []})
                for m in lst:
                    for kk in d:
                        d[kk].append(m[kk])
    return res


EXPLORE = [("G1", nm, kw, 40000 + 1000 * i) for i, (nm, kw) in enumerate(K.DEV_CONDITIONS.items())] + \
          [("G2", nm, kw, 600000 + 1000 * i) for i, (nm, kw) in enumerate(K.CONFIRM_CONDITIONS.items())]


def explore():
    out, allres = {}, {}
    for gen, nm, kw, s0 in EXPLORE:
        res = run(gen, kw, 20, s0)
        allres[nm] = res
        out[nm] = K.summary(res)
        best = sorted(PAIRS, key=lambda p: -out[nm][p]["worst"])[:3]
        print(nm, [(b, round(out[nm][b]["worst"], 3)) for b in best], flush=True)
    crit = {p: float(np.mean([out[c][p]["worst"] for c in out])) for p in PAIRS}
    order = sorted(PAIRS, key=lambda p: -crit[p])
    top = crit[order[0]]
    chosen = next(p for p in PAIRS if crit[p] >= top - 0.005)
    pooled = K._pool_conditions(allres)
    tests = {p: K.paired_boot(pooled, chosen, p, "worst") for p in order[:8] if p != chosen}
    single = {nm: float(np.mean([out[c][nm]["worst"] for c in out])) for nm in MARGINAL + SPATIAL + ["contextual LOF ([u, neighbor mean])", "OR(LOF, Shekhar)",
                                                                                                    "fusion: max (rarity, B)", "fusion: max (isolation forest, Shekhar)"]}
    K._save(dict(label="POST HOC exploration after the failed stage-1 confirmatory test", conditions=out, criterion=crit, ranking=order, chosen=chosen, tests_vs_chosen=tests,
                 reference_detectors_mean_worst=single), "kbs2_explore.json")


CONFIRM2 = {
    "G3 oriented texture, shuffled regions": ("G3", dict(arr="shuffle")),
    "G3 oriented texture, rotated patches": ("G3", dict(arr="rotate")),
    "G2 replication (fresh seeds), phi=40, delta=1": ("G2", dict(phi=40.0, delta=1.0)),
    "G1 replication (fresh seeds), blocks 3x3, s=2": ("G1", dict(s=2.0, f_shape="block", f_block=3)),
}
SEED2 = 900000


def freeze():
    ex = json.load(open(os.path.join(RES, "kbs2_explore.json")))
    sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    fz = dict(frozen_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), code_sha256=sha, stage1_code_sha256=hashlib.sha256(open(K.__file__, "rb").read()).hexdigest(),
              chosen=ex["chosen"], conditions={k: [g, kw] for k, (g, kw) in CONFIRM2.items()}, lattices_per_condition=30, seed0=SEED2,
              primary_endpoint="worst-type AP of the chosen fusion, per condition and pooled over the two G3 conditions; paired bootstrap against each comparator",
              comparators=["fusion: max (rarity, B)", "OR(LOF, Shekhar)", "contextual LOF ([u, neighbor mean])", "fusion: max (isolation forest, Shekhar)",
                           "fusion: cohort selection (tail index)"] + [x for x in ex["ranking"][:4] if x != ex["chosen"]] + MARGINAL + SPATIAL,
              g3_note="gen3 written before this freeze; a smoke test generated one lattice (seed 1) of each arrangement type to check that the code runs, no score computed")
    K._save(fz, "kbs2_frozen.json")


def confirm():
    fz = json.load(open(os.path.join(RES, "kbs2_frozen.json")))
    if hashlib.sha256(open(__file__, "rb").read()).hexdigest() != fz["code_sha256"]:
        sys.exit("kbs_stage2.py changed after the second freeze; refusing to run")
    ch = fz["chosen"]
    out, allres = {}, {}
    for i, (nm, (g, kw)) in enumerate(fz["conditions"].items()):
        res = run(g, kw, fz["lattices_per_condition"], fz["seed0"] + 1000 * i)
        allres[nm] = res
        sm = K.summary(res)
        comps = [c for c in fz["comparators"] if c != ch and c in res]
        out[nm] = dict(summary=sm, tests={c: K.paired_boot(res, ch, c, "worst") for c in comps})
        print(nm, ch, round(sm[ch]["worst"], 3), sorted(((round(sm[c]["worst"], 3), c[:30]) for c in comps), reverse=True)[:5], flush=True)
    g3 = K._pool_conditions({k: v for k, v in allres.items() if k.startswith("G3")})
    comps = [c for c in fz["comparators"] if c != ch]
    K._save(dict(frozen=fz, chosen=ch, conditions=out, g3_pooled_summary=K.summary(g3),
                 g3_pooled_tests={c: {key: K.paired_boot(g3, ch, c, key) for key in ("worst", "feature", "arrangement", "all")} for c in comps}), "kbs2_confirm.json")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    {"explore": explore, "freeze": freeze, "confirm": confirm}[sys.argv[1]]()
