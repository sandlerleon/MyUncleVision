# -*- coding: utf-8 -*-
"""Detectability and complementarity of feature and spatial-order anomalies on embedding lattices (study for the Knowledge-Based Systems manuscript).

Two synthetic generators, written before any confirmatory evaluation:
    G1 (development)   Gaussian random-field background (correlation length `corr`) normalized to the sphere; FEATURE anomalies add a common random shift to the embeddings of a region
                       (strength s in background standard deviations per coordinate); ARRANGEMENT anomalies permute the embeddings of a fraction delta of the sites of 6 x 6 blocks among
                       themselves (cyclic derangement: every chosen site receives another site's embedding).
    G2 (confirmatory)  sphere-native: Voronoi domains, each with its own mean direction, a linear drift of that direction across the domain and its own von Mises-Fisher concentration (heterogeneous
                       background coherence; domain walls present); FEATURE anomalies are irregular random-walk blobs drawn around a novel direction at angle phi from the local mean;
                       ARRANGEMENT anomalies exchange the embeddings of a fraction delta of the sites of two distant irregular regions of equal size and reassign them at random (the multiset of
                       the lattice's embeddings is exactly unchanged).
Labels: 0 normal, 1 feature anomaly, 2 arrangement anomaly.

Scores (all label-free):  rarity (kNN cosine distance, k = 5, to a pool drawn from the OTHER lattices of the cohort), its cohort rank r; fixed w x w mean of r; similarity-weighted aggregation of r
(channel A, gating.py); directional disorder (channel B, two_channel.py); isotropic disorder D; center-to-neighbor distance; baselines LOF and isolation forest on the embeddings of a lattice,
the spatial-outlier statistic ||u_i - mean of the 8 neighbors|| (Shekhar et al.), contextual LOF on [u_i, neighbor mean].
Fusion of the cohort ranks a (channel A) and b (channel B): select (cohort tail-index selection), max, OR = 1-(1-a)(1-b), zmax, and two locally adaptive rules: lcOR (channel B standardized
against its own 15 x 15 neighborhood median and MAD before ranking) and relmax = max(a, (1-g) b) with g the cohort rank of the 15 x 15 median of B (unreliability of B in incoherent background).

    python -m myuncle_vision.kbs_study pilot                 calibration of the G1 strength levels (development seeds only)
    python -m myuncle_vision.kbs_study phase                 Experiment A, G1 -> results/kbs_phase.json
    python -m myuncle_vision.kbs_study dev                   Experiment B, G1, rule selection -> results/kbs_dev.json
    python -m myuncle_vision.kbs_study freeze                writes results/kbs_frozen.json (selected rule, G2 conditions, seeds, endpoints, sha256 of this file)
    python -m myuncle_vision.kbs_study confirm               Experiment C, G2, frozen protocol -> results/kbs_confirm.json
    python -m myuncle_vision.kbs_study robust                Experiment D, G2 -> results/kbs_robust.json
    python -m myuncle_vision.kbs_study props                 numerical checks of the propositions -> results/kbs_props.json
"""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import vonmises_fisher
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from . import gating as G
from . import local_geometry as LG
from . import two_channel as T

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
N_PER_COHORT = 10
K_NN = 5
POOL = 3000


# ================================================================================================ generators
def _place(mask, rng, pixels_fn, tries=4000):
    for _ in range(tries):
        px = pixels_fn()
        ys, xs = px[:, 0], px[:, 1]
        n = mask.shape[0]
        if ys.min() < 0 or xs.min() < 0 or ys.max() >= n or xs.max() >= n:
            continue
        if mask[max(0, ys.min() - 1):ys.max() + 2, max(0, xs.min() - 1):xs.max() + 2].any():
            continue
        return px
    return None


def _block(rng, n, b):
    y0, x0 = rng.integers(0, n - b + 1, 2)
    yy, xx = np.mgrid[y0:y0 + b, x0:x0 + b]
    return np.stack([yy.ravel(), xx.ravel()], 1)


def _blob(rng, n, size):
    """irregular connected region grown by a random walk on the 4-neighborhood"""
    y, x = rng.integers(2, n - 2, 2)
    pts = {(int(y), int(x))}
    while len(pts) < size:
        py, px = list(pts)[rng.integers(len(pts))]
        dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[rng.integers(4)]
        pts.add((py + dy, px + dx))
    return np.array(sorted(pts))


def _derange(rng, m):
    """random permutation without fixed points (a random cyclic shift of a random order)"""
    order = rng.permutation(m)
    perm = np.empty(m, int)
    perm[order] = np.roll(order, 1)
    return perm


def gen1(seed, n=32, d=16, corr=3.0, noise=0.2, scale=1.5, s=1.0, delta=1.0, prev_f=0.015, prev_a=0.035, f_shape="block", f_block=3, a_block=6):
    """the smooth field occupies the first d/2 dimensions (normal sites lie near that subspace, plus iid noise of SD `noise` in all d); a feature anomaly region is shifted by s along one random
    unit direction of the complementary dimensions (s in absolute units; the noise norm in the complementary dimensions is noise * sqrt(d/2))"""
    rng = np.random.default_rng(seed)
    d_field = max(1, d // 2)
    f = ndi.gaussian_filter(rng.normal(size=(n, n, d)), sigma=(corr, corr, 0), mode="wrap")
    f = (f - f.mean((0, 1))) / f.std((0, 1))
    f[..., d_field:] = 0.0
    X = scale * f + rng.normal(0, noise, (n, n, d))
    kind = np.zeros((n, n), int)
    occ = np.zeros((n, n), bool)
    if delta > 0 and prev_a > 0:
        k = max(1, int(round(prev_a * n * n / (a_block * a_block * max(delta, 1e-9)))))
        for _ in range(k):
            px = _place(occ, rng, lambda: _block(rng, n, a_block))
            if px is None:
                break
            occ[px[:, 0], px[:, 1]] = True
            m = max(2, int(round(delta * len(px))))
            sel = px[rng.choice(len(px), m, replace=False)]
            vals = X[sel[:, 0], sel[:, 1]].copy()
            X[sel[:, 0], sel[:, 1]] = vals[_derange(rng, m)]
            kind[sel[:, 0], sel[:, 1]] = 2
    if s > 0 and prev_f > 0:
        area = f_block * f_block if f_shape == "block" else f_block
        k = max(1, int(round(prev_f * n * n / area)))
        for _ in range(k):
            fn = (lambda: _block(rng, n, f_block)) if f_shape == "block" else (lambda: _blob(rng, n, f_block))
            px = _place(occ, rng, fn)
            if px is None:
                break
            occ[px[:, 0], px[:, 1]] = True
            u = np.zeros(d)
            u[d_field:] = rng.normal(size=d - d_field) if d > d_field else 0.0
            if d == d_field:
                u = rng.normal(size=d)
            X[px[:, 0], px[:, 1]] += s * u / np.linalg.norm(u)
            kind[px[:, 0], px[:, 1]] = 1
    U = X / np.linalg.norm(X, axis=-1, keepdims=True)
    return U.astype(np.float64), kind


def _rotate_toward(mu, angle, rng):
    v = rng.normal(size=mu.shape)
    v -= v.dot(mu) * mu
    v /= np.linalg.norm(v)
    return np.cos(angle) * mu + np.sin(angle) * v


def gen2(seed, n=32, d=16, n_dom=(5, 10), kappa_log=(np.log(40.0), np.log(400.0)), drift=0.6, phi=40.0, delta=1.0, prev_f=0.015, prev_a=0.04, f_size=(1, 12), a_size=(16, 32),
         kappa_scale=1.0):
    """phi in degrees; kappa_scale multiplies every domain's concentration (background coherence)"""
    rng = np.random.default_rng(seed)
    K = int(rng.integers(n_dom[0], n_dom[1] + 1))
    seeds = rng.uniform(0, n, (K, 2))
    yy, xx = np.mgrid[0:n, 0:n]
    dist = (yy[..., None] - seeds[:, 0]) ** 2 + (xx[..., None] - seeds[:, 1]) ** 2
    dom = dist.argmin(-1)
    mus = rng.normal(size=(K, d))
    mus /= np.linalg.norm(mus, axis=1, keepdims=True)
    kap = np.exp(rng.uniform(*kappa_log, K)) * kappa_scale
    U = np.zeros((n, n, d))
    for k in range(K):
        idx = np.argwhere(dom == k)
        g = rng.normal(size=d)
        g -= g.dot(mus[k]) * mus[k]
        g /= np.linalg.norm(g)
        ang = rng.uniform(0, 2 * np.pi)
        proj = ((idx[:, 0] - seeds[k, 0]) * np.cos(ang) + (idx[:, 1] - seeds[k, 1]) * np.sin(ang)) / n
        for (y, x), t in zip(idx, proj):
            m = mus[k] * np.cos(drift * t) + g * np.sin(drift * t)
            U[y, x] = vonmises_fisher(m, kap[k]).rvs(1, random_state=rng)[0]
    kind = np.zeros((n, n), int)
    occ = np.zeros((n, n), bool)
    if delta > 0 and prev_a > 0:
        target = prev_a * n * n
        got, guard = 0, 0
        while got < target and guard < 50:
            guard += 1
            sz = int(rng.integers(a_size[0], a_size[1] + 1))
            r1 = _place(occ, rng, lambda: _blob(rng, n, sz))
            if r1 is None:
                continue
            occ[r1[:, 0], r1[:, 1]] = True
            r2 = None
            for _ in range(200):
                c = _place(occ, rng, lambda: _blob(rng, n, sz))
                if c is not None and np.hypot(*(c.mean(0) - r1.mean(0))) > n / 3:
                    r2 = c
                    break
            if r2 is None:
                occ[r1[:, 0], r1[:, 1]] = False
                continue
            occ[r2[:, 0], r2[:, 1]] = True
            m = max(1, int(round(delta * sz)))
            s1 = r1[rng.choice(sz, m, replace=False)]
            s2 = r2[rng.choice(sz, m, replace=False)]
            v1, v2 = U[s1[:, 0], s1[:, 1]].copy(), U[s2[:, 0], s2[:, 1]].copy()
            U[s1[:, 0], s1[:, 1]] = v2[rng.permutation(m)]
            U[s2[:, 0], s2[:, 1]] = v1[rng.permutation(m)]
            kind[s1[:, 0], s1[:, 1]] = 2
            kind[s2[:, 0], s2[:, 1]] = 2
            got += 2 * m
    if phi > 0 and prev_f > 0:
        target = prev_f * n * n
        got, guard = 0, 0
        while got < target and guard < 200:
            guard += 1
            sz = int(rng.integers(f_size[0], f_size[1] + 1))
            px = _place(occ, rng, lambda: _blob(rng, n, sz))
            if px is None:
                continue
            occ[px[:, 0], px[:, 1]] = True
            k = int(np.bincount(dom[px[:, 0], px[:, 1]]).argmax())
            mu_f = _rotate_toward(mus[k], np.deg2rad(phi), rng)
            U[px[:, 0], px[:, 1]] = vonmises_fisher(mu_f, kap[k]).rvs(len(px), random_state=rng)
            kind[px[:, 0], px[:, 1]] = 1
            got += len(px)
    return U, kind, dom


GENERATORS = {"G1": gen1, "G2": lambda seed, **kw: gen2(seed, **kw)[:2]}


# ================================================================================================ scores
def _neighbor_mean(U):
    s = np.zeros(U.shape)
    n = np.zeros(U.shape[:2])
    v = np.ones(U.shape[:2], bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            s += G._shift(U, dy, dx, 0.0)
            n += G._shift(v.astype(float), dy, dx, 0.0)
    return s / np.maximum(n, 1)[..., None]


def _local_robust(x, w=15):
    """local median and MAD over a w x w window (w = 15, larger than the anomalies, so that the background and not the anomaly sets the local reference)"""
    med = ndi.median_filter(np.nan_to_num(x, nan=np.nanmedian(x)), size=w, mode="reflect")
    mad = ndi.median_filter(np.abs(np.nan_to_num(x, nan=np.nanmedian(x)) - med), size=w, mode="reflect")
    return (x - med) / (1.4826 * mad + 1e-6), med


def _crank(fields):
    return T.cohort_rank(fields)


def cohort_scores(Us, window=3, rng_seed=0, with_baselines=True, timing=None):
    """score lattices for one cohort (list of (n, n, d) unit-embedding arrays).  Returns a dict name -> list of (n, n) arrays (higher = more anomalous)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(rng_seed)
    n_l = len(Us)
    flat = [U.reshape(-1, U.shape[-1]) for U in Us]
    valid = [np.ones(U.shape[:2], bool) for U in Us]
    raw = []
    for i in range(n_l):
        others = np.concatenate([flat[j] for j in range(n_l) if j != i])
        pool = others[rng.choice(len(others), min(POOL, len(others)), replace=False)]
        nn = NearestNeighbors(n_neighbors=K_NN).fit(pool)
        dist, _ = nn.kneighbors(flat[i])
        raw.append((dist ** 2 / 2.0).mean(1).reshape(Us[i].shape[:2]))           # mean cosine distance to the k nearest
    t_r = time.perf_counter()
    r = _crank(raw)
    tau = G.estimate_tau(Us, valid)
    items = [dict(r=r[i], U=Us[i], valid=valid[i]) for i in range(n_l)]
    h = window // 2
    out = {"rarity": r, "rarity, fixed mean": [G._mean_window(r[i], valid[i], window) for i in range(n_l)],
           "channel A (similarity-weighted rarity)": G.cohort_apply(items, "simw", window, 0, 0, 0, tau)}
    t_a = time.perf_counter()
    dd = [T.directional_disorder(Us[i], valid[i], h) for i in range(n_l)]
    B = [x[0] for x in dd]
    out["channel B (directional disorder)"] = B
    t_b = time.perf_counter()
    out["mean quadrant disorder"] = [x[1] for x in dd]
    out["isotropic disorder D"] = [LG.disorder_map(Us[i], valid[i]) for i in range(n_l)]
    out["center-to-neighbor distance"] = [T.center_neighbor_distance(Us[i], valid[i]) for i in range(n_l)]
    if with_baselines:
        nm = [_neighbor_mean(U) for U in Us]
        out["spatial outlier statistic (Shekhar)"] = [np.linalg.norm(Us[i] - nm[i], axis=-1) for i in range(n_l)]
        out["LOF (embeddings)"] = [-LocalOutlierFactor(n_neighbors=20).fit(flat[i]).negative_outlier_factor_.reshape(Us[i].shape[:2]) for i in range(n_l)]
        out["isolation forest (embeddings)"] = [-IsolationForest(n_estimators=100, random_state=rng_seed + i).fit(flat[i]).score_samples(flat[i]).reshape(Us[i].shape[:2]) for i in range(n_l)]
        out["contextual LOF ([u, neighbor mean])"] = [
            -LocalOutlierFactor(n_neighbors=20).fit(np.concatenate([flat[i], nm[i].reshape(len(flat[i]), -1)], 1)).negative_outlier_factor_.reshape(Us[i].shape[:2]) for i in range(n_l)]
        lr, ls = _crank(out["LOF (embeddings)"]), _crank(out["spatial outlier statistic (Shekhar)"])
        out["OR(LOF, Shekhar)"] = [1 - (1 - x) * (1 - y) for x, y in zip(lr, ls)]
    # fusion of channel A and channel B
    A = out["channel A (similarity-weighted rarity)"]
    a, b = _crank(A), [np.nan_to_num(x, nan=0.0) for x in _crank(B)]
    out["fusion: cohort selection (tail index)"] = T.fuse(A, raw, B, "select")[0]
    out["fusion: max"] = [np.fmax(x, y) for x, y in zip(a, b)]
    out["fusion: OR"] = [1 - (1 - x) * (1 - y) for x, y in zip(a, b)]
    out["fusion: zmax"] = T.fuse(A, raw, B, "zmax")[0]
    cn = [np.nan_to_num(x, nan=0.0) for x in _crank(out["center-to-neighbor distance"])]
    out["fusion: max (A, neighbor distance)"] = [np.fmax(x, y) for x, y in zip(a, cn)]
    out["fusion: OR (A, neighbor distance)"] = [1 - (1 - x) * (1 - y) for x, y in zip(a, cn)]
    rr = _crank(r)
    out["fusion: max (rarity, B)"] = [np.fmax(x, y) for x, y in zip(rr, b)]
    if with_baselines:
        fi, fs = _crank(out["isolation forest (embeddings)"]), _crank(out["spatial outlier statistic (Shekhar)"])
        out["fusion: max (isolation forest, Shekhar)"] = [np.fmax(x, y) for x, y in zip(fi, fs)]
    lc = [_local_robust(x)[0] for x in B]
    blc = [np.nan_to_num(x, nan=0.0) for x in _crank(lc)]
    out["fusion: lcOR (locally calibrated B)"] = [1 - (1 - x) * (1 - y) for x, y in zip(a, blc)]
    gmed = _crank([_local_robust(x)[1] for x in B])
    out["fusion: relmax (reliability-weighted B)"] = [np.fmax(x, (1 - g) * y) for x, y, g in zip(a, b, gmed)]
    t_f = time.perf_counter()
    if timing is not None:
        timing.update(rarity=(t_r - t0) / n_l, channelA=(t_a - t_r) / n_l, channelB=(t_b - t_a) / n_l, total=(t_f - t0) / n_l)
    return out


# ================================================================================================ metrics
def _ap(y, s):
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(average_precision_score(y, s))


def lattice_metrics(kind, score):
    k, s = kind.ravel(), np.asarray(score, float).ravel()
    out = {}
    for t, nm in ((1, "feature"), (2, "arrangement")):
        m = (k == t) | (k == 0)
        out[nm] = _ap((k[m] == t).astype(int), s[m]) if (k == t).any() else np.nan
    out["all"] = _ap((k > 0).astype(int), s)
    return out


def _cohort_job(args):
    gen, seeds, kw, window, with_baselines = args
    Us, kinds = [], []
    for sd in seeds:
        U, kd = GENERATORS[gen](sd, **kw)
        Us.append(U)
        kinds.append(kd)
    tm = {}
    sc = cohort_scores(Us, window=window, rng_seed=int(seeds[0]), with_baselines=with_baselines, timing=tm)
    per = {nm: [lattice_metrics(kinds[i], v[i]) for i in range(len(Us))] for nm, v in sc.items()}
    return per, tm


def run_condition(gen, kw, n_lattices, seed0, window=3, with_baselines=True, workers=8):
    """returns {detector: {"feature": [AP per lattice], "arrangement": [...], "all": [...]}}, mean timing per lattice"""
    seeds = [seed0 + i for i in range(n_lattices)]
    jobs = [(gen, seeds[c:c + N_PER_COHORT], kw, window, with_baselines) for c in range(0, n_lattices, N_PER_COHORT)]
    res, tms = {}, []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for per, tm in ex.map(_cohort_job, jobs):
            tms.append(tm)
            for nm, lst in per.items():
                d = res.setdefault(nm, {"feature": [], "arrangement": [], "all": []})
                for m in lst:
                    for kk in d:
                        d[kk].append(m[kk])
    timing = {k: float(np.mean([t[k] for t in tms])) for k in tms[0]}
    return res, timing


def summary(res):
    out = {}
    for nm, d in res.items():
        f, a, al = np.nanmean(d["feature"]), np.nanmean(d["arrangement"]), np.nanmean(d["all"])
        out[nm] = dict(feature=float(f), arrangement=float(a), all=float(al), worst=float(np.nanmin([f, a])))
    return out


def paired_boot(res, x, y, key="worst", B=2000, seed=0):
    """paired bootstrap over lattices of the difference x - y in mean AP (key feature / arrangement / all) or in the worst-type AP"""
    rng = np.random.default_rng(seed)
    n = len(res[x]["all"])
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)

        def val(nm):
            f = np.nanmean(np.asarray(res[nm]["feature"])[idx])
            a = np.nanmean(np.asarray(res[nm]["arrangement"])[idx])
            if key == "worst":
                return min(f, a)
            return dict(feature=f, arrangement=a, all=np.nanmean(np.asarray(res[nm]["all"])[idx]))[key]
        diffs.append(val(x) - val(y))
    d = np.array(diffs)
    return dict(mean=float(np.mean(d)), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], share_positive=float(np.mean(d > 0)))


def _save(obj, name):
    os.makedirs(RES, exist_ok=True)
    json.dump(obj, open(os.path.join(RES, name), "w"), indent=1)
    print("saved", name, flush=True)


# ================================================================================================ experiments
S_LEVELS = (1.0, 1.5, 2.0, 2.5, 3.0)                # G1 feature strength levels (set by the pilot: unsmoothed-rarity AP 0.07 ... 0.86 on development seeds)
DELTA_LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0)
DEV_CONDITIONS = {                                    # Experiment B: mixed lattices from G1 (development)
    "G1 mixed, blocks 3x3, s=2": dict(s=2.0, f_shape="block", f_block=3),
    "G1 mixed, scattered, s=2.5": dict(s=2.5, f_shape="block", f_block=1),
    "G1 mixed, blobs 6, s=2, delta=0.6": dict(s=2.0, f_shape="blob", f_block=6, delta=0.6),
    "G1 mixed, short correlation (corr 1.5), s=2": dict(s=2.0, f_shape="block", f_block=3, corr=1.5),
}
FUSION_CANDIDATES = ["fusion: cohort selection (tail index)", "fusion: max", "fusion: OR", "fusion: zmax", "fusion: lcOR (locally calibrated B)", "fusion: relmax (reliability-weighted B)",
                     "fusion: max (rarity, B)"]
SIMPLE = ["fusion: max", "fusion: OR", "fusion: max (rarity, B)"]
# Development run 1 (results/kbs_dev_run1.json) had only max and OR of (A, B) as simple candidates; it showed the unsmoothed rarity above channel A for feature anomalies, so max(rarity, B) was added
# as a candidate, and the generic pair max(isolation forest, Shekhar) as a comparator, before run 2 and before the freeze.


def pilot():
    """G1 feature-only and arrangement-only lattices: AP of the unsmoothed rarity per strength level, and of channel B per delta (development seeds 9000+)"""
    out = {}
    for s in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        res, _ = run_condition("G1", dict(s=s, delta=0.0, f_shape="block", f_block=3), 10, 9000, with_baselines=False)
        out["s=%g" % s] = summary(res)["rarity"]["feature"]
        print("pilot s", s, out["s=%g" % s], flush=True)
    for dl in DELTA_LEVELS:
        res, _ = run_condition("G1", dict(s=0.0, delta=dl), 10, 9100, with_baselines=False)
        out["delta=%g" % dl] = summary(res)["channel B (directional disorder)"]["arrangement"]
        print("pilot delta", dl, out["delta=%g" % dl], flush=True)
    _save(out, "kbs_pilot.json")


def phase():
    """Experiment A: per (s, delta) on mixed G1 lattices (blocks 3 x 3), 20 lattices each; background-coherence axis corr in {1, 1.5, 3, 6} at s = 2, delta = 1"""
    dets = ["rarity", "channel A (similarity-weighted rarity)", "channel B (directional disorder)", "center-to-neighbor distance", "fusion: max", "fusion: OR",
            "fusion: cohort selection (tail index)"]
    grid = {}
    for i, s in enumerate(S_LEVELS):
        for j, dl in enumerate(DELTA_LEVELS):
            res, _ = run_condition("G1", dict(s=s, delta=dl, f_shape="block", f_block=3), 20, 20000 + 100 * (5 * i + j), with_baselines=False)
            sm = summary(res)
            grid["s=%g,delta=%g" % (s, dl)] = {k: sm[k] for k in dets}
            print("phase", s, dl, {k[:12]: round(sm[k]["worst"], 3) for k in dets}, flush=True)
    coh = {}
    for c in (1.0, 1.5, 3.0, 6.0):
        res, _ = run_condition("G1", dict(s=2.0, delta=1.0, f_shape="block", f_block=3, corr=c), 20, 23000 + int(10 * c), with_baselines=False)
        sm = summary(res)
        coh["corr=%g" % c] = {k: sm[k] for k in dets}
        print("coherence", c, {k[:12]: round(sm[k]["worst"], 3) for k in dets}, flush=True)
    _save(dict(s_levels=S_LEVELS, delta_levels=DELTA_LEVELS, grid=grid, coherence=coh, lattices_per_cell=20), "kbs_phase.json")


def dev():
    """Experiment B on G1 (development): 30 lattices per condition, all detectors; selection of the fusion rule by the pre-declared criterion"""
    allres, out = {}, {}
    for c, (nm, kw) in enumerate(DEV_CONDITIONS.items()):
        res, tm = run_condition("G1", kw, 30, 30000 + 1000 * c)
        allres[nm] = res
        out[nm] = dict(summary=summary(res), timing=tm)
        print(nm, {k[:16]: round(v["worst"], 3) for k, v in out[nm]["summary"].items()}, flush=True)
    crit = {f: float(np.mean([out[nm]["summary"][f]["worst"] for nm in out])) for f in FUSION_CANDIDATES}
    best_simple = max(SIMPLE, key=lambda f: round(crit[f], 4))          # ties: the first listed
    adaptive = ["fusion: lcOR (locally calibrated B)", "fusion: relmax (reliability-weighted B)"]
    pooled = _pool_conditions(allres)
    tests = {f: paired_boot(pooled, f, best_simple, "worst") for f in adaptive + [x for x in FUSION_CANDIDATES if x not in adaptive and x != best_simple]}
    chosen = best_simple
    for f in adaptive:
        if crit[f] > crit[chosen] and tests[f]["ci"][0] > 0:
            chosen = f
    out_all = dict(conditions=out, criterion_mean_worst_type_AP=crit, best_simple=best_simple, tests_vs_best_simple=tests, chosen=chosen,
                   rule="choose the better of max and OR by the mean over conditions of the worst-type AP; replace it by an adaptive rule only if that rule is better AND its paired bootstrap "
                        "95% interval of the gain (lattices pooled over conditions) excludes zero")
    _save(out_all, "kbs_dev.json")
    return out_all


def _pool_conditions(allres):
    pooled = {}
    for res in allres.values():
        for nm, d in res.items():
            p = pooled.setdefault(nm, {"feature": [], "arrangement": [], "all": []})
            for k in p:
                p[k] += list(d[k])
    return pooled


CONFIRM_CONDITIONS = {             # Experiment C on G2; declared in the freeze file before any G2 lattice is generated
    "G2 mixed, phi=40, delta=1": dict(phi=40.0, delta=1.0),
    "G2 mixed, phi=60, delta=0.6": dict(phi=60.0, delta=0.6),
    "G2 mixed, low coherence (kappa x0.25), phi=40, delta=1": dict(phi=40.0, delta=1.0, kappa_scale=0.25),
}
CONFIRM_LATTICES = 30
CONFIRM_SEED0 = 500000
PRIMARY_COMPARATORS = ["channel A (similarity-weighted rarity)", "channel B (directional disorder)", "fusion: cohort selection (tail index)", "OR(LOF, Shekhar)",
                       "contextual LOF ([u, neighbor mean])", "fusion: max (A, neighbor distance)", "fusion: max (isolation forest, Shekhar)", "isolation forest (embeddings)",
                       "spatial outlier statistic (Shekhar)", "fusion: max", "fusion: max (rarity, B)"]
# The spatial channel of the method is fixed a priori as the directional disorder B (Proposition 3: insensitive to straight domain walls); G1 has no walls, so the development data cannot
# decide between B and the plain neighbor distance.  The max fusion of A with the neighbor distance is therefore a pre-declared comparator of the confirmatory test, not a candidate.


def freeze():
    devres = json.load(open(os.path.join(RES, "kbs_dev.json")))
    sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    fz = dict(frozen_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), code_sha256=sha, chosen_fusion=devres["chosen"], window=3, k_nn=K_NN, pool=POOL,
              generator="G2 (gen2, written before this freeze, never run before it)", conditions=CONFIRM_CONDITIONS, lattices_per_condition=CONFIRM_LATTICES, seed0=CONFIRM_SEED0,
              primary_endpoint="worst-type AP = min(mean feature-anomaly AP, mean arrangement-anomaly AP) per condition and pooled over the three conditions; "
                               "paired bootstrap (2000 resamples of lattices) of the frozen fusion minus each comparator",
              comparators=PRIMARY_COMPARATORS, secondary="per-type AP, all-anomaly AP, all other detectors, runtime",
              development_history="development run 1 (kbs_dev_run1.json) -> candidates max(rarity, B) and comparator max(isolation forest, Shekhar) added -> development run 2 (kbs_dev.json) selected the rule; G2 smoke test before the freeze: one lattice (seed 1) generated to check that the code runs; no score computed")
    _save(fz, "kbs_frozen.json")


def confirm():
    fz = json.load(open(os.path.join(RES, "kbs_frozen.json")))
    sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    if sha != fz["code_sha256"]:
        sys.exit("kbs_study.py changed after the freeze; refusing to run the confirmatory test")
    chosen = fz["chosen_fusion"]
    allres, out = {}, {}
    for c, (nm, kw) in enumerate(fz["conditions"].items()):
        res, tm = run_condition("G2", kw, fz["lattices_per_condition"], fz["seed0"] + 1000 * c)
        allres[nm] = res
        out[nm] = dict(summary=summary(res), timing=tm, tests={cmp: paired_boot(res, chosen, cmp, "worst") for cmp in fz["comparators"] if cmp != chosen})
        print(nm, {k[:16]: round(v["worst"], 3) for k, v in out[nm]["summary"].items()}, flush=True)
    pooled = _pool_conditions(allres)
    tests = {cmp: {key: paired_boot(pooled, chosen, cmp, key) for key in ("worst", "feature", "arrangement", "all")} for cmp in fz["comparators"] if cmp != chosen}
    tests_all = {nm: paired_boot(pooled, chosen, nm, "worst") for nm in pooled if nm != chosen}
    _save(dict(frozen=fz, chosen=chosen, conditions=out, pooled_summary=summary(pooled), pooled_tests=tests, pooled_tests_all=tests_all,
               per_lattice={nm: pooled[nm] for nm in pooled}), "kbs_confirm.json")


def robust():
    """Experiment D (descriptive, after the freeze): the frozen method and its channels on G2 under one-at-a-time variation; 10 lattices per setting; runtime per lattice"""
    fz = json.load(open(os.path.join(RES, "kbs_frozen.json")))
    chosen = fz["chosen_fusion"]
    keep = ["rarity", "channel A (similarity-weighted rarity)", "channel B (directional disorder)", chosen, "fusion: cohort selection (tail index)", "OR(LOF, Shekhar)",
            "contextual LOF ([u, neighbor mean])"]
    base = dict(phi=40.0, delta=1.0)
    settings = [("dimension", "d", (2, 16, 64, 384)), ("lattice size", "n", (32, 64, 128)), ("feature prevalence", "prev_f", (0.01, 0.03, 0.05)),
                ("window", "window", (3, 5, 7)), ("coherence (kappa scale)", "kappa_scale", (0.25, 0.5, 1.0, 2.0))]
    out = {}
    seed = 700000
    for label, key, vals in settings:
        out[label] = {}
        for v in vals:
            kw = dict(base)
            win = 3
            if key == "window":
                win = v
            else:
                kw[key] = v
            if key == "n":
                kw["n_dom"] = (max(2, int(5 * (v / 32) ** 2)), max(4, int(10 * (v / 32) ** 2)))
            res, tm = run_condition("G2", kw, 10, seed, window=win, with_baselines=True, workers=4 if (key == "n" and v >= 128) or (key == "d" and v >= 384) else 8)
            seed += 1000
            sm = summary(res)
            out[label][str(v)] = dict(summary={k: sm[k] for k in keep if k in sm}, timing=tm)
            print(label, v, {k[:14]: round(sm[k]["worst"], 3) for k in keep if k in sm}, "t/lattice %.2fs" % tm["total"], flush=True)
    _save(dict(chosen=chosen, base=base, settings=out), "kbs_robust.json")


def props():
    """numerical checks: P1 (marginal detectors at chance on arrangement anomalies), P2 (occupancy-based survival of the fixed mean), P3 (domain walls and channel B)"""
    out = {}
    # P1: AUC of the unsmoothed rarity and of LOF for arrangement anomalies, arrangement-only lattices of both generators (fresh seeds)
    for gen, kw, s0 in (("G1", dict(s=0.0, delta=1.0), 800000), ("G2", dict(phi=0.0, delta=1.0), 810000)):
        aucs = {"rarity": [], "LOF": []}
        for c in range(3):
            Us, kinds = [], []
            for i in range(N_PER_COHORT):
                U, kd = GENERATORS[gen](s0 + 100 * c + i, **kw)
                Us.append(U)
                kinds.append(kd)
            sc = cohort_scores(Us, with_baselines=True, rng_seed=s0 + c)
            for i in range(len(Us)):
                y = (kinds[i] == 2).ravel().astype(int)
                aucs["rarity"].append(float(roc_auc_score(y, sc["rarity"][i].ravel())))
                aucs["LOF"].append(float(roc_auc_score(y, sc["LOF (embeddings)"][i].ravel())))
        out["P1_" + gen] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v)), n=len(v)) for k, v in aucs.items()}
        print("P1", gen, out["P1_" + gen], flush=True)
    # P2: predicted survival (share of anomalous sites with occupancy >= p*) vs measured survival after the fixed 3 x 3 mean, feature-only G1 lattices of several shapes
    p2 = {}
    for shape, blk in (("block", 1), ("block", 3), ("block", 6), ("blob", 6), ("blob", 12)):
        Us, kinds = [], []
        for i in range(20):
            U, kd = gen1(820000 + 37 * blk + i + (500 if shape == "blob" else 0), s=1.5, delta=0.0, f_shape=shape, f_block=blk, prev_f=0.03)
            Us.append(U)
            kinds.append(kd)
        valid = [np.ones(U.shape[:2], bool) for U in Us]
        sc = cohort_scores(Us[:10], with_baselines=False, rng_seed=1) ["rarity"] + cohort_scores(Us[10:], with_baselines=False, rng_seed=2)["rarity"]
        sm = [G._mean_window(sc[i], valid[i], 3) for i in range(20)]
        thr = np.nanquantile(np.concatenate([x.ravel() for x in sm]), 0.97)
        ra = float(np.nanmean(np.concatenate([sc[i][kinds[i] == 1] for i in range(20)])))
        rn = float(np.nanmean(np.concatenate([sc[i][kinds[i] == 0] for i in range(20)])))
        pstar = (thr - rn) / (ra - rn)
        occ = [ndi.uniform_filter((kinds[i] == 1).astype(float), 3, mode="constant") * 9 / np.maximum(ndi.uniform_filter(np.ones_like(sc[i]), 3, mode="constant") * 9, 1) for i in range(20)]
        pred = float(np.mean(np.concatenate([occ[i][kinds[i] == 1] >= pstar for i in range(20)])))
        meas = float(np.mean(np.concatenate([sm[i][kinds[i] == 1] >= thr for i in range(20)])))
        p2["%s %d" % (shape, blk)] = dict(r_a=ra, r_n=rn, threshold=float(thr), p_star=float(pstar), predicted_survival=pred, measured_survival=meas)
        print("P2", shape, blk, p2["%s %d" % (shape, blk)], flush=True)
    out["P2"] = p2
    # P3: G2 background only (no anomalies): share of domain-wall sites among the top 3% of each disorder statistic
    walls, tops = [], {"channel B (directional disorder)": [], "isotropic disorder D": [], "center-to-neighbor distance": [], "spatial outlier statistic (Shekhar)": []}
    frac_wall = []
    for c in range(2):
        Us, doms = [], []
        for i in range(N_PER_COHORT):
            U, kd, dom = gen2(830000 + 100 * c + i, phi=0.0, delta=0.0)
            Us.append(U)
            doms.append(dom)
        sc = cohort_scores(Us, with_baselines=True, rng_seed=c)
        for i in range(len(Us)):
            dm = doms[i]
            wall = (ndi.maximum_filter(dm, 3) != ndi.minimum_filter(dm, 3))
            frac_wall.append(wall.mean())
            for k in tops:
                s = sc[k][i]
                t = np.nanquantile(s, 0.97)
                sel = np.isfinite(s) & (s >= t)
                tops[k].append(float(wall[sel].mean()))
    out["P3"] = dict(wall_fraction=float(np.mean(frac_wall)), share_of_top3pct_on_walls={k: float(np.mean(v)) for k, v in tops.items()})
    print("P3", out["P3"], flush=True)
    _save(out, "kbs_props.json")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    {"pilot": pilot, "phase": phase, "dev": dev, "freeze": freeze, "confirm": confirm, "robust": robust, "props": props}[sys.argv[1]]()
