# -*- coding: utf-8 -*-
"""Benchmark harness: development cohort, held-out cohort, ablations, baselines, spatial-coherence and region-level metrics.

Two synthetic cohorts (NOT kidney tissue, authored by the same person as the method):
    dev   the eight slides used while the method was developed (validate.SPECS)
    held  sixteen slides with three anomaly types that were never used in development, a different prevalence (about 1% of the patches), stain / scanner shifts and different artifacts.
          It is evaluated ONCE, with settings frozen in results/frozen_settings.json (its SHA-256 is written into the held-out result).

Every method is scored on the same patches and the same embeddings:
    axes and ablations   appearance rarity (1% scale / multi-scale), state-wise disorder, contact rarity, all two- and three-axis combinations, optional lattice smoothing
    baselines            Isolation Forest, Local Outlier Factor, k-nearest-neighbour distance (k = 10), Mahalanobis distance, local disorder D alone
"""
import hashlib
import json
import os
import time

import cv2
import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

from . import cohort as C
from . import synthetic as S
from . import wsi
from .encoder import get_encoder
from .pipeline import RunConfig, load_slide
from .validate import SPECS as DEV_SPECS

ART = ("blur", "pen", "fold")
DEV_ANOM = ("disorganized", "novel_state")
HELD_ANOM = ("dense_tubules", "nuclear_crowding", "glomerular_cluster")
HELD_SPECS = [   # (seed, anomalies, artifacts, palette_shift, soft_blur)
    (301, ("dense_tubules",), ("blur",), 0.0, 0.8),
    (302, ("nuclear_crowding",), ("pen",), 0.6, 0.8),
    (303, ("glomerular_cluster",), (), -0.6, 0.8),
    (304, ("dense_tubules", "nuclear_crowding"), ("fold",), 0.3, 0.8),
    (305, ("glomerular_cluster",), ("blur", "fold"), 0.0, 1.2),
    (306, ("nuclear_crowding",), (), 0.6, 1.2),
    (307, (), (), 0.0, 0.8), (308, (), ("pen",), 0.6, 0.8), (309, (), (), -0.6, 0.8), (310, (), ("blur",), 0.3, 1.2),
    (311, (), (), 0.0, 0.8), (312, (), ("fold",), 0.6, 0.8), (313, (), (), -0.6, 1.2), (314, (), ("blur", "pen"), 0.0, 0.8),
    (315, (), (), 0.3, 0.8), (316, (), ("pen", "fold"), -0.3, 1.2),
]
SCALES = (0.005, 0.01, 0.02, 0.05, 0.10)


# ------------------------------------------------------------------------------------------------ cohorts on disk
def build_cohort(kind, root, size=4480, log=print):
    """write the slides of a cohort once and keep downsampled truth masks beside them; returns the list of slide paths"""
    d = os.path.join(root, kind)
    os.makedirs(os.path.join(d, "slides"), exist_ok=True)
    os.makedirs(os.path.join(d, "truth"), exist_ok=True)
    paths = []
    specs = [(s, a, r, 0.0, 0.8) for s, a, r in DEV_SPECS] if kind == "dev" else HELD_SPECS
    radius = int(size * (0.075 if kind == "dev" else 0.065))
    for seed, an, ar, shift, soft in specs:
        name = "%s_%d" % (kind, seed)
        p = os.path.join(d, "slides", name + ".tif")
        t = os.path.join(d, "truth", name + ".npz")
        if not (os.path.exists(p) and os.path.exists(t)):
            img, tr = S.make_slide(seed=seed, size=size, anomalies=an, artifacts=ar, anomaly_radius=radius, palette_shift=shift, soft_blur=soft)
            S.write_tiff(p, img, 0.5)
            small = {k: cv2.resize(v.astype(np.uint8), (560, 560), interpolation=cv2.INTER_AREA) for k, v in tr.items() if k in ("tissue",) + DEV_ANOM + HELD_ANOM + ART}
            np.savez_compressed(t, **small)
            log("[bench] wrote %s anomalies=%s artifacts=%s" % (name, an, ar))
        paths.append(p)
    return paths


def load_cohort(kind, root, encoder, cfg=None, log=print):
    """SlideData for every slide (embeddings cached under root/<kind>/cache) and the per-patch truth overlap arrays"""
    cfg = cfg or RunConfig(encoder=encoder.name if hasattr(encoder, "name") else "auto", figures=False)
    paths = build_cohort(kind, root, log=log)
    out = os.path.join(root, kind, "cache_%s_%s" % (encoder.name, ("mpp%g_px%d" % (cfg.mpp, cfg.patch_px))))
    slides, truth = [], []
    for p in paths:
        s = load_slide(p, cfg, encoder, out)
        z = np.load(os.path.join(root, kind, "truth", os.path.splitext(os.path.basename(p))[0] + ".npz"))
        t = {}
        for k in z.files:
            if k == "tissue":
                continue
            fr = cv2.resize(z[k].astype(np.float32), (s.shape[1], s.shape[0]), interpolation=cv2.INTER_AREA)
            t[k] = fr[s.rows, s.cols]
        slides.append(s)
        truth.append(t)
    return slides, truth


# ------------------------------------------------------------------------------------------------ scores
def or_combine(pe, ph, pa, w):
    w = np.asarray(w, float) / np.sum(w)
    return 1.0 - ((1 - pe) ** w[0]) * ((1 - np.nan_to_num(ph, nan=0.5)) ** w[1]) * ((1 - np.nan_to_num(pa, nan=0.5)) ** w[2])


def all_scores(slides, weights=(0.4, 0.4, 0.2), seed=0, n_components=16, n_states=12):
    """dict method -> list of per-slide per-patch score arrays (NaN for artifact patches).  One cohort analysis computes every axis and every rarity scale;
    the OR combinations below use the axes with the stated weights (the frozen MyUncleVision score is 'appearance, multi-scale, lattice-smoothed')."""
    for sl in slides:
        sl.extra = {k: v for k, v in sl.extra.items() if k in ("path", "backend", "dimensions", "mpp_assumed", "mpp_eff")}
    C.analyse(slides, C.Config(seed=seed, weights=weights, n_components=n_components, n_states=n_states))
    msk = lambda sc, i: np.where(slides[i].ok, sc, np.nan)
    ex = lambda key: [sl.extra[key] for sl in slides]
    pe1, pems, ph, pa, Dd = ex("p_emb"), ex("p_emb_ms"), ex("p_het"), ex("p_adj"), ex("D")
    comb = lambda pe_list, w: [msk(or_combine(pe_list[i], ph[i], pa[i], w), i) for i in range(len(slides))]
    sm = lambda arrs: [C.smooth_lattice(sl, a) for sl, a in zip(slides, arrs)]
    embms = [msk(p, i) for i, p in enumerate(pems)]
    full_ms = comb(pems, weights)
    M = {
        "MyUncleVision frozen: appearance, multi-scale, lattice-smoothed": sm(embms),
        "appearance, 1% scale": [msk(p, i) for i, p in enumerate(pe1)],
        "appearance, multi-scale": embms,
        "state-wise disorder": [msk(p, i) for i, p in enumerate(ph)],
        "contact rarity": [msk(p, i) for i, p in enumerate(pa)],
        "three-axis OR, 1% scale (original default)": comb(pe1, (0.4, 0.4, 0.2)),
        "appearance + disorder": comb(pems, (0.5, 0.5, 0.0)),
        "appearance + contact": comb(pems, (0.67, 0.0, 0.33)),
        "disorder + contact": comb(pems, (0.0, 0.67, 0.33)),
        "three-axis OR, multi-scale": full_ms,
        "three-axis OR, multi-scale, lattice-smoothed": sm(full_ms),
        "local disorder D alone": [msk(p, i) for i, p in enumerate(Dd)],
    }
    # baselines on the same unit PCA embeddings of the non-artifact patches, raw and lattice-smoothed
    ok_l = [sl.ok for sl in slides]
    U = np.concatenate([sl.extra["U"][o] for sl, o in zip(slides, ok_l)])
    sid = np.concatenate([np.full(int(o.sum()), i) for i, o in enumerate(ok_l)])
    iso = -IsolationForest(n_estimators=200, random_state=seed).fit(U).score_samples(U)
    lof = -LocalOutlierFactor(n_neighbors=20).fit(U).negative_outlier_factor_
    mu = U.mean(0)
    cov = np.cov(U.T) + 1e-6 * np.eye(U.shape[1])
    maha = np.einsum("ij,jk,ik->i", U - mu, np.linalg.inv(cov), U - mu)
    from sklearn.neighbors import NearestNeighbors
    dist, _ = NearestNeighbors(n_neighbors=11).fit(U).kneighbors(U)
    knn = dist[:, 1:].mean(1)
    for name, vec in (("Isolation Forest", iso), ("Local Outlier Factor", lof), ("Mahalanobis distance", maha), ("kNN distance, k = 10", knn)):
        arrs = []
        for i, o in enumerate(ok_l):
            x = np.full(len(o), np.nan)
            x[o] = vec[sid == i]
            arrs.append(x)
        M[name] = arrs
        # same processing as the MyUncleVision score: cohort percentile rank, then lattice smoothing (apples to apples); the raw-score smoothing is reported separately
        allv = np.concatenate([x[np.isfinite(x)] for x in arrs])
        ranked = [np.where(np.isfinite(x), C._rank(x, allv), np.nan) for x in arrs]
        M[name + ", lattice-smoothed (rank)"] = sm(ranked)
        if name in ("Isolation Forest", "Mahalanobis distance", "kNN distance, k = 10"):
            M[name + ", lattice-smoothed (raw score)"] = sm(arrs)
    return M


# ------------------------------------------------------------------------------------------------ metrics
def _components(slide, mask_patch, min_size=3):
    g = np.zeros(slide.shape, bool)
    g[slide.rows, slide.cols] = mask_patch
    lab, n = ndi.label(g, structure=np.ones((3, 3), int))
    sizes = ndi.sum(g, lab, range(1, n + 1)) if n else []
    return lab, n, np.asarray(sizes)


def evaluate(slides, truth, scores, anom, q=0.03, boot=200, seed=0):
    ok = np.concatenate([s.ok for s in slides])
    sc = np.concatenate(scores)
    T = lambda k: np.concatenate([t.get(k, np.zeros(len(s.rows))) for s, t in zip(slides, truth)])
    an_any = np.max([T(a) for a in anom], axis=0) >= 0.5
    art = np.max([T(a) for a in ART], axis=0) >= 0.25
    clean = (np.max([T(a) for a in anom], axis=0) < 0.05) & (np.max([T(a) for a in ART], axis=0) < 0.05)
    sel = ok & np.isfinite(sc) & (an_any | clean) & ~art
    y = an_any[sel].astype(int)
    out = dict(n_normal=int((sel & clean).sum()), n_anomalous=int((sel & an_any).sum()))
    out["auroc"] = float(roc_auc_score(y, sc[sel])) if 0 < y.sum() < len(y) else float("nan")
    out["average_precision"] = float(average_precision_score(y, sc[sel])) if y.sum() else float("nan")
    out["auroc_by_type"] = {}
    for a in anom:
        pos = T(a) >= 0.5
        s2 = ok & np.isfinite(sc) & ((pos) | clean) & ~art
        yy = pos[s2].astype(int)
        out["auroc_by_type"][a] = float(roc_auc_score(yy, sc[s2])) if 0 < yy.sum() < len(yy) else float("nan")
    # slide bootstrap
    sid = np.concatenate([np.full(s.n, i) for i, s in enumerate(slides)])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        pick = rng.integers(0, len(slides), len(slides))
        idx = np.concatenate([np.where((sid == p) & sel)[0] for p in pick])
        yy = an_any[idx].astype(int)
        if 0 < yy.sum() < len(yy):
            bs.append(roc_auc_score(yy, sc[idx]))
    out["auroc_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None
    # top-q: localization, coherence, region recall, false regions
    cand = np.zeros(len(sc), bool)
    valid = ok & np.isfinite(sc)
    k = max(1, int(round(q * valid.sum())))
    thr = np.sort(sc[valid])[::-1][k - 1]
    cand[valid] = sc[valid] >= thr
    off = 0
    near, near_all, coher, hits, instances, false_regions, comps_total = [], [], [], [], 0, 0, 0
    for i, (s, t) in enumerate(zip(slides, truth)):
        n = s.n
        c = cand[off:off + n]
        anm = np.max([t.get(a, np.zeros(n)) for a in anom], axis=0) >= 0.5
        g = np.zeros(s.shape, bool)
        g[s.rows, s.cols] = anm
        gd = ndi.binary_dilation(g, structure=np.ones((3, 3), bool))
        near.append(gd[s.rows, s.cols][c])
        near_all.append(gd[s.rows, s.cols][ok[off:off + n] & np.isfinite(sc[off:off + n])])
        lab, nc, sizes = _components(s, c)
        big = np.zeros(s.shape, bool)
        for j in range(1, nc + 1):
            if sizes[j - 1] >= 3:
                big |= lab == j
        coher.append(big[s.rows, s.cols][c])
        artm = np.max([t.get(a, np.zeros(n)) for a in ART], axis=0) > 0.05
        artg = np.zeros(s.shape, bool)
        artg[s.rows, s.cols] = artm
        artd = ndi.binary_dilation(artg, structure=np.ones((3, 3), bool))
        for j in range(1, nc + 1):
            if sizes[j - 1] >= 3:
                comp = lab == j
                comps_total += 1
                if not (comp & gd).any() and not (comp & artd).any():
                    false_regions += 1
        for a in anom:
            m = t.get(a, np.zeros(n)) >= 0.5
            if m.sum() >= 3:
                instances += 1
                hits.append(bool(c[m].any()))
        off += n
    near, near_all, coher = np.concatenate(near), np.concatenate(near_all), np.concatenate(coher)
    out.update(top_q=q, top_k=int(k), precision_within_one_patch=float(near.mean()) if len(near) else float("nan"), base_rate_within_one_patch=float(near_all.mean()) if len(near_all) else float("nan"),
               spatial_coherence=float(coher.mean()) if len(coher) else float("nan"), region_recall=float(np.mean(hits)) if hits else float("nan"), n_instances=int(instances),
               false_regions=int(false_regions), regions_total=int(comps_total), false_regions_per_slide=float(false_regions) / max(len(slides), 1))
    return out


def run_bench(kind, root, encoder_name, outdir, weights=(0.4, 0.4, 0.2), q=0.03, seed=0, log=print):
    t0 = time.time()
    enc = get_encoder(encoder_name, "cpu")
    slides, truth = load_cohort(kind, root, enc, log=log)
    M = all_scores(slides, weights=weights, seed=seed)
    anom = DEV_ANOM if kind == "dev" else HELD_ANOM
    res = {name: evaluate(slides, truth, sc, anom, q=q) for name, sc in M.items()}
    prev = float(np.mean(np.concatenate([np.max([t.get(a, np.zeros(len(s.rows))) for a in anom], axis=0) >= 0.5 for s, t in zip(slides, truth)])))
    out = dict(kind=kind, encoder=enc.name, n_slides=len(slides), n_patches=int(sum(s.n for s in slides)), anomalous_patch_share=prev, q=q, weights=list(weights), seconds=round(time.time() - t0, 1), methods=res)
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, "bench_%s_%s.json" % (kind, enc.name)), "w"), indent=1, default=float)
    return out


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------ robustness
def jaccard_top(a_list, b_list, q):
    """Jaccard overlap of the top-q patch sets of two score lists (same patches); the stability of what the method would flag"""
    a, b = np.concatenate(a_list), np.concatenate(b_list)
    v = np.isfinite(a) & np.isfinite(b)
    k = max(1, int(round(q * v.sum())))
    ta = set(np.argsort(-np.where(v, a, -np.inf))[:k])
    tb = set(np.argsort(-np.where(v, b, -np.inf))[:k])
    return len(ta & tb) / float(len(ta | tb))


def final_score(slides, weights=(1.0, 0.0, 0.0), seed=0, n_components=16, n_states=12, smooth=True):
    """the MyUncleVision score of one configuration (one cohort analysis)"""
    for sl in slides:
        sl.extra = {k: v for k, v in sl.extra.items() if k in ("path", "backend", "dimensions", "mpp_assumed", "mpp_eff")}
    C.analyse(slides, C.Config(seed=seed, weights=weights, n_components=n_components, n_states=n_states, smooth=smooth))
    return [sl.extra["novelty"].copy() for sl in slides]


def robustness(kind, root, encoder_name, outdir, q=0.03, log=print):
    """stability of the frozen method under seeds, embedding dimension, number of states, axis weights, no smoothing and patch size: Jaccard of the top-q set and the metrics against the default"""
    enc = get_encoder(encoder_name, "cpu")
    anom = DEV_ANOM if kind == "dev" else HELD_ANOM
    slides, truth = load_cohort(kind, root, enc, log=log)
    keys = ("auroc", "average_precision", "spatial_coherence", "region_recall")
    base = final_score(slides)
    rows = [dict(variant="default (frozen)", jaccard=1.0, **{k: evaluate(slides, truth, base, anom, q, boot=0)[k] for k in keys})]
    variants = [("seed 1", dict(seed=1)), ("seed 2", dict(seed=2)), ("seed 3", dict(seed=3)), ("8 PCA components", dict(n_components=8)), ("32 PCA components", dict(n_components=32)),
                ("6 states", dict(n_states=6)), ("24 states", dict(n_states=24)), ("no lattice smoothing", dict(smooth=False)), ("three-axis weights 0.4/0.4/0.2", dict(weights=(0.4, 0.4, 0.2))), ("equal axis weights", dict(weights=(1, 1, 1)))]
    for name, kw in variants:
        sc = final_score(slides, **kw)
        m = evaluate(slides, truth, sc, anom, q, boot=0)
        rows.append(dict(variant=name, jaccard=jaccard_top(base, sc, q), **{k: m[k] for k in keys}))
        log("[robust] %s %s: jaccard %.2f auroc %.3f" % (enc.name, name, rows[-1]["jaccard"], m["auroc"]))
    # coarser patches (0.75 um/px): another grid, the truth is recomputed from the saved masks
    cfg2 = RunConfig(encoder=encoder_name, mpp=0.75, figures=False)
    sl2, tr2 = load_cohort(kind, root, enc, cfg2, log=log)
    m2 = evaluate(sl2, tr2, final_score(sl2), anom, q, boot=0)
    rows.append(dict(variant="coarser patches (0.75 um/px)", jaccard=float("nan"), **{k: m2[k] for k in keys}))
    out = dict(kind=kind, encoder=enc.name, q=q, rows=rows)
    json.dump(out, open(os.path.join(outdir, "robustness_%s_%s.json" % (kind, enc.name)), "w"), indent=1, default=float)
    return out
