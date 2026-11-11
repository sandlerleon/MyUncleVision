# -*- coding: utf-8 -*-
"""Controlled study of lattice smoothing against anomaly morphology, on an independent synthetic generator (embedding lattices, neither tissue images nor MyUncle simulations).

Everything but the spatial size of the anomalies is held constant: the prevalence of anomalous sites (3%), their feature separability (a fixed shift of the embedding, in units of the noise standard
deviation), the background (smooth spatial domains with different embedding means), the lattice size and the number of images.  The anomalies are placed as b x b blocks: b = 1 is scattered single
sites, larger b is compact regions.  Compared: no smoothing, the frozen configuration (3 x 3 mean), a 5 x 5 mean, and the champion configurations found by the GaaR search (gaar_opt).

    python -m myuncle_vision.morphology --out results/morphology_study.json
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score

from . import cohort as C
from . import gaar_opt as GO

SIZES = (1, 2, 3, 4, 6)
GEN = dict(d=16, shift=10.0, noise=0.3, domain_scale=1.5)       # generator settings (separability calibrated on the unsmoothed score only, see the manuscript)
CONFIGS = {
    "no smoothing": dict(GO.DEFAULT, smooth="none"),
    "frozen (mean 3)": dict(GO.DEFAULT),
    "mean 5": dict(GO.DEFAULT, smooth="mean5"),
    "GaaR restart champion": {"axes": "app+het 50/50", "max_scale": 0.1, "pca": 8, "smooth": "mean3"},
    "exhaustive development optimum": {"axes": "app+het 50/50", "max_scale": 0.1, "pca": 8, "smooth": "gauss5"},
}


def _pixels(rng, shape, size, n, dom):
    """coordinates of one anomaly of the given shape: block (size x size), bar (1 x size, either orientation), irregular (random-walk cluster of `size` sites), boundary (3 x 3 block on a domain boundary)"""
    if shape == "block":
        y, x = rng.integers(0, n - size + 1), rng.integers(0, n - size + 1)
        return [(y + i, x + j) for i in range(size) for j in range(size)]
    if shape == "bar":
        horiz = rng.random() < 0.5
        y, x = (rng.integers(0, n), rng.integers(0, n - size + 1)) if horiz else (rng.integers(0, n - size + 1), rng.integers(0, n))
        return [(y, x + k) for k in range(size)] if horiz else [(y + k, x) for k in range(size)]
    if shape == "irregular":
        cur = {(int(rng.integers(2, n - 2)), int(rng.integers(2, n - 2)))}
        while len(cur) < size:
            y, x = list(cur)[int(rng.integers(0, len(cur)))]
            dy, dx = [(1, 0), (-1, 0), (0, 1), (0, -1)][int(rng.integers(0, 4))]
            if 0 <= y + dy < n and 0 <= x + dx < n:
                cur.add((y + dy, x + dx))
        return list(cur)
    if shape == "boundary":
        edge = np.argwhere((dom != np.roll(dom, -1, 0)) | (dom != np.roll(dom, -1, 1)))
        edge = edge[(edge[:, 0] > 1) & (edge[:, 0] < n - 2) & (edge[:, 1] > 1) & (edge[:, 1] < n - 2)]
        y, x = edge[int(rng.integers(0, len(edge)))]
        return [(y + i, x + j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    raise ValueError(shape)


def make_lattice(seed, n=32, d=16, prevalence=0.03, block=1, shift=3.0, domains=4, noise=1.0, domain_scale=1.5, shape="block"):
    """one embedding lattice: (n, n, d) embeddings and the (n, n) boolean anomaly mask.  `block` is the size of the anomaly (side of a square, length of a bar, number of sites of an irregular cluster)"""
    rng = np.random.default_rng(seed)
    # background: smooth spatial domains (nearest of a few random centers) with different embedding means
    cent = rng.uniform(0, n, (domains, 2))
    means = rng.normal(0, domain_scale, (domains, d))
    yy, xx = np.mgrid[0:n, 0:n]
    dom = np.argmin((yy[..., None] - cent[:, 0]) ** 2 + (xx[..., None] - cent[:, 1]) ** 2, axis=-1)
    X = means[dom] + rng.normal(0, noise, (n, n, d))
    area = {"block": block * block, "bar": block, "irregular": block, "boundary": 9}[shape]
    k = max(1, int(round(prevalence * n * n / area)))
    mask = np.zeros((n, n), bool)
    placed, tries = 0, 0
    while placed < k and tries < 20000:
        tries += 1
        px = _pixels(rng, shape, block, n, dom)
        ys, xs = [p[0] for p in px], [p[1] for p in px]
        if min(ys) < 0 or min(xs) < 0 or max(ys) >= n or max(xs) >= n:
            continue
        if mask[max(0, min(ys) - 1):max(ys) + 2, max(0, min(xs) - 1):max(xs) + 2].any():
            continue
        u = rng.normal(0, 1, d)                          # every anomaly has its own direction: the anomalies are diverse, not one recurring state
        u /= np.linalg.norm(u)
        for y, x in px:
            mask[y, x] = True
            X[y, x] += shift * u
        placed += 1
    return X, mask


def _slide(name, X):
    n = X.shape[0]
    rr, cc = np.divmod(np.arange(n * n), n)
    z = np.zeros(n * n)
    qc = dict(blur=z, pen=z, dark=z, white=z, artifact=np.zeros(n * n, bool), reason=np.array([""] * (n * n)))
    stain = dict(h_mean=z, e_mean=z, nuc_frac=z, od_sd=z)
    return C.SlideData(name=name, shape=(n, n), rows=rr, cols=cc, x0=cc.astype(float), y0=rr.astype(float), side0=1.0, mpp=1.0, emb=X.reshape(n * n, -1).astype(np.float32), qc=qc, stain=stain)


def run_cohort(seed, block, n_images=10, **kw):
    slides, masks = [], []
    for i in range(n_images):
        X, m = make_lattice(seed * 1000 + i, block=block, **kw)
        slides.append(_slide("m%d_%d" % (seed, i), X))
        masks.append(m.reshape(-1).astype(int))
    out = {}
    for name, g in CONFIGS.items():
        for s in slides:
            s.extra = {}
        C.analyse(slides, GO.to_config(g, n_states=6, ref_per_slide=300, ref_max=3000))
        out[name] = [(masks[i], np.asarray(s.extra["novelty"], float)) for i, s in enumerate(slides)]
    return out


def study(out_path, cohorts=5, n_images=10, B=500, seed=0):
    rng = np.random.default_rng(seed)
    res = dict(sizes=list(SIZES), configs=list(CONFIGS), cohorts=cohorts, images_per_cohort=n_images, prevalence=0.03, shift=3.0, lattice=32, dim=16, results={})
    for block in SIZES:
        pooled = {nm: [] for nm in CONFIGS}
        for c in range(cohorts):
            r = run_cohort(1 + c, block, n_images, **GEN)
            for nm in CONFIGS:
                pooled[nm].extend(r[nm])
        ap = lambda parts: float(average_precision_score(np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])))
        entry = {}
        n_img = len(pooled["frozen (mean 3)"])
        draws = [rng.integers(0, n_img, n_img) for _ in range(B)]
        base = {nm: [ap([pooled[nm][i] for i in dr]) for dr in draws] for nm in CONFIGS}
        for nm in CONFIGS:
            a = np.array(base[nm])
            d = a - np.array(base["frozen (mean 3)"])
            entry[nm] = dict(ap=ap(pooled[nm]), ci=[float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))], diff_vs_frozen_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
        res["results"][str(block)] = entry
        print("block", block, {nm: round(entry[nm]["ap"], 3) for nm in CONFIGS}, flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
    return res


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default="results/morphology_study.json")
    ap_.add_argument("--cohorts", type=int, default=5)
    a = ap_.parse_args()
    study(a.out, cohorts=a.cohorts)
