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


def make_lattice(seed, n=32, d=16, prevalence=0.03, block=1, shift=3.0, domains=4, noise=1.0, domain_scale=1.5):
    """one embedding lattice: (n, n, d) embeddings and the (n, n) boolean anomaly mask"""
    rng = np.random.default_rng(seed)
    # background: smooth spatial domains (nearest of a few random centers) with different embedding means
    cent = rng.uniform(0, n, (domains, 2))
    means = rng.normal(0, domain_scale, (domains, d))
    yy, xx = np.mgrid[0:n, 0:n]
    dom = np.argmin((yy[..., None] - cent[:, 0]) ** 2 + (xx[..., None] - cent[:, 1]) ** 2, axis=-1)
    X = means[dom] + rng.normal(0, noise, (n, n, d))
    # anomalies: block x block squares, total prevalence fixed
    k = max(1, int(round(prevalence * n * n / (block * block))))
    mask = np.zeros((n, n), bool)
    placed, tries = 0, 0
    while placed < k and tries < 5000:
        tries += 1
        y, x = rng.integers(0, n - block + 1), rng.integers(0, n - block + 1)
        if mask[max(0, y - 1):y + block + 1, max(0, x - 1):x + block + 1].any():
            continue
        mask[y:y + block, x:x + block] = True
        u = rng.normal(0, 1, d)                          # every block has its own direction: the anomalies are diverse, not one recurring state
        u /= np.linalg.norm(u)
        X[y:y + block, x:x + block] += shift * u
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
