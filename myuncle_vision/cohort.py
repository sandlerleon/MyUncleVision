# -*- coding: utf-8 -*-
"""Cohort analysis: embeddings -> sphere -> tissue lattice -> order, entropy, transition and novelty scores.

Per patch, three axes (all rank-based inside the cohort, 0 to 1, high = unusual) and their weighted probabilistic OR, the novelty score 1 - prod (1 - p_axis)^w_axis (default weights: appearance only, multi-scale, lattice-smoothed; selected on the development cohort):
    emb   how rare the patch's appearance is: mean cosine distance to its k nearest reference patches, with k = 1% of the reference set (so a state that recurs but covers
          under 1% of the cohort is still rare), leaving out the same slide's neighbourhood
    het   how disordered its neighbourhood is FOR ITS TISSUE STATE: MyUncle local disorder D (1 - resultant length of the neighbours' unit embeddings),
          ranked among patches of the same discrete state, so that routine interfaces between compartments do not count as unusual
    adj   how rare the contacts it makes are: -log of the cohort frequency of (own state, neighbour state) pairs, averaged over its neighbours
Artifact patches (QC) are excluded everywhere: they are not part of the lattice, the reference or the ranks.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from . import lattice as lat


@dataclass
class SlideData:
    name: str
    shape: tuple                      # (nrows, ncols) of the patch grid
    rows: np.ndarray
    cols: np.ndarray
    x0: np.ndarray
    y0: np.ndarray
    side0: float
    mpp: float
    emb: np.ndarray                   # (N, D) raw embeddings
    qc: dict                          # arrays: blur, pen, dark, white, artifact (bool), reason
    stain: dict                       # arrays: h_mean, e_mean, nuc_frac, od_sd
    thumb: np.ndarray = None
    mask: np.ndarray = None
    scale: float = 1.0
    extra: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.rows)

    @property
    def ok(self):
        return ~np.asarray(self.qc["artifact"], bool)


@dataclass
class Config:
    n_components: int = 16
    n_states: int = 12
    neighbors: int = 4
    k_nn: int = 10                            # minimum number of neighbours
    rarity_fraction: float = 0.01             # novelty is judged at the scale of this fraction of the reference set: states rarer than that are 'novel' even if they recur
    exclude_radius: int = 6
    ref_per_slide: int = 4000
    ref_max: int = 40000
    weights: tuple = (1.0, 0.0, 0.0)          # emb, het, adj.  FROZEN default: the appearance axis alone (selected on the development cohort, results/frozen_settings.json); the other axes are computed as diagnostics
    rarity_scales: tuple = (0.005, 0.01, 0.02, 0.05, 0.10)   # appearance rarity is judged at all of these scales and the highest rank counts (the prevalence of an interesting state is not known in advance)
    multiscale: bool = True
    scorer: str = "knn"                        # appearance scorer feeding the novelty score: 'knn' (multi-scale nearest-neighbour rarity, frozen candidate A) or 'mahalanobis' (frozen candidate B)
    null_permutations: int = 8                # spatial-shuffle null of the slide-level order statistic (0 = skip; the search scenarios do not need it)
    smooth_window: int = 3                    # lattice smoothing window (1 = none); 3 is the frozen default
    smooth_type: str = "mean"                  # 'mean' (frozen default), 'gaussian' or 'median'
    smooth: bool = True                       # 3 x 3 lattice smoothing of the novelty score: spatially coherent regions are boosted, isolated patches suppressed
    aggregation: str = "mean"                 # 'mean': the fixed window above (the frozen default of the earlier versions); 'simw': similarity-weighted aggregation; 'two_channel': similarity-weighted
                                              # rarity (channel A) or the directional local disorder (channel B), selected per cohort by a label-free tail index (two_channel.py)
    seed: int = 0


def fit_space(slides, cfg):
    """PCA (cohort-centred) on the non-artifact embeddings and k-means prototype states.  Returns dict(mu, V, centers)"""
    rng = np.random.default_rng(cfg.seed)
    X = []
    for s in slides:
        e = s.emb[s.ok]
        if len(e) > cfg.ref_per_slide:
            e = e[rng.choice(len(e), cfg.ref_per_slide, replace=False)]
        X.append(e)
    X = np.concatenate(X).astype(np.float64)
    d = int(min(cfg.n_components, X.shape[1], max(2, len(X) - 1)))
    pca = PCA(n_components=d, random_state=cfg.seed).fit(X)
    U = lat.to_unit(pca.transform(X))
    km = MiniBatchKMeans(n_clusters=int(min(cfg.n_states, len(U))), random_state=cfg.seed, n_init=3, batch_size=2048).fit(U)
    return dict(pca=pca, km=km, d=d, K=km.n_clusters)


def project(space, s):
    """unit embeddings (N, d) and prototype states (N,) of one slide"""
    U = lat.to_unit(space["pca"].transform(s.emb.astype(np.float64)))
    return U, space["km"].predict(U)


def _knn_novelty(Uq, sq, posq, ref, k, excl, which=None, rarity=0.0):
    """mean cosine distance to the k nearest reference patches that are not within `excl` lattice steps on the same slide"""
    R, Rs, Rp = ref["U"], ref["slide"], ref["pos"]
    if which is not None:
        R, Rs, Rp = R[which], Rs[which], Rp[which]
    k = int(max(k, round(rarity * len(R))))
    kq = int(min(len(R), k + 6 * (2 * excl + 1) ** 2 // 4 + 20))
    nn = NearestNeighbors(n_neighbors=kq, algorithm="auto").fit(R)
    out = np.empty(len(Uq))
    for lo in range(0, len(Uq), 4000):                                  # chunked: k can be in the hundreds for a large reference set
        hi = min(len(Uq), lo + 4000)
        dist, idx = nn.kneighbors(Uq[lo:hi])
        near = (Rs[idx] == sq[lo:hi, None]) & (np.maximum(np.abs(Rp[idx][..., 0] - posq[lo:hi, None, 0]), np.abs(Rp[idx][..., 1] - posq[lo:hi, None, 1])) <= excl)
        dist = np.where(near, np.inf, dist)
        dist.sort(axis=1)
        top = dist[:, :k]
        fin = np.isfinite(top)
        cos = np.where(fin, top ** 2 / 2.0, 0.0)
        cnt = fin.sum(1)
        out[lo:hi] = np.where(cnt > 0, cos.sum(1) / np.maximum(cnt, 1), np.nan)
    return out


def _knn_multi(Uq, sq, posq, ref, scales, k_min, excl):
    """mean cosine distance to the k nearest reference patches for several k at once (k = fraction of the reference set, at least k_min), same-slide neighbours within `excl` lattice steps excluded.
    Returns (Q, len(scales))"""
    R, Rs, Rp = ref["U"], ref["slide"], ref["pos"]
    ks = [int(min(len(R), max(k_min, round(f * len(R))))) for f in scales]
    kq = int(min(len(R), max(ks) + 6 * (2 * excl + 1) ** 2 // 4 + 20))
    nn = NearestNeighbors(n_neighbors=kq, algorithm="auto").fit(R)
    out = np.empty((len(Uq), len(ks)))
    for lo in range(0, len(Uq), 1000):
        hi = min(len(Uq), lo + 1000)
        dist, idx = nn.kneighbors(Uq[lo:hi])
        near = (Rs[idx] == sq[lo:hi, None]) & (np.maximum(np.abs(Rp[idx][..., 0] - posq[lo:hi, None, 0]), np.abs(Rp[idx][..., 1] - posq[lo:hi, None, 1])) <= excl)
        dist = np.where(near, np.inf, dist)
        dist.sort(axis=1)
        fin = np.isfinite(dist)
        cum = np.cumsum(np.where(fin, dist ** 2 / 2.0, 0.0), axis=1)
        cnt = np.cumsum(fin, axis=1)
        for j, k in enumerate(ks):
            out[lo:hi, j] = np.where(cnt[:, k - 1] > 0, cum[:, k - 1] / np.maximum(cnt[:, k - 1], 1), np.nan)
    return out


def smooth_lattice(slide, scores, window=3, kind="mean"):
    """lattice smoothing of per-patch scores over a (window x window) neighborhood, NaN-aware: the mean (frozen default, window 3), Gaussian-weighted mean or median of the valid scores
    in the window of every site; NaN where the patch itself has no score.  window = 1 returns the scores unchanged."""
    if window <= 1:
        return np.asarray(scores, float).copy()
    from scipy.ndimage import uniform_filter, gaussian_filter, generic_filter
    a = np.full(slide.shape, np.nan)
    a[slide.rows, slide.cols] = scores
    w = np.isfinite(a).astype(float)
    if kind == "mean":
        num = uniform_filter(np.nan_to_num(a), window, mode="constant")
        den = uniform_filter(w, window, mode="constant")
        sm = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    elif kind == "gaussian":
        sig = window / 4.0 if window > 3 else 0.75
        tr = (window // 2) / sig
        num = gaussian_filter(np.nan_to_num(a), sig, mode="constant", truncate=tr)
        den = gaussian_filter(w, sig, mode="constant", truncate=tr)
        sm = np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)
    elif kind == "median":
        sm = generic_filter(a, np.nanmedian, size=window, mode="constant", cval=np.nan)
    else:
        raise ValueError("unknown smoothing type " + str(kind))
    return np.where(np.isfinite(scores), sm[slide.rows, slide.cols], np.nan)


def _rank(x, ref):
    """fraction of the reference values below x (empirical percentile, 0..1)"""
    ref = np.sort(ref[np.isfinite(ref)])
    out = np.full(np.shape(x), np.nan)
    ok = np.isfinite(x)
    if len(ref):
        out[ok] = np.searchsorted(ref, x[ok], side="right") / float(len(ref))
    return out


def analyse(slides, cfg=None, space=None):
    """Run the cohort analysis.  Adds to every slide .extra: U, state, maps (lattice arrays), per-patch arrays d_emb, D, H_chi, T_rare, p_emb, p_het, p_adj, novelty, p_emb_a, p_emb_b,
    plus slide-level order statistics and a spatial-shuffle null.  Returns (space, summary dict)."""
    cfg = cfg or Config()
    rng = np.random.default_rng(cfg.seed)
    space = space or fit_space(slides, cfg)
    K, d = space["K"], space["d"]
    # ---- lattices
    A = np.zeros((K, K))
    for i, s in enumerate(slides):
        U, st = project(space, s)
        s.extra["U"], s.extra["state"] = U, st
        nr, nc = s.shape
        valid = np.zeros((nr, nc), bool)
        valid[s.rows[s.ok], s.cols[s.ok]] = True
        L = np.zeros((nr, nc, d))
        L[s.rows, s.cols] = U
        lab = np.zeros((nr, nc), int)
        lab[s.rows, s.cols] = st
        s.extra.update(valid=valid, lattice=L, labels=lab)
        A += lat.adjacency_counts(lab, valid, K, cfg.neighbors)
    P = (A + 1.0) / (A + 1.0).sum(1, keepdims=True)                       # Laplace-smoothed P(neighbour state | own state)
    # ---- per slide maps
    for s in slides:
        valid, L, lab = s.extra["valid"], s.extra["lattice"], s.extra["labels"]
        m = lat.local_order_maps(L, valid, cfg.neighbors)
        m["state_entropy"] = lat.state_entropy(lab, valid, K, 1)
        m["T_rare"] = lat.transition_rarity(lab, valid, P, cfg.neighbors)
        s.extra["maps"] = m
        # spatial-shuffle null for the order of the whole slide: permute the embeddings among the valid sites
        idx = np.argwhere(valid)
        nulls = []
        for _ in range(cfg.null_permutations):
            perm = rng.permutation(len(idx))
            Ln = np.zeros_like(L)
            Ln[idx[:, 0], idx[:, 1]] = L[idx[perm, 0], idx[perm, 1]]
            nulls.append(np.nanmean(lat.local_order_maps(Ln, valid, cfg.neighbors)["Rbar"]))
        obs = float(np.nanmean(m["Rbar"]))
        s.extra["order"] = dict(Rbar_obs=obs, Rbar_null_mean=float(np.mean(nulls)) if nulls else float("nan"), Rbar_null_sd=float(np.std(nulls) + 1e-9) if nulls else float("nan"),
                                z=float((obs - np.mean(nulls)) / (np.std(nulls) + 1e-9)) if nulls else float("nan"), H_chi_mean=float(np.nanmean(m["H_chi"])), D_mean=float(np.nanmean(m["D"])))
    # ---- reference set for the novelty axis
    refU, refS, refP = [], [], []
    for i, s in enumerate(slides):
        sel = np.where(s.ok)[0]
        if len(sel) > cfg.ref_per_slide:
            sel = rng.choice(sel, cfg.ref_per_slide, replace=False)
        refU.append(s.extra["U"][sel])
        refS.append(np.full(len(sel), i))
        refP.append(np.stack([s.rows[sel], s.cols[sel]], 1))
    ref = dict(U=np.concatenate(refU), slide=np.concatenate(refS), pos=np.concatenate(refP))
    if len(ref["U"]) > cfg.ref_max:
        keep = rng.choice(len(ref["U"]), cfg.ref_max, replace=False)
        ref = {k_: v[keep] for k_, v in ref.items()}
    half = rng.random(len(ref["U"])) < 0.5
    scales = tuple(sorted(set(cfg.rarity_scales) | {cfg.rarity_fraction}))
    j1 = scales.index(cfg.rarity_fraction)
    per = []
    for i, s in enumerate(slides):
        ok = s.ok
        pos = np.stack([s.rows, s.cols], 1)
        sq = np.full(s.n, i)
        dms = np.full((s.n, len(scales)), np.nan)
        da = np.full(s.n, np.nan)
        db = np.full(s.n, np.nan)
        if ok.any():
            dms[ok] = _knn_multi(s.extra["U"][ok], sq[ok], pos[ok], ref, scales, cfg.k_nn, cfg.exclude_radius)
            da[ok] = _knn_novelty(s.extra["U"][ok], sq[ok], pos[ok], ref, cfg.k_nn, cfg.exclude_radius, which=np.where(half)[0], rarity=cfg.rarity_fraction)
            db[ok] = _knn_novelty(s.extra["U"][ok], sq[ok], pos[ok], ref, cfg.k_nn, cfg.exclude_radius, which=np.where(~half)[0], rarity=cfg.rarity_fraction)
        m = s.extra["maps"]
        s.extra["d_emb_by_scale"] = dms
        s.extra["d_emb"], s.extra["d_emb_a"], s.extra["d_emb_b"] = dms[:, j1], da, db
        s.extra["D"] = np.where(ok, m["D"][s.rows, s.cols], np.nan)
        s.extra["H_chi"] = np.where(ok, m["H_chi"][s.rows, s.cols], np.nan)
        s.extra["S_state"] = np.where(ok, m["state_entropy"][s.rows, s.cols], np.nan)
        s.extra["T_rare"] = np.where(ok, m["T_rare"][s.rows, s.cols], np.nan)
        per.append(s)
    # ---- cohort ranks
    allv = lambda key: np.concatenate([s.extra[key] for s in per])
    ref_by_scale = [np.concatenate([s.extra["d_emb_by_scale"][:, j] for s in per]) for j in range(len(scales))]
    ref_ema, ref_emb_b, ref_adj, Dall = allv("d_emb_a"), allv("d_emb_b"), allv("T_rare"), allv("D")
    Sall = np.concatenate([s.extra["state"] for s in per])
    w = np.asarray(cfg.weights, float) / np.sum(cfg.weights)
    Uok = np.concatenate([s.extra["U"][s.ok] for s in per])
    mu_, cov_ = Uok.mean(0), np.cov(Uok.T) + 1e-6 * np.eye(Uok.shape[1])
    inv_ = np.linalg.inv(cov_)
    d2 = [np.where(s.ok, np.einsum("ij,jk,ik->i", s.extra["U"] - mu_, inv_, s.extra["U"] - mu_), np.nan) for s in per]
    d2_ref = np.concatenate(d2)
    for s, x in zip(per, d2):
        s.extra["p_maha"] = _rank(x, d2_ref)
    for s in per:
        pes = np.stack([_rank(s.extra["d_emb_by_scale"][:, j], ref_by_scale[j]) for j in range(len(scales))], 1)
        s.extra["p_emb_by_scale"] = dict(zip(scales, pes.T))
        s.extra["p_emb"] = pes[:, j1]
        allnan = np.all(np.isnan(pes), axis=1)
        s.extra["p_emb_ms"] = np.where(allnan, np.nan, np.nanmax(np.where(np.isnan(pes), -1.0, pes), axis=1))
        s.extra["p_emb_a"] = _rank(s.extra["d_emb_a"], ref_ema)
        s.extra["p_emb_b"] = _rank(s.extra["d_emb_b"], ref_emb_b)
        ph = np.full(s.n, np.nan)
        for k in range(K):
            sel = s.extra["state"] == k
            ref_k = Dall[(Sall == k)]
            if sel.any():
                ph[sel] = _rank(s.extra["D"][sel], ref_k)
        s.extra["p_het"] = ph
        s.extra["p_adj"] = _rank(s.extra["T_rare"], ref_adj)
        pe = (s.extra["p_emb_ms"] if cfg.multiscale else s.extra["p_emb"]) if cfg.scorer == "knn" else s.extra["p_maha"]
        ph_, pa = np.nan_to_num(s.extra["p_het"], nan=0.5), np.nan_to_num(s.extra["p_adj"], nan=0.5)
        nov = 1.0 - ((1.0 - pe) ** w[0]) * ((1.0 - ph_) ** w[1]) * ((1.0 - pa) ** w[2])        # weighted probabilistic OR of the axes (the default weights keep the appearance axis only)
        nov[~s.ok] = np.nan
        s.extra["novelty_raw"] = nov
        s.extra["novelty"] = smooth_lattice(s, nov, cfg.smooth_window, cfg.smooth_type) if cfg.smooth else nov
    summary = dict(K=K, d=d, adjacency=A.tolist(), pair_prob=P.tolist(), n_reference=int(len(ref["U"])), weights=w.tolist(), aggregation=cfg.aggregation)
    if cfg.aggregation != "mean":
        if cfg.aggregation not in ("simw", "two_channel"):
            raise ValueError("aggregation must be 'mean', 'simw' or 'two_channel'")
        from .two_channel import aggregate_slides
        summary["two_channel"] = aggregate_slides(slides, cfg.aggregation, cfg.smooth_window if cfg.smooth_window > 1 else 3)
    return space, summary
