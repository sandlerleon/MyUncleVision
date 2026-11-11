# -*- coding: utf-8 -*-
"""Local order and local manifold statistics of a lattice of unit embeddings (label-free): detectors of spatial ARRANGEMENT anomalies, which the rarity score cannot see because they keep the marginal
distribution of the embeddings.

    disorder_map(U, valid)                 D_i = 1 - |mean of the four neighbors' unit vectors|  (MyUncle's local disorder; its rank is the rank of H_chi)
    manifold_inconsistency(U, valid, ...)  e_i = |(I - P_i)(u_i - mu_i)|^2 with mu_i the mean of the window neighbors of i (excluding i) and P_i the projector on their m leading principal directions
    rank_field(x, valid)                   cohort percentile of a score field
"""
import numpy as np

from . import lattice as lat


def rank_field(x, valid):
    out = np.full(x.shape, np.nan)
    v = valid & np.isfinite(x)
    vals = np.sort(x[v])
    out[v] = np.searchsorted(vals, x[v], side="right") / max(len(vals), 1)
    return out


def disorder_map(U, valid):
    return lat.local_order_maps(U, valid, neighbors=4, min_neighbors=2)["D"]


def manifold_inconsistency(U, valid, window=5, m=4):
    """squared distance of every site's embedding from the affine local tangent subspace estimated from its neighbors (leave-one-out)"""
    H, W, d = U.shape
    h = window // 2
    out = np.full((H, W), np.nan)
    for y in range(H):
        for x in range(W):
            if not valid[y, x]:
                continue
            ys, xs = np.mgrid[max(0, y - h):min(H, y + h + 1), max(0, x - h):min(W, x + h + 1)]
            ys, xs = ys.ravel(), xs.ravel()
            keep = valid[ys, xs] & ~((ys == y) & (xs == x))
            if keep.sum() < m + 2:
                continue
            Z = U[ys[keep], xs[keep]]
            mu = Z.mean(0)
            _, _, Vt = np.linalg.svd(Z - mu, full_matrices=False)
            P = Vt[:m]
            r = U[y, x] - mu
            out[y, x] = float(r @ r - np.sum((P @ r) ** 2))
    return out
