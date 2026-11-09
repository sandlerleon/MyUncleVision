# -*- coding: utf-8 -*-
"""Tissue lattice: patch embeddings on a 2-D grid, local order and entropy.

The maintenance lattice of MyUncle has a rotor (angle theta) on every site; its local disorder is D = 1 - |mean of the neighbours' unit vectors|
(MyUncle_core.MC.local_disorder, four neighbours) and its order-entropy bridge is the von Mises entropy S(kappa) with A(kappa) = I1/I0 = U.
Here the rotor becomes the unit-normalized embedding of a tissue patch, a point on the sphere S^(d-1):

    * D   = 1 - ||mean of the neighbours' unit embeddings||     (identical to MyUncle for d = 2 and the same neighbourhood)
    * kappa solves A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa) = mean resultant length   (von Mises-Fisher, d dimensions)
    * S_vmf is the vMF entropy and H_chi = S_unif - S_vmf the information held in the local order (MyUncle's H_chi at d = 2).

These are the same definitions for d = 2, and the tests check that numerically against the vendored core.
"""
import math

import numpy as np
from scipy.special import gammaln, ive
from scipy.ndimage import uniform_filter


# ------------------------------------------------------------------------------------------------ geometry on the sphere
def to_unit(Z, eps=1e-12):
    Z = np.asarray(Z, float)
    return Z / np.maximum(np.linalg.norm(Z, axis=-1, keepdims=True), eps)


def _shift(a, dy, dx, periodic):
    """a shifted so that out[y, x] = a[y + dy, x + dx]; zeros outside the array unless periodic"""
    if periodic:
        return np.roll(np.roll(a, -dy, 0), -dx, 1)
    out = np.zeros_like(a)
    H, W = a.shape[:2]
    ys, yd = (slice(max(0, -dy), H - max(0, dy)), slice(max(0, dy), H - max(0, -dy)))
    xs, xd = (slice(max(0, -dx), W - max(0, dx)), slice(max(0, dx), W - max(0, -dx)))
    out[ys, xs] = a[yd, xd]
    return out


OFFSETS = {4: [(1, 0), (-1, 0), (0, 1), (0, -1)], 8: [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]}
HALF = {4: [(1, 0), (0, 1)], 8: [(1, 0), (0, 1), (1, 1), (1, -1)]}       # one offset of every neighbour pair


def neighbor_mean(U, valid, neighbors=4, periodic=False):
    """mean of the unit vectors of the valid neighbours of every site, and the number of valid neighbours.  U: (H, W, d), valid: (H, W) bool"""
    Uv = np.where(valid[..., None], U, 0.0)
    s = np.zeros_like(Uv)
    n = np.zeros(valid.shape)
    for dy, dx in OFFSETS[neighbors]:
        s += _shift(Uv, dy, dx, periodic)
        n += _shift(valid.astype(float), dy, dx, periodic)
    return s / np.maximum(n[..., None], 1.0), n


# ------------------------------------------------------------------------------------------------ von Mises-Fisher in d dimensions
def _logI(nu, k):
    """log of the modified Bessel function I_nu(k), stable for large k"""
    return np.log(np.maximum(ive(nu, k), 1e-300)) + k


def A_d(k, d):
    """mean resultant length of a vMF distribution with concentration k in d dimensions (d = 2: I1/I0 = MyUncle's Bridge.A)"""
    k = np.asarray(k, float)
    return np.where(k > 0, ive(d / 2.0, k) / np.maximum(ive(d / 2.0 - 1.0, k), 1e-300), 0.0)


def kappa_from_rbar(R, d, hi=1e6, iters=100):
    """invert A_d(kappa) = R by vectorized bisection (R in [0, 1))"""
    R = np.clip(np.asarray(R, float), 0.0, 1.0 - 1e-12)
    lo = np.zeros_like(R)
    up = np.full_like(R, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + up)
        small = A_d(mid, d) < R
        lo = np.where(small, mid, lo)
        up = np.where(small, up, mid)
    return 0.5 * (lo + up)


def entropy_uniform(d):
    """differential entropy of the uniform distribution on S^(d-1)  (d = 2: log 2 pi = MyUncle's Bridge.Smax)"""
    return math.log(2.0) + 0.5 * d * math.log(math.pi) - float(gammaln(d / 2.0))


def vmf_entropy(k, d):
    """differential entropy (nats) of the vMF distribution on S^(d-1);  d = 2 reproduces MyUncle's Bridge.S"""
    k = np.asarray(k, float)
    nu = d / 2.0 - 1.0
    kk = np.maximum(k, 1e-12)
    S = -nu * np.log(kk) + (d / 2.0) * math.log(2.0 * math.pi) + _logI(nu, kk) - kk * A_d(kk, d)
    return np.where(k > 1e-8, S, entropy_uniform(d))


def order_information(k, d):
    """H_chi = S_uniform - S_vmf  (>= 0; d = 2 reproduces MyUncle's Bridge.H)"""
    return entropy_uniform(d) - vmf_entropy(k, d)


# ------------------------------------------------------------------------------------------------ maps
def local_order_maps(U, valid, neighbors=4, min_neighbors=2, periodic=False):
    """local order of the tissue lattice.  Returns dict of (H, W) maps (NaN where a site is not tissue or has fewer than min_neighbors tissue neighbours):
    Rbar (mean resultant length), D (= 1 - Rbar, MyUncle's local disorder), kappa, S_vmf, H_chi, n_nb"""
    d = U.shape[-1]
    m, n = neighbor_mean(U, valid, neighbors, periodic)
    R = np.linalg.norm(m, axis=-1)
    ok = valid & (n >= min_neighbors)
    k = np.full(R.shape, np.nan)
    S = np.full(R.shape, np.nan)
    Hc = np.full(R.shape, np.nan)
    if ok.any():
        k[ok] = kappa_from_rbar(R[ok], d)
        S[ok] = vmf_entropy(k[ok], d)
        Hc[ok] = entropy_uniform(d) - S[ok]
    nan = np.where(ok, 0.0, np.nan)
    return dict(Rbar=R + nan, D=1.0 - R + nan, kappa=k, S_vmf=S, H_chi=Hc, n_nb=n)


def state_entropy(labels, valid, K, win=1):
    """Shannon entropy (nats) of the distribution of discrete tissue states in the (2 win + 1)^2 window around every site (embedding-state analogue of a spin-state entropy)"""
    size = 2 * win + 1
    H = np.zeros(labels.shape)
    cnt = uniform_filter(valid.astype(float), size=size, mode="constant") * size * size
    for k in range(K):
        pk = uniform_filter(((labels == k) & valid).astype(float), size=size, mode="constant") * size * size
        p = pk / np.maximum(cnt, 1.0)
        H -= np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0)
    return np.where(valid, H, np.nan)


def adjacency_counts(labels, valid, K, neighbors=4):
    """symmetric K x K matrix of how often two states are adjacent on the lattice (each unordered pair counted once per adjacent site pair)"""
    M = np.zeros((K, K))
    for dy, dx in HALF[neighbors]:
        a = labels[max(0, -dy):labels.shape[0] - max(0, dy), max(0, -dx):labels.shape[1] - max(0, dx)]
        b = labels[max(0, dy):labels.shape[0] - max(0, -dy), max(0, dx):labels.shape[1] - max(0, -dx)]
        va = valid[max(0, -dy):valid.shape[0] - max(0, dy), max(0, -dx):valid.shape[1] - max(0, dx)]
        vb = valid[max(0, dy):valid.shape[0] - max(0, -dy), max(0, dx):valid.shape[1] - max(0, -dx)]
        ok = va & vb
        np.add.at(M, (a[ok], b[ok]), 1.0)
        np.add.at(M, (b[ok], a[ok]), 1.0)
    return M


def transition_rarity(labels, valid, P_pair, neighbors=4):
    """mean over the valid neighbours of -log P(neighbour state | own state) under the cohort adjacency matrix P_pair (rows sum to 1): high where a site touches states that rarely touch it"""
    s = np.zeros(labels.shape)
    n = np.zeros(labels.shape)
    for dy, dx in OFFSETS[neighbors]:
        lab_n = _shift(labels.astype(float), dy, dx, False).astype(int)
        v_n = _shift(valid.astype(float), dy, dx, False) > 0
        ok = valid & v_n
        p = P_pair[labels, lab_n]
        s += np.where(ok, -np.log(np.maximum(p, 1e-9)), 0.0)
        n += ok
    return np.where(valid & (n > 0), s / np.maximum(n, 1), np.nan)
