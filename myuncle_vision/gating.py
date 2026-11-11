# -*- coding: utf-8 -*-
"""Spatial aggregation rules for a per-site anomaly score on a lattice of unit embeddings.

The detector first produces a rarity score r_i (the percentile rank of the multi-scale appearance rarity, cohort-wide) for every site.  A *rule* turns r into the final score s on the lattice.
All rules use only the score r and the unit embeddings u (no labels):

    none           s = r                                                   (no spatial information)
    mean           s_i = mean of r over the (w x w) window                 (the frozen configuration for w = 3)
    simw           s_i = sum_j c_ij r_j / sum_j c_ij  over the window      (similarity-weighted: an edge-preserving filter in embedding space)
    og_cos         s_i = (1 - g_i) r_i + g_i * simw_i,  g_i = logistic(alpha (A_i - theta)),  A_i = rank of the mean compatibility of the neighbors of i
    og_disorder    same gate with A_i = rank of the resultant length R_i = 1 - D_i of the neighbors (MyUncle's local order)
    og_vmf         same gate with A_i = rank of the von Mises-Fisher order information H_chi of the neighbors (the d-dimensional entropy quantity of the paper)
    onesided       s_i = r_i + max(0, m_i - r_i)  with m_i = simw over the neighbors only: neighbors can only add evidence, never remove it
                   (an asymmetric rule, as used for a failure mode that has a sign)
    onesided_gate  the one-sided rule inside the og_cos gate
    ngh            nucleation and growth with hysteresis: seeds (rank above a threshold) keep their score; weaker sites next to a seed are raised to gamma * compatibility * the seed's score, never lowered

c_ij = exp(-(1 - u_i.u_j)/tau) with tau = the median of (1 - u_i.u_j) over lattice-adjacent pairs of the cohort (label-free).
Persistence (optional, weight lam): P_i = share of the windows l in {1, 3, 5, 7} for which the rule's score at that window exceeds its cohort top-3% threshold,  S_i = s_i ((1 - lam) + lam P_i).
"""
import numpy as np

from . import lattice as lat

RULES = ("none", "mean", "simw", "og_cos", "og_disorder", "og_vmf", "onesided", "onesided_gate")
EXTRA_RULES = ("ngh",)          # nucleation-and-growth with hysteresis (evaluated separately, see og_extra.py)
WINDOWS = (1, 3, 5, 7)


def _shift(a, dy, dx, fill=np.nan):
    out = np.full_like(a, fill, dtype=float) if a.dtype.kind == "f" else np.zeros_like(a)
    H, W = a.shape[:2]
    ys, yd = slice(max(0, -dy), H - max(0, dy)), slice(max(0, dy), H - max(0, -dy))
    xs, xd = slice(max(0, -dx), W - max(0, dx)), slice(max(0, dx), W - max(0, -dx))
    out[ys, xs] = a[yd, xd]
    return out


def estimate_tau(slides_U, slides_valid):
    """median of 1 - cos over lattice-adjacent valid pairs of the cohort"""
    d = []
    for U, v in zip(slides_U, slides_valid):
        for dy, dx in ((0, 1), (1, 0)):
            Us = _shift(U, dy, dx, 0.0)
            vs = _shift(v.astype(float), dy, dx, 0.0) > 0
            m = v & vs
            d.append(1.0 - np.sum(U[m] * Us[m], axis=-1))
    d = np.concatenate(d) if d else np.array([0.5])
    return float(max(np.median(d), 1e-3))


def _rank(x, valid):
    out = np.full(x.shape, np.nan)
    v = valid & np.isfinite(x)
    vals = np.sort(x[v])
    out[v] = np.searchsorted(vals, x[v], side="right") / max(len(vals), 1)
    return out


def _window_sums(r, U, valid, window, tau, include_self):
    """sum_j c_ij r_j, sum_j c_ij and the mean compatibility over the window around every site (neighbors only if not include_self)"""
    h = window // 2
    num = np.zeros(r.shape)
    den = np.zeros(r.shape)
    n_nb = np.zeros(r.shape)
    comp = np.zeros(r.shape)
    rr = np.where(valid & np.isfinite(r), r, 0.0)
    vv = valid & np.isfinite(r)
    for dy in range(-h, h + 1):
        for dx in range(-h, h + 1):
            if dy == 0 and dx == 0 and not include_self:
                continue
            Us = _shift(U, dy, dx, 0.0)
            rs = _shift(rr, dy, dx, 0.0)
            vs = _shift(vv.astype(float), dy, dx, 0.0) > 0
            cos = np.sum(U * Us, axis=-1)
            c = np.exp(-(1.0 - cos) / tau) * (vs & vv)
            num += c * rs
            den += c
            if not (dy == 0 and dx == 0):
                n_nb += (vs & vv)
                comp += c
    return num, den, np.where(n_nb > 0, comp / np.maximum(n_nb, 1), np.nan)


def _mean_window(r, valid, window):
    h = window // 2
    s = np.zeros(r.shape)
    n = np.zeros(r.shape)
    vv = valid & np.isfinite(r)
    rr = np.where(vv, r, 0.0)
    for dy in range(-h, h + 1):
        for dx in range(-h, h + 1):
            vs = _shift(vv.astype(float), dy, dx, 0.0) > 0
            s += _shift(rr, dy, dx, 0.0) * vs
            n += vs
    return np.where(vv & (n > 0), s / np.maximum(n, 1), np.nan)


def _resultant(U, valid, window):
    h = window // 2
    S = np.zeros(U.shape)
    n = np.zeros(U.shape[:2])
    for dy in range(-h, h + 1):
        for dx in range(-h, h + 1):
            if dy == 0 and dx == 0:
                continue
            vs = _shift(valid.astype(float), dy, dx, 0.0) > 0
            S += _shift(U, dy, dx, 0.0) * vs[..., None]
            n += vs
    R = np.linalg.norm(S / np.maximum(n[..., None], 1), axis=-1)
    return np.where(valid & (n > 0), R, np.nan)


def _nucleation_growth(r, U, valid, window, tau, nuc, gamma):
    """nucleation and growth with hysteresis: a site whose rarity r exceeds the nucleation threshold `nuc` (a cohort rank) is a seed and keeps its score; a weaker site next to a seed is raised to
    gamma * (compatibility with the seed) * (the seed's score), never lowered (growth into compatible neighbors, the two-threshold logic of hysteresis thresholding, with the compatibility of the
    embeddings as the growth probability).  s_i = max(r_i, gamma * max_j c_ij 1[r_j > nuc] r_j)."""
    h = window // 2
    best = np.zeros(r.shape)
    vv = valid & np.isfinite(r)
    rr = np.where(vv, r, 0.0)
    seed = (rr > nuc) & vv
    for dy in range(-h, h + 1):
        for dx in range(-h, h + 1):
            if dy == 0 and dx == 0:
                continue
            Us = _shift(U, dy, dx, 0.0)
            rs = _shift(rr, dy, dx, 0.0)
            ss = _shift(seed.astype(float), dy, dx, 0.0) > 0
            c = np.exp(-(1.0 - np.sum(U * Us, axis=-1)) / tau)
            best = np.maximum(best, c * rs * ss)
    return np.maximum(rr, gamma * best)


def closure_gap(r, valid, window=3, q=0.03):
    """label-free diagnostic from the closure test of a mean-field reduction: aggregating a steep function (here the top-q exceedance) over a window flattens it when the neighborhood is heterogeneous.
    Returns the share of the cohort's top-q sites (by r) that are still in the top-q after the fixed mean over the window ('retention'): near 1 when the high scores are spatially coherent, near 0 when they are isolated spikes."""
    m = _mean_window(r, valid, window)
    ok = valid & np.isfinite(r) & np.isfinite(m)
    qr = np.quantile(r[ok], 1.0 - q)
    qm = np.quantile(m[ok], 1.0 - q)
    top = ok & (r >= qr)
    return float(np.mean(m[top] >= qm)) if top.any() else float("nan")


def apply_rule(r, U, valid, rule, window=3, theta=0.5, alpha=5.0, tau=0.5):
    """final score lattice s (NaN where r is NaN).  r: (H, W) rarity ranks; U: (H, W, d) unit embeddings; valid: (H, W) bool"""
    ok = valid & np.isfinite(r)
    if rule == "none" or window <= 1:
        return np.where(ok, r, np.nan)
    if rule == "mean":
        return np.where(ok, _mean_window(r, valid, window), np.nan)
    if rule == "ngh":
        return np.where(ok, _nucleation_growth(r, U, valid, window, tau, nuc=theta, gamma=alpha), np.nan)
    num_s, den_s, comp = _window_sums(r, U, valid, window, tau, include_self=True)
    simw = np.where(den_s > 0, num_s / np.maximum(den_s, 1e-12), np.nan)
    if rule == "simw":
        return np.where(ok, simw, np.nan)
    num_n, den_n, _ = _window_sums(r, U, valid, window, tau, include_self=False)
    nb = np.where(den_n > 0, num_n / np.maximum(den_n, 1e-12), np.nan)
    if rule == "onesided":
        return np.where(ok, r + np.maximum(0.0, np.nan_to_num(nb, nan=-np.inf) - r).clip(0, None) * np.isfinite(nb), np.nan)
    # gates
    if rule in ("og_cos", "onesided_gate"):
        A = _rank(comp, ok)
    elif rule == "og_disorder":
        A = _rank(_resultant(U, valid, window), ok)
    elif rule == "og_vmf":
        R = _resultant(U, valid, window)
        k = np.full(R.shape, np.nan)
        m = np.isfinite(R)
        d = U.shape[-1]
        k[m] = lat.kappa_from_rbar(R[m], d, iters=40)
        H = np.full(R.shape, np.nan)
        H[m] = lat.order_information(k[m], d)
        A = _rank(H, ok)
    else:
        raise ValueError("unknown rule " + rule)
    g = 1.0 / (1.0 + np.exp(-alpha * (np.nan_to_num(A, nan=0.0) - theta)))
    if rule == "onesided_gate":
        agg = r + np.maximum(0.0, np.nan_to_num(nb, nan=-np.inf) - r).clip(0, None) * np.isfinite(nb)
    else:
        agg = simw
    s = (1.0 - g) * r + g * np.where(np.isfinite(agg), agg, r)
    return np.where(ok, s, np.nan)


def cohort_apply(slides, rule, window=3, theta=0.5, alpha=5.0, lam=0.0, tau=0.5, q=0.03, windows=(1, 3, 5, 7)):
    """apply a rule (and optional persistence) to every slide of a cohort.  slides: list of dict(r, U, valid).  Returns a list of score lattices.
    Persistence thresholds are cohort-wide top-q quantiles of the rule score at each window (label-free)."""
    base = [apply_rule(s["r"], s["U"], s["valid"], rule, window, theta, alpha, tau) for s in slides]
    if lam <= 0:
        return base
    ws = [w for w in windows if w <= window]
    per = {w: [apply_rule(s["r"], s["U"], s["valid"], rule, w, theta, alpha, tau) for s in slides] for w in ws}
    thr = {w: np.nanquantile(np.concatenate([p[np.isfinite(p)] for p in per[w]]), 1.0 - q) for w in ws}
    out = []
    for i in range(len(slides)):
        P = np.mean([(per[w][i] > thr[w]).astype(float) for w in ws], axis=0)
        out.append(base[i] * ((1.0 - lam) + lam * P))
    return out
