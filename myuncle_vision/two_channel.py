# -*- coding: utf-8 -*-
"""Two label-free anomaly channels on a lattice of unit embeddings and their fusion.

Channel A (WHAT is there): the rarity of a site's embedding in the cohort (kNN distance; see cohort.py), aggregated with similarity weights (gating.py).
Channel B (HOW it is ARRANGED): the directional local disorder, the minimum over the four quadrant sub-windows that contain the site of MyUncle's local disorder D = 1 - |mean unit vector|.
    A wall between two ordered domains is ordered on at least one side, so the minimum stays low there; a disorganized region is disordered in every direction.
    Arrangement anomalies (the same embeddings in the wrong places) leave the distribution of the embeddings unchanged, so channel A cannot see them, whatever the aggregation.

Fusion (label-free): each channel's raw score is standardized over the cohort with the median and the MAD (robust z); the tail index of a channel is the 97th percentile of its robust z (1.88 for a
Gaussian bulk without anomalies).  Rules:
    select      one channel per cohort: channel B if its tail index is larger than that of channel A AND larger than `guard` times the Gaussian value (a comparison of two tail indices at the
                Gaussian level carries no information, so channel A is the default); guard = 0 gives the unguarded rule
    tailw       max(w_A a_i, w_B b_i) with a, b the cohort ranks of the two channels and w_c proportional to (tail index)^p
    zmax        max(z_A, z_B) site by site
    rankmax     max(a_i, b_i)

    directional_disorder(U, valid, h=1)     (min, mean) over the quadrants
    center_neighbor_distance(U, valid)      mean cosine distance of a site to its 8 neighbors (a simple baseline for channel B)
    robust_z, cohort_rank, tail_index       cohort statistics of a list of score lattices
    fuse(a, zA, b_raw, rule, p)             fused score lattices of a cohort
    aggregate_slides(slides, mode)          the same for slides analysed by cohort.analyse (Config.aggregation = "simw" or "two_channel")
"""
import numpy as np

from .gating import _shift

T_GAUSS = 1.8808          # 97th percentile of a standard normal
RULES = ("select", "select0", "tailw", "zmax", "rankmax")
GUARD = 2.0


def directional_disorder(U, valid, h=1):
    """minimum and mean over the four quadrant sub-windows ((h + 1) x (h + 1), each containing the site) of the local disorder 1 - |mean unit vector|.  Quadrants with a missing site do not count."""
    H, W, _ = U.shape
    Uv = U * valid[..., None]
    vf = valid.astype(float)
    res = []
    for sy in (-1, 1):
        for sx in (-1, 1):
            S = np.zeros(U.shape)
            n = np.zeros((H, W))
            for a in range(h + 1):
                for b in range(h + 1):
                    S += _shift(Uv, sy * a, sx * b, 0.0)
                    n += _shift(vf, sy * a, sx * b, 0.0)
            R = np.linalg.norm(S / np.maximum(n[..., None], 1), axis=-1)
            res.append(np.where(valid & (n >= (h + 1) ** 2), 1.0 - R, np.nan))
    res = np.stack(res)
    fin = np.isfinite(res)
    anyq = fin.any(0)
    mn = np.where(anyq, np.min(np.where(fin, res, np.inf), 0), np.nan)
    mean = np.where(anyq, np.sum(np.where(fin, res, 0.0), 0) / np.maximum(fin.sum(0), 1), np.nan)
    return mn, mean


def center_neighbor_distance(U, valid):
    s = np.zeros(valid.shape)
    n = np.zeros(valid.shape)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            vs = _shift(valid.astype(float), dy, dx, 0.0) > 0
            s += (1.0 - np.sum(U * _shift(U, dy, dx, 0.0), -1)) * vs
            n += vs
    return np.where(valid & (n > 0), s / np.maximum(n, 1), np.nan)


def _pool(fields):
    return np.concatenate([f[np.isfinite(f)] for f in fields])


def robust_z(fields):
    a = _pool(fields)
    med = np.median(a)
    mad = 1.4826 * np.median(np.abs(a - med)) + 1e-12
    return [(f - med) / mad for f in fields]


def cohort_rank(fields):
    a = np.sort(_pool(fields))
    return [np.where(np.isfinite(f), np.searchsorted(a, f, side="right") / max(len(a), 1), np.nan) for f in fields]


def tail_index(fields, q=0.97):
    return float(np.quantile(_pool(robust_z(fields)), q))


def fuse(a, zA, b_raw, rule="select", p=2.0, q=0.97, guard=GUARD):
    """a: channel-A score lattices of a cohort (any monotone score, e.g. similarity-weighted rarity; ranked here); zA: raw rarity lattices (kNN distances) for the tail index and the z fusion;
    b_raw: channel-B raw lattices (directional disorder).  Sites where channel B is undefined keep the channel-A score.  Returns (scores, info)."""
    ar, br = cohort_rank(a), [np.nan_to_num(x, nan=0.0) for x in cohort_rank(b_raw)]
    za, zb = robust_z(zA), robust_z(b_raw)
    tA, tB = float(np.quantile(_pool(za), q)), float(np.quantile(_pool(zb), q))
    info = dict(tail_A=tA, tail_B=tB)
    keep = lambda s, ref: [np.where(np.isfinite(r), x, np.nan) for x, r in zip(s, ref)]
    if rule in ("select", "select0"):
        useB = tB > tA and tB > (guard if rule == "select" else 0.0) * T_GAUSS
        info["channel"] = "B" if useB else "A"
        return keep(br if useB else ar, ar), info
    if rule == "tailw":
        wa, wb = max(tA, 1e-6) ** p, max(tB, 1e-6) ** p
        wa, wb = wa / (wa + wb), wb / (wa + wb)
        info.update(w_A=wa, w_B=wb)
        return keep([np.fmax(wa * x, wb * y) for x, y in zip(ar, br)], ar), info
    if rule == "zmax":
        return keep([np.fmax(x, np.where(np.isfinite(y), y, -np.inf)) for x, y in zip(za, zb)], ar), info
    if rule == "rankmax":
        return keep([np.fmax(x, y) for x, y in zip(ar, br)], ar), info
    raise ValueError(rule)


def aggregate_slides(slides, mode="two_channel", window=3, rule="select"):
    """for slides already analysed by cohort.analyse: channel A (similarity-weighted rarity), channel B (directional disorder) and, for mode 'two_channel', the cohort-level channel selection.
    Writes extra['channel_A'], extra['channel_B'] (cohort ranks per patch), extra['novelty'] (mode 'simw': channel A; 'two_channel': the selected channel) and returns the cohort information
    (tail indices and the selected channel)."""
    from .gating import cohort_apply, estimate_tau
    items = []
    for s in slides:
        H, W = s.shape

        def grid(v, fill=np.nan):
            g = np.full((H, W) + np.shape(v)[1:], fill)
            g[s.rows, s.cols] = v
            return g
        valid = np.zeros((H, W), bool)
        valid[s.rows, s.cols] = s.ok
        items.append(dict(r=grid(s.extra["novelty_raw"]), U=grid(s.extra["U"], 0.0), valid=valid, d=grid(s.extra["d_emb"])))
    tau = estimate_tau([it["U"] for it in items], [it["valid"] for it in items])
    a = cohort_apply(items, "simw", window, 0.0, 0.0, 0.0, tau)
    b = [directional_disorder(it["U"], it["valid"], 1)[0] for it in items]
    ar = cohort_rank(a)
    have_b = any(np.isfinite(x).any() for x in b)
    br = cohort_rank(b) if have_b else [np.full(x.shape, np.nan) for x in b]
    info = dict(mode=mode, tau=tau, channel="A")
    fused = ar
    if mode == "two_channel" and have_b:
        fused, inf = fuse(a, [it["d"] for it in items], b, rule)
        info.update(inf)
    for s, x, y, f in zip(slides, ar, br, fused):
        s.extra["channel_A"], s.extra["channel_B"] = x[s.rows, s.cols], y[s.rows, s.cols]
        nov = f[s.rows, s.cols].astype(float)
        nov[~s.ok] = np.nan
        s.extra["novelty"] = nov
    return info
