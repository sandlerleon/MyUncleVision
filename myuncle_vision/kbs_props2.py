# -*- coding: utf-8 -*-
"""Further numerical checks for the Knowledge-Based Systems manuscript (run after both freezes; descriptive).

    P2b  occupancy model of the fixed-window loss with strong feature anomalies (s = 3, rarity ranks near saturation), G1 feature-only lattices of five shapes
    P4   cost of max fusion at a fixed rank threshold (top 3%): TPR and FPR per anomaly type of max(isolation forest, Shekhar) vs the two channels, G3 shuffled-region lattices
    P5   rotated patches: AUC of rotation-invariant local statistics on the interior and on the border of rotated patches, G3 rotated-patch lattices

    python -m myuncle_vision.kbs_props2 -> results/kbs_props2.json
"""
import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score

from . import gating as G
from . import kbs_stage2 as S
from . import kbs_study as K


def p2b():
    out = {}
    for shape, blk in (("block", 1), ("block", 3), ("block", 6), ("blob", 6), ("blob", 12)):
        Us, kinds = [], []
        for i in range(20):
            U, kd = K.gen1(1300000 + 37 * blk + i + (500 if shape == "blob" else 0), s=3.0, delta=0.0, f_shape=shape, f_block=blk, prev_f=0.03)
            Us.append(U)
            kinds.append(kd)
        sc = K.cohort_scores(Us[:10], with_baselines=False, rng_seed=1)["rarity"] + K.cohort_scores(Us[10:], with_baselines=False, rng_seed=2)["rarity"]
        valid = [np.ones(U.shape[:2], bool) for U in Us]
        sm = [G._mean_window(sc[i], valid[i], 3) for i in range(20)]
        sw = []
        for half in (0, 10):
            items = [dict(r=sc[i], U=Us[i], valid=valid[i]) for i in range(half, half + 10)]
            sw += G.cohort_apply(items, "simw", 3, 0, 0, 0, G.estimate_tau([it["U"] for it in items], [it["valid"] for it in items]))
        thr = np.nanquantile(np.concatenate([x.ravel() for x in sm]), 0.97)
        thr_w = np.nanquantile(np.concatenate([x.ravel() for x in sw]), 0.97)
        ra = float(np.mean(np.concatenate([sc[i][kinds[i] == 1] for i in range(20)])))
        rn = float(np.mean(np.concatenate([sc[i][kinds[i] == 0] for i in range(20)])))
        pstar = (thr - rn) / (ra - rn)
        occ = [ndi.uniform_filter((kinds[i] == 1).astype(float), 3, mode="constant") / np.maximum(ndi.uniform_filter(np.ones((32, 32)), 3, mode="constant"), 1e-9)
               for i in range(20)]
        pred = float(np.mean(np.concatenate([occ[i][kinds[i] == 1] >= pstar - 1e-9 for i in range(20)])))
        meas = float(np.mean(np.concatenate([sm[i][kinds[i] == 1] >= thr for i in range(20)])))
        meas_raw = float(np.mean(np.concatenate([sc[i][kinds[i] == 1] >= np.quantile(np.concatenate([x.ravel() for x in sc]), 0.97) for i in range(20)])))
        meas_w = float(np.mean(np.concatenate([sw[i][kinds[i] == 1] >= thr_w for i in range(20)])))
        out["%s %d" % (shape, blk)] = dict(r_a=ra, r_n=rn, threshold=float(thr), p_star=float(pstar), predicted_survival=pred, measured_survival_mean=meas,
                                         measured_survival_unsmoothed=meas_raw, measured_survival_similarity_weighted=meas_w)
        print("P2b", shape, blk, {k: round(v, 3) for k, v in out["%s %d" % (shape, blk)].items()}, flush=True)
    return out


def p4():
    acc = {k: {"TPR_f": [], "TPR_a": [], "FPR": []} for k in ("isolation forest", "Shekhar", "max fusion")}
    for c in range(3):
        Us, kinds = [], []
        for i in range(10):
            U, kd = S.gen3(1400000 + 100 * c + i, arr="shuffle")
            Us.append(U)
            kinds.append(kd)
        sc = K.cohort_scores(Us, with_baselines=True, rng_seed=c)
        ri, rs = K._crank(sc["isolation forest (embeddings)"]), K._crank(sc["spatial outlier statistic (Shekhar)"])
        rf = [np.fmax(x, y) for x, y in zip(ri, rs)]
        for nm, ranks in (("isolation forest", ri), ("Shekhar", rs), ("max fusion", rf)):
            for i in range(10):
                flag = ranks[i] > 0.97
                acc[nm]["TPR_f"].append(float(flag[kinds[i] == 1].mean()))
                acc[nm]["TPR_a"].append(float(flag[kinds[i] == 2].mean()))
                acc[nm]["FPR"].append(float(flag[kinds[i] == 0].mean()))
    out = {nm: {k: float(np.mean(v)) for k, v in d.items()} for nm, d in acc.items()}
    print("P4", out, flush=True)
    return out


def p5():
    res = {k: {"interior": [], "border": []} for k in ("spatial outlier statistic (Shekhar)", "channel B (directional disorder)", "center-to-neighbor distance",
                                                         "contextual LOF ([u, neighbor mean])")}
    for c in range(3):
        Us, kinds = [], []
        for i in range(10):
            U, kd = S.gen3(1500000 + 100 * c + i, arr="rotate", prev_f=0.0)
            Us.append(U)
            kinds.append(kd)
        sc = K.cohort_scores(Us, with_baselines=True, rng_seed=c)
        for i in range(10):
            m = kinds[i] == 2
            interior = m & (ndi.minimum_filter(m.astype(int), 3, mode="constant") == 1)
            border = m & ~interior
            normal = ~ndi.binary_dilation(m, iterations=1)
            for k in res:
                s = sc[k][i]
                for part, sel in (("interior", interior), ("border", border)):
                    y = np.r_[np.ones(sel.sum()), np.zeros(normal.sum())]
                    v = np.r_[s[sel], s[normal]]
                    ok = np.isfinite(v)
                    res[k][part].append(float(roc_auc_score(y[ok], v[ok])))
    out = {k: {p: dict(mean=float(np.mean(v)), sd=float(np.std(v)), n=len(v)) for p, v in d.items()} for k, d in res.items()}
    print("P5", out, flush=True)
    return out


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    K._save(dict(P2b=p2b(), P4=p4(), P5=p5()), "kbs_props2.json")
