# -*- coding: utf-8 -*-
"""Second testbed: lattices simulated with MyUncle (unmodified core), ground truth = the damaged sites.  Same protocol as bench.py: settings frozen (results/frozen_settings.json,
no tuning here), a development set (one preset) and a held-out set (three other presets), baselines on the same embeddings.

    python -m myuncle_vision.bridge_bench --out results

The embedding of a site never contains the intact flag: 'rotor' = (cos theta, sin theta); 'rotor + D' = (cos theta, sin theta, local disorder D).  The damaged sites are
the sites that lost their coupling (g0 << g_intact in MyUncle), so they are ordinary physics, not a pattern drawn to resemble the method's score; the prevalence of damaged sites
is NOT small (reported), which stresses the method's assumption that interesting states are rare.
"""
import argparse
import hashlib
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import bridge as B
from . import bench as BN

DEV = dict(presets=("chiral_scaffold_biology",), seeds=(0, 1, 2, 3))
HELD = dict(presets=("polymer_photodegradation", "liquid_crystal", "magnetic_storage_decay"), seeds=(10, 11, 12, 13))
SNAP_STEPS = (20, 40, 60)
FROZEN = "MyUncleVision frozen: appearance, multi-scale, lattice-smoothed"
METHODS = (FROZEN, "Isolation Forest, lattice-smoothed (rank)", "Local Outlier Factor, lattice-smoothed (rank)", "Mahalanobis distance, lattice-smoothed (rank)",
           "kNN distance, k = 10, lattice-smoothed (rank)", "Isolation Forest", "Local Outlier Factor", "Mahalanobis distance", "kNN distance, k = 10", "local disorder D alone")


def make_states(spec, L=48, killed_fraction=0.25):
    states, truth, meta = [], [], []
    for preset in spec["presets"]:
        for seed in spec["seeds"]:
            snaps, _ = B.simulate(preset, L=L, seed=seed, steps=max(SNAP_STEPS), every=1, killed_fraction=killed_fraction)
            for step, th, it in snaps:
                if step in SNAP_STEPS:
                    states.append(("%s_s%d_t%d" % (preset, seed, step), th, it))
                    truth.append(~it)
                    meta.append(dict(preset=preset, seed=seed, step=step, damaged_fraction=float(np.mean(~it))))
    return states, truth, meta


def _slides(states, embedding):
    sl = []
    for nm, th, it in states:
        s = B._state_to_slide(nm, th, it)
        e = s.emb                                       # (cos, sin, intact, D)
        s.emb = (e[:, [0, 1]] if embedding == "rotor" else e[:, [0, 1, 3]]).astype(np.float32)
        sl.append(s)
    return sl


def _boot_ci(metric, y_by_state, s_by_state, group=len(SNAP_STEPS), n=300, seed=0):
    """95% interval from resampling whole simulation runs (the snapshots of one run are correlated, so states are not resampled one by one)"""
    rng = np.random.default_rng(seed)
    k = len(y_by_state) // group
    vals = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        sel = [j for i in idx for j in range(i * group, (i + 1) * group)]
        y = np.concatenate([y_by_state[i] for i in sel])
        s = np.concatenate([s_by_state[i] for i in sel])
        if 0 < y.sum() < len(y):
            vals.append(metric(y, s))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def evaluate_set(spec, embedding, L=48, log=print):
    states, truth, meta = make_states(spec, L)
    slides = _slides(states, embedding)
    M = BN.all_scores(slides, weights=(1.0, 0.0, 0.0), seed=0, n_components=16, n_states=6)
    out = dict(embedding=embedding, n_states=len(states), L=L, damaged_fraction_mean=float(np.mean([m["damaged_fraction"] for m in meta])),
               damaged_fraction_by_preset={p: float(np.mean([m["damaged_fraction"] for m in meta if m["preset"] == p])) for p in spec["presets"]}, methods={})
    for name in METHODS:
        if name not in M:
            continue
        ys, ss = [], []
        for t, a in zip(truth, M[name]):
            ok = np.isfinite(a)
            tt = t.reshape(-1)
            aa = a
            ys.append(tt[ok]); ss.append(aa[ok])
        y, s = np.concatenate(ys), np.concatenate(ss)
        out["methods"][name] = dict(auroc=float(roc_auc_score(y, s)), average_precision=float(average_precision_score(y, s)),
                                    auroc_ci=_boot_ci(roc_auc_score, ys, ss),
                                    average_precision_ci=_boot_ci(average_precision_score, ys, ss))
        log("   %-12s %-62s AUROC %.2f  AP %.2f" % (embedding, name, out["methods"][name]["auroc"], out["methods"][name]["average_precision"]))
    out["prevalence_all"] = float(np.mean(np.concatenate([t.reshape(-1) for t in truth])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    fz = os.path.join(a.out, "frozen_settings.json")
    fz_hash = hashlib.sha256(open(fz, "rb").read()).hexdigest() if os.path.exists(fz) else None
    for kind, spec in (("dev", DEV), ("held", HELD)):
        for emb in ("rotor", "rotor_D"):
            r = evaluate_set(spec, emb)
            r.update(kind=kind, presets=list(spec["presets"]), seeds=list(spec["seeds"]), snapshots=list(SNAP_STEPS), frozen_settings_sha256=fz_hash,
                     note="settings of the image benchmark were used unchanged (appearance only, weights 1,0,0; multi-scale rarity; 3x3 smoothing); n_states = 6 because the embedding has 2 or 3 dimensions; the intact flag is not part of the embedding")
            json.dump(r, open(os.path.join(a.out, "bridge_bench_%s_%s.json" % (kind, emb)), "w"), indent=1)


if __name__ == "__main__":
    main()
