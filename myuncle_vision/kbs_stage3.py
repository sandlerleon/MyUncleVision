# -*- coding: utf-8 -*-
"""Stage 3 of the Knowledge-Based Systems study: a corrected, stationary generator G4 and a third prospective freeze.

Why: Experiment D showed that G3's background becomes irregular on large lattices (its stripe phase is computed from absolute coordinates with a varying orientation), so G3 cannot test
scaling.  G4 replaces the texture by a STATIONARY one: the stripe phase is the argument of a complex Gaussian random field band-pass filtered around wavelength 8 sites (a labyrinth texture
whose statistics do not depend on position or lattice size), mapped into d dimensions by a fixed linear encoder that is different from G3's.  Anomalies as in G3 (out-of-range texture code;
shuffled exchange between two blobs, or 90-degree rotated 6 x 6 patches).

Nothing is selected here: the rule frozen in stage 2 (results/kbs2_frozen.json) and its comparators are applied unchanged.

    python -m myuncle_vision.kbs_stage3 freeze    -> results/kbs3_frozen.json (conditions, seeds, endpoint, sha256 of this file and of kbs_stage2.py / kbs_study.py)
    python -m myuncle_vision.kbs_stage3 confirm   -> results/kbs3_confirm.json
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

from . import kbs_stage2 as S
from . import kbs_study as K

G4_MAP_SEED = 5151


def _stationary_phase(rng, n, wavelength=8.0, bandwidth=0.25):
    """phase of a complex Gaussian random field whose spectrum is a ring of radius 1/wavelength (relative width `bandwidth`); stationary by construction (periodic FFT filtering)"""
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.fftfreq(n)[None, :]
    kr = np.hypot(ky, kx)
    k0 = 1.0 / wavelength
    filt = np.exp(-0.5 * ((kr - k0) / (bandwidth * k0)) ** 2)
    z = np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) * filt)
    return np.angle(z)


def gen4(seed, n=32, d=16, noise=0.08, psi_range=(0.0, 0.6), dpsi=1.2, prev_f=0.015, prev_a=0.04, arr="shuffle", f_size=(2, 10), a_size=(16, 30), a_block=6):
    rng = np.random.default_rng(seed)
    P = np.random.default_rng(G4_MAP_SEED).normal(size=(4, d))
    phi = _stationary_phase(rng, n)
    g = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n))) * np.exp(-0.5 * (np.hypot(np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]) / (1 / 30.0)) ** 2)))
    g = (g - g.mean()) / (g.std() + 1e-12)
    psi = psi_range[0] + (psi_range[1] - psi_range[0]) * (0.5 + 0.5 * np.tanh(g))       # stationary texture code inside the normal range
    kind = np.zeros((n, n), int)
    occ = np.zeros((n, n), bool)
    target, got, guard = prev_f * n * n, 0, 0
    while got < target and guard < 50 * max(1, n // 32) ** 2:
        guard += 1
        sz = int(rng.integers(f_size[0], f_size[1] + 1))
        px = K._place(occ, rng, lambda: K._blob(rng, n, sz))
        if px is None:
            continue
        occ[px[:, 0], px[:, 1]] = True
        psi[px[:, 0], px[:, 1]] = psi_range[1] + dpsi + rng.uniform(-0.1, 0.1)
        kind[px[:, 0], px[:, 1]] = 1
        got += len(px)
    F = np.stack([np.cos(phi), np.sin(phi), 0.6 * np.cos(psi), 0.6 * np.sin(psi)], -1)
    X = F @ P + rng.normal(0, noise, (n, n, d))
    U = X / np.linalg.norm(X, axis=-1, keepdims=True)
    if arr == "shuffle":
        target, got, guard = prev_a * n * n, 0, 0
        while got < target and guard < 30 * max(1, n // 32) ** 2:
            guard += 1
            sz = int(rng.integers(a_size[0], a_size[1] + 1))
            r1 = K._place(occ, rng, lambda: K._blob(rng, n, sz))
            if r1 is None:
                continue
            occ[r1[:, 0], r1[:, 1]] = True
            r2 = None
            for _ in range(200):
                c = K._place(occ, rng, lambda: K._blob(rng, n, sz))
                if c is not None and np.hypot(*(c.mean(0) - r1.mean(0))) > 10:
                    r2 = c
                    break
            if r2 is None:
                occ[r1[:, 0], r1[:, 1]] = False
                continue
            occ[r2[:, 0], r2[:, 1]] = True
            v1, v2 = U[r1[:, 0], r1[:, 1]].copy(), U[r2[:, 0], r2[:, 1]].copy()
            U[r1[:, 0], r1[:, 1]] = v2[rng.permutation(sz)]
            U[r2[:, 0], r2[:, 1]] = v1[rng.permutation(sz)]
            kind[r1[:, 0], r1[:, 1]] = 2
            kind[r2[:, 0], r2[:, 1]] = 2
            got += 2 * sz
    elif arr == "rotate":
        k = max(1, int(round(prev_a * n * n / (a_block * a_block))))
        for _ in range(k):
            px = K._place(occ, rng, lambda: K._block(rng, n, a_block))
            if px is None:
                break
            occ[px[:, 0], px[:, 1]] = True
            y0, x0 = px[:, 0].min(), px[:, 1].min()
            U[y0:y0 + a_block, x0:x0 + a_block] = np.rot90(U[y0:y0 + a_block, x0:x0 + a_block].copy())
            kind[y0:y0 + a_block, x0:x0 + a_block] = 2
    return U, kind


def _job3(args):
    seeds, kw = args
    Us, kinds = [], []
    for sd in seeds:
        U, kd = gen4(sd, **kw)
        Us.append(U)
        kinds.append(kd)
    sc = S.scores2(Us, rng_seed=int(seeds[0]))
    return {nm: [K.lattice_metrics(kinds[i], v[i]) for i in range(len(Us))] for nm, v in sc.items()}


def run(kw, n_lat, seed0, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    seeds = [seed0 + i for i in range(n_lat)]
    jobs = [(seeds[c:c + K.N_PER_COHORT], kw) for c in range(0, n_lat, K.N_PER_COHORT)]
    res = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for per in ex.map(_job3, jobs):
            for nm, lst in per.items():
                d = res.setdefault(nm, {"feature": [], "arrangement": [], "all": []})
                for m in lst:
                    for kk in d:
                        d[kk].append(m[kk])
    return res


CONDITIONS = {
    "G4 stationary texture, shuffled regions, 32x32": (dict(arr="shuffle"), 30),
    "G4 stationary texture, rotated patches, 32x32": (dict(arr="rotate"), 30),
    "G4 stationary texture, shuffled regions, 64x64": (dict(arr="shuffle", n=64), 20),
    "G4 stationary texture, shuffled regions, 128x128": (dict(arr="shuffle", n=128), 20),
}
SEED3 = 2000000
HERE = os.path.abspath(__file__)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def freeze():
    fz2 = json.load(open(os.path.join(K.RES, "kbs2_frozen.json")))
    fz = dict(frozen_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), code_sha256=sha(HERE), stage2_code_sha256=sha(S.__file__), stage1_code_sha256=sha(K.__file__),
              rule=fz2["chosen"], comparators=fz2["comparators"], conditions={k: [kw, n] for k, (kw, n) in CONDITIONS.items()}, seed0=SEED3,
              primary_endpoint="worst-type AP of the stage-2 rule pooled over the two 32x32 G4 conditions; paired bootstrap against each comparator. Secondary: per-condition results, "
                               "and the change of worst-type AP with lattice size (shuffled regions, 32 -> 64 -> 128)",
              note="gen4 written before this freeze; a smoke test generated lattices with seeds 1-3 (anomaly-free, n = 32, 64, 128) and seed 1 (each arrangement type and size) and printed only the label counts and the mean neighbor "
                   "difference of anomaly-free lattices (stationarity check); no detector score was computed")
    K._save(fz, "kbs3_frozen.json")


def confirm():
    fz = json.load(open(os.path.join(K.RES, "kbs3_frozen.json")))
    if sha(HERE) != fz["code_sha256"] or sha(S.__file__) != fz["stage2_code_sha256"] or sha(K.__file__) != fz["stage1_code_sha256"]:
        sys.exit("code changed after the third freeze; refusing to run")
    ch = fz["rule"]
    out, allres = {}, {}
    for i, (nm, (kw, nl)) in enumerate(fz["conditions"].items()):
        res = run(kw, nl, fz["seed0"] + 1000 * i, workers=4 if kw.get("n", 32) >= 128 else 8)
        allres[nm] = res
        sm = K.summary(res)
        comps = [c for c in fz["comparators"] if c != ch and c in res]
        out[nm] = dict(summary=sm, tests={c: K.paired_boot(res, ch, c, "worst") for c in comps})
        print(nm, ch[:20], round(sm[ch]["worst"], 3), sorted(((round(sm[c]["worst"], 3), c[:30]) for c in comps), reverse=True)[:5], flush=True)
    base = K._pool_conditions({k: v for k, v in allres.items() if "32x32" in k})
    comps = [c for c in fz["comparators"] if c != ch]
    K._save(dict(frozen=fz, rule=ch, conditions=out, pooled32_summary=K.summary(base),
                 pooled32_tests={c: {key: K.paired_boot(base, ch, c, key) for key in ("worst", "feature", "arrangement", "all")} for c in comps}), "kbs3_confirm.json")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    {"freeze": freeze, "confirm": confirm}[sys.argv[1]]()
