# -*- coding: utf-8 -*-
"""Bridge between MyUncle (maintenance-lattice simulation) and MyUncleVision (lattice analysis of images), in both directions.

    MyUncle  ->  MyUncleVision     analyse_myuncle_states(states)       MyUncle lattice states (rotor angle theta + intact mask) become "slides": the rotor
                                                                          order maps (D, kappa, S_vmf, H_chi; identical to MyUncle's own for d = 2, periodic) and a label-free
                                                                          novelty map that flags unusual local organization (defects, domain walls, damage fronts) across states.
    MyUncleVision  ->  MyUncle     slide_to_myuncle(slide)               an analysed tissue slide (or any lattice of embeddings) becomes the initial state of a MyUncle
                                                                          kinetic Monte Carlo (the exploratory scenario analysis of progression.py uses this).
    check                          verify_bridge(mc)                     numerical equality of the two implementations on a given MyUncle state.

MyUncle itself is used unmodified through myuncle_vision._core (the vendored, hash-checked core).  Nothing here is a claim about real tissue: the second direction is a
modelling analogy, the first a way to use MyUncleVision's anomaly discovery on simulation output of any MyUncle preset.
"""
import numpy as np

from . import _core, lattice as lat
from . import cohort as C


# ------------------------------------------------------------------------------------------------ MyUncle -> MyUncleVision
def rotor_field(theta):
    """MyUncle rotor angles (L, L) -> unit vectors (L, L, 2): the d = 2 case of the lattice of unit embeddings used for tissue patches"""
    th = np.asarray(theta, float)
    return np.stack([np.cos(th), np.sin(th)], -1)


def order_maps(theta, valid=None):
    """MyUncleVision's order maps for a MyUncle state (periodic lattice, four neighbours, as in MyUncle): Rbar, D, kappa, S_vmf, H_chi.
    For d = 2 these equal MyUncle.MC.local_disorder and the Bridge functions (see verify_bridge)."""
    U = rotor_field(theta)
    valid = np.ones(U.shape[:2], bool) if valid is None else np.asarray(valid, bool)
    return lat.local_order_maps(U, valid, neighbors=4, min_neighbors=0, periodic=True)


def simulate(preset="chiral_scaffold_biology", L=32, seed=0, steps=60, every=15, killed_fraction=0.25, **overrides):
    """run a MyUncle kinetic Monte Carlo (unmodified core) from a damaged start and return snapshots [(step, theta, intact), ...] and the MC object"""
    core = _core.load()
    c = core.PRESETS[preset].copy(L=L, seed=seed, equil_sweeps=0, meas_sweeps=0, **overrides)
    mc = core.MC(c).reset(killed_fraction=killed_fraction)
    snaps = [(0, mc.th.copy(), mc.intact.copy())]
    for t in range(1, steps + 1):
        mc.step()
        if t % every == 0:
            snaps.append((t, mc.th.copy(), mc.intact.copy()))
    return snaps, mc


def _state_to_slide(name, theta, intact):
    """one MyUncle state as a SlideData: every site is a 'patch' whose embedding is (cos theta, sin theta, intact, local disorder D)"""
    L = np.asarray(theta).shape[0]
    D = order_maps(theta)["D"]
    emb = np.stack([np.cos(theta).ravel(), np.sin(theta).ravel(), np.asarray(intact, float).ravel(), D.ravel()], 1).astype(np.float32)
    rr, cc = np.divmod(np.arange(L * L), L)
    n = L * L
    qc = dict(blur=np.zeros(n), pen=np.zeros(n), dark=np.zeros(n), white=np.zeros(n), artifact=np.zeros(n, bool), reason=np.array([""] * n))
    stain = dict(h_mean=np.zeros(n), e_mean=np.zeros(n), nuc_frac=np.zeros(n), od_sd=np.zeros(n))
    return C.SlideData(name=name, shape=(L, L), rows=rr, cols=cc, x0=cc.astype(float), y0=rr.astype(float), side0=1.0, mpp=1.0, emb=emb, qc=qc, stain=stain)


def analyse_myuncle_states(states, cfg=None):
    """label-free anomaly analysis of MyUncle lattice states with the MyUncleVision cohort pipeline (appearance rarity at several prevalence scales + lattice smoothing).
    states: list of (name, theta, intact) or the snapshots of simulate() as (step, theta, intact).  Rarity is judged against ALL the given states together, so give it several
    (different times, seeds or presets): a state is unusual relative to the others.  Returns a list of dicts with the exact rotor maps and the novelty map (L, L), NaN where undefined."""
    slides = [_state_to_slide(str(nm), th, it) for nm, th, it in states]
    cfg = cfg or C.Config(n_components=4, n_states=6, ref_per_slide=2000, exclude_radius=3)
    C.analyse(slides, cfg)
    out = []
    for s, (nm, th, it) in zip(slides, states):
        nov = np.full(s.shape, np.nan)
        nov[s.rows, s.cols] = s.extra["novelty"]
        out.append(dict(name=str(nm), novelty=nov, order=order_maps(th), intact_fraction=float(np.mean(it)), order_parameter=float(abs(np.mean(np.exp(1j * np.asarray(th)))))))
    return out


# ------------------------------------------------------------------------------------------------ MyUncleVision -> MyUncle
def slide_to_myuncle(slide, Ls=32, damaged_pct=0.75, preset="chiral_scaffold_biology", seed=0, R=None):
    """initial state of a MyUncle MC from an analysed slide (MODELLING ANALOGY, not prognosis): site angle = angle of the first two principal components of the patch embedding,
    damaged sites = patches whose local disorder is high for their state.  Returns the MC object (use mc.step(), mc.order(), mc.intact) and the observed damaged fraction."""
    from .progression import slide_to_lattice
    core = _core.load()
    base = core.PRESETS[preset]
    theta0, intact0, _ = slide_to_lattice(slide, Ls, damaged_pct)
    ph = np.nan_to_num(slide.extra["p_het"], nan=0.0)
    I_obs = float(np.clip(np.mean((ph > damaged_pct)[slide.ok]) if slide.ok.any() else 0.5, 1e-3, 1 - 1e-3))
    rate = float(np.clip(((1 - I_obs) / I_obs) * base.lam_eff, 0.0, 1.0)) if R is None else float(R)
    mc = core.MC(base.copy(L=Ls, seed=seed, R=rate, equil_sweeps=0, meas_sweeps=0))
    mc.th = theta0.copy()
    mc.intact = intact0.copy()
    mc.g = np.where(mc.intact, mc.c.g_intact, mc.c.g0)
    return mc, I_obs


# ------------------------------------------------------------------------------------------------ check
def verify_bridge(mc):
    """largest absolute differences between MyUncleVision's order quantities and MyUncle's own on the state of `mc` (small for d = 2; the entropy comparison skips sites whose kappa is so large that MyUncle's own I1/I0 overflows)"""
    core = _core.load()
    m = order_maps(mc.th)
    d_D = float(np.max(np.abs(m["D"] - mc.local_disorder())))
    Rbar = np.clip(m["Rbar"], 0.0, 1.0 - 1e-9)
    with np.errstate(all="ignore"):
        k_my = core.Bridge.kappa_of_U(Rbar)
        H_my, S_my = core.Bridge.H(k_my), core.Bridge.S(k_my)
    ok = np.isfinite(H_my) & np.isfinite(S_my) & np.isfinite(m["H_chi"]) & (k_my < 300)        # MyUncle's I1/I0 overflows for very large kappa (almost perfectly aligned sites)
    d_D = float(np.max(np.abs(m["D"] - mc.local_disorder())))
    d_H = float(np.max(np.abs(m["H_chi"][ok] - H_my[ok]))) if ok.any() else float("nan")
    d_S = float(np.max(np.abs(m["S_vmf"][ok] - S_my[ok]))) if ok.any() else float("nan")
    return dict(max_abs_diff_D=d_D, max_abs_diff_H_chi=d_H, max_abs_diff_S_vmf=d_S, equal=bool(d_D < 1e-10 and d_H < 1e-5 and d_S < 1e-5))


# ------------------------------------------------------------------------------------------------ command-line report
def json_report(preset, L, steps, every, seed, outdir):
    """simulate -> analyse -> write bridge_report.json and bridge_maps.png (rows: rotor angle, local disorder D, order information H_chi, novelty; one column per snapshot)"""
    import json
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    snaps, mc = simulate(preset, L, seed, steps, every)
    res = analyse_myuncle_states(snaps)
    check = verify_bridge(mc)
    rep = dict(preset=preset, L=L, steps=steps, seed=seed, check=check, core_unmodified=bool(_core.verify()),
               snapshots=[dict(step=int(sn[0]), intact_fraction=r["intact_fraction"], order_parameter=r["order_parameter"], mean_D=float(np.nanmean(r["order"]["D"])),
                               mean_H_chi=float(np.nanmean(r["order"]["H_chi"])), novelty_p95=float(np.nanpercentile(r["novelty"], 95))) for sn, r in zip(snaps, res)],
               note="MyUncle simulation (unmodified core) analysed with MyUncleVision's lattice quantities and label-free novelty; novelty is relative to the snapshots given together")
    json.dump(rep, open(os.path.join(outdir, "bridge_report.json"), "w"), indent=1)
    n = len(res)
    fig, ax = plt.subplots(4, n, figsize=(2.6 * n, 10), squeeze=False)
    for j, (sn, r) in enumerate(zip(snaps, res)):
        panels = [(sn[1] % (2 * np.pi), "twilight", "angle"), (r["order"]["D"], "viridis", "D"), (r["order"]["H_chi"], "magma", "H_chi"), (r["novelty"], "inferno", "novelty")]
        for i, (img, cm, nm) in enumerate(panels):
            ax[i, j].imshow(img, cmap=cm)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if j == 0:
                ax[i, j].set_ylabel(nm)
        ax[0, j].set_title("step %d" % sn[0])
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "bridge_maps.png"), dpi=110)
    plt.close(fig)
    return rep
