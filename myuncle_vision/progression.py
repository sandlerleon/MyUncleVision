# -*- coding: utf-8 -*-
"""Tissue progression scenarios with the MyUncle simulation (Phase 13).  MODEL-BASED SCENARIOS, NOT PROGNOSIS: no clinical outcome is used.

A slide is mapped onto MyUncle's maintenance lattice:
    angle theta of a site   = angle of its first two principal components (the rotor)
    intact / damaged site   = not disordered / disordered for its state (p_het above `damaged_pct`)
    I_obs                   = fraction of disordered sites;  rho_hat = (1 - I_obs) / I_obs from the linear steady state I* = 1 / (1 + rho)
MyUncle's mean-field reduction has a bistable window in rho: there two maintained states coexist and a slide can sit in either.  The lattice kinetic Monte Carlo (MyUncle_core.MC) is then
started from the observed slide and run for three repair rates (half, equal, double) to show how the observed order would evolve under that model.
"""
import numpy as np
import cv2

from . import _core


def slide_to_lattice(s, Ls=32, damaged_pct=0.75):
    z = np.zeros((s.shape[0], s.shape[1], 2))
    U = s.extra["U"]
    z[s.rows, s.cols, 0] = U[:, 0]
    z[s.rows, s.cols, 1] = U[:, 1]
    bad = np.zeros(s.shape, np.float32)
    ph = np.nan_to_num(s.extra["p_het"], nan=0.0)
    bad[s.rows, s.cols] = (ph > damaged_pct) & s.ok
    valid = s.extra["valid"].astype(np.float32)
    rs = lambda a: cv2.resize(a.astype(np.float32), (Ls, Ls), interpolation=cv2.INTER_AREA)
    vv = rs(valid)
    zx, zy = rs(z[..., 0] * valid), rs(z[..., 1] * valid)
    theta = np.arctan2(zy, zx)
    dmg = rs(bad * valid) / np.maximum(vv, 1e-6) > 0.5
    return theta, ~dmg, vv > 0.25


def scenarios(s, Ls=32, steps=60, seeds=2, damaged_pct=0.75, preset="chiral_scaffold_biology"):
    core = _core.load()
    base = core.PRESETS[preset]
    mf = core.MeanField(base)
    win = mf.bistable_window()
    ph = np.nan_to_num(s.extra["p_het"], nan=0.0)
    I_obs = float(np.mean((ph > damaged_pct)[s.ok])) if s.ok.any() else float("nan")
    I_obs = min(max(I_obs, 1e-3), 1 - 1e-3)
    rho_hat = (1.0 - I_obs) / I_obs
    theta0, intact0, tissue = slide_to_lattice(s, Ls, damaged_pct)
    out = dict(I_obs=I_obs, rho_hat=float(rho_hat), bistable_window=list(win) if win else None, in_bistable_window=bool(win and win[0] <= rho_hat <= win[1]),
               mean_field_stable_states=[float(1 - r) for r in mf.fixed_points(rho_hat) if mf.stable(r, rho_hat)], scenarios={})
    for name, fac in (("repair_x0.5", 0.5), ("repair_x1", 1.0), ("repair_x2", 2.0)):
        traj = []
        for sd in range(seeds):
            c = base.copy(L=Ls, seed=sd, R=float(np.clip(rho_hat * base.lam_eff * fac, 0.0, 1.0)), equil_sweeps=0, meas_sweeps=0)
            mc = core.MC(c)
            mc.th = theta0.copy()
            mc.intact = intact0.copy()
            mc.g = np.where(mc.intact, c.g_intact, c.g0)
            ts = [(float(mc.intact.mean()), float(mc.order()))]
            for _ in range(steps):
                mc.step()
                ts.append((float(mc.intact.mean()), float(mc.order())))
            traj.append(ts)
        T = np.array(traj).mean(0)
        out["scenarios"][name] = dict(intact_start=float(T[0, 0]), intact_end=float(T[-1, 0]), order_start=float(T[0, 1]), order_end=float(T[-1, 1]), R=float(c.R))
    return out
