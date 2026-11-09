#!/usr/bin/env python3
"""MyUncle_core — compact maintenance-lattice framework (Leon Sandler).
Order eroded by a state-dependent flux, restored by repair; reusable across STEM.
Invoke: "use MyUncle to generate a draft paper on X".
API: from MyUncle_core import reproduce_paper; reproduce_paper('Title','sub',preset='chiral_scaffold_biology')
CLI: python MyUncle_core.py --demo | --grid-size | --reproduce --title .. --subtitle .. --preset ..
Needs numpy, scipy, matplotlib. Full house-styled .docx version: MyUncle.py. Spec: MyUncle_MASTER_REFERENCE.md."""
from __future__ import annotations
import argparse, itertools, math, os
from dataclasses import dataclass, asdict
import numpy as np
from scipy.special import i0, i1

# ---------- config ----------
@dataclass
class Config:
    L: int = 48; seed: int = 0; n_seeds: int = 3
    T: float = 0.45; J0: float = 1.0; h0: float = 0.30
    g_intact: float = 1.0; g0: float = 0.12
    k: float = 1.0; G: float = 1.0; s: float = 0.75; u_s: float = 0.0
    lam0: float = 0.02; beta: float = 20.0; hill_h: float = 6.0; hill_K: float = 0.5
    R: float = 0.10; eps_reverse: float = 1e-3
    relax_per_step: int = 2; equil_sweeps: int = 260; meas_sweeps: int = 260
    sample_every: int = 5; theta_init_sigma: float = 0.10; label: str = "default"
    @property
    def N(self): return self.L * self.L
    @property
    def lam_eff(self): return self.lam0 * self.k * self.G * (1 - self.s * self.u_s)
    @property
    def rho(self): return self.R / max(self.lam_eff, 1e-12)
    def copy(self, **kw):
        d = asdict(self); d.update(kw); return Config(**d)

# ---------- (A) order-entropy bridge ----------
class Bridge:
    @staticmethod
    def A(k): k = np.asarray(k, float); return np.where(k > 0, i1(k) / i0(k), 0.0)
    @staticmethod
    def S(k): k = np.asarray(k, float); return np.log(2*np.pi*i0(k)) - k*Bridge.A(k)
    Smax = math.log(2*math.pi)
    @staticmethod
    def H(k): return Bridge.Smax - Bridge.S(k)
    @staticmethod
    def dS_dU(k): return -np.asarray(k, float)          # exact
    @staticmethod
    def H_asymptote(U): U = np.asarray(U, float); return -0.5*np.log(1-U) + 0.5*(np.log(np.pi)-1)
    @staticmethod
    def kappa_of_U(U, hi=1e6):
        def solve(u):
            if u <= 0: return 0.0
            if u >= 1: return hi
            lo, h = 0.0, hi
            for _ in range(200):
                m = 0.5*(lo+h); lo, h = (m, h) if Bridge.A(m) < u else (lo, m)
            return 0.5*(lo+h)
        return np.vectorize(solve)(U)

# ---------- (B) mean-field reduction + bistability ----------
class MeanField:
    def __init__(self, c): self.c = c
    def mu(self, I):
        c = self.c; I = np.asarray(I, float)
        return 1 + c.beta * I**c.hill_h / (c.hill_K**c.hill_h + I**c.hill_h)
    def rhs(self, I, rho): return self.mu(I)*(1-I) - rho*I
    def I_star_linear(self, rho=None): rho = self.c.rho if rho is None else rho; return 1/(1+rho)
    def fixed_points(self, rho, n=4000):
        I = np.linspace(1e-4, 1-1e-4, n); g = self.rhs(I, rho)
        return I[np.where(np.diff(np.sign(g)) != 0)[0]]
    def stable(self, I, rho, d=1e-5): return (self.rhs(I+d, rho)-self.rhs(I-d, rho))/(2*d) < 0
    def bistable_window(self, n=6000):
        I = np.linspace(1e-3, 0.999, n); F = self.mu(I)*(1-I)/I
        t = np.where(np.diff(np.sign(np.diff(F))) != 0)[0]
        if len(t) >= 2: r = sorted(F[t]); return (float(r[0]), float(r[-1]))
        return None
    def integrate(self, I0, rho, T=80.0, dt=0.02):
        t = np.arange(0, T, dt); I = np.empty_like(t); I[0] = I0
        for k in range(1, len(t)): I[k] = min(1, max(0, I[k-1] + dt*self.rhs(I[k-1], rho)))
        return t, I

# ---------- (C+D) site-diluted XY kinetic Monte Carlo ----------
class MC:
    def __init__(self, c, seed_off=0):
        self.c = c; self.rng = np.random.default_rng(c.seed + seed_off)
        self.idx = (np.add.outer(np.arange(c.L), np.arange(c.L))) % 2; self.reset()
    def reset(self, killed_fraction=None):
        c = self.c; self.th = self.rng.normal(0, c.theta_init_sigma, (c.L, c.L))
        self.g = np.full((c.L, c.L), c.g_intact); self.intact = np.ones((c.L, c.L), bool)
        if killed_fraction is not None:
            v = self.rng.random((c.L, c.L)) < killed_fraction; self.g[v] = c.g0; self.intact[v] = False
        return self
    def metropolis(self):
        c = self.c; th, g = self.th, self.g
        for col in (0, 1):
            m = self.idx == col; prop = th + self.rng.uniform(-np.pi, np.pi, (c.L, c.L))
            e0 = np.zeros((c.L, c.L)); e1 = np.zeros((c.L, c.L))
            for sh, ax in [(1, 0), (-1, 0), (1, 1), (-1, 1)]:
                J = g*np.roll(g, sh, ax); tj = np.roll(th, sh, ax)
                e0 -= J*np.cos(th-tj); e1 -= J*np.cos(prop-tj)
            e0 -= c.h0*g*np.cos(th); e1 -= c.h0*g*np.cos(prop)
            acc = (self.rng.random((c.L, c.L)) < np.exp(-(e1-e0)/c.T)) & m
            th = np.where(acc, prop, th)
        self.th = th; return self
    def local_disorder(self):
        z = np.exp(1j*self.th)
        nb = (np.roll(z, 1, 0)+np.roll(z, -1, 0)+np.roll(z, 1, 1)+np.roll(z, -1, 1))/4
        return 1 - np.abs(nb)
    def kmc(self):
        c = self.c; D = self.local_disorder()
        lam = c.lam_eff*(1 + c.beta*D**c.hill_h/(c.hill_K**c.hill_h + D**c.hill_h))
        kill = self.intact & (self.rng.random(D.shape) < lam)
        self.intact &= ~kill; self.g[kill] = c.g0; nk = int(kill.sum())
        rep = (~self.intact) & (self.rng.random(D.shape) < c.R)
        self.intact |= rep; self.g[rep] = c.g_intact; self.th[rep] = 0.0
        return nk, int(rep.sum())
    def step(self):
        for _ in range(self.c.relax_per_step): self.metropolis()
        return self.kmc()
    def order(self):
        w = self.g if self.c.g0 == 0 else np.ones_like(self.g)
        return abs(np.sum(w*np.exp(1j*self.th))/max(w.sum(), 1))
    def helicity(self):
        c = self.c; th, g = self.th, self.g; u = []
        for ax in (0, 1):
            J = g*np.roll(g, -1, ax); d = th-np.roll(th, -1, ax)
            u.append(np.sum(J*np.cos(d))/c.N - np.sum(J*np.sin(d))**2/(c.N*c.T))
        return 0.5*(u[0]+u[1])
    def correlation(self):
        L = self.c.L; z = np.exp(1j*self.th); rs = np.arange(1, L//2); G = []
        for r in rs:
            G.append(np.real(np.mean(z*np.conj(np.roll(z, r, 0)))+np.mean(z*np.conj(np.roll(z, r, 1))))/2)
        return rs, np.array(G)
    def run_ness(self):
        c = self.c
        for _ in range(c.equil_sweeps//c.relax_per_step): self.step()
        U = []; nk = nr = 0
        for _ in range(c.meas_sweeps//c.relax_per_step):
            a, b = self.step(); nk += a; nr += b; U.append(self.order())
        U = np.array(U); tail = U[-max(len(U)//3, 1):]
        den = (c.meas_sweeps//c.relax_per_step)*c.N
        return dict(U=float(tail.mean()), U_sd=float(tail.std()), J_kill=nk/den, J_rep=nr/den)

# ---------- (E) analyses (numbers + optional figure) ----------
C = dict(lo="#2c7fb8", mid="#e08e0b", hi="#c0392b", g="#1a7a4c", p="#6a3d9a")
class Analyses:
    def __init__(self, c): self.c = c
    def entropy_bridge(self, save=None):
        import matplotlib.pyplot as plt
        k = np.concatenate([np.linspace(1e-3, 2, 200), np.linspace(2, 80, 600)])
        U = Bridge.A(k); H = Bridge.H(k); Ha = Bridge.H_asymptote(U)
        f, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(U, Bridge.S(k), color=C["hi"], lw=2.4); ax[0].axhline(Bridge.Smax, ls="--", color="k")
        ax[0].set_xlabel("U_chi"); ax[0].set_ylabel("S (nats)"); ax[0].set_title("dS/dU = -kappa", loc="left")
        ax[1].plot(U, H, color=C["g"], lw=2.4, label="H_chi"); m = U > .55
        ax[1].plot(U[m], Ha[m], color=C["mid"], ls="--", lw=1.7, label="asymptote")
        ax[1].set_xlabel("U_chi"); ax[1].set_ylabel("H_chi"); ax[1].legend(fontsize=8)
        for a in ax: a.grid(True, color="#ddd")
        f.tight_layout(); f.savefig(save, dpi=150, bbox_inches="tight") if save else None; plt.close(f)
        return dict(figure=save)
    def bifurcation(self, save=None):
        import matplotlib.pyplot as plt
        mf = MeanField(self.c); win = mf.bistable_window(); rr = np.linspace(.3, 15, 400); hi, lo = [], []
        for rho in rr:
            st = [1-r for r in mf.fixed_points(rho) if mf.stable(r, rho)]
            hi.append(max(st) if st else np.nan); lo.append(min(st) if len(st) > 1 else np.nan)
        f, ax = plt.subplots(figsize=(6, 4.2)); ax.plot(rr, hi, color=C["g"], lw=2); ax.plot(rr, lo, color=C["g"], lw=2)
        if win: ax.axvspan(*win, color=C["mid"], alpha=.12)
        ax.set_xlabel("rho"); ax.set_ylabel("steady U_chi"); ax.set_ylim(0, 1); ax.set_title("bistability", loc="left")
        ax.grid(True, color="#ddd"); f.tight_layout(); f.savefig(save, dpi=150, bbox_inches="tight") if save else None; plt.close(f)
        return dict(bistable_window=win, figure=save)
    def dimensionless_collapse(self, n=500, save=None):
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(self.c.seed); rho, I = [], []
        for _ in range(n):
            G = rng.uniform(.3, 3); R = rng.uniform(.02, 1.5); u = rng.uniform(0, 1)
            r = R/(self.c.k*G*(1-self.c.s*u)); rho.append(r); I.append(1/(1+r))
        f, ax = plt.subplots(figsize=(6, 4.2)); ax.scatter(rho, I, s=9, alpha=.35, color=C["lo"])
        x = np.logspace(-1.6, 1.7, 200); ax.plot(x, 1/(1+x), "k-", lw=2)
        ax.set_xscale("log"); ax.set_xlabel("rho"); ax.set_ylabel("I* = 1/(1+rho)"); ax.set_title("universal collapse", loc="left")
        ax.grid(True, color="#ddd", which="both"); f.tight_layout(); f.savefig(save, dpi=150, bbox_inches="tight") if save else None; plt.close(f)
        return dict(figure=save)
    def entropy_production(self, save=None):
        import matplotlib.pyplot as plt
        eps = self.c.eps_reverse; rho = np.logspace(-1.5, 1.5, 300); lam = 1.0; R = rho*lam
        pO = (eps+R)/(lam+2*eps+R); J = lam*pO - eps*(1-pO); sig = J*np.log(lam*R/eps**2)
        f, ax = plt.subplots(figsize=(6, 4.2)); ax.plot(rho, sig, color=C["hi"], lw=2.4)
        ax.set_xscale("log"); ax.set_xlabel("rho"); ax.set_ylabel("EPR sigma"); ax.set_title("Schnakenberg EPR", loc="left")
        ax.grid(True, color="#ddd", which="both"); f.tight_layout(); f.savefig(save, dpi=150, bbox_inches="tight") if save else None; plt.close(f)
        return dict(sigma_positive=bool(np.all(sig > 0)), figure=save)
    def finite_size_scaling(self, Ls=(32, 48, 64), fs=None, save=None):
        import matplotlib.pyplot as plt
        fs = np.array([0, .1, .2, .28, .34, .4, .46, .55]) if fs is None else np.asarray(fs)
        NK = 2*self.c.T/np.pi; out = {}
        for L in Ls:
            U, Y = [], []
            for fk in fs:
                us, ys = [], []
                for sd in range(max(self.c.n_seeds, 1)):
                    mc = MC(self.c.copy(L=L, h0=0.0, g0=0.0, seed=self.c.seed+sd+int(fk*131)))
                    mc.reset(killed_fraction=fk)
                    for _ in range(self.c.equil_sweeps): mc.metropolis()
                    mm = []
                    for t in range(self.c.meas_sweeps):
                        mc.metropolis()
                        if t % self.c.sample_every == 0: mm.append(mc.order()); ys.append(mc.helicity())
                    us.append(np.mean(mm))
                U.append(np.mean(us)); Y.append(np.mean(ys))
            out[L] = dict(U=np.array(U), Y=np.array(Y))
        def cross(x, y, lv):
            for i in range(len(x)-1):
                if (y[i]-lv)*(y[i+1]-lv) < 0: return float(x[i]+(lv-y[i])*(x[i+1]-x[i])/(y[i+1]-y[i]))
            return float("nan")
        fKT = {L: cross(fs, out[L]["Y"], NK) for L in Ls}
        cols = [C["lo"], C["mid"], C["hi"], C["p"]]
        f, ax = plt.subplots(1, 2, figsize=(11, 4.2)); pc = 1-0.5927
        for i, L in enumerate(Ls):
            ax[0].plot(fs, out[L]["U"], "o-", color=cols[i % 4], label=f"L={L}")
            ax[1].plot(fs, out[L]["Y"], "o-", color=cols[i % 4], label=f"L={L}")
        ax[1].axhline(NK, ls="--", color="k")
        for a, t in zip(ax, ["U_chi", "helicity Y vs 2T/pi"]):
            a.axvline(pc, ls=":", color="k"); a.set_xlabel("f"); a.set_ylabel(t); a.legend(fontsize=8); a.grid(True, color="#ddd")
        f.tight_layout(); f.savefig(save, dpi=150, bbox_inches="tight") if save else None; plt.close(f)
        return dict(f_KT=fKT, percolation=pc, figure=save)

# ---------- (F) presets + 6000-point grid ----------
PRESETS = {
    "magnetism_xy": Config(label="magnetism_xy", h0=0.0, beta=0.0, T=0.45),
    "polymer_photodegradation": Config(label="polymer_photodegradation", beta=15, s=0.6, u_s=0.4, R=0.08),
    "liquid_crystal": Config(label="liquid_crystal", h0=0.2, beta=10, T=0.5),
    "chiral_scaffold_biology": Config(label="chiral_scaffold_biology", beta=20, s=0.75, u_s=0.5, R=0.15, hill_h=6, hill_K=0.5),
    "magnetic_storage_decay": Config(label="magnetic_storage_decay", h0=0.35, beta=8, R=0.20, T=0.4),
    "superconducting_scaffold": Config(label="superconducting_scaffold", h0=0.4, beta=5, T=0.35),
}
PARAMETER_GRID = dict(
    T=[0.30, 0.45, 0.60, 0.75, 0.90], R=[0.02, 0.05, 0.10, 0.20, 0.40, 0.80],
    G=[0.5, 1.0, 1.5, 2.0, 3.0], u_s=[0.0, 0.33, 0.66, 1.0],
    beta=[0.0, 5.0, 10.0, 20.0, 30.0], hill_h=[4.0, 6.0])
def grid_size():
    n = 1
    for v in PARAMETER_GRID.values(): n *= len(v)
    return n
def full_parameter_grid(base=None):
    base = base or Config(); keys = list(PARAMETER_GRID)
    for combo in itertools.product(*[PARAMETER_GRID[k] for k in keys]):
        yield base.copy(**dict(zip(keys, combo)))

# ---------- (G) one-call paper reproduction (figures + markdown) ----------
def reproduce_paper(title, subtitle, preset="chiral_scaffold_biology", outdir="myuncle_paper",
                    author="Leon Sandler", email="leonsandler@alumni.swinburne.edu",
                    zenodo="https://doi.org/10.5281/zenodo.xxxx", include=("bridge", "bifurcation", "epr", "collapse")):
    """Generate the figure suite + a markdown draft (front matter + captions).
    Render to .docx with:  pandoc manuscript.md -o paper.docx  (use full MyUncle.py for the house-styled docx)."""
    import matplotlib; matplotlib.use("Agg")
    c = PRESETS[preset]; os.makedirs(outdir, exist_ok=True); A = Analyses(c)
    latt = c.copy(L=20, n_seeds=1, equil_sweeps=40, meas_sweeps=40)
    lab = {"bridge": "Order-entropy bridge (dS/dU = -kappa).", "bifurcation": "Bistability and saddle-node bifurcations.",
           "epr": "Schnakenberg entropy production sigma(rho) > 0.", "collapse": "Universal collapse I* = 1/(1+rho).",
           "fss": "Finite-size scaling; helicity modulus vs 2T/pi."}
    figs = {}
    for key in [x for x in ("bridge", "bifurcation", "fss", "epr", "collapse") if x in include]:
        p = os.path.join(outdir, "fig_%s.png" % key)
        {"bridge": A.entropy_bridge, "bifurcation": A.bifurcation, "epr": A.entropy_production,
         "collapse": A.dimensionless_collapse,
         "fss": lambda save: Analyses(latt).finite_size_scaling(Ls=(16, 24), save=save)}[key](save=p)
        figs[key] = p
    md = ["# %s" % title, "", "*%s*" % subtitle, "",
          "%s | Independent Researcher | %s" % (author, email), "", "Zenodo: %s" % zenodo, "",
          "**Abstract.** We study %s in a maintenance-lattice framework: a directionally ordered medium eroded by a "
          "state-dependent flux and restored by repair. An order-entropy bridge links U_chi to entropy; the mean-field "
          "reduction dI/dtau = mu(I)(1-I) - rho I is bistable under feedback; the NESS carries positive entropy "
          "production." % subtitle.lower(), ""]
    for i, key in enumerate(figs, 1):
        md += ["## %d. %s" % (i, lab[key].rstrip(".")), "",
               "![](%s)" % os.path.abspath(figs[key]), "", "*Figure %d. %s*" % (i, lab[key]), ""]
    md += ["## References", "", "[1] Mardia & Jupp, Directional Statistics, Wiley 2000.",
           "[2] Schnakenberg, Rev. Mod. Phys. 48:571, 1976."]
    mdp = os.path.join(outdir, "manuscript.md"); open(mdp, "w").write("\n".join(md))
    return {"markdown": mdp, **{"fig_"+k: v for k, v in figs.items()}}

# ---------- demo / CLI ----------
def demo():
    import matplotlib; matplotlib.use("Agg")
    c = Config(L=16, n_seeds=1, equil_sweeps=30, meas_sweeps=30)
    print("dS/dU(3) =", float(Bridge.dS_dU(3)))
    print("bistable window:", MeanField(c).bistable_window(), "| I*_lin:", round(MeanField(c).I_star_linear(), 3))
    print("NESS:", {k: round(v, 4) for k, v in MC(c).run_ness().items()})
    print("EPR>0:", Analyses(c).entropy_production()["sigma_positive"])
    print("grid:", grid_size())
    r = reproduce_paper("Demo Draft", "a one-call test", preset="magnetism_xy", outdir="_demo")
    print("reproduce_paper ->", {k: os.path.basename(v) if v else None for k, v in r.items()})
    print("DEMO OK")
def main():
    ap = argparse.ArgumentParser(description="MyUncle_core")
    ap.add_argument("--preset", choices=list(PRESETS), default="chiral_scaffold_biology")
    ap.add_argument("--analysis", choices=["bridge", "bifurcation", "collapse", "epr", "fss", "ness"], default="ness")
    ap.add_argument("--reproduce", action="store_true"); ap.add_argument("--title", default="Information Maintenance in a Rotational Lattice")
    ap.add_argument("--subtitle", default="an order-entropy bridge and its mean-field reduction")
    ap.add_argument("--out", default=None); ap.add_argument("--demo", action="store_true"); ap.add_argument("--grid-size", action="store_true")
    a = ap.parse_args()
    if a.grid_size: print(grid_size()); return
    if a.demo: demo(); return
    if a.reproduce:
        print(reproduce_paper(a.title, a.subtitle, preset=a.preset, outdir=a.out or "myuncle_paper", include=("bridge", "bifurcation", "fss", "epr", "collapse"))); return
    c = PRESETS[a.preset]
    if a.analysis == "ness": print(MC(c).run_ness())
    else:
        import matplotlib; matplotlib.use("Agg")
        print(getattr(Analyses(c), {"bridge": "entropy_bridge", "bifurcation": "bifurcation", "collapse": "dimensionless_collapse",
              "epr": "entropy_production", "fss": "finite_size_scaling"}[a.analysis])(save=a.out or a.analysis+".png"))
if __name__ == "__main__": main()
