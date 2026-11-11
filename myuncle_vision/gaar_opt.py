# -*- coding: utf-8 -*-
"""Configuration search for the MyUncleVision score with GaaR (Goal-Aware Adaptive Regulation, github.com/sandlerleon/GaaR, vendored unmodified in third_party/gaar).

What this is, and what it is not.  GaaR searches a small grid of scoring configurations (lattice smoothing type and window, axis weights, PCA dimension, largest rarity scale) for the one with the best
fitness on DEVELOPMENT scenarios whose labels are known.  That is supervised hyperparameter selection: the selected configuration is fixed afterwards and applied to unlabeled data.  It is not label-free
adaptation to a new image.  The champion is then evaluated once on fresh test scenarios (new seeds, one unseen MyUncle preset) that no search ever saw.

Fitness over the development scenarios s (average precision AP_s):   F = 0.5 * mean_s(AP_s) + 0.5 * min_s(AP_s)      (a lower-tail objective: no scenario may be sacrificed).
"""
import importlib.util
import itertools
import json
import os
import random
import sys

import numpy as np
from sklearn.metrics import average_precision_score

from . import bench as BN
from . import bridge as BR
from . import bridge_bench as BB
from . import cohort as C
from .encoder import get_encoder

HERE = os.path.dirname(os.path.abspath(__file__))
GAAR_PATH = os.path.join(HERE, "..", "third_party", "gaar", "GaaR_core.py")
BENCH_ROOT = os.environ.get("MUV_BENCH_ROOT", r"C:\Users\Leon\Downloads\MyUncleVision_work\bench")

SMOOTH = {"none": ("mean", 1), "mean3": ("mean", 3), "mean5": ("mean", 5), "mean7": ("mean", 7), "gauss5": ("gaussian", 5), "gauss7": ("gaussian", 7), "median3": ("median", 3), "median5": ("median", 5)}
AXES = {"appearance": (1.0, 0.0, 0.0), "app+het 75/25": (0.75, 0.25, 0.0), "app+het 50/50": (0.5, 0.5, 0.0), "app+adj 75/25": (0.75, 0.0, 0.25), "three-axis 40/40/20": (0.4, 0.4, 0.2)}
SCALES_ALL = (0.005, 0.01, 0.02, 0.05, 0.10)
GRID = {"smooth": list(SMOOTH), "axes": list(AXES), "pca": [8, 16, 32], "max_scale": [0.02, 0.05, 0.10]}
DEFAULT = {"smooth": "mean3", "axes": "appearance", "pca": 16, "max_scale": 0.10}          # the frozen MyUncleVision configuration


def load_gaar():
    spec = importlib.util.spec_from_file_location("GaaR_core", GAAR_PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["GaaR_core"] = m
    spec.loader.exec_module(m)
    return m


def to_config(g, n_states=12, seed=0, **extra):
    kind, w = SMOOTH[g["smooth"]]
    return C.Config(null_permutations=0, weights=AXES[g["axes"]], n_components=int(g["pca"]), rarity_scales=tuple(s for s in SCALES_ALL if s <= g["max_scale"] + 1e-9),
                    smooth=w > 1, smooth_window=w, smooth_type=kind, n_states=n_states, seed=seed, **extra)


# ------------------------------------------------------------------------------------------------ scenarios
class Scenario:
    """slides with embeddings + a function that returns (y, score) pooled over the evaluable patches after C.analyse"""

    def __init__(self, name, slides, truth_fn, n_states=12, cfg_extra=None, unit=1):
        self.name, self.slides, self.truth_fn, self.n_states, self.cfg_extra = name, slides, truth_fn, n_states, cfg_extra or {}
        self.unit = unit                  # slides per independent unit (1 for images; 3 snapshots of one simulation run for lattices)

    def units(self, g):
        """per-slide (y, score) after analysing the scenario with configuration g"""
        for s in self.slides:
            s.extra = {k: v for k, v in s.extra.items() if k in ("path", "backend", "dimensions", "mpp_assumed", "mpp_eff")}
        C.analyse(self.slides, to_config(g, self.n_states, **self.cfg_extra))
        out = []
        for i, s in enumerate(self.slides):
            y, sel = self.truth_fn(i, s)
            sc = np.asarray(s.extra["novelty"], float)
            m = sel & np.isfinite(sc)
            out.append((y[m], sc[m]))
        return out

    def ap(self, g):
        u = self.units(g)
        y, sc = np.concatenate([a for a, _ in u]), np.concatenate([b for _, b in u])
        return float(average_precision_score(y, sc)) if 0 < y.sum() < len(y) else float("nan")


def image_scenario(kind, encoder_name):
    enc = get_encoder("handcrafted" if encoder_name == "handcrafted" else "dinov2")
    slides, truth = BN.load_cohort(kind, BENCH_ROOT, enc, log=lambda *a: None)
    anom = BN.DEV_ANOM if kind == "dev" else BN.HELD_ANOM

    def fn(i, s):
        t = truth[i]
        T = lambda k: t.get(k, np.zeros(len(s.rows)))
        an = np.max([T(a) for a in anom], axis=0) >= 0.5
        art = np.max([T(a) for a in BN.ART], axis=0) >= 0.25
        clean = (np.max([T(a) for a in anom], axis=0) < 0.05) & (np.max([T(a) for a in BN.ART], axis=0) < 0.05)
        return an.astype(int), s.ok & (an | clean) & ~art
    return Scenario("images-%s-%s" % (kind, encoder_name), slides, fn)


def sim_scenario(spec, embedding, name):
    states, truth, meta = BB.make_states(spec, L=48)
    slides = BB._slides(states, embedding)
    flat = [t.reshape(-1).astype(int) for t in truth]
    return Scenario(name, slides, lambda i, s: (flat[i], np.ones(len(flat[i]), bool)), n_states=6, cfg_extra=dict(ref_per_slide=350, ref_max=4200), unit=len(BB.SNAP_STEPS))     # reference sample of about 4,000 sites, as in the image scenarios


TEST_SIM = dict(presets=("superconducting_scaffold", "polymer_photodegradation", "liquid_crystal", "magnetic_storage_decay"), seeds=(20, 21, 22, 23))


def dev_scenarios():
    return [image_scenario("dev", "handcrafted"), image_scenario("dev", "dinov2_vits14"), sim_scenario(BB.DEV, "rotor", "sim-dev-rotor"), sim_scenario(BB.DEV, "rotor_D", "sim-dev-rotor_D")]


def test_scenarios():
    return [image_scenario("test", "handcrafted"), image_scenario("test", "dinov2_vits14"), sim_scenario(TEST_SIM, "rotor", "sim-test-rotor"), sim_scenario(TEST_SIM, "rotor_D", "sim-test-rotor_D")]


def fitness_from(aps):
    a = np.asarray(aps, float)
    return float(0.5 * np.nanmean(a) + 0.5 * np.nanmin(a))


# ------------------------------------------------------------------------------------------------ GaaR domain over a precomputed grid table
def make_domain(G, table):
    """GaaR Domain over the grid with a seeded random-number generator; fitness comes from the table of exhaustively evaluated development configurations (key = JSON of the genome)"""
    class SeededGrid(G.GridDomain):
        def __init__(self):
            super().__init__(GRID, lambda g: table[json.dumps(g, sort_keys=True)], name="myunclevision")
            self._rng = random.Random(0)

        def reseed(self, seed):
            self._rng = random.Random(seed)

        def seed_genome(self):
            return {k: self._rng.choice(v) for k, v in self.grid.items()}
    return SeededGrid()


def all_genomes():
    keys = list(GRID)
    return [dict(zip(keys, c)) for c in itertools.product(*[GRID[k] for k in keys])]


# ------------------------------------------------------------------------------------------------ command line
_SC = None


def _init_worker():
    import warnings
    warnings.filterwarnings("ignore")
    global _SC
    _SC = dev_scenarios()


def _eval_genome(g):
    aps = [sc.ap(g) for sc in _SC]
    return json.dumps(g, sort_keys=True), aps, fitness_from(aps)


def run_grid(out, workers=3):
    """evaluate every configuration of the grid on the development scenarios (resumable); the table is the fitness oracle of the search comparison"""
    import multiprocessing as mp
    done = json.load(open(out)) if os.path.exists(out) else {}
    todo = [g for g in all_genomes() if json.dumps(g, sort_keys=True) not in done]
    print("grid: %d configurations, %d to do" % (len(all_genomes()), len(todo)), flush=True)
    with mp.Pool(workers, initializer=_init_worker) as pool:
        for i, (k, aps, f) in enumerate(pool.imap_unordered(_eval_genome, todo), 1):
            done[k] = dict(aps=aps, fitness=f)
            if i % 10 == 0 or i == len(todo):
                json.dump(done, open(out, "w"))
                print("  %d/%d" % (i, len(todo)), flush=True)
    return done


STRATEGIES = ("greedy", "restart", "population", "random")


def run_search(grid_json, out, budget=40, repeats=100):
    """equal-budget comparison of the GaaR strategies on the exhaustively evaluated development table (a fixed-budget search finds how good a configuration?)"""
    G = load_gaar()
    table = {k: v["fitness"] for k, v in json.load(open(grid_json)).items()}
    vals = np.array(sorted(table.values(), reverse=True))
    best_all = float(vals[0])
    q = lambda f: float((vals > f).mean())
    res = dict(n_configs=len(vals), budget=budget, repeats=repeats, optimum=best_all, optimum_genomes=[k for k, v in table.items() if v == best_all], default_fitness=table[json.dumps(DEFAULT, sort_keys=True)],
               default_rank_fraction=q(table[json.dumps(DEFAULT, sort_keys=True)]), strategies={})
    thr1, thr5 = float(np.quantile(vals, 0.99)), float(np.quantile(vals, 0.95))
    for strat in STRATEGIES:
        bests, hit1, hit5, to5, champs = [], 0, 0, [], []
        for r in range(repeats):
            dom = make_domain(G, table)
            dom.reseed(1000 + r)
            out_ = G.GaaR(dom, strategy=strat, seed=r, db_path=os.devnull, log=False).run(budget=budget, verbose=False)
            curve = np.array(out_["curve"])
            bests.append(out_["best_score"]); champs.append(out_["best_genome"])
            hit1 += out_["best_score"] >= thr1; hit5 += out_["best_score"] >= thr5
            reach = np.where(curve >= thr5)[0]
            to5.append(int(reach[0]) + 1 if len(reach) else None)
        reached = [t for t in to5 if t is not None]
        res["strategies"][strat] = dict(median_best=float(np.median(bests)), mean_best=float(np.mean(bests)), q10_best=float(np.percentile(bests, 10)), hit_top1pct=hit1 / repeats, hit_top5pct=hit5 / repeats,
                                        median_evals_to_top5pct=float(np.median(reached)) if reached else None, share_reaching_top5pct=len(reached) / repeats, champion_seed0=champs[0], best_score_seed0=float(bests[0]))
        print(strat, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["strategies"][strat].items() if k != "champion_seed0"}, flush=True)
    json.dump(res, open(out, "w"), indent=1)
    return res


def moran_I(slide, scores):
    """Moran's I (queen contiguity) of a per-patch score on the patch lattice: label-free spatial autocorrelation of the score field"""
    a = np.full(slide.shape, np.nan)
    a[slide.rows, slide.cols] = scores
    v = np.isfinite(a)
    if v.sum() < 10:
        return float("nan")
    z = np.where(v, a - np.nanmean(a), 0.0)
    num = 0.0
    wsum = 0.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            A = np.roll(np.roll(z, dy, 0), dx, 1)
            V = np.roll(np.roll(v, dy, 0), dx, 1) & v
            num += float((z * A * V).sum())
            wsum += float(V.sum())
    den = float((z ** 2).sum())
    return float((v.sum() / max(wsum, 1)) * num / den) if den > 0 else float("nan")


def raw_score_moran(sc):
    """Moran's I of the unsmoothed multi-scale appearance rarity (label-free), averaged over the slides of a scenario"""
    for s in sc.slides:
        s.extra = {k: v for k, v in s.extra.items() if k in ("path", "backend", "dimensions", "mpp_assumed", "mpp_eff")}
    C.analyse(sc.slides, to_config(dict(DEFAULT, smooth="none"), sc.n_states, **sc.cfg_extra))
    vals = [moran_I(s, s.extra["p_emb_ms"]) for s in sc.slides]
    return float(np.nanmean(vals))


def bootstrap_test(units, group, names, B=300, seed=0, ref="frozen default"):
    """95% intervals by resampling independent units (images; simulation runs) within every scenario, the same draws for every configuration (paired).
    Returns, per configuration, the test fitness interval and the interval of its difference from the frozen default."""
    rng = np.random.default_rng(seed)
    n_sc = len(group)
    n_units = [len(units[names[0]][k]) // group[k] for k in range(n_sc)]
    F = {nm: [] for nm in names}
    AP = {nm: [[] for _ in range(n_sc)] for nm in names}
    for _ in range(B):
        draw = [rng.integers(0, n_units[k], n_units[k]) for k in range(n_sc)]
        for nm in names:
            aps = []
            for k in range(n_sc):
                parts = [units[nm][k][u * group[k] + j] for u in draw[k] for j in range(group[k])]
                y, f = np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])
                aps.append(float(average_precision_score(y, f)) if 0 < y.sum() < len(y) else float("nan"))
                AP[nm][k].append(aps[-1])
            F[nm].append(fitness_from(aps))
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    out = dict(B=B, units=n_units)
    for nm in names:
        d = np.array(F[nm]) - np.array(F[ref])
        out[nm] = dict(fitness_ci=ci(F[nm]), diff_vs_default_ci=ci(d), share_better_than_default=float(np.mean(d > 0)), ap_ci=[ci(AP[nm][k]) for k in range(n_sc)])
    return out


def run_test(search_json, out, grid_json):
    """evaluate the frozen default, the exhaustive development optimum and the champions of the searches once on the fresh test scenarios (never seen by any search), with the label-free statistic"""
    srch = json.load(open(search_json))
    grid = json.load(open(grid_json))
    key = lambda g: json.dumps(g, sort_keys=True)
    cands = {"frozen default": DEFAULT, "no smoothing": dict(DEFAULT, smooth="none"), "exhaustive dev optimum": json.loads(srch["optimum_genomes"][0])}
    for st in STRATEGIES:
        cands["GaaR %s (seed 0, budget %d)" % (st, srch["budget"])] = srch["strategies"][st]["champion_seed0"]
    scs = test_scenarios()
    res = dict(candidates={k: v for k, v in cands.items()}, test={}, dev={})
    units = {}
    for name, g in cands.items():
        res["dev"][name] = grid[key(g)]
        aps = []
        units[name] = []
        for sc in scs:
            u = sc.units(g)
            units[name].append(u)
            y, f = np.concatenate([a for a, _ in u]), np.concatenate([b for _, b in u])
            aps.append(float(average_precision_score(y, f)))
            print(name, sc.name, round(aps[-1], 3), flush=True)
        res["test"][name] = dict(aps=aps, fitness=fitness_from(aps))
    res["bootstrap"] = bootstrap_test(units, [sc.unit for sc in scs], list(cands))
    res["test_scenarios"] = [sc.name for sc in scs]
    res["moran_test"] = {sc.name: raw_score_moran(sc) for sc in scs}
    json.dump(res, open(out, "w"), indent=1)
    return res


def run_labelfree(grid_json, test_json, out):
    """is the effect of smoothing predictable from a label-free statistic?  effect = AP(mean3) - AP(none) at the default axes; statistic = Moran's I of the unsmoothed appearance score"""
    grid = json.load(open(grid_json))
    key = lambda g: json.dumps(g, sort_keys=True)
    dev_names = [sc.name for sc in dev_scenarios()]
    rows = []
    dev = dev_scenarios()
    for j, sc in enumerate(dev):
        rows.append(dict(scenario=sc.name, split="dev", effect=grid[key(DEFAULT)]["aps"][j] - grid[key(dict(DEFAULT, smooth="none"))]["aps"][j], moran=raw_score_moran(sc)))
    t = json.load(open(test_json))
    for j, nm in enumerate(t["test_scenarios"]):
        rows.append(dict(scenario=nm, split="test", effect=t["test"]["frozen default"]["aps"][j] - t["test"]["no smoothing"]["aps"][j], moran=t["moran_test"][nm]))
    json.dump(dict(rows=rows), open(out, "w"), indent=1)
    for r in rows:
        print("%-28s %-5s effect of mean3 smoothing on AP %+.3f   Moran I of raw score %.3f" % (r["scenario"], r["split"], r["effect"], r["moran"]))
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grid", "search", "test", "labelfree"])
    ap.add_argument("--out", default="results/gaar_grid_dev.json")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore")
    if a.cmd == "grid":
        run_grid(a.out, a.workers)
    elif a.cmd == "search":
        run_search("results/gaar_grid_dev.json", "results/gaar_search_comparison.json")
    elif a.cmd == "test":
        run_test("results/gaar_search_comparison.json", "results/gaar_test.json", "results/gaar_grid_dev.json")
    elif a.cmd == "labelfree":
        run_labelfree("results/gaar_grid_dev.json", "results/gaar_test.json", "results/gaar_labelfree.json")
