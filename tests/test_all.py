# -*- coding: utf-8 -*-
"""Automated checks:  python tests/test_all.py   (or python -m pytest tests -q)"""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from myuncle_vision import _core, lattice as lat, wsi, synthetic as S   # noqa: E402
from myuncle_vision import cohort, discover                              # noqa: E402

CORE = _core.load()


# ------------------------------------------------------------------------------------------------ the vendored MyUncle core is untouched and reused faithfully
def test_core_is_unmodified():
    assert _core.verify(), "third_party/myuncle/MyUncle_core.py differs from the recorded release"


def test_local_disorder_equals_myuncle_for_d2():
    L = 24
    rng = np.random.default_rng(3)
    th = rng.uniform(-np.pi, np.pi, (L, L))
    mc = CORE.MC(CORE.Config(L=L, seed=1))
    mc.th = th
    ref = mc.local_disorder()                                                           # MyUncle: 1 - |mean of the four neighbours' unit vectors|, periodic
    U = np.stack([np.cos(th), np.sin(th)], -1)
    m = lat.local_order_maps(U, np.ones((L, L), bool), neighbors=4, min_neighbors=0, periodic=True)
    assert np.allclose(m["D"], ref, atol=1e-12)


def test_vmf_entropy_equals_myuncle_bridge_for_d2():
    k = np.array([0.0, 0.05, 0.5, 1.0, 3.0, 10.0, 80.0])
    assert np.allclose(lat.vmf_entropy(k, 2), CORE.Bridge.S(k), atol=1e-9)
    assert np.allclose(lat.order_information(k, 2), CORE.Bridge.H(k), atol=1e-9)
    assert np.allclose(lat.A_d(k, 2), CORE.Bridge.A(k), atol=1e-12)
    assert abs(lat.entropy_uniform(2) - CORE.Bridge.Smax) < 1e-12


def test_kappa_inversion_roundtrip_high_dimension():
    for d in (2, 16, 384):
        k = np.array([0.5, 5.0, 50.0, 500.0]) * (d / 16.0 if d > 2 else 1.0)
        R = lat.A_d(k, d)
        assert np.allclose(lat.kappa_from_rbar(R, d), k, rtol=2e-3), d


def test_entropy_decreases_with_concentration():
    for d in (2, 16, 64):
        s = lat.vmf_entropy(np.array([0.1, 1.0, 10.0, 100.0]), d)
        assert np.all(np.diff(s) < 0) and s[0] <= lat.entropy_uniform(d) + 1e-9


def test_order_maps_separate_aligned_from_random():
    rng = np.random.default_rng(0)
    d, L = 16, 20
    v = lat.to_unit(rng.normal(size=d))
    aligned = lat.to_unit(v[None, None, :] + 0.02 * rng.normal(size=(L, L, d)))
    rand = lat.to_unit(rng.normal(size=(L, L, d)))
    valid = np.ones((L, L), bool)
    Da = np.nanmean(lat.local_order_maps(aligned, valid)["D"])
    Dr = np.nanmean(lat.local_order_maps(rand, valid)["D"])
    Ha = np.nanmean(lat.local_order_maps(aligned, valid)["H_chi"])
    Hr = np.nanmean(lat.local_order_maps(rand, valid)["H_chi"])
    assert Da < 0.01 and 0.4 < Dr < 0.6 and Ha > 5 * Hr                          # random unit vectors: the mean of 4 has length ~ 1/2, so D ~ 0.5 in high dimension


def test_invalid_sites_are_excluded():
    rng = np.random.default_rng(1)
    U = lat.to_unit(rng.normal(size=(10, 10, 8)))
    valid = np.ones((10, 10), bool)
    valid[:, 5] = False
    m = lat.local_order_maps(U, valid, min_neighbors=2)
    assert np.all(np.isnan(m["D"][:, 5]))
    U2 = U.copy()
    U2[:, 5] = 9.0                                                                       # whatever sits on an invalid site must not leak into its neighbours
    assert np.allclose(np.nan_to_num(lat.local_order_maps(U2, valid, min_neighbors=2)["D"]), np.nan_to_num(m["D"]))


def test_state_entropy_and_adjacency():
    lab = np.zeros((8, 8), int)
    lab[:, 4:] = 1
    valid = np.ones((8, 8), bool)
    H = lat.state_entropy(lab, valid, 2, 1)
    assert H[0, 0] == 0.0 and H[3, 3] > 0.3
    A = lat.adjacency_counts(lab, valid, 2)
    assert A[0, 1] == A[1, 0] == 8 and A[0, 0] > 0


# ------------------------------------------------------------------------------------------------ slides, tissue, grid, QC
def _tiny_slide(tmp, seed=1, an=(), ar=(), size=1792):
    img, truth = S.make_slide(seed=seed, size=size, anomalies=an, artifacts=ar, anomaly_radius=130)
    p = os.path.join(tmp, "s%d.tif" % seed)
    S.write_tiff(p, img, 0.5)
    return p, img, truth


def test_slide_backend_mpp_and_tissue_mask():
    with tempfile.TemporaryDirectory() as tmp:
        p, img, truth = _tiny_slide(tmp)
        sl = wsi.Slide(p)
        assert sl.mpp == 0.5 and not sl.mpp_assumed and sl.dimensions == (1792, 1792)
        g = wsi.patch_grid(sl, 224, 0.5)
        inter = (g["mask"] & cv2_resize(truth["tissue"], g["mask"].shape)).sum() / max(g["mask"].sum(), 1)
        assert inter > 0.9
        assert g["shape"] == (8, 8) and np.all(g["x0"] == g["cols"] * 224.0)
        patch = sl.read(g["x0"][0], g["y0"][0], g["side0"], 224)
        assert patch.shape == (224, 224, 3)
        ref = img[int(g["y0"][0]):int(g["y0"][0]) + 224, int(g["x0"][0]):int(g["x0"][0]) + 224]
        assert np.array_equal(patch[:ref.shape[0], :ref.shape[1]], ref[:224, :224])


def cv2_resize(m, shape):
    import cv2
    return cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def test_patch_qc_flags_artifacts_and_keeps_tissue():
    rng = np.random.default_rng(0)
    tissue = np.clip(np.array([228, 168, 198]) + rng.normal(0, 6, (224, 224, 3)), 0, 255).astype(np.uint8)
    pen = tissue.copy()
    pen[60:110, :] = (28, 58, 190)
    fold = (tissue * 0.2).astype(np.uint8)
    glass = np.full((224, 224, 3), 247, np.uint8)
    assert not wsi.patch_qc(tissue)["artifact"]
    assert wsi.patch_qc(pen)["artifact"] and "pen" in wsi.patch_qc(pen)["reason"]
    assert "fold_or_debris" in wsi.patch_qc(fold)["reason"] and "glass" in wsi.patch_qc(glass)["reason"]
    qc = dict(blur=np.array([50.0, 55.0, 48.0, 0.5]), pen=np.zeros(4), dark=np.zeros(4), white=np.zeros(4), artifact=np.zeros(4, bool), reason=np.array([""] * 4, dtype=object))
    f = wsi.finalize_qc(qc)
    assert list(f["artifact"]) == [False, False, False, True] and f["reason"][3] == "blur"


# ------------------------------------------------------------------------------------------------ encoders and cohort
def test_handcrafted_encoder_is_deterministic_and_discriminates():
    from myuncle_vision.encoder import HandcraftedEncoder
    rng = np.random.default_rng(0)
    a = np.clip(np.array([228, 168, 198]) + rng.normal(0, 6, (4, 224, 224, 3)), 0, 255).astype(np.uint8)
    b = np.clip(np.array([172, 190, 214]) + rng.normal(0, 6, (4, 224, 224, 3)), 0, 255).astype(np.uint8)
    e = HandcraftedEncoder()
    Za, Zb = e.encode(a), e.encode(b)
    assert np.array_equal(Za, e.encode(a)) and Za.shape == (4, 64) and np.all(np.isfinite(Za))
    assert np.linalg.norm(Za.mean(0) - Zb.mean(0)) > 5 * np.linalg.norm(Za.std(0))


def test_dinov2_encoder_if_available():
    """the real encoder: 384-d, deterministic, finite; skipped when torch/timm/weights are missing"""
    try:
        from myuncle_vision.encoder import DinoV2Encoder
        enc = DinoV2Encoder("cpu")
    except Exception as e:
        print("   (skipped: DINOv2 unavailable: %s)" % type(e).__name__)
        return
    rng = np.random.default_rng(0)
    x = np.clip(np.array([228, 168, 198]) + rng.normal(0, 12, (3, 224, 224, 3)), 0, 255).astype(np.uint8)
    z = enc.encode(x)
    assert z.shape == (3, 384) and np.all(np.isfinite(z)) and np.allclose(z, enc.encode(x), atol=1e-4)
    y = np.clip(np.array([60, 80, 150]) + rng.normal(0, 12, (3, 224, 224, 3)), 0, 255).astype(np.uint8)
    assert np.linalg.norm(z.mean(0) - enc.encode(y).mean(0)) > 1.0                      # different colour/texture, different embedding


def test_frozen_defaults_and_scorer_option():
    """frozen settings: appearance only, multi-scale, 3 x 3 smoothing; the scorer option and the smoothing change the score as documented"""
    import json as _j
    fz = _j.load(open(os.path.join(os.path.dirname(HERE), "results", "frozen_settings.json")))
    cfg = cohort.Config()
    assert tuple(cfg.weights) == (1.0, 0.0, 0.0) and cfg.multiscale and cfg.smooth and cfg.scorer == "knn"
    assert list(cfg.rarity_scales) == fz["shared"]["rarity_scales"] and cfg.n_components == fz["shared"]["pca_components"] and cfg.n_states == fz["shared"]["n_states"]
    with tempfile.TemporaryDirectory() as tmp:
        ind, slides, truths, rc = _mini_cohort(tmp)
        out = {}
        for name, kw in (("default", {}), ("raw", dict(smooth=False)), ("maha", dict(scorer="mahalanobis")), ("three", dict(weights=(0.4, 0.4, 0.2)))):
            for s in slides:
                s.extra = {k: v for k, v in s.extra.items() if k in ("path", "backend", "dimensions", "mpp_assumed", "mpp_eff")}
            cohort.analyse(slides, cohort.Config(**kw))
            out[name] = [s.extra["novelty"].copy() for s in slides]
        s0 = slides[0]
        for name in ("default", "raw", "maha", "three"):
            assert not np.any(np.isfinite(out[name][0][~s0.ok]))                       # artifact patches never get a score
        a, b = np.concatenate(out["default"]), np.concatenate(out["raw"])
        ok = np.isfinite(a) & np.isfinite(b)
        assert np.abs(a[ok] - b[ok]).max() > 0.05 and np.nanmax(a) <= 1.0              # smoothing changes the score and keeps it a rank in [0, 1]
        assert not np.allclose(np.nan_to_num(np.concatenate(out["maha"])), np.nan_to_num(a))
        assert not np.allclose(np.nan_to_num(np.concatenate(out["three"])), np.nan_to_num(a))


def test_smooth_lattice_is_a_local_mean():
    class Sl:
        shape = (4, 4)
        rows = np.arange(4).repeat(4)
        cols = np.tile(np.arange(4), 4)
    sc = np.zeros(16)
    sc[5] = 9.0                                                                          # one site at (1, 1)
    out = cohort.smooth_lattice(Sl, sc)
    assert abs(out[5] - 1.0) < 1e-9 and abs(out[0] - 9.0 / 4) < 1e-9 and abs(out[15]) < 1e-12   # centre: 9/9, neighbour corner (0, 0): 9/4 (window cut by the border), far corner 0


def _mini_cohort(tmp):
    from myuncle_vision.pipeline import RunConfig, load_slide
    from myuncle_vision.encoder import HandcraftedEncoder
    specs = [(11, ("disorganized", "novel_state"), ("blur",)), (12, ("novel_state",), ()), (13, ("disorganized",), ("fold",)), (14, (), ())]
    truths, slides = {}, []
    ind = os.path.join(tmp, "in")
    os.makedirs(ind)
    for seed, an, ar in specs:
        img, tr = S.make_slide(seed=seed, size=3136, anomalies=an, artifacts=ar, anomaly_radius=230)
        S.write_tiff(os.path.join(ind, "m%d.tif" % seed), img, 0.5)
        truths["m%d" % seed] = tr
    cfg = RunConfig(encoder="handcrafted", figures=False, progression=False)
    enc = HandcraftedEncoder()
    for f in sorted(os.listdir(ind)):
        slides.append(load_slide(os.path.join(ind, f), cfg, enc, tmp))
    return ind, slides, truths, cfg


def test_cohort_pipeline_ranks_planted_anomalies_and_excludes_artifacts():
    from sklearn.metrics import roc_auc_score
    from myuncle_vision.validate import _truth_arrays
    with tempfile.TemporaryDirectory() as tmp:
        ind, slides, truths, cfg = _mini_cohort(tmp)
        cohort.analyse(slides, cohort.Config(seed=0))
        y, sc, art, qc_art, art_truth = [], [], [], [], []
        for s in slides:
            t = _truth_arrays(s, truths[s.name])
            an = np.maximum(t["disorganized"], t["novel_state"]) >= 0.5
            a = (t["blur"] >= 0.9) | (t["fold"] >= 0.3)                          # artifacts that cover the patch (a partly blurred patch keeps the variance of its sharp part and escapes the QC)
            y.append(an)
            sc.append(s.extra["novelty"])
            qc_art.append(~s.ok)
            art_truth.append(a)
        y, sc, qa, at = np.concatenate(y), np.concatenate(sc), np.concatenate(qc_art), np.concatenate(art_truth)
        ok = np.isfinite(sc)
        assert roc_auc_score(y[ok], sc[ok]) > 0.7
        assert at.sum() >= 2 and qa[at].mean() > 0.9                            # patches covered by an artifact are excluded
        assert not np.any(np.isfinite(sc[qa]))                                  # excluded patches get no novelty score


def test_findings_are_deterministic_and_regions_geometry_is_consistent():
    with tempfile.TemporaryDirectory() as tmp:
        ind, slides, truths, cfg = _mini_cohort(tmp)
        out = []
        for _ in range(2):
            for s in slides:
                s.extra = {k: v for k, v in s.extra.items() if k in ("path", "backend", "dimensions", "mpp_assumed", "mpp_eff")}
            cohort.analyse(slides, cohort.Config(seed=0))
            thr, regions = discover.candidates(slides, 0.04, 3)
            out.append((round(thr, 10), [(r["slide"], r["n"], round(r["novelty"], 8)) for r in regions]))
        assert out[0] == out[1]
        for r in regions:
            s = slides[r["slide"]]
            assert r["x1"] - r["x0"] >= s.side0 - 1e-6 and r["y0"] >= 0 and r["x1"] <= s.side0 * s.shape[1] + 1e-6


def test_progression_uses_the_myuncle_simulation():
    from myuncle_vision import progression
    with tempfile.TemporaryDirectory() as tmp:
        ind, slides, truths, cfg = _mini_cohort(tmp)
        cohort.analyse(slides, cohort.Config(seed=0))
        out = progression.scenarios(slides[0], Ls=16, steps=10, seeds=1)
        assert 0 < out["I_obs"] < 1 and out["rho_hat"] > 0 and out["bistable_window"] is not None
        sc = out["scenarios"]
        assert set(sc) == {"repair_x0.5", "repair_x1", "repair_x2"}
        assert sc["repair_x2"]["intact_end"] >= sc["repair_x0.5"]["intact_end"] - 0.05      # more repair does not lower the intact fraction


# ------------------------------------------------------------------------------------------------ the single command
def test_cli_run_single_command_writes_all_outputs():
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        ind = os.path.join(tmp, "in")
        os.makedirs(ind)
        for seed, an in ((21, ("disorganized",)), (22, ("novel_state",)), (23, ())):
            img, _ = S.make_slide(seed=seed, size=2240, anomalies=an, anomaly_radius=170)
            S.write_tiff(os.path.join(ind, "c%d.tif" % seed), img, 0.5)
        out = os.path.join(tmp, "out")
        r = subprocess.run([sys.executable, "-m", "myuncle_vision", "run", "--input", ind, "--output", out, "--encoder", "handcrafted", "--q", "0.05", "--scenarios"], cwd=os.path.dirname(HERE), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        for f in ("findings.json", "findings.csv", "images_used.csv", "summary_statistics.csv", "run_manifest.json"):
            assert os.path.exists(os.path.join(out, f)), f
        man = json.load(open(os.path.join(out, "run_manifest.json")))
        assert man["myuncle_core_unmodified"] is True and man["n_slides_analysed"] == 3
        assert json.load(open(os.path.join(out, "slides", "c21", "progression_scenarios.json")))["exploratory"] is True       # the scenario analysis is labelled exploratory
        for n in ("c21", "c22", "c23"):
            for f in ("annotations.json", "patch_scores.csv", "novelty_mask.png", "heatmaps.png", "progression_scenarios.json"):
                assert os.path.exists(os.path.join(out, "slides", n, f)), (n, f)
        used = open(os.path.join(out, "images_used.csv")).read().splitlines()
        assert len(used) == 4 and all("used" in u for u in used[1:])


def test_python_api_with_custom_encoder_on_arrays():
    """discover_arrays + CallableEncoder: a user-supplied embedding function on in-memory images, no files prepared by the user"""
    import myuncle_vision as muv
    rng = np.random.default_rng(0)
    imgs = []
    for seed in (21, 22, 23):
        img, _ = S.make_slide(seed=seed, size=1792, anomalies=(), anomaly_radius=150)
        imgs.append(np.asarray(img))
    calls = []

    def fn(p):
        calls.append(len(p))
        return np.stack([p.reshape(len(p), -1, 3).mean(1)[:, 0], p.reshape(len(p), -1, 3).std(1)[:, 1], p.reshape(len(p), -1, 3).mean(1)[:, 2]], 1) + 0.0 * rng.random((len(p), 3))
    enc = muv.CallableEncoder(fn, dim=3, name="toy")
    with tempfile.TemporaryDirectory() as tmp:
        res = muv.discover_arrays(imgs, os.path.join(tmp, "o"), encoder=enc, log=lambda *a: None, figures=False, progression=False)
        assert sum(calls) > 0
        used = open(os.path.join(tmp, "o", "images_used.csv")).read().splitlines()
        assert len(used) == 4
        man = json.load(open(os.path.join(tmp, "o", "run_manifest.json")))
        assert "toy" in json.dumps(man)


def test_encoder_module_loader_and_demo_command():
    import subprocess
    from myuncle_vision.encoder import load_encoder_module
    enc = load_encoder_module(os.path.join(HERE, "..", "examples", "custom_encoder.py") + ":encode", 48)
    z = enc.encode(np.random.default_rng(0).integers(0, 255, (5, 32, 32, 3), dtype=np.uint8))
    assert z.shape == (5, 48) and np.all(np.isfinite(z))
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, "-m", "myuncle_vision", "demo", "--output", os.path.join(tmp, "d")], cwd=os.path.join(HERE, ".."), capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-800:]
        assert os.path.exists(os.path.join(tmp, "d", "findings.csv")) and os.path.exists(os.path.join(tmp, "d", "run_manifest.json"))


def test_bridge_myuncle_to_vision_and_back():
    """MyUncle state -> MyUncleVision order maps equal MyUncle's own; novelty map is defined; slide -> MyUncle gives a runnable MC state; CLI report is written"""
    from myuncle_vision import bridge as B
    snaps, mc = B.simulate("chiral_scaffold_biology", L=20, seed=1, steps=20, every=10)
    chk = B.verify_bridge(mc)
    assert chk["equal"] and chk["max_abs_diff_D"] < 1e-12, chk
    res = B.analyse_myuncle_states(snaps)
    assert len(res) == len(snaps) and all(np.isfinite(r["novelty"]).mean() > 0.95 for r in res)
    with tempfile.TemporaryDirectory() as tmp:
        ind, slides, truths, cfg = _mini_cohort(tmp)
        cohort.analyse(slides, cohort.Config(seed=0))
        mc2, I_obs = B.slide_to_myuncle(slides[0], Ls=16)
        o0 = mc2.order()
        for _ in range(3):
            mc2.step()
        assert mc2.th.shape == (16, 16) and 0 < I_obs < 1 and np.isfinite(o0) and np.isfinite(mc2.order())
        rep = B.json_report("chiral_scaffold_biology", 16, 10, 5, 0, os.path.join(tmp, "br"))
        assert rep["check"]["equal"] and os.path.exists(os.path.join(tmp, "br", "bridge_maps.png")) and rep["core_unmodified"]


def test_bridge_bench_states_truth_and_run_level_bootstrap():
    from myuncle_vision import bridge_bench as BB
    from sklearn.metrics import roc_auc_score
    states, truth, meta = BB.make_states(dict(presets=("chiral_scaffold_biology",), seeds=(0, 1)), L=16)
    assert len(states) == 2 * len(BB.SNAP_STEPS) and all(t.shape == (16, 16) and t.dtype == bool for t in truth)
    assert all(abs(m["damaged_fraction"] - t.mean()) < 1e-12 for m, t in zip(meta, truth))
    sl = BB._slides(states, "rotor")
    assert sl[0].emb.shape[1] == 2 and BB._slides(states, "rotor_D")[0].emb.shape[1] == 3         # the intact flag is not part of the embedding
    rng = np.random.default_rng(0)
    ys = [(rng.random(200) < 0.2).astype(int) for _ in range(6)]
    ss = [y + rng.normal(0, 1, 200) for y in ys]
    lo, hi = BB._boot_ci(roc_auc_score, ys, ss, n=50)
    assert 0.5 < lo < hi <= 1.0


def test_smoothing_options_and_gaar_vendored_search():
    """smoothing types/windows are backward compatible; GaaR is vendored unmodified and searches the configuration grid reproducibly"""
    import hashlib
    from myuncle_vision import gaar_opt as GO
    from myuncle_vision.cohort import smooth_lattice
    rec = open(os.path.join(HERE, "..", "third_party", "gaar", "GaaR_core.sha256")).read().split()[0]
    assert hashlib.sha256(open(GO.GAAR_PATH, "rb").read()).hexdigest() == rec, "vendored GaaR_core.py differs from the recorded copy"

    class Sl:
        shape = (6, 6)
        rows = np.repeat(np.arange(6), 6)
        cols = np.tile(np.arange(6), 6)
    sc = np.random.default_rng(1).random(36)
    sc[5] = np.nan
    assert np.allclose(smooth_lattice(Sl, sc, 1), sc, equal_nan=True)
    ref = smooth_lattice(Sl, sc)                                                              # the frozen default: mean over 3 x 3
    assert np.allclose(smooth_lattice(Sl, sc, 3, "mean"), ref, equal_nan=True)
    for kind in ("gaussian", "median"):
        out = smooth_lattice(Sl, sc, 5, kind)
        assert np.isnan(out[5]) and np.isfinite(np.delete(out, 5)).all()
    cfg = GO.to_config(GO.DEFAULT)
    assert cfg.smooth_window == 3 and cfg.smooth_type == "mean" and cfg.weights == (1.0, 0.0, 0.0) and cfg.n_components == 16 and max(cfg.rarity_scales) == 0.10
    assert len(GO.all_genomes()) == 360
    G = GO.load_gaar()
    rng = np.random.default_rng(0)
    table = {json.dumps(g, sort_keys=True): float(rng.random()) for g in GO.all_genomes()}
    runs = []
    for _ in range(2):
        dom = GO.make_domain(G, table)
        dom.reseed(7)
        runs.append(G.GaaR(dom, strategy="greedy", seed=3, db_path=os.devnull, log=False).run(budget=30, verbose=False)["best_score"])
    assert runs[0] == runs[1]                                                                 # seeded: reproducible
    assert abs(GO.fitness_from([0.4, 0.8]) - (0.5 * 0.6 + 0.5 * 0.4)) < 1e-12
    assert np.isfinite(GO.moran_I(Sl, np.random.default_rng(2).random(36)))


def test_morphology_generator_holds_prevalence_and_separability_fixed():
    from myuncle_vision import morphology as M
    prev = []
    for b in (1, 2, 6):
        X, m = M.make_lattice(5, block=b, **M.GEN)
        assert X.shape == (32, 32, M.GEN["d"]) and m.dtype == bool
        prev.append(m.mean())
        assert 0.02 < m.mean() < 0.04                                        # prevalence held near 3% at every block size
    X1, m1 = M.make_lattice(5, block=1, **M.GEN)
    X1b, m1b = M.make_lattice(5, block=1, **M.GEN)
    assert np.array_equal(X1, X1b) and np.array_equal(m1, m1b)               # seeded
    X6, m6 = M.make_lattice(5, block=6, **M.GEN)
    lab = __import__("scipy.ndimage", fromlist=["label"]).label(m6)[1]
    assert lab >= 1 and m6.sum() % 36 == 0                                   # whole 6 x 6 blocks


def test_openslide_svs_backend_if_sample_present():
    """reads a real Aperio SVS when tests/data/*.svs exists (the public OpenSlide test file CMU-1-Small-Region.svs); skipped otherwise"""
    d = os.path.join(HERE, "data")
    svs = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".svs")] if os.path.isdir(d) else []
    if not svs or wsi.openslide is None:
        print("   (skipped: no SVS sample or no openslide)")
        return
    sl = wsi.Slide(svs[0])
    assert sl.backend == "openslide" and sl.dimensions[0] > 100
    g = wsi.patch_grid(sl, 224, 0.5)
    assert len(g["rows"]) > 0
    p = sl.read(g["x0"][0], g["y0"][0], g["side0"], 224)
    assert p.shape == (224, 224, 3) and p.std() > 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("ok  ", f.__name__)
    print("%d checks passed" % len(fns))
