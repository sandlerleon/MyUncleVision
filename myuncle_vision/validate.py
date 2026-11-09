# -*- coding: utf-8 -*-
"""Validation on synthetic slides with planted anomalies and artifacts (see synthetic.py: NOT kidney tissue).

What this measures, and what it does not.  It measures whether the full pipeline, run exactly as in production, (i) ranks planted anomalous patches above normal patches, (ii) keeps artifacts out,
(iii) does not invent findings on slides with nothing planted, (iv) is reproducible.  It does not measure performance on kidney tissue.
"""
import json
import os
import tempfile
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from . import synthetic as S
from .pipeline import RunConfig, run

SPECS = [   # (seed, anomalies, artifacts)
    (101, ("disorganized", "novel_state"), ("blur", "pen")),
    (102, ("disorganized", "novel_state"), ("fold",)),
    (103, ("novel_state",), ("blur", "fold")),
    (104, ("disorganized",), ("pen",)),
    (105, ("novel_state",), ()),
    (106, ("disorganized",), ()),
    (107, (), ()),
    (108, (), ("blur", "pen", "fold")),
]
ANOM = ("disorganized", "novel_state")
ART = ("blur", "pen", "fold")


def _auc(y, s):
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def _truth_arrays(slide, truth):
    from .synthetic import patch_truth
    grid = dict(rows=slide.rows, cols=slide.cols, side0=slide.side0, shape=slide.shape)
    names = [n for n in ANOM + ART if n in truth]
    pt = patch_truth(truth, grid, names)
    z = np.zeros(slide.n)
    out = {n: pt.get(n, z) for n in ANOM + ART}
    return out


def validate(outdir="results", encoder="handcrafted", size=4480, specs=SPECS, workdir=None, q=0.03, log=print):
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    work = workdir or tempfile.mkdtemp(prefix="muv_synth_")
    ind = os.path.join(work, "input")
    os.makedirs(ind, exist_ok=True)
    truths = {}
    for seed, an, ar in specs:
        img, tr = S.make_slide(seed=seed, size=size, anomalies=an, artifacts=ar, anomaly_radius=int(size * 0.075))
        name = "synth_%d" % seed
        S.write_tiff(os.path.join(ind, name + ".tif"), img, 0.5)
        truths[name] = tr
        log("[synthetic] %s anomalies=%s artifacts=%s" % (name, an, ar))
    cfg = RunConfig(encoder=encoder, q=q, figures=False, progression=False)
    res = run(ind, os.path.join(work, "out"), cfg, log=lambda *a: None)
    slides = res["slides"]
    # ---- patch-level arrays
    rows = []
    for s in slides:
        t = _truth_arrays(s, truths[s.name])
        e = s.extra
        rows.append(dict(name=s.name, ok=s.ok, an={k: t[k] for k in ANOM}, art={k: t[k] for k in ART}, **{k: e[k] for k in ("novelty", "p_emb", "p_het", "p_adj", "D", "d_emb", "H_chi")}))
    cat = lambda f: np.concatenate([f(r) for r in rows])
    ok = cat(lambda r: r["ok"])
    art_any = cat(lambda r: np.max([r["art"][k] for k in ART], axis=0) >= 0.25)
    an_any = cat(lambda r: np.max([r["an"][k] for k in ANOM], axis=0) >= 0.5)
    normal = ok & ~art_any & cat(lambda r: np.max([r["an"][k] for k in ANOM], axis=0) < 0.05) & cat(lambda r: np.max([r["art"][k] for k in ART], axis=0) < 0.05)
    out = dict(encoder=res["manifest"]["encoder"], n_slides=len(slides), n_patches=int(len(ok)), n_analysed=int(ok.sum()), n_normal=int(normal.sum()), n_anomalous=int((an_any & ok).sum()))
    # ---- ranking quality (anomalous vs normal patches, artifacts excluded as the pipeline does)
    scores = ("novelty", "p_emb", "p_het", "p_adj", "D", "d_emb")
    auc = {}
    for nm in ("any",) + ANOM:
        pos = an_any if nm == "any" else cat(lambda r: r["an"][nm] >= 0.5)
        sel = ok & (pos | normal)
        auc[nm] = {sc: _auc(pos[sel].astype(int), cat(lambda r: r[sc])[sel]) for sc in scores}
    out["auroc_vs_normal"] = auc
    nov = cat(lambda r: r["novelty"])
    k = max(1, int(round(q * ok.sum())))
    top = np.argsort(-np.where(ok & np.isfinite(nov), nov, -1))[:k]
    # localization: a top patch counts when it lies within one lattice step of a planted anomalous patch (the rim of a region is where a score changes)
    from scipy.ndimage import binary_dilation
    near = []
    for s, r in zip(slides, rows):
        g = np.zeros(s.shape, bool)
        g[s.rows, s.cols] = np.max([r["an"][a] for a in ANOM], axis=0) >= 0.5
        gd = binary_dilation(g, structure=np.ones((3, 3), bool))
        near.append(gd[s.rows, s.cols])
    near = np.concatenate(near)
    out["top_q"] = dict(q=q, k=int(k), precision_planted_anomaly=float(an_any[top].mean()), precision_within_one_patch=float(near[top].mean()), precision_artifact=float(art_any[top].mean()), base_rate_anomaly=float(an_any[ok].mean()),
                        base_rate_within_one_patch=float(near[ok].mean()))
    # slide-level bootstrap of the AUROC of the novelty score (planted anomalies vs normal patches)
    rng = np.random.default_rng(0)
    sid = np.concatenate([np.full(len(r["ok"]), i) for i, r in enumerate(rows)])
    nvl = cat(lambda r: r["novelty"])
    sel_all = ok & (an_any | normal)
    boots = []
    for _ in range(300):
        pick = rng.integers(0, len(rows), len(rows))
        idx = np.concatenate([np.where((sid == p_) & sel_all)[0] for p_ in pick])
        if 0 < an_any[idx].sum() < len(idx):
            boots.append(roc_auc_score(an_any[idx].astype(int), nvl[idx]))
    out["auroc_novelty_any_ci95_slide_bootstrap"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None
    # ---- artifacts
    qc_art = cat(lambda r: ~r["ok"])
    out["qc"] = dict(artifact_patches_planted=int(art_any.sum()), flagged_among_planted=float(qc_art[art_any].mean()) if art_any.any() else float("nan"),
                     false_flag_rate_normal=float(qc_art[~art_any & ~an_any].mean()), false_flag_rate_anomaly=float(qc_art[an_any & ~art_any].mean()) if (an_any & ~art_any).any() else float("nan"))
    # ---- region level: does a finding region cover every planted anomaly instance?
    inst = []
    for s in slides:
        t = _truth_arrays(s, truths[s.name])
        cand = s.extra["cand_mask"][s.rows, s.cols]
        for a in ANOM:
            m = t[a] >= 0.5
            if m.sum() >= 3:
                inst.append(dict(slide=s.name, type=a, n_patches=int(m.sum()), hit=bool(cand[m].any()), frac_candidate=float(cand[m].mean())))
    out["instances"] = inst
    out["instance_recall"] = {a: float(np.mean([i["hit"] for i in inst if i["type"] == a])) if any(i["type"] == a for i in inst) else float("nan") for a in ANOM}
    # ---- null slides
    nulls = {}
    for s in slides:
        sp = [sp for sp in specs if "synth_%d" % sp[0] == s.name][0]
        if not sp[1]:
            nulls[s.name] = dict(n_candidate_patches=int(s.extra["cand_mask"].sum()), expected_by_construction=float(q * s.ok.sum()), order_z=s.extra["order"]["z"],
                                 n_regions_in_findings=int(sum(1 for f in res["findings"] for r in f["regions"] if r["slide_name"] == s.name)))
    out["null_slides"] = nulls
    out["findings"] = [dict(id=f["id"], n_slides=f["n_slides"], n_patches=f["n_patches"], type=f["type"], novelty=f["novelty"], repro=f["repro"],
                            planted=_finding_truth(f, slides, truths)) for f in res["findings"]]
    out["order"] = {s.name: dict(order_z=s.extra["order"]["z"], D_mean=s.extra["order"]["D_mean"], rho_hat=res["summary"][i]["rho_hat"], in_window=res["summary"][i]["in_bistable_window"]) for i, s in enumerate(slides)}
    # ---- ablation: no QC (artifacts left in the lattice and the reference)
    from . import cohort as C
    import copy
    sl2 = []
    for s in slides:
        s2 = copy.copy(s)
        s2.qc = dict(s.qc)
        s2.qc["artifact"] = np.zeros(s.n, bool)
        s2.extra = {}
        sl2.append(s2)
    C.analyse(sl2, C.Config(seed=0))
    nov2 = np.concatenate([s.extra["novelty"] for s in sl2])
    top2 = np.argsort(-np.where(np.isfinite(nov2), nov2, -1))[:k]
    out["ablation_no_qc"] = dict(precision_artifact_in_top_q=float(art_any[top2].mean()), precision_planted_anomaly_in_top_q=float(an_any[top2].mean()))
    try:
        _example_figure(slides[0], truths[slides[0].name], os.path.join(outdir, "validation_example_%s.png" % out["encoder"]))
    except Exception as e:                                                         # the figure is a convenience, never a reason to lose the numbers
        log("[validate] example figure skipped: %s" % type(e).__name__)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    out["headline"] = dict(encoder=out["encoder"], auroc_novelty_any=auc["any"]["novelty"], auroc_novelty_disorganized=auc["disorganized"]["novelty"], auroc_novelty_novel_state=auc["novel_state"]["novelty"],
                           precision_top_q=out["top_q"]["precision_planted_anomaly"], artifact_in_top_q=out["top_q"]["precision_artifact"], artifact_in_top_q_without_qc=out["ablation_no_qc"]["precision_artifact_in_top_q"],
                           instance_recall=out["instance_recall"], qc_recall=out["qc"]["flagged_among_planted"])
    with open(os.path.join(outdir, "validation_%s.json" % out["encoder"]), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    out["work"] = work
    out["result"] = res
    out["truths"] = truths
    return out


def _finding_truth(f, slides, truths):
    """fraction of the finding's patches that lie in each planted type"""
    names = {s.name: s for s in slides}
    cnt = {a: 0.0 for a in ANOM + ART + ("normal",)}
    tot = 0.0
    for r in f["regions"]:
        s = names[r["slide_name"]]
        t = _truth_arrays(s, truths[s.name])
        for pi in r["patches"]:
            tot += 1
            hit = False
            for a in ANOM + ART:
                if t[a][pi] >= 0.3:
                    cnt[a] += 1
                    hit = True
            cnt["normal"] += (not hit)
    return {k: round(v / max(tot, 1), 3) for k, v in cnt.items()}


def _example_figure(s, truth, path):
    """planted truth (anomaly outlines in green, artifacts in red) against the three axes and the novelty score of one synthetic slide"""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = s.thumb
    sc = th.shape[1] / float(truth["tissue"].shape[1])
    def outline(ax, mask, col):
        m = cv2.resize(mask.astype(np.uint8), (th.shape[1], th.shape[0]), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            ax.plot(c[:, 0, 0], c[:, 0, 1], color=col, lw=1.3)
    ext = [0, s.shape[1] * s.side0 / s.scale, s.shape[0] * s.side0 / s.scale, 0]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = [("planted truth", None), ("appearance rarity p_emb", "p_emb"), ("neighbourhood disorder p_het", "p_het"), ("novelty score", "novelty")]
    for a, (t, key) in zip(ax, panels):
        a.imshow(th)
        if key:
            m = np.full(s.shape, np.nan)
            m[s.rows, s.cols] = s.extra[key]
            a.imshow(np.ma.masked_invalid(m), extent=ext, cmap="magma", alpha=0.8, vmin=0, vmax=1)
        for n in ("disorganized", "novel_state"):
            if n in truth:
                outline(a, truth[n], "lime")
        for n in ("blur", "pen", "fold"):
            if n in truth:
                outline(a, truth[n], "red")
        a.set_title(t, fontsize=9)
        a.axis("off")
    fig.suptitle("synthetic slide %s: green = planted anomalies, red = planted artifacts" % s.name, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def rarity_sweep(outdir="results", encoder="handcrafted", fractions=(0.005, 0.01, 0.02, 0.05, 0.10), size=4480, specs=SPECS, workdir=None, log=print):
    """AUROC of the novelty score and of the appearance axis as a function of the rarity scale (fraction of the reference set used as k).  A state can look rare only if it is
    rarer than the scale: the planted anomalies cover about 3% of the benchmark patches, so scales below that are expected to fail for encoders that embed them tightly."""
    from . import cohort as C
    work = workdir or tempfile.mkdtemp(prefix="muv_sweep_")
    ind = os.path.join(work, "input")
    os.makedirs(ind, exist_ok=True)
    truths = {}
    for seed, an, ar in specs:
        name = "synth_%d" % seed
        pth = os.path.join(ind, name + ".tif")
        img, tr = S.make_slide(seed=seed, size=size, anomalies=an, artifacts=ar, anomaly_radius=int(size * 0.075))
        if not os.path.exists(pth):
            S.write_tiff(pth, img, 0.5)
        truths[name] = tr
    res = run(ind, os.path.join(work, "out"), RunConfig(encoder=encoder, figures=False, progression=False), log=lambda *a: None)
    slides = res["slides"]
    rows = []
    out = dict(encoder=res["manifest"]["encoder"], fractions=list(fractions), results=[])
    for fr in fractions:
        for s in slides:
            s.extra = {k: v for k, v in s.extra.items() if k in ("path", "backend", "dimensions", "mpp_assumed", "mpp_eff")}
        C.analyse(slides, C.Config(seed=0, rarity_fraction=fr))
        ys, ok_all, nov, pe, hd = [], [], [], [], []
        for s in slides:
            t = _truth_arrays(s, truths[s.name])
            an = np.max([t[a] for a in ANOM], axis=0) >= 0.5
            art = np.max([t[a] for a in ART], axis=0) >= 0.25
            nm = (np.max([t[a] for a in ANOM], axis=0) < 0.05) & (np.max([t[a] for a in ART], axis=0) < 0.05)
            sel = s.ok & (an | nm) & ~art
            ys.append(an[sel])
            nov.append(s.extra["novelty"][sel])
            pe.append(s.extra["p_emb"][sel])
        y = np.concatenate(ys).astype(int)
        out["results"].append(dict(rarity_fraction=fr, k_reference=int(round(fr * min(res["manifest"]["cohort"]["n_reference"], 40000))), auroc_novelty=_auc(y, np.concatenate(nov)), auroc_appearance=_auc(y, np.concatenate(pe)),
                                   anomalous_share=float(y.mean())))
        log("[sweep] %s rarity %.3f: AUROC novelty %.3f appearance %.3f" % (out["encoder"], fr, out["results"][-1]["auroc_novelty"], out["results"][-1]["auroc_appearance"]))
    with open(os.path.join(outdir, "rarity_sweep_%s.json" % out["encoder"]), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    return out
