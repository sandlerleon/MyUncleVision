# -*- coding: utf-8 -*-
"""End-to-end pipeline: a directory of images (whole-slide images first) in, a directory of results out (one command, no manual step).

    slides -> tissue mask -> patch grid -> QC -> embeddings -> cohort space -> lattice maps (order, entropy, transitions) -> novelty -> regions -> findings
           -> per-slide annotations / masks / heat maps / patch scores, findings with explanations, progression scenarios, images_used.csv, run_manifest.json
"""
import csv
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass, asdict

import numpy as np

from . import __version__, _core, cohort, discover, explain, render, progression, wsi


@dataclass
class RunConfig:
    patch_px: int = 224
    mpp: float = 0.5                 # target micrometres per pixel of a patch (about 20x)
    min_tissue: float = 0.5
    encoder: str = "auto"            # auto | dinov2 | handcrafted
    device: str = "auto"
    max_patches: int = 0             # 0 = no limit; otherwise the grid is made coarser until a slide has at most this many patches
    q: float = 0.02                  # top fraction of the novelty scores that become candidate patches
    min_region: int = 3
    link_threshold: float = 0.25
    seed: int = 0
    n_components: int = 16
    n_states: int = 12
    weights: tuple = (1.0, 0.0, 0.0)  # novelty axis weights: appearance, state-wise disorder, contact rarity (frozen default: appearance only; see results/frozen_settings.json)
    scorer: str = "knn"               # appearance scorer: knn (multi-scale nearest-neighbour rarity) or mahalanobis
    rarity_fraction: float = 0.01     # scale at which appearance rarity is judged (fraction of the reference set)
    aggregation: str = "mean"         # mean (fixed 3 x 3 window, the frozen default) | simw (similarity-weighted) | two_channel (rarity or directional local order, selected per cohort)
    progression: bool = False        # EXPLORATORY MyUncle scenario analysis (opt-in, --scenarios): a modelling analogy, not evidence about disease progression
    figures: bool = True
    hash_files: bool = False         # SHA-256 of every input file (slow for TB-scale input)
    data_source: str = "images supplied by the user (the user confirms the right to use them)"    # written into images_used.csv; set it to document the provenance of the data (e.g. a public data release)
    batch: int = 128


def _cache_key(path, cfg, enc_name, mpp_eff):
    st = os.stat(path)
    h = hashlib.sha1(("%s|%d|%d|%s|%d|%.4f|%.3f" % (os.path.abspath(path), st.st_size, int(st.st_mtime), enc_name, cfg.patch_px, mpp_eff, cfg.min_tissue)).encode()).hexdigest()[:16]
    return h


def load_slide(path, cfg, encoder, outdir):
    """patches -> QC -> embeddings for one slide (cached in <outdir>/cache)"""
    sl = wsi.Slide(path)
    mpp_eff = cfg.mpp
    grid = wsi.patch_grid(sl, cfg.patch_px, mpp_eff, cfg.min_tissue)
    if cfg.max_patches and len(grid["rows"]) > cfg.max_patches:
        mpp_eff = cfg.mpp * float(np.sqrt(len(grid["rows"]) / float(cfg.max_patches))) * 1.02
        grid = wsi.patch_grid(sl, cfg.patch_px, mpp_eff, cfg.min_tissue)
    key = _cache_key(path, cfg, encoder.name, mpp_eff)
    cache = os.path.join(outdir, "cache", key + ".npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        emb, qc, stain = z["emb"], z["qc"].item(), z["stain"].item()
    else:
        N = len(grid["rows"])
        emb = None
        qcl, stl = [], []
        for i in range(0, N, cfg.batch):
            sel = range(i, min(N, i + cfg.batch))
            P = np.stack([sl.read(grid["x0"][j], grid["y0"][j], grid["side0"], cfg.patch_px) for j in sel])
            for p in P:
                qcl.append(wsi.patch_qc(p))
                stl.append(wsi.stain_descriptors(p))
            e = encoder.encode(P)
            emb = e if emb is None else np.concatenate([emb, e])
        qc = {k: np.array([q[k] for q in qcl]) for k in ("blur", "pen", "dark", "white", "artifact", "reason")}
        stain = {k: np.array([q[k] for q in stl]) for k in ("h_mean", "e_mean", "nuc_frac", "od_sd")}
        if emb is None:
            emb = np.zeros((0, encoder.dim), np.float32)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, emb=emb.astype(np.float16), qc=np.array(qc, dtype=object), stain=np.array(stain, dtype=object))
        emb = emb.astype(np.float16)
    qc = wsi.finalize_qc(qc)
    sd = cohort.SlideData(name=sl.name, shape=grid["shape"], rows=grid["rows"], cols=grid["cols"], x0=grid["x0"], y0=grid["y0"], side0=grid["side0"], mpp=sl.mpp,
                          emb=np.asarray(emb, np.float32), qc=qc, stain=stain, thumb=grid["thumb"], mask=grid["mask"], scale=grid["scale"])
    sd.extra.update(path=path, backend=sl.backend, dimensions=sl.dimensions, mpp_assumed=sl.mpp_assumed, mpp_eff=mpp_eff)
    sl.close()
    return sd


def run(input_dir, output_dir, cfg=None, log=print, encoder=None):
    cfg = cfg or RunConfig()
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    paths = wsi.find_slides(input_dir)
    if not paths:
        raise SystemExit("no slide files (%s) found in %s" % ("/".join(wsi.SLIDE_EXT), input_dir))
    from .encoder import get_encoder
    encoder = encoder if encoder is not None else get_encoder(cfg.encoder, cfg.device)       # any object with name, dim, model_files and encode(uint8 patches) -> embeddings
    log("[run] %d slides, encoder %s" % (len(paths), encoder.name))
    slides, used, errors = [], [], []
    for p in paths:
        row = dict(slide_id=os.path.splitext(os.path.basename(p))[0], file_name=os.path.basename(p), relative_path=os.path.relpath(p, input_dir), size_bytes=os.path.getsize(p))
        if cfg.hash_files:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk)
            row["sha256"] = h.hexdigest()
        try:
            s = load_slide(p, cfg, encoder, output_dir)
            row.update(backend=s.extra["backend"], width=s.extra["dimensions"][0], height=s.extra["dimensions"][1], mpp=s.mpp, mpp_assumed=s.extra["mpp_assumed"], mpp_used_for_patches=round(s.extra["mpp_eff"], 4),
                       n_patches=s.n, n_artifact=int((~s.ok).sum()), n_analysed=int(s.ok.sum()), status="used")
            slides.append(s)
            log("[slide] %s: %d patches (%d artifact)" % (s.name, s.n, int((~s.ok).sum())))
        except Exception as e:                                         # a corrupt file must not stop the run; it is listed with its error
            row.update(status="failed: %s" % type(e).__name__)
            errors.append("%s: %s" % (row["slide_id"], e))
            log("[slide] %s FAILED: %s" % (row["slide_id"], e))
        row["how_used"] = "whole slide; tissue patches at the stated resolution; unsupervised (no labels, no clinical data); source: " + cfg.data_source
        used.append(row)
    if errors and not slides:
        raise SystemExit("every slide failed; first error: " + errors[0])
    slides = [s for s in slides if s.ok.sum() >= 5]
    if not slides:
        raise SystemExit("no slide had enough analysable tissue")
    ccfg = cohort.Config(n_components=cfg.n_components, n_states=cfg.n_states, seed=cfg.seed, weights=tuple(cfg.weights), rarity_fraction=cfg.rarity_fraction, scorer=cfg.scorer, aggregation=cfg.aggregation)
    space, csum = cohort.analyse(slides, ccfg)
    if "two_channel" in csum:
        log("[cohort] aggregation %s: channel %s (tail index A %s, B %s)" % (cfg.aggregation, csum["two_channel"].get("channel"), csum["two_channel"].get("tail_A"), csum["two_channel"].get("tail_B")))
    thr, regions = discover.candidates(slides, cfg.q, cfg.min_region)
    findings = discover.cluster_regions(regions, len(slides), cfg.link_threshold)
    ref = explain.cohort_reference(slides)
    for f in findings:
        f["explanation"] = explain.explain_finding(f, slides, ref)
    # ---- write
    sdir = os.path.join(output_dir, "slides")
    summary = []
    for s in slides:
        d = os.path.join(sdir, s.name)
        os.makedirs(d, exist_ok=True)
        regs = [dict(finding=f["id"], x=r["x0"], y=r["y0"], width=r["x1"] - r["x0"], height=r["y1"] - r["y0"], n_patches=r["n"], area_mm2=r["area_mm2"], novelty=r["novelty"])
                for f in findings for r in f["regions"] if r["slide_name"] == s.name]
        json.dump(dict(slide=s.name, coordinate_system="level-0 pixels of the source slide", mpp=s.mpp, regions=regs), open(os.path.join(d, "annotations.json"), "w"), indent=1)
        with open(os.path.join(d, "patch_scores.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["row", "col", "x0", "y0", "side0", "artifact", "qc_reason", "state", "novelty", "p_emb", "p_het", "p_adj", "D", "H_chi", "state_entropy", "transition_rarity", "candidate"])
            cand = s.extra["cand_mask"][s.rows, s.cols]
            for i in range(s.n):
                e = s.extra
                w.writerow([s.rows[i], s.cols[i], round(s.x0[i], 1), round(s.y0[i], 1), round(s.side0, 1), int(not s.ok[i]), s.qc["reason"][i], int(e["state"][i])] +
                           ["%.4f" % e[k][i] for k in ("novelty", "p_emb", "p_het", "p_adj", "D", "H_chi", "S_state", "T_rare")] + [int(cand[i])])
        render.novelty_mask_png(s, os.path.join(d, "novelty_mask.png"))
        np.save(os.path.join(d, "novelty_map.npy"), _grid(s, "novelty"))
        if cfg.figures:
            render.slide_figure(s, findings, os.path.join(d, "heatmaps.png"))
        prog = progression.scenarios(s) if cfg.progression else None
        if prog:
            prog = dict(exploratory=True, note="Model-based scenario analysis with the MyUncle simulation; a modelling analogy that is not calibrated against any biological measurement and is not evidence of disease progression or repair dynamics.", **prog)
            json.dump(prog, open(os.path.join(d, "progression_scenarios.json"), "w"), indent=1)
        o = s.extra["order"]
        summary.append(dict(slide=s.name, n_patches=s.n, n_analysed=int(s.ok.sum()), n_artifact=int((~s.ok).sum()), n_candidate_patches=int(s.extra["cand_mask"].sum()), n_regions=len(regs),
                            Rbar=o["Rbar_obs"], Rbar_shuffled=o["Rbar_null_mean"], order_z_vs_shuffle=o["z"], D_mean=o["D_mean"], H_chi_mean=o["H_chi_mean"],
                            rho_hat=(prog or {}).get("rho_hat"), in_bistable_window=(prog or {}).get("in_bistable_window")))
    with open(os.path.join(output_dir, "summary_statistics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    cols = ["slide_id", "file_name", "relative_path", "size_bytes", "sha256", "backend", "width", "height", "mpp", "mpp_assumed", "mpp_used_for_patches", "n_patches", "n_artifact", "n_analysed", "status", "how_used"]
    with open(os.path.join(output_dir, "images_used.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(used)
    fj = []
    for f in findings:
        fj.append(dict(id=f["id"], type=f["type"], n_slides=f["n_slides"], n_regions=f["n_regions"], n_patches=f["n_patches"], area_mm2=f["area_mm2"], novelty=f["novelty"], axes=f["axes"], slide_entropy=f["slide_entropy"],
                       reproducibility=f["repro"], score=f["score"], explanation=f["explanation"],
                       regions=[dict(slide=r["slide_name"], x=r["x0"], y=r["y0"], width=r["x1"] - r["x0"], height=r["y1"] - r["y0"], n_patches=r["n"]) for r in f["regions"]]))
    json.dump(dict(novelty_threshold=thr, q=cfg.q, n_slides=len(slides), findings=fj), open(os.path.join(output_dir, "findings.json"), "w"), indent=1)
    with open(os.path.join(output_dir, "findings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["finding", "type", "n_slides", "n_regions", "n_patches", "area_mm2", "novelty", "p_emb", "p_het", "p_adj", "slide_entropy", "reproducibility", "score"])
        for f in findings:
            w.writerow([f["id"], f["type"], f["n_slides"], f["n_regions"], f["n_patches"], "%.3f" % f["area_mm2"], "%.3f" % f["novelty"], *["%.3f" % f["axes"][a] for a in ("emb", "het", "adj")],
                        "%.3f" % f["slide_entropy"], "%.3f" % f["repro"], "%.3f" % f["score"]])
    man = dict(software="MyUncleVision", version=__version__, run_seconds=round(time.time() - t0, 1), config=asdict(cfg), encoder=encoder.name, model_files=list(getattr(encoder, "model_files", [])),
               myuncle_core_sha256=_core.sha256(), myuncle_core_unmodified=_core.verify(), cohort=csum | dict(novelty_threshold=thr), python=platform.python_version(), platform=platform.platform(),
               device=getattr(encoder, "device", "cpu"), n_slides_listed=len(paths), n_slides_analysed=len(slides), n_findings=len(findings))
    json.dump(man, open(os.path.join(output_dir, "run_manifest.json"), "w"), indent=1, default=str)
    log("[run] done in %.1f s: %d slides, %d findings -> %s" % (time.time() - t0, len(slides), len(findings), output_dir))
    return dict(slides=slides, findings=findings, manifest=man, summary=summary)


def _grid(s, key):
    m = np.full(s.shape, np.nan, np.float32)
    m[s.rows, s.cols] = s.extra[key]
    return m
