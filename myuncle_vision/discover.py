# -*- coding: utf-8 -*-
"""Discovery: candidate patches -> regions on the lattice -> cross-slide candidate states ("findings").

Candidate patches: the highest-scoring fraction `q` of the cohort's novelty scores (a cohort-relative threshold, so the false-discovery protection is NOT the threshold but the
filters that follow): not an artifact, in a connected region of at least `min_region` patches, and (for the ranking) recurrence across slides and reproducibility under two disjoint
halves of the reference set.  Regions are then clustered across slides by the mean embedding of their patches; a cluster is a finding.
"""
import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import AgglomerativeClustering

from . import lattice as lat

AXES = ("emb", "het", "adj")
AXIS_TEXT = {"emb": "unusual appearance", "het": "disordered neighbourhood for its tissue state", "adj": "rare contacts between tissue states (transition zone)"}


def candidates(slides, q=0.02, min_region=3, connectivity=8):
    """mark candidate patches and label the connected regions; returns the novelty threshold and a list of region dicts"""
    nov = np.concatenate([s.extra["novelty"][s.ok] for s in slides])
    thr = float(np.quantile(nov[np.isfinite(nov)], 1.0 - q))
    regions = []
    st = np.ones((3, 3), int) if connectivity == 8 else None
    for si, s in enumerate(slides):
        cand = np.zeros(s.shape, bool)
        ok = s.ok & np.isfinite(s.extra["novelty"]) & (s.extra["novelty"] >= thr)
        cand[s.rows[ok], s.cols[ok]] = True
        lab, n = ndi.label(cand, structure=st)
        s.extra["cand_mask"] = cand
        s.extra["region_label"] = lab
        grid_idx = -np.ones(s.shape, int)
        grid_idx[s.rows, s.cols] = np.arange(s.n)
        for r in range(1, n + 1):
            pi = grid_idx[lab == r]
            if len(pi) < min_region:
                continue
            U = s.extra["U"][pi]
            m = U.mean(0)
            regions.append(dict(slide=si, slide_name=s.name, patches=pi, n=len(pi), centroid=m / max(np.linalg.norm(m), 1e-12),
                                novelty=float(np.mean(s.extra["novelty"][pi])), p_emb=float(np.mean(s.extra["p_emb"][pi])), p_het=float(np.nanmean(s.extra["p_het"][pi])),
                                p_adj=float(np.nanmean(s.extra["p_adj"][pi])), repro=float(np.mean(np.minimum(s.extra["p_emb_a"][pi], s.extra["p_emb_b"][pi]) >= 0.5)),
                                x0=float(s.x0[pi].min()), y0=float(s.y0[pi].min()), x1=float(s.x0[pi].max() + s.side0), y1=float(s.y0[pi].max() + s.side0),
                                area_mm2=float(len(pi) * (s.side0 * s.mpp / 1000.0) ** 2)))
    return thr, regions


def cluster_regions(regions, n_slides, link_threshold=0.25):
    """cluster regions by cosine distance of their centroid embeddings (average linkage); returns a list of finding dicts ranked by score"""
    if not regions:
        return []
    if len(regions) == 1:
        labels = np.zeros(1, int)
    else:
        C = np.stack([r["centroid"] for r in regions])
        labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=link_threshold).fit_predict(C)
    findings = []
    for g in np.unique(labels):
        rs = [r for r, l in zip(regions, labels) if l == g]
        ns = len({r["slide"] for r in rs})
        w = np.array([r["n"] for r in rs], float)
        mean = lambda key: float(np.average([r[key] for r in rs], weights=w))
        axes = {a: mean("p_" + a) for a in AXES}
        slide_counts = np.bincount([r["slide"] for r in rs], weights=w, minlength=n_slides)
        p = slide_counts[slide_counts > 0] / slide_counts.sum()
        ent = float(-(p * np.log(p)).sum()) if len(p) > 1 else 0.0
        dom = max(axes, key=axes.get)
        f = dict(regions=rs, n_regions=len(rs), n_slides=ns, n_patches=int(w.sum()), area_mm2=float(sum(r["area_mm2"] for r in rs)), novelty=mean("novelty"), axes=axes, dominant_axis=dom,
                 slide_entropy=ent, recurrence=ns / float(max(n_slides, 1)), repro=mean("repro"))
        f["score"] = f["novelty"] * (1.0 + np.log(ns)) * (0.5 + 0.5 * f["repro"])
        findings.append(f)
    findings.sort(key=lambda f: -f["score"])
    for i, f in enumerate(findings, 1):
        f["id"] = "F%d" % i
        f["type"] = AXIS_TEXT[f["dominant_axis"]]
    return findings
