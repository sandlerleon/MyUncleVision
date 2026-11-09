# -*- coding: utf-8 -*-
"""Human-readable explanations.  Every sentence is a number measured in the run; no pathology term is assigned (the platform has no labels).  Terms such as 'nuclear-density proxy'
are image statistics, not diagnoses, and every explanation ends with the request for expert review."""
import numpy as np


def _pct(v, ref):
    ref = ref[np.isfinite(ref)]
    return 100.0 * float(np.mean(ref <= v)) if len(ref) else float("nan")


def cohort_reference(slides):
    ok = lambda s, k: np.asarray(s.stain[k])[s.ok]
    ref = {k: np.concatenate([ok(s, k) for s in slides]) for k in ("h_mean", "e_mean", "nuc_frac")}
    ref["D"] = np.concatenate([s.extra["D"][s.ok] for s in slides])
    ref["state_share"] = np.bincount(np.concatenate([s.extra["state"][s.ok] for s in slides]))
    ref["state_share"] = ref["state_share"] / max(ref["state_share"].sum(), 1)
    return ref


def explain_finding(f, slides, ref):
    idx = {}
    for r in f["regions"]:
        idx.setdefault(r["slide"], []).append(r["patches"])
    pi = {si: np.concatenate(v) for si, v in idx.items()}
    stain = {k: np.mean(np.concatenate([np.asarray(slides[si].stain[k])[p] for si, p in pi.items()])) for k in ("h_mean", "e_mean", "nuc_frac")}
    D = float(np.nanmean(np.concatenate([slides[si].extra["D"][p] for si, p in pi.items()])))
    states = np.concatenate([slides[si].extra["state"][p] for si, p in pi.items()])
    top = np.bincount(states).argmax()
    share = float(np.mean(states == top))
    ax = f["axes"]
    parts = [
        "%s: a candidate state of %d patches (%.2f mm2) in %d of the cohort's slides; ranked by novelty %.2f (cohort percentile scale 0 to 1) and driven mainly by %s." % (f["id"], f["n_patches"], f["area_mm2"], f["n_slides"], f["novelty"], f["type"]),
        "Axes (mean cohort percentile): appearance %.2f, local disorder for its state %.2f, rarity of contacts %.2f." % (ax["emb"], ax["het"], ax["adj"]),
        "Local disorder D = %.2f (cohort median %.2f); hematoxylin density at the %.0fth percentile, eosin at the %.0fth, nuclear-density proxy at the %.0fth." % (
            D, float(np.nanmedian(ref["D"])), _pct(stain["h_mean"], ref["h_mean"]), _pct(stain["e_mean"], ref["e_mean"]), _pct(stain["nuc_frac"], ref["nuc_frac"])),
        "Most patches (%.0f%%) belong to prototype state %d, which holds %.1f%% of the cohort's tissue." % (100 * share, top, 100 * ref["state_share"][top]),
    ]
    if f["n_slides"] > 1:
        parts.append("It recurs across slides (slide entropy %.2f nats), which argues against a single-slide staining or scanning artifact, but batch and site effects are not excluded." % f["slide_entropy"])
    else:
        parts.append("It occurs in a single slide, so a slide-specific artifact cannot be excluded.")
    parts.append("Reproducibility under two disjoint halves of the reference set: %.0f%% of its patches. Expert review of the regions is required before any biological interpretation." % (100 * f["repro"]))
    return " ".join(parts)
