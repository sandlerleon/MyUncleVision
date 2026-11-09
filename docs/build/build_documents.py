# -*- coding: utf-8 -*-
"""Builds the manuscript (Elsevier-style), the cover letter, the highlights and the HeroX submission pack from the result files, so that every number is the code's.

    python docs/build/build_documents.py [--journal "Pattern Recognition Letters"] [--kpmp-run DIR] [--skip-herox]

Inputs: results/bench_{dev,held}_{handcrafted,dinov2_vits14}.json, results/robustness_*.json, results/frozen_settings.json (and, optionally, the output directory of a real KPMP run).
Key qualitative sentences are guarded by asserts: if the numbers change so that a sentence is no longer true, the build stops.
"""
import argparse
import csv
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import docx_engine as E                                   # noqa: E402
from refs import LIB                                      # noqa: E402

FIGS = os.path.join(ROOT, "docs", "figures")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs")
TITLE = "MyUncleVision: An Interpretable Lattice-Based Framework for Unsupervised Anomaly Discovery in Whole-Slide Histopathology Images"
EMAIL = "sandler.leon@gmail.com"
GITHUB = "https://github.com/sandlerleon/MyUncleVision"
ZEN_CODE, ZEN_PAPER = "10.5281/zenodo.24066260", "10.5281/zenodo.24066256"

ap = argparse.ArgumentParser()
ap.add_argument("--kpmp-run", default=None)
ap.add_argument("--journal", default="Pattern Recognition Letters")
ap.add_argument("--skip-herox", action="store_true")
ARGS = ap.parse_args()
SLUGS = {"Journal of Pathology Informatics": "JPI", "Computers in Biology and Medicine": "CBM", "Computerized Medical Imaging and Graphics": "CMIG", "Computer Methods and Programs in Biomedicine": "CMPB", "Pattern Recognition Letters": "PRL"}
JOURNAL = ARGS.journal
SLUG = SLUGS.get(JOURNAL, "".join(w[0] for w in JOURNAL.split() if w[0].isupper()) or "J")
HIGHLIGHTS = ["Lattice smoothing makes any anomaly score spatially coherent",
              "Settings frozen on a development set, tested once on a held-out cohort",
              "Frozen score best with DINOv2, poor with handcrafted features; both reported",
              "MyUncle order and entropy extend to d dimensions; MyUncle simulations as 2nd testbed",
              "One-command Docker pipeline; real-histology validation planned as future work"]
assert all(len(h) <= 85 for h in HIGHLIGHTS) and 3 <= len(HIGHLIGHTS) <= 5

J = lambda name: json.load(open(os.path.join(RES, name))) if os.path.exists(os.path.join(RES, name)) else None
DH, DD = J("bench_dev_handcrafted.json"), J("bench_dev_dinov2_vits14.json")
BH, BD = J("bench_held_handcrafted.json"), J("bench_held_dinov2_vits14.json")
RH, RD = J("robustness_held_handcrafted.json"), J("robustness_held_dinov2_vits14.json")
FZ = J("frozen_settings.json")
BRD = {(k, e): J("bridge_bench_%s_%s.json" % (k, e)) for k in ("dev", "held") for e in ("rotor", "rotor_D")}
assert all(v is not None for v in BRD.values()), "bridge benchmark results are missing"
assert all(x is not None for x in (DH, DD, BH, BD, FZ)), "benchmark results are missing"
KP = None
if ARGS.kpmp_run:
    KP = dict(findings=json.load(open(os.path.join(ARGS.kpmp_run, "findings.json"))), summary=list(csv.DictReader(open(os.path.join(ARGS.kpmp_run, "summary_statistics.csv")))),
              manifest=json.load(open(os.path.join(ARGS.kpmp_run, "run_manifest.json"))))

n = lambda x, d=2: ("%." + str(d) + "f") % x
pc = lambda x, d=0: ("%." + str(d) + "f%%") % (100 * x)
FROZEN = "MyUncleVision frozen: appearance, multi-scale, lattice-smoothed"
ORIG = "three-axis OR, 1% scale (original default)"
MAHA, MAHA_S = "Mahalanobis distance", "Mahalanobis distance, lattice-smoothed (rank)"
met = lambda R, m, k: R["methods"][m][k]
best = lambda R, k: max(R["methods"], key=lambda m: R["methods"][m][k] if R["methods"][m][k] == R["methods"][m][k] else -1)

# ---- guards: qualitative claims that the text makes
assert best(BD, "average_precision") == FROZEN, "text says the frozen score is the best on the held-out cohort with DINOv2"
assert met(BH, FROZEN, "average_precision") < met(BH, "appearance, multi-scale", "average_precision"), "text says smoothing hurt on the held-out cohort with handcrafted features"
assert met(BH, FROZEN, "region_recall") < met(BH, "appearance, multi-scale", "region_recall")
assert all(met(R, FROZEN, "spatial_coherence") > met(R, "appearance, multi-scale", "spatial_coherence") for R in (DH, DD, BH, BD)), "text says smoothing raises spatial coherence"
assert met(DH, MAHA_S, "average_precision") > met(DH, FROZEN, "average_precision") and met(DD, MAHA_S, "average_precision") > met(DD, FROZEN, "average_precision"), "text says smoothed Mahalanobis beats the frozen score on the development cohort"
assert met(DD, ORIG, "auroc") < 0.5, "text says the original default failed with DINOv2"
assert all(met(R, ORIG, "spatial_coherence") < met(R, FROZEN, "spatial_coherence") for R in (DH, DD))

_bm = lambda k, e, m, key="average_precision": BRD[(k, e)]["methods"][m][key]
_BASE = ("Isolation Forest", "Mahalanobis distance", "kNN distance, k = 10")
_OTHERS = [m for m in BRD[("held", "rotor_D")]["methods"] if m not in (FROZEN, "local disorder D alone")]
for _k in ("dev", "held"):
    for _b in _BASE:
        assert _bm(_k, "rotor", _b + ", lattice-smoothed (rank)") < _bm(_k, "rotor", _b), "text says smoothing lowered the AP of every baseline on the simulated lattices (rotor embedding)"
    assert _bm(_k, "rotor", FROZEN) < max(_bm(_k, "rotor", _b) for _b in _BASE), "text says the frozen score was below the unsmoothed baselines on the rotor embedding"
    assert _bm(_k, "rotor_D", FROZEN) > _bm(_k, "rotor", FROZEN), "text says the local disorder as a feature raised the AP of the frozen score"
assert _bm("held", "rotor_D", FROZEN) >= max(_bm("held", "rotor_D", m) for m in _OTHERS) - 0.02, "text says the frozen score was within 0.02 of the best AP on the held-out simulated lattices with the rotor + D embedding"
assert max(_bm("dev", "rotor_D", m) for m in _OTHERS) > _bm("dev", "rotor_D", FROZEN) + 0.05, "text says the frozen score was clearly not the best on the development simulations with the rotor + D embedding"
assert BRD[("held", "rotor")]["prevalence_all"] > 0.15 and BRD[("dev", "rotor")]["prevalence_all"] > 0.05, "text says the damaged share is not small"

T3 = "three-axis OR, multi-scale, lattice-smoothed"
assert met(DH, T3, "average_precision") > met(DH, FROZEN, "average_precision") and met(BH, T3, "average_precision") > met(BH, FROZEN, "average_precision"), "text says the lattice axes helped with handcrafted features"
assert met(DD, T3, "average_precision") < met(DD, FROZEN, "average_precision") and met(BD, T3, "average_precision") < met(BD, FROZEN, "average_precision"), "text says the lattice axes hurt with DINOv2"
for _R in (DH, DD, BH, BD):
    for _b in ("Isolation Forest", "Local Outlier Factor", "Mahalanobis distance", "kNN distance, k = 10"):
        assert met(_R, _b + ", lattice-smoothed (rank)", "spatial_coherence") > met(_R, _b, "spatial_coherence"), "text says smoothing raised the coherence of every baseline"

ROWS = [(ORIG, "three-axis OR, 1% scale (v0.1 default)"), ("appearance, 1% scale", "appearance, 1% scale"), ("appearance, multi-scale", "appearance, multi-scale"), (FROZEN, "MyUncleVision frozen (appearance, multi-scale, smoothed)"),
        ("three-axis OR, multi-scale, lattice-smoothed", "three-axis OR, multi-scale, smoothed"), ("state-wise disorder", "state-wise disorder"), ("contact rarity", "contact rarity"), ("local disorder D alone", "local disorder D alone"),
        ("Isolation Forest", "Isolation Forest"), ("Isolation Forest, lattice-smoothed (rank)", "  + smoothing"), ("Local Outlier Factor", "Local Outlier Factor"), ("Mahalanobis distance", "Mahalanobis distance"),
        (MAHA_S, "  + smoothing"), ("kNN distance, k = 10", "kNN distance, k = 10"), ("kNN distance, k = 10, lattice-smoothed (rank)", "  + smoothing")]


def table_rows(Rh, Rd):
    out = []
    for key, label in ROWS:
        out.append([label] + [n(met(R, key, k)) for R in (Rh, Rd) for k in ("auroc", "average_precision", "spatial_coherence", "region_recall")])
    return out


HDR = ["Method", "AUROC", "AP", "coh.", "rec.", "AUROC", "AP", "coh.", "rec."]
WID = [2.8, 0.78, 0.55, 0.55, 0.55, 0.78, 0.55, 0.55, 0.55]

# ================================================================================================ manuscript
B = []
add = B.append
add(("title", TITLE))
add(("author", "Leon Sandler"))
add(("small", "Independent Researcher, Northbrook, Illinois, USA"))
add(("small", "ORCID: https://orcid.org/0009-0007-4584-808X"))
add(("small", "Corresponding author: " + EMAIL))
add(("blank",))
add(("h1", "Abstract"))
abs_ = ("Unsupervised discovery in whole-slide images must separate unusual tissue organization from artifacts and routine histology and give experts places to look. "
        "We present MyUncleVision, an open pipeline that places patch embeddings on a tissue lattice, judges appearance rarity at several prevalence scales, smooths the score over the lattice to favor spatially coherent regions, and ranks and explains regions across slides. "
        "Its order and entropy definitions generalize those of the MyUncle maintenance-lattice framework to d dimensions and equal them at d = 2 (tested against the unmodified core). "
        "We compare ablations and baselines (Isolation Forest, Local Outlier Factor, nearest-neighbor, Mahalanobis) on synthetic slides with planted anomalies: a development cohort used to choose settings and a held-out cohort of 16 slides with three unseen anomaly types, about %s prevalence and stain shifts, evaluated once with frozen settings. "
        "Lattice smoothing raised the spatial coherence of every scorer. On the held-out cohort the frozen score reached an average precision of %s with DINOv2 (AUROC %s, the best of all methods) but %s with handcrafted features, where smoothing diluted small anomalies; "
        "on the development cohort a Mahalanobis baseline with the same smoothing scored higher (%s against %s with DINOv2). The disorder and contact axes of the lattice helped with handcrafted features and hurt with DINOv2, so they are optional diagnostics. "
        "Simulated MyUncle lattices gave similarly mixed results. Validation on real histopathology, with expert review, is planned (Section 5); the evidence is synthetic. Code and results are open."
        % (pc(BD["anomalous_patch_share"], 1), n(met(BD, FROZEN, "average_precision")), n(met(BD, FROZEN, "auroc")), n(met(BH, FROZEN, "average_precision")), n(met(DD, MAHA_S, "average_precision")), n(met(DD, FROZEN, "average_precision"))))
add(("p0", abs_))
add(("p0", "Keywords: whole-slide imaging; unsupervised anomaly detection; histopathology; spatial smoothing; von Mises-Fisher entropy; open-source software"))
add(("h1", "1. Introduction"))
add(("p", "Digital pathology produces gigapixel images, most of which are examined at a small fraction of their area, and the models that analyze them are usually trained for a task defined in advance: a diagnosis, a grade, or a segmentation [@Campanella2019;@Lu2021]. "
          "A different question is what organization of tissue exists that nobody has defined yet. Astronomy faced the same question for large surveys and developed outlier-detection and active anomaly detection methods [@Baron2017;@Lochner2021]; "
          "in medical imaging, unsupervised anomaly detection has mostly meant detecting deviations from a healthy appearance with generative models [@Schlegl2019]. "
          "The Kidney Precision Medicine Project (KPMP) has produced a large public set of kidney whole-slide images next to molecular atlases [@Lake2023], and a public challenge asks for previously unrecognized or undercharacterized patterns in them."))
add(("p", "A discovery method has to keep artifacts out to avoid reporting routine histology as unusual, and to give an expert a place, a measure and an explanation. This paper describes MyUncleVision and, above all, tests what its lattice contributes. "
          "Contributions: (1) a generalization of the order and entropy quantities of the maintenance-lattice framework MyUncle [@MyUncle2026] from two to d dimensions, verified against the unmodified core; "
          "(2) a score that judges appearance rarity at several prevalence scales and smooths it over the tissue lattice; (3) an open, containerized pipeline from SVS files to ranked, mapped, explained candidates with an audit trail of every image used; "
          "(4) an evaluation with ablations, baselines, a held-out cohort with frozen settings, a robustness suite and a second testbed of lattices simulated with MyUncle, whose results are mixed and are reported as such; (5) a prespecified plan for validation on real histopathology, presented as future work (Section 5); this paper reports synthetic evidence only."))
add(("h1", "2. Method"))
add(("h2", "2.1 Slides, patches and quality control"))
add(("p", "Slides are read with OpenSlide for SVS and other vendor formats [@Goode2013] and as plain images otherwise. A tissue mask is made on a thumbnail (glass is bright and nearly colorless). The slide is cut into nonoverlapping square patches of 224 px at 0.5 µm/px; "
          "the patch grid is the lattice, and a patch is kept when at least half of it is tissue. Quality control marks a patch as an artifact when it is out of focus (Laplacian variance below an absolute floor or below 15% of the slide's own median), contains pen ink (saturated blue and green hues), "
          "is dark debris or a fold, or is mostly glass. These thresholds are heuristics that must be tuned on the slides at hand; a patch that is only partly blurred keeps the variance of its sharp part and escapes the rule. Artifact patches are excluded from the lattice, the reference set and the ranks."))
add(("h2", "2.2 Embeddings and the sphere"))
add(("p", "Each patch is encoded by DINOv2 ViT-S/14 [@Oquab2023] (384-d class-token embedding; natural-image pretraining, no fine-tuning) or, offline, by a deterministic handcrafted encoder of stain, color, gradient-orientation and texture statistics (64-d) built on color deconvolution [@Ruifrok2001]. "
          "Embeddings are centered and reduced by a cohort PCA (16 components, scikit-learn [@Pedregosa2011]) and normalized to unit length, so every patch is a point $u$ on a sphere $S^{d-1}$."))
add(("h2", "2.3 Order and entropy of the tissue lattice"))
add(("p", "MyUncle [@MyUncle2026] describes a lattice of rotors whose order is eroded by a state-dependent flux and restored by repair. Its local disorder is one minus the length of the mean of the four neighbors' unit vectors, and its order-entropy bridge is the von Mises entropy of the angle at a given order. "
          "The same definitions hold for unit vectors in any dimension. With $N_i$ the valid neighbors of site $i$ and $u_j$ the unit embedding of neighbor $j$,"))
add(("eq", r"R_i = \left| \frac{1}{|N_i|} \sum_{j \in N_i} u_j \right|, \qquad D_i = 1 - R_i", 1))
add(("p", "where $D_i$ is MyUncle's local disorder. The concentration $\\kappa_i$ of the von Mises-Fisher distribution on $S^{d-1}$ [@Banerjee2005;@Mardia2000] that has mean resultant length $R_i$ solves"))
add(("eq", r"A_d(\kappa) = \frac{I_{d/2}(\kappa)}{I_{d/2-1}(\kappa)} = R_i", 2))
add(("p", "and the entropy of that distribution and the information held in the local order are"))
add(("eq", r"S_{\mathrm{vMF}}(\kappa, d) = -\left(\frac{d}{2}-1\right) \ln \kappa + \frac{d}{2} \ln (2\pi) + \ln I_{d/2-1}(\kappa) - \kappa A_d(\kappa), \qquad H_\chi = S_{\mathrm{unif}} - S_{\mathrm{vMF}}", 3))
add(("p", "For $d = 2$ these are MyUncle's $U = A(\\kappa) = I_1/I_0$, $S(\\kappa)$ and $H_\\chi$. The tests compare them with the vendored, unmodified core (Fig. @F:bridge@): the local disorder on a periodic lattice agrees to $10^{-12}$ and the entropy to $10^{-9}$. "
          "With a neighborhood of four patches the resultant length is a noisy estimate of the local order; this is a stated limitation."))
add(("fig", "fig_bridge.png", "Fig. @F:bridge@. (a) The order-information bridge in d dimensions, per dimension: MyUncle's $H_\\chi$ (d = 2, red) and the vMF generalization for d = 2, 16, 64 and 384. (b) Difference between the d = 2 vMF entropy and the vendored MyUncle Bridge.S.", 6.0))
add(("h2", "2.4 The frozen novelty score and the diagnostic axes"))
add(("p", "Three rank-based axes are computed for every nonartifact patch (empirical percentile in the cohort, 0 to 1): appearance rarity, state-wise disorder and contact rarity. "
          "Appearance rarity $p_{emb}$ is the mean cosine distance to the $k$ nearest patches of a reference set (a random sample of the cohort, same-slide neighbors within six lattice steps excluded). Because the prevalence of an interesting state is not known in advance, "
          "it is evaluated with $k$ equal to 0.5, 1, 2, 5 and 10% of the reference set and the highest rank counts: a state that recurs but covers less than the scale is still rare at that scale. "
          "State-wise disorder $p_{het}$ ranks $D_i$ among the patches of the same prototype state (k-means with 12 states), so that routine interfaces between compartments are not unusual. Contact rarity $p_{adj}$ ranks the mean of $-\\ln P(s_j \\mid s_i)$ over the neighbors, "
          "with $P$ the cohort-wide frequency with which states touch. The novelty score of a patch is the mean of the valid scores of its 3 × 3 lattice window (lattice smoothing) of a weighted probabilistic OR of the axes,"))
add(("eq", r"N_i = \frac{1}{|W_i|} \sum_{j \in W_i} q_j, \qquad q_j = 1 - (1 - p_{emb,j})^{w_{emb}} (1 - p_{het,j})^{w_{het}} (1 - p_{adj,j})^{w_{adj}}", 4))
add(("p", "with the frozen weights $(w_{emb}, w_{het}, w_{adj}) = (1, 0, 0)$: the default score is the smoothed multi-scale appearance rarity. The disorder and contact axes were part of the original design (weights 0.4, 0.4, 0.2, one rarity scale, no smoothing) and are retained as diagnostics and as options (--weights), because in the evaluation below their effect depended on the encoder: they helped with handcrafted features and hurt with DINOv2, where they were inverted. "
          "A Mahalanobis-distance scorer, smoothed in the same way, is available as an alternative (--scorer mahalanobis)."))
add(("h2", "2.5 Discovery, ranking and explanation"))
add(("p", "Patches in the top fraction $q$ of the novelty scores (default 2%) are candidates. Connected candidates of at least three patches form regions; regions are clustered across slides by the cosine distance of their mean embeddings (average linkage). "
          "A cluster is a finding, ranked by mean novelty × (1 + ln of the number of slides) × (0.5 + 0.5 × reproducibility), where reproducibility is the fraction of its patches that stay above the median rank of appearance rarity when the reference set is split into two disjoint halves. "
          "The score is a ranking, not a probability. Each finding carries a plain-language explanation made only of measured numbers, an explicit statement when it occurs in one slide only, and the request for expert review; no pathological term is assigned."))
add(("fig", "fig_pipeline.png", "Fig. @F:pipe@. MyUncleVision: from whole-slide images to ranked, mapped and explained candidate states, with the unmodified MyUncle core supplying the reference definitions.", 6.4))
add(("h2", "2.6 Software"))
add(("p", "Python (numpy, scipy, scikit-learn, OpenCV, timm and PyTorch for the encoder). One command, `python -m myuncle_vision run --input DIR --output DIR`, also the default command of the Docker image, writes findings, per-slide annotations in level-0 pixel coordinates, masks, heat maps, patch scores, "
          "a CSV listing every image used and how, and a manifest with versions, configuration, model files and the hash of the MyUncle core. The code is archived and open (Data availability). A bridge module connects the two code bases in both directions (simulate any MyUncle preset and analyse its lattice states with the same order maps and novelty score, with a numerical equivalence check; Section 4.5), and an exploratory analysis that maps a slide onto MyUncle's simulation is opt-in and described in Appendix A."))
add(("h1", "3. Evaluation design"))
add(("p", "No real kidney image was available for this work. The evaluation uses synthetic slides (4,480 × 4,480 px at 0.5 µm/px) that imitate what the method measures, not kidney tissue: interstitium, tubule-like rings with nuclei, glomerulus-like discs, vessels, stain variation, planted anomalies and three artifact types (an out-of-focus disc, a pen stroke, a dark fold). "
          "They were written by the author of the method, which limits what they can show."))
add(("bullet", "The development cohort (8 slides, %d patches, %s of them anomalous) was used while the method was built: two anomaly types (a patchwork of small tiles of different structure; a coherent striped state in an unusual hue). Settings were chosen on it." % (DH["n_patches"], pc(DH["anomalous_patch_share"], 1))))
add(("bullet", "The held-out cohort (16 slides, %d patches, %s anomalous) has three anomaly types never used in development (tightly packed small tubules, nuclear crowding, a cluster of glomerulus-like discs), seven planted instances, a prevalence near 1%%, stain and scanner shifts and different artifacts. "
               "The settings were frozen and recorded (results/frozen_settings.json, written before this cohort existed) and the cohort was evaluated once. Two candidates were frozen: the score of Section 2.4 (selected by a rule fixed beforehand: the highest mean average precision, over the two encoders, among the MyUncleVision variants) and a smoothed Mahalanobis scorer, "
               "which was added after the development results showed that it scored higher and which the rule did not select." % (BD["n_patches"], pc(BD["anomalous_patch_share"], 1))))
add(("p", "Methods compared: the original default of v0.1 (three-axis OR at one scale, no smoothing), every axis alone and in pairs, multi-scale appearance with and without smoothing, local disorder alone, and four baselines on the same embeddings: Isolation Forest, Local Outlier Factor, nearest-neighbor distance with k = 10 and Mahalanobis distance, each raw and smoothed in the same way as the proposed score (percentile rank, then the 3 × 3 window). "
          "Metrics, on patches that pass QC, with artifact patches excluded: AUROC and average precision (AP) of planted anomalous against clean patches; for the top 3% of scores, the spatial coherence (the share that lie in connected groups of at least three patches), the region recall (planted instances that contain a candidate patch) and the number of connected candidate groups per slide that touch neither a planted anomaly nor an artifact (false regions). "
          "A robustness suite varies seeds, embedding dimension, number of states, axis weights, smoothing and patch size. Everything was run with two encoders, DINOv2 and the handcrafted one."))
add(("h1", "4. Results on synthetic slides"))
add(("h2", "4.1 Development cohort"))
add(("table", "Table 1. Development cohort: AUROC, average precision (AP), spatial coherence (coh.) and region recall (rec.) of the methods, handcrafted encoder (columns 2 to 5) and DINOv2 (columns 6 to 9). '+ smoothing' is the method above it with percentile ranks smoothed over the 3 × 3 lattice window.",
     HDR, table_rows(DH, DD), WID, True, 14))
add(("p", "The original default (three-axis OR at one rarity scale) had an AUROC of %s with handcrafted features and %s with DINOv2, below chance. The cause is a scale mismatch and an encoder property: the planted states covered %s of the patches, more than the 1%% rarity scale, and DINOv2 embeds them tightly, so the appearance axis alone is at chance at that scale (%s); "
          "in addition, the DINOv2 embeddings of the planted regions are more homogeneous than those of normal tissue, so the disorder and contact axes point the wrong way (AUROC %s and %s). Judging rarity at several scales fixed the appearance axis (AUROC %s with DINOv2). "
          "Smoothing over the lattice raised the spatial coherence of every score: for the frozen score the coherence rose from %s to %s (handcrafted) and from %s to %s (DINOv2) and the region recall to %s and %s. With smoothing, adding the disorder and contact axes to appearance (three-axis OR) raised the AP from %s to %s with handcrafted features and lowered it from %s to %s with DINOv2; the held-out cohort showed the same sign flip (Section 4.2)."
     % (n(met(DH, ORIG, "auroc")), n(met(DD, ORIG, "auroc")), pc(DD["anomalous_patch_share"], 1), n(met(DD, "appearance, 1% scale", "auroc")), n(met(DD, "state-wise disorder", "auroc")), n(met(DD, "contact rarity", "auroc")), n(met(DD, "appearance, multi-scale", "auroc")),
        n(met(DH, "appearance, multi-scale", "spatial_coherence")), n(met(DH, FROZEN, "spatial_coherence")), n(met(DD, "appearance, multi-scale", "spatial_coherence")), n(met(DD, FROZEN, "spatial_coherence")), n(met(DH, FROZEN, "region_recall")), n(met(DD, FROZEN, "region_recall")),
        n(met(DH, FROZEN, "average_precision")), n(met(DH, T3, "average_precision")), n(met(DD, FROZEN, "average_precision")), n(met(DD, T3, "average_precision")))))
add(("p", "Against the baselines the result is less flattering. A plain Mahalanobis distance, a single Gaussian model of the embeddings, reaches an AUROC of %s and %s and, smoothed in the same way, an AP of %s and %s (handcrafted, DINOv2) against %s and %s for the frozen score. "
          "The frozen score is better than the nonparametric baselines (Isolation Forest, Local Outlier Factor, nearest-neighbor distance), whose smoothed AP with DINOv2 was %s, %s and %s. Smoothing raised the spatial coherence of every baseline in all four cases (cohort by encoder); it lowered the region recall of one of them (Isolation Forest, held-out, handcrafted) and not always the AUROC. "
          "Smoothing the raw Mahalanobis distance instead of its percentile rank did better still (AP %s and %s); this was not adopted, because it was found after the settings were chosen and would need a new held-out cohort."
     % (n(met(DH, MAHA, "auroc")), n(met(DD, MAHA, "auroc")), n(met(DH, MAHA_S, "average_precision")), n(met(DD, MAHA_S, "average_precision")), n(met(DH, FROZEN, "average_precision")), n(met(DD, FROZEN, "average_precision")),
        n(met(DD, "Isolation Forest, lattice-smoothed (rank)", "average_precision")), n(met(DD, "Local Outlier Factor, lattice-smoothed (rank)", "average_precision")), n(met(DD, "kNN distance, k = 10, lattice-smoothed (rank)", "average_precision")),
        n(met(DH, "Mahalanobis distance, lattice-smoothed (raw score)", "average_precision")), n(met(DD, "Mahalanobis distance, lattice-smoothed (raw score)", "average_precision")))))
add(("h2", "4.2 Held-out cohort, settings frozen"))
add(("table", "Table 2. Held-out cohort (settings frozen beforehand, evaluated once); columns as in Table 1.", HDR, table_rows(BH, BD), WID, True, 14))
add(("fig", "fig_heldout.png", "Fig. @F:held@. AUROC and average precision of the main methods on the held-out cohort, both encoders.", 6.4))
add(("p", "With DINOv2 the frozen score was the best of all methods in average precision (%s, AUROC %s, spatial coherence %s, region recall %s, %s false regions per slide), ahead of Mahalanobis with smoothing (AP %s) and of the unsmoothed multi-scale appearance (AP %s). "
          "With handcrafted features it did badly: AP %s, region recall %s, although its AUROC was %s and its spatial coherence %s. The unsmoothed multi-scale appearance score had an AP of %s and a region recall of %s. "
          "The held-out anomalies are small (about five patches) and their prevalence is below 1%%; with handcrafted features a 3 × 3 window mixes each of them with normal neighbors and dilutes the percentile ranks, which saturate. "
          "The choice to smooth, made on the development cohort, therefore did not generalize across encoders. The lattice axes behaved as on the development cohort: with smoothing they raised the AP with handcrafted features (%s against %s) and lowered it with DINOv2 (%s against %s). All methods had a low AP in absolute terms with handcrafted features (the best, %s, was %s)."
     % (n(met(BD, FROZEN, "average_precision")), n(met(BD, FROZEN, "auroc")), n(met(BD, FROZEN, "spatial_coherence")), n(met(BD, FROZEN, "region_recall")), n(met(BD, FROZEN, "false_regions_per_slide")), n(met(BD, MAHA_S, "average_precision")), n(met(BD, "appearance, multi-scale", "average_precision")),
        n(met(BH, FROZEN, "average_precision")), n(met(BH, FROZEN, "region_recall")), n(met(BH, FROZEN, "auroc")), n(met(BH, FROZEN, "spatial_coherence")), n(met(BH, "appearance, multi-scale", "average_precision")), n(met(BH, "appearance, multi-scale", "region_recall")),
        n(met(BH, T3, "average_precision")), n(met(BH, FROZEN, "average_precision")), n(met(BD, T3, "average_precision")), n(met(BD, FROZEN, "average_precision")), best(BH, "average_precision"), n(met(BH, best(BH, "average_precision"), "average_precision")))))
add(("fig", "fig_heldout_example.png", "Fig. @F:ex@. One held-out slide (DINOv2): planted anomalies (green), planted artifact (red), the unsmoothed multi-scale appearance score, the frozen score and the smoothed Mahalanobis score on the patch lattice.", 6.6))
if RH and RD:
    rows3 = []
    for a, b in zip(RH["rows"], RD["rows"]):
        rows3.append([a["variant"], "n/a" if a["jaccard"] != a["jaccard"] else n(a["jaccard"]), n(a["average_precision"]), "n/a" if b["jaccard"] != b["jaccard"] else n(b["jaccard"]), n(b["average_precision"])])
    add(("h2", "4.3 Robustness"))
    add(("table", "Table 3. Held-out cohort: stability of the frozen method under changes of configuration. Jaccard = overlap of the top-3% patch set with the default configuration; AP = average precision. Handcrafted (columns 2, 3) and DINOv2 (columns 4, 5).",
         ["Variant", "Jacc.", "AP", "Jacc.", "AP"], rows3, [3.2, 0.8, 0.8, 0.8, 0.8], True, 15))
    benign = lambda R: [r for r in R["rows"][1:8]]
    pca = lambda R: [r["jaccard"] for r in R["rows"] if "PCA" in r["variant"]]
    design = lambda R: [r["jaccard"] for r in R["rows"] if r["variant"] in ("no lattice smoothing", "three-axis weights 0.4/0.4/0.2", "equal axis weights")]
    coarse = lambda R: [r for r in R["rows"] if r["variant"].startswith("coarser")][0]
    add(("p", "Seeds and the number of states did not change the flagged set (Jaccard 1.00), but this is not evidence of stability: at this cohort size the reference sample is the whole cohort and the default weights do not use the states, so those two settings have no effect here. "
              "The embedding dimension changed the top-3%% set moderately (Jaccard %s to %s over both encoders). Switching smoothing off or adding the disorder and contact axes changed it almost completely (Jaccard %s to %s), as the earlier tables imply: those are design choices, not noise. "
              "A coarser patch size (0.75 µm/px) was tolerated with DINOv2 (AUROC %s, AP %s, region recall %s) but not with handcrafted features (AUROC %s, AP %s, region recall %s). "
              "Stain and scanner shifts are built into the held-out cohort (palette and softness differ between slides); a systematic test of other scanners, patients and stains needs real slides."
         % (n(min(pca(RH) + pca(RD))), n(max(pca(RH) + pca(RD))), n(min(design(RH) + design(RD))), n(max(design(RH) + design(RD))),
            n(coarse(RD)["auroc"]), n(coarse(RD)["average_precision"]), n(coarse(RD)["region_recall"]), n(coarse(RH)["auroc"]), n(coarse(RH)["average_precision"]), n(coarse(RH)["region_recall"]))))
add(("h2", "4.4 What the lattice adds and what it does not"))
add(("p", "Added: the lattice makes the score spatially coherent. For every scorer, smoothing over the lattice raised the share of top-ranked patches that form connected regions, and with DINOv2 it turned a noisy patch-level score into compact regions (Fig. @F:ex@). That is the property that matters for an expert who has to look at places. "
          "Encoder-dependent: the two lattice-based axes, disorder and contact rarity, added detection power with handcrafted features and removed it with DINOv2 on both cohorts; because the sign depends on the embedding, they are diagnostics and options, not part of the default score. "
          "Not shown: that the specific appearance score is better than a Mahalanobis distance (it is not, on the development cohort), and any benefit on real tissue."))
add(("h2", "4.5 A second testbed: lattices simulated with MyUncle"))
_dv, _hl = BRD[("dev", "rotor")], BRD[("held", "rotor")]
_pf = _hl["damaged_fraction_by_preset"]
add(("p", "The synthetic slides were written by the author of the method. A second testbed uses lattices that are not images and come from the simulation MyUncle was built for, through the bridge module: MyUncle's kinetic Monte Carlo (unmodified core) was run from a damaged start, "
          "and snapshots at steps 20, 40 and 60 (48 × 48 sites) were analysed as slides whose sites are patches. The sites that have lost their coupling in the simulation (the damaged sites) are the ground truth; the intact flag is never part of the embedding. "
          "Two embeddings were used: the rotor (cos θ, sin θ) and the rotor with the local disorder D of the site. The development set was one preset (chiral scaffold, four runs, damaged share %s); the held-out set was three other presets (polymer photodegradation, liquid crystal, magnetic storage decay; four runs each; damaged share %s, %s and %s). "
          "The settings of the image benchmark were used unchanged, with six states because the embedding has two or three dimensions, and the baselines were those of Section 3. The damaged share is not small, so this set also tests the assumption that interesting states are rare. "
          "Intervals in the json result files are 95%% intervals over whole simulation runs, not over snapshots of one run, and overlap for the methods whose held-out averages are close (for example the frozen score and smoothed Isolation Forest with the rotor + D embedding)."
     % (pc(_dv["prevalence_all"]), pc(_pf["polymer_photodegradation"]), pc(_pf["liquid_crystal"]), pc(_pf["magnetic_storage_decay"]))))
_T4 = [(FROZEN, "MyUncleVision frozen (smoothed)"), ("Isolation Forest", "Isolation Forest"), ("Isolation Forest, lattice-smoothed (rank)", "  + smoothing"), ("Mahalanobis distance", "Mahalanobis distance"), (MAHA_S, "  + smoothing"),
       ("kNN distance, k = 10", "kNN distance, k = 10"), ("kNN distance, k = 10, lattice-smoothed (rank)", "  + smoothing"), ("local disorder D alone", "local disorder D alone")]
_rows4 = [[lab] + [n(_bm("held", e, m, k)) for e in ("rotor", "rotor_D") for k in ("auroc", "average_precision")] + [n(_bm("dev", e, m)) for e in ("rotor", "rotor_D")] for m, lab in _T4]
add(("table", "Table 4. Lattices simulated with MyUncle, damaged sites as ground truth. Held-out presets: AUROC and AP with the rotor embedding (columns 2, 3) and the rotor + D embedding (columns 4, 5); AP on the development preset (columns 6, 7).",
     ["Method", "AUROC", "AP", "AUROC", "AP", "AP dev", "AP dev"], _rows4, [2.6, 0.8, 0.6, 0.8, 0.6, 0.7, 0.7], True, 15))
add(("p", "With the rotor alone, the frozen score was no better than smoothed Isolation Forest or nearest-neighbor distance (held-out AP %s, AUROC %s) and clearly below the unsmoothed baselines (Mahalanobis AP %s, Isolation Forest %s) and the smoothed Mahalanobis score (AUROC %s). "
          "Smoothing lowered the AP of every baseline on both sets (for example Isolation Forest %s to %s on the held-out set): the damaged sites are not compact regions that a 3 × 3 window preserves, which is the dilution already seen with handcrafted features on the images, although here that explanation is a plausible reading and was not tested separately. "
          "Adding the local disorder as a feature raised the AP of the frozen score on both sets (development %s to %s, held-out %s to %s); on the held-out presets it was then within 0.02 of the best AP (%s, tied with smoothed Isolation Forest), "
          "while on the development preset an unsmoothed Isolation Forest was clearly better (AP %s against %s). The disorder map alone ranked the damaged sites poorly (held-out AP %s with the rotor + D embedding), so the gain comes from combining it with the appearance of the site. "
          "The results are mixed in the same way as on the images: the lattice smoothing is helpful when the anomaly forms regions and harmful when it does not, and a plain Gaussian or tree-based baseline is a strong competitor. "
          "The simulation is the author's own, the damaged sites are an easy kind of anomaly (a site that lost its coupling), and four runs per preset are few; this testbed shows that the pipeline and its baselines run on a lattice that is not an image and that its limits are the same ones, not that the method detects damage in a real system."
     % (n(_bm("held", "rotor", FROZEN)), n(_bm("held", "rotor", FROZEN, "auroc")), n(_bm("held", "rotor", "Mahalanobis distance")), n(_bm("held", "rotor", "Isolation Forest")), n(_bm("held", "rotor", MAHA_S, "auroc")),
        n(_bm("held", "rotor", "Isolation Forest")), n(_bm("held", "rotor", "Isolation Forest, lattice-smoothed (rank)")),
        n(_bm("dev", "rotor", FROZEN)), n(_bm("dev", "rotor_D", FROZEN)), n(_bm("held", "rotor", FROZEN)), n(_bm("held", "rotor_D", FROZEN)), n(_bm("held", "rotor_D", FROZEN)),
        n(_bm("dev", "rotor_D", "Isolation Forest")), n(_bm("dev", "rotor_D", FROZEN)), n(_bm("held", "rotor_D", "local disorder D alone")))))
add(("h1", "5. Future work: validation on real histopathology"))
add(("p", "The evidence in this paper is synthetic: the slides were generated by the author of the method, with simple textures and sharply bounded anomalies, and they cannot show that the method finds anything biologically meaningful in tissue. "
          "Validation on real histopathology is therefore the principal next step, and the study is specified here in advance so that its design cannot be shaped by its results. It has not been started."))
add(("h2", "5.1 Data and design"))
add(("p", "The first real-image study will use publicly released whole-slide images of the Kidney Precision Medicine Project [@Lake2023] (the images of its first public release are distributed to registered participants of the NIDDK Kidney AI Discovery Challenge), which are deidentified; controlled-access data will not be used. "
          "A pilot of 10 to 20 slides from different patients will be used to inspect the quality-control exclusions, the heat maps and the candidate regions and to fix the settings. The settings, the list of baselines and the analysis plan will then be frozen and published in the repository before an independent cohort of slides from other patients, sites and stains is analysed once, "
          "following the procedure used here for the synthetic held-out cohort. Challenge participation will be disclosed and will not be presented as independent validation."))
add(("h2", "5.2 Comparators and endpoints"))
add(("p", "On the same slides and embeddings the frozen score will be compared with the baselines of Section 3 (Isolation Forest, Local Outlier Factor, nearest-neighbor and Mahalanobis distances, raw and smoothed) and with alternative spatial postprocessing (Gaussian and median filters, other window sizes), with the encoders DINOv2 and, where available, a pathology-pretrained encoder. "
          "Primary endpoint: the share of the top-ranked regions that at least two kidney pathologists, blinded to the algorithm and to the ranking, judge to be an unexplained candidate rather than known histology or an artifact, with interrater agreement (Cohen's kappa) and slide-level bootstrap confidence intervals, compared between methods by paired resampling of slides. "
          "Secondary endpoints: the fraction of artifact regions among the top-ranked ones; the stability of the flagged set under another seed, reference sample, encoder and patch size (Jaccard overlap); the recurrence of each finding across slides, patients and sites against a null model in which the site or scanner is the explanatory variable; computing time. "
          "The success criteria will be written into the analysis plan before the independent cohort is analysed."))
add(("h2", "5.3 Further planned work"))
add(("p", "A second synthetic generator with different textures, anomaly boundaries and artifacts, evaluated once with the frozen settings; slide-level bootstrap intervals for the average precision and for the differences from the Mahalanobis baseline, with more planted instances than the seven of the present held-out cohort; "
          "a direct comparison of the lattice smoothing with Gaussian and median filtering and other neighborhood sizes; smoothing of the raw score instead of its rank; stain normalization [@Macenko2009]; and a pathology-pretrained encoder [@Chen2024;@Vorontsov2024]. "
          "If the real-image study shows no advantage over the baselines, that result will be reported."))
add(("h1", "6. Discussion"))
add(("p", "What the evaluation supports. The lattice smoothing gives spatially coherent candidate regions for any scorer, the multi-scale rarity removes the dependence on a prevalence chosen in advance, and with DINOv2 the frozen score was the best method on an unseen cohort. "
          "The order and entropy definitions are MyUncle's, verified at d = 2, and extend to the embedding space; the explanation of a finding is a list of measured numbers that a reviewer can check against the image."))
add(("p", "What it does not support. The slides and the simulated lattices are synthetic, written by the author of the method (the simulation is MyUncle's, with an easy kind of anomaly), with simple textures and sharply bounded anomalies; nothing here shows that the method finds anything in a kidney or that what it finds is biologically meaningful. "
          "The held-out cohort was used once but is of the same family as the development cohort. The two encoders disagree on whether smoothing helps and on whether the lattice axes help; a single Gaussian model scored higher than the frozen score on the development cohort; and the frozen choice (the nonparametric score) is justified for real tissue only by an untested argument that tissue is multimodal. "
          "Novelty is not pathology: a rare state may be a rare preparation, a scanner, a stain batch or a known entity that is merely uncommon in the cohort. The embedding is trained on natural images and may group patches by stain or sharpness. "
          "Quality control flagged only part of the planted artifact patches in the development runs (42%), and partly blurred patches escape it."))
add(("p", "Improvements: a pathology-pretrained encoder (foundation models trained on large slide collections have been reported [@Chen2024;@Vorontsov2024], but were not available for this work), stain normalization [@Macenko2009], smoothing of the raw score instead of its rank, neighborhoods larger than four patches, a calibrated null for the novelty score, and active expert feedback in the manner of anomaly detection systems for astronomy [@Lochner2021]. "
          "Computing: the encoder dominates; a slide of ten thousand patches takes about ten minutes on one CPU core and minutes on a GPU, so the 2.4 TB of the first KPMP release is a cluster workload; the pipeline offers a patch budget per slide and a resumable cache."))
add(("h1", "7. Conclusions"))
add(("p", "MyUncleVision turns a whole-slide image into a lattice of tissue patches, scores the rarity of their appearance across prevalence scales and smooths the score over the lattice, which yields spatially coherent, explainable candidate regions. In synthetic evaluations with frozen settings and a held-out cohort, "
          "the frozen score was the best method with DINOv2 and a poor one with handcrafted features, a simple Mahalanobis baseline was competitive, and the disorder and contact axes of the lattice added detection power with handcrafted features but removed it with DINOv2. Its value for kidney and other histopathology is untested; validation on real slides with expert review is the planned next step (Section 5)."))
add(("h1", "Appendix A. Exploratory scenario analysis (not evaluated)"))
add(("p", "An opt-in option (--scenarios) maps a slide onto the lattice of MyUncle: the rotor angle of a site is the angle of its first two principal components, a site is damaged when its state-wise disorder is above the 75th percentile, and the damaged fraction gives a repair parameter from the linear steady state of MyUncle's mean-field reduction, whose bistable window and kinetic Monte Carlo are then run from the slide under half, equal and double repair. "
          "This is a modeling analogy. It is not calibrated against any biological measurement, no result of it is reported in this paper, and its output is labeled exploratory; it is no evidence about disease progression or repair dynamics."))
add(("h1", "Declarations"))
add(("p0", "Funding: this work received no external funding."))
add(("p0", "Competing interests: the author declares no competing interests. [To be confirmed by the author.]"))
add(("p0", "Author contributions (CRediT): Leon Sandler, sole author: conceptualization, methodology, software, validation, formal analysis, investigation, visualization, writing – original draft, writing – review and editing."))
add(("p0", "Data and code availability: the code, the tests, the synthetic benchmarks and their results are open at %s (MIT license) and archived on Zenodo (software: https://doi.org/%s, manuscript: https://doi.org/%s, all versions). The MyUncle core is vendored unmodified (doi:10.5281/zenodo.21223570). No patient data were used; the KPMP images are not part of this work." % (GITHUB, ZEN_CODE, ZEN_PAPER)))
add(("p0", "Declaration of generative AI and AI-assisted technologies in the writing process: during the preparation of this work the author used Claude (Anthropic) to assist with writing and testing the code, running the analyses and drafting the text and figures. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the publication."))
add(("refs",))

# ================================================================================================ build
cit = E.Citations(LIB)
for b in B:
    for x in b[1:]:
        if isinstance(x, str):
            cit.scan(x)
FIGNUM = {k: i + 1 for i, k in enumerate(["bridge", "pipe", "held", "ex"])}


def resolve(s):
    for k, v in FIGNUM.items():
        s = s.replace("@F:%s@" % k, str(v))
    return cit.resolve(s)


def build(path, blocks, journal):
    doc = E.Doc(cit, journal, True)
    for b in blocks:
        k = b[0]
        if k == "title":
            doc.title(b[1])
        elif k == "author":
            doc.centered(b[1], 24)
        elif k == "small":
            doc.centered(b[1], 19)
        elif k == "blank":
            doc.blank()
        elif k == "h1":
            doc.h1(b[1])
        elif k == "h2":
            doc.h2(b[1])
        elif k == "p":
            doc.p(resolve(b[1]))
        elif k in ("p0", "bullet"):
            doc.p(("• " if k == "bullet" else "") + resolve(b[1]), indent=False)
        elif k == "eq":
            doc.eq(b[1], b[2])
        elif k == "fig":
            doc.fig(os.path.join(FIGS, b[1]), resolve(b[2]), b[3] if len(b) > 3 else 6.4)
        elif k == "table":
            doc.table(b[1], b[2], b[3], b[4], b[5], b[6] if len(b) > 6 else 16)
        elif k == "refs":
            doc.h1("References")
            for i, text in cit.entries():
                par = doc.d.add_paragraph()
                E._spacing(par, 0, 60)
                if journal:
                    doc._ds(par)
                par.paragraph_format.left_indent = E.Emu(400 * 635)
                par.paragraph_format.first_line_indent = E.Emu(-400 * 635)
                E.add_runs(par, "[%d] %s" % (i, text), 24 if journal else 18)
    doc.save(path, title=TITLE)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    build(os.path.join(OUT, "MyUncleVision_Manuscript_%s.docx" % SLUG), B, True)
    words = sum(len(b[1].split()) for b in B if b[0] in ("p", "p0", "bullet"))
    print("manuscript built; %d references; abstract %d words; ~%d words" % (len(cit.order), len(abs_.split()), words))
    assert len(abs_.split()) <= 250
    exec(open(os.path.join(HERE, "build_herox.py"), encoding="utf-8").read())
    exec(open(os.path.join(HERE, "build_cover.py"), encoding="utf-8").read())
