# -*- coding: utf-8 -*-
"""Builds the manuscript (Elsevier-style, Journal of Pathology Informatics), the HeroX submission pack and the cover letter from the validation JSON files, so that every number is the code's.

    python docs/build/build_documents.py [--kpmp-run DIR]

--kpmp-run: output directory of a real KPMP run (findings.json, summary_statistics.csv, images_used.csv).  Without it the HeroX narratives carry an explicit 'KPMP run pending' notice and
claim nothing about kidney tissue.
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
TITLE = "MyUncleVision: entropy-guided, label-free discovery of unusual tissue organization in whole-slide images, with a maintenance-lattice order–entropy framework"
EMAIL = "sandler.leon@gmail.com"
GITHUB = "https://github.com/sandlerleon/MyUncleVision"

ap = argparse.ArgumentParser()
ap.add_argument("--kpmp-run", default=None)
ap.add_argument("--journal", default="Journal of Pathology Informatics", help="target Elsevier journal (cover letter, file names)")
ap.add_argument("--skip-herox", action="store_true")
ARGS = ap.parse_args()

SLUGS = {"Journal of Pathology Informatics": "JPI", "Computers in Biology and Medicine": "CBM", "Computerized Medical Imaging and Graphics": "CMIG", "Computer Methods and Programs in Biomedicine": "CMPB"}
JOURNAL = ARGS.journal
SLUG = SLUGS.get(JOURNAL, "".join(w[0] for w in JOURNAL.split() if w[0].isupper()) or "J")
HIGHLIGHTS = ["Embedding lattice measures tissue order with MyUncle's definitions",
              "Disorder and entropy extend to d dimensions and equal MyUncle's at d = 2",
              "Three-axis novelty ranks unusual states, not artifacts or routine histology",
              "Synthetic test: works with handcrafted features, fails with DINOv2 defaults",
              "One-command open container; kidney evaluation protocol given, not yet run"]
assert all(len(h) <= 85 for h in HIGHLIGHTS) and 3 <= len(HIGHLIGHTS) <= 5

V = {}
for enc in ("handcrafted", "dinov2_vits14"):
    p = os.path.join(RES, "validation_%s.json" % enc)
    if os.path.exists(p):
        V[enc] = json.load(open(p))
H, D = V.get("handcrafted"), V.get("dinov2_vits14")
PRIMARY = H or D
SW = {}
for enc in ("handcrafted", "dinov2_vits14"):
    p_ = os.path.join(RES, "rarity_sweep_%s.json" % enc)
    if os.path.exists(p_):
        SW[enc] = json.load(open(p_))
KP = None
if ARGS.kpmp_run:
    KP = dict(findings=json.load(open(os.path.join(ARGS.kpmp_run, "findings.json"))), summary=list(csv.DictReader(open(os.path.join(ARGS.kpmp_run, "summary_statistics.csv")))),
              used=list(csv.DictReader(open(os.path.join(ARGS.kpmp_run, "images_used.csv")))), manifest=json.load(open(os.path.join(ARGS.kpmp_run, "run_manifest.json"))))

n = lambda x, d=2: ("%." + str(d) + "f") % x
pc = lambda x, d=0: ("%." + str(d) + "f%%") % (100 * x)
A = lambda r, t, a: r["auroc_vs_normal"][t][a]


def sw(enc, fr, key):
    return next(r[key] for r in SW[enc]["results"] if abs(r["rarity_fraction"] - fr) < 1e-9)


def dino_clause():
    """one honest paragraph on the DINOv2 result, from the JSON files"""
    if not (D and "dinov2_vits14" in SW):
        return ""
    return ("With DINOv2 the novelty score does not separate the planted structure: its AUROC is %s, below chance. Two things cause it. At the default rarity scale of 1%% of the reference set the planted states, which cover %s of the benchmark patches, are not rare, "
            "and DINOv2 embeds them tightly: the appearance axis is at chance (%s) and recovers only as the scale exceeds their prevalence (Table 2: %s at a scale of 0.5%%, %s at 5%% and %s at 10%%). "
            "In addition, in the DINOv2 embedding the planted regions are more homogeneous than normal tissue, so the disorder and contact axes point the wrong way (AUROC %s and %s; below 0.5 means the planted patches are less unusual than normal ones), which pulls the combination down (%s at a scale of 10%%); on the slides with nothing planted %d region(s) entered a finding with DINOv2. "
            "The mechanism is a property of the encoder and of how the synthetic structures are built (the tiles of the patchwork are smaller than a patch, so a patch averages them); it is not evidence about kidney tissue, and it shows that the default weights and scale are not validated for DINOv2 on this benchmark."
            % (n(A(D, "any", "novelty")), pc(D["n_anomalous"] / float(D["n_analysed"]), 1), n(A(D, "any", "p_emb")), n(sw("dinov2_vits14", 0.005, "auroc_appearance")), n(sw("dinov2_vits14", 0.05, "auroc_appearance")), n(sw("dinov2_vits14", 0.10, "auroc_appearance")),
               n(A(D, "any", "p_het")), n(A(D, "any", "p_adj")), n(sw("dinov2_vits14", 0.10, "auroc_novelty")), sum(v["n_regions_in_findings"] for v in D["null_slides"].values())))


def ci(r):
    c = r.get("auroc_novelty_any_ci95_slide_bootstrap")
    return "%s–%s" % (n(c[0]), n(c[1])) if c else "n/a"


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
abs_ = ("Unsupervised discovery in whole-slide images (WSI) has to separate unusual tissue organization from artifacts and routine histology, and its outputs must be checkable by experts. "
        "We present MyUncleVision, an open pipeline that places patch embeddings on a tissue lattice and measures their organization with the quantities of the maintenance-lattice framework MyUncle: local disorder, neighbourhood entropy, transitions between tissue states, "
        "and a three-axis novelty score (appearance rarity, state-wise disorder, contact rarity). Regions are clustered across slides, ranked, mapped and explained with measured numbers. "
        "The order and entropy definitions generalize MyUncle's from two to d dimensions and reproduce its local disorder and von Mises entropy exactly at d = 2, which automated tests verify against the unmodified core. "
        "On eight synthetic slides with planted anomalies and artifacts, the novelty score reaches an AUROC of %s (slide-bootstrap interval %s) with a handcrafted encoder but %s with DINOv2, which is below chance: the planted states cover %s of the patches, more than the 1%% rarity scale, "
        "and DINOv2 embeds them more homogeneously than normal tissue. The appearance axis alone recovers with DINOv2 once the scale exceeds the prevalence (%s at a scale of 10%%). "
        "Quality control removed %s of the planted artifact patches. The platform has not been run on kidney tissue and no pathologist has reviewed any output; the evaluation protocol for the Kidney Precision Medicine Project images is specified. Code, tests and results are open."
        % (n(A(H, "any", "novelty")), ci(H), n(A(D, "any", "novelty")), pc(D["n_anomalous"] / float(D["n_analysed"]), 1), n(sw("dinov2_vits14", 0.10, "auroc_appearance")), pc(H["qc"]["flagged_among_planted"])))
add(("p0", abs_))
add(("p0", "Keywords: whole-slide imaging; unsupervised discovery; anomaly detection; kidney pathology; von Mises-Fisher entropy; maintenance lattice"))
add(("h1", "1. Introduction"))
add(("p", "Digital pathology produces gigapixel images of which most are examined at a small fraction of their area, and the models that analyse them are usually trained for a task defined in advance: a diagnosis, a grade, a segmentation [@Campanella2019;@Lu2021]. "
          "A different question is what organization of tissue exists that nobody has defined yet. Astronomy faced the same question for large surveys and developed outlier-detection and active anomaly-detection methods for it [@Baron2017;@Lochner2021]; "
          "in medical imaging, unsupervised anomaly detection has mostly meant detecting deviations from healthy appearance with generative models [@Schlegl2019]. "
          "The Kidney Precision Medicine Project (KPMP) has produced a large public set of kidney whole-slide images next to molecular atlases [@Lake2023], and a public challenge asks for previously unrecognized or under-characterized patterns in them."))
add(("p", "Three requirements follow. A discovery method must (i) not confuse artifacts with findings, (ii) not report routine histology, whose compartments differ from each other everywhere, as unusual, and (iii) give an expert something to look at: a place, a measure, an explanation. "
          "This paper describes MyUncleVision, which meets them with one idea: treat the tissue as a lattice of local states whose order is measured, and look for places where order, disorder or the contacts between states are unusual for what the tissue there is."))
add(("p", "Contributions: (1) a generalization of the order and entropy quantities of the maintenance-lattice framework MyUncle [@MyUncle2026] from two to d dimensions, verified against the unmodified core; (2) a three-axis novelty score that is rank-based inside a cohort, "
          "conditions local disorder on the tissue state and judges rarity at the scale of a fraction of the cohort; (3) an open, containerized pipeline from SVS files to ranked, mapped, explained candidates with an audit trail of every image used; "
          "(4) a validation on synthetic slides with planted anomalies and artifacts that reports where the method works and where it does not; (5) a protocol for the evaluation on KPMP images, which this paper does not report."))
add(("h1", "2. Method"))
add(("h2", "2.1 Slides, patches and quality control"))
add(("p", "Slides are read with OpenSlide for SVS and other vendor formats [@Goode2013] and as plain images otherwise. A tissue mask is obtained on a thumbnail (Otsu threshold on saturation, morphological clean-up). The slide is cut into non-overlapping square patches of 224 px at a target resolution of 0.5 µm/px; "
          "the patch grid is the lattice, and a patch is kept when at least half of it is tissue. Quality control marks a patch as an artifact when it is out of focus (Laplacian variance below an absolute floor or below 15% of the slide's own median), contains pen ink (saturated blue and green hues), "
          "is dark debris or a fold, or is mostly glass. These thresholds are heuristics and must be tuned on the slides at hand; every patch keeps its QC numbers. Artifact patches are excluded from the lattice, the reference set and the ranks."))
add(("h2", "2.2 Embeddings and the sphere"))
add(("p", "Each patch is encoded by DINOv2 ViT-S/14 [@Oquab2023] (384-d class-token embedding; natural-image pretraining, no fine-tuning) or, offline, by a deterministic handcrafted encoder of stain, colour, gradient-orientation and texture statistics (64-d) built on colour deconvolution [@Ruifrok2001]. "
          "Embeddings are centred and reduced by a cohort PCA (16 components, scikit-learn [@Pedregosa2011]) and normalized to unit length, so every patch is a point $u$ on a sphere $S^{d-1}$."))
add(("h2", "2.3 Order and entropy of the tissue lattice"))
add(("p", "MyUncle [@MyUncle2026] describes a lattice of rotors whose order is eroded by a state-dependent flux and restored by repair. Its local disorder is one minus the length of the mean of the four neighbours' unit vectors, and its order-entropy bridge is the von Mises entropy of the angle at a given order. "
          "The same definitions hold for unit vectors in any dimension. With $N_i$ the valid neighbours of site $i$ and $u_j$ the unit embedding of neighbour $j$,"))
add(("eq", r"R_i = \left| \frac{1}{|N_i|} \sum_{j \in N_i} u_j \right|, \qquad D_i = 1 - R_i", 1))
add(("p", "where $D_i$ is MyUncle's local disorder. The concentration $\\kappa_i$ of the von Mises-Fisher distribution on $S^{d-1}$ [@Banerjee2005;@Mardia2000] that has mean resultant length $R_i$ solves"))
add(("eq", r"A_d(\kappa) = \frac{I_{d/2}(\kappa)}{I_{d/2-1}(\kappa)} = R_i", 2))
add(("p", "and the entropy of that distribution and the information held in the local order are"))
add(("eq", r"S_{\mathrm{vMF}}(\kappa, d) = -\left(\frac{d}{2}-1\right) \ln \kappa + \frac{d}{2} \ln (2\pi) + \ln I_{d/2-1}(\kappa) - \kappa A_d(\kappa), \qquad H_\chi = S_{\mathrm{unif}} - S_{\mathrm{vMF}}", 3))
add(("p", "For $d = 2$ these are MyUncle's $U = A(\\kappa) = I_1/I_0$, $S(\\kappa)$ and $H_\\chi$. The tests compare them with the vendored, unmodified core (Fig. @F:bridge@): the local disorder on a periodic lattice agrees to $10^{-12}$ and the entropy to $10^{-9}$. "
          "A site with fewer than two valid neighbours has no order value. With the neighbourhood of four patches the resultant length is a noisy estimate of the true local order; this is a stated limitation."))
add(("fig", "fig_bridge.png", "Fig. @F:bridge@. (a) The order-information bridge in d dimensions: MyUncle's $H_\\chi$ (d = 2, red) and the vMF generalization for d = 2, 16, 64 and 384. (b) Difference between the d = 2 vMF entropy and the vendored MyUncle Bridge.S.", 6.0))
add(("h2", "2.4 Tissue states, transitions and the novelty score"))
add(("p", "Prototype states are k-means clusters of the unit embeddings (12 states). From them: the Shannon entropy of the states in a 3 × 3 window, and the rarity of a site's contacts, the mean over its neighbours of $-\\ln P(s_j \\mid s_i)$, "
          "where $P$ is the cohort-wide, Laplace-smoothed frequency with which states touch. Three axes are computed for every non-artifact patch and ranked inside the cohort (empirical percentile, 0 to 1):"))
add(("bullet", "appearance rarity $p_{emb}$: the mean cosine distance to the $k$ nearest patches of a reference set (a random sample of the cohort, same-slide neighbours within six lattice steps excluded), with $k$ equal to 1% of the reference set, so that a state that recurs but covers under 1% of the cohort is still rare;"))
add(("bullet", "state-wise disorder $p_{het}$: $D_i$ ranked among the patches of the same prototype state, so that the routine interfaces between compartments are not unusual;"))
add(("bullet", "contact rarity $p_{adj}$: the rank of the transition rarity."))
add(("p", "The novelty score is a weighted probabilistic OR, so that one strongly unusual axis suffices, as for a coherent region that is ordered but unlike anything else:"))
add(("eq", r"N = 1 - \prod_{a} (1 - p_a)^{w_a}, \qquad (w_{emb}, w_{het}, w_{adj}) = (0.4, 0.4, 0.2)", 4))
add(("h2", "2.5 Discovery, ranking and explanation"))
add(("p", "Patches in the top fraction $q$ of the cohort's novelty scores (default 2%) are candidates. Connected candidates of at least three patches form regions; regions are clustered across slides by the cosine distance of their mean embeddings (average linkage). "
          "A cluster is a finding, ranked by mean novelty × (1 + ln of the number of slides) × (0.5 + 0.5 × reproducibility), where reproducibility is the fraction of its patches that stay above the median rank of appearance rarity when the reference set is split into two disjoint halves. "
          "The score is a ranking, not a probability. Each finding carries a plain-language explanation made only of measured numbers (axes, local disorder against the cohort median, stain percentiles, the most frequent prototype state and its share of the cohort, recurrence), "
          "an explicit statement when a finding occurs in one slide only, and the request for expert review; no pathological term is assigned."))
add(("h2", "2.6 Progression scenarios with the MyUncle simulation"))
add(("p", "A slide is mapped onto MyUncle's lattice: the rotor angle of a site is the angle of its first two principal components, a site is damaged when its state-wise disorder is above the 75th percentile, and the observed damaged fraction $I$ gives $\\hat\\rho = (1 - I)/I$ from the linear steady state $I^* = 1/(1+\\rho)$ of the mean-field reduction. "
          "MyUncle's bistable window in $\\rho$ tells whether two maintained states are possible; its kinetic Monte Carlo, started from the slide, runs under half, equal and double repair. These are scenarios of a model; no clinical outcome is used."))
add(("fig", "fig_pipeline.png", "Fig. @F:pipe@. MyUncleVision: from whole-slide images to ranked, mapped and explained candidate states, with the unmodified MyUncle core supplying the reference definitions and the scenario engine.", 6.4))
add(("h2", "2.7 Software"))
add(("p", "Python (numpy, scipy, scikit-learn, OpenCV, timm and PyTorch for the encoder). One command, `python -m myuncle_vision run --input DIR --output DIR`, also the default command of the Docker image, writes findings, per-slide annotations in level-0 pixel coordinates, masks, heat maps, patch scores, scenario files, "
          "a CSV listing every image used and how, and a manifest with versions, configuration, model files and the hash of the MyUncle core. The code is archived and open (Data availability)."))
add(("h1", "3. Validation design"))
add(("p", "Real kidney images were not available for this work. The pipeline is therefore validated on synthetic slides (4,480 × 4,480 px at 0.5 µm/px) that imitate what the method measures, not kidney tissue: interstitium, tubule-like rings with nuclei, glomerulus-like discs, vessels, slide-to-slide stain variation, "
          "two planted anomaly types and three artifact types. The disorganized anomaly replaces the architecture by a patchwork of small tiles of different structure; the novel state is a coherent striped lattice in an unusual hue. Artifacts are an out-of-focus disc, a pen stroke and a dark fold. "
          "Eight slides carry different combinations; two carry no anomaly, one of these only artifacts. The production pipeline is run unchanged and judged against the planted truth. The same suite checks, with automated tests, the equalities with MyUncle at d = 2, "
          "the exclusion of invalid sites, determinism, the SVS reader on a public Aperio file (OpenSlide test data, not kidney) and the single command. The synthetic anomalies were designed after a first run showed that a region of random texture is not disordered in the sense of the method "
          "(adjacent patches look alike); the thresholds and the rarity scale were also adjusted after seeing synthetic results, so the numbers below are development results, not an untouched test."))
add(("h1", "4. Results on synthetic slides"))
tbl = []
for typ, lab in (("any", "any planted anomaly"), ("disorganized", "disorganized patchwork"), ("novel_state", "coherent novel state")):
    row = [lab]
    for r in (H, D):
        if r:
            row += [n(A(r, typ, "novelty")), n(A(r, typ, "p_emb")), n(A(r, typ, "p_het")), n(A(r, typ, "p_adj")), n(A(r, typ, "D"))]
        else:
            row += ["–"] * 5
    tbl.append(row)
add(("table", "Table 1. AUROC of the novelty score and of its axes for planted anomalous patches against normal patches (artifact patches excluded as in the pipeline). Columns 2 to 6: handcrafted encoder; columns 7 to 11: DINOv2. nov. = novelty score (probabilistic OR); app. = appearance rarity; dis. = state-wise disorder; con. = contact rarity; D = local disorder alone, without the state-wise rank (baseline).",
     ["Anomaly", "nov.", "app.", "dis.", "con.", "D", "nov.", "app.", "dis.", "con.", "D"], tbl, [2.2] + [0.7] * 10, True, 15))
p_ = PRIMARY
add(("p", "Table 1 reports %d slides with %s patches, %d of which were analysed after QC; %d are normal and %d lie in planted anomalies. With the %s encoder the novelty score separates planted from normal patches with an AUROC of %s (slide-bootstrap interval %s). "
          "The appearance axis alone is the best single score (%s), and local disorder alone reaches %s; the combination is therefore not better than its best axis on this benchmark. It is kept because the axes see different things: "
          "the patchwork (AUROC of the novelty score %s) is seen by appearance and disorder, and the rims of every anomaly by the state-wise disorder and contact axes."
     % (p_["n_slides"], "{:,}".format(p_["n_patches"]), p_["n_analysed"], p_["n_normal"], p_["n_anomalous"], p_["encoder"], n(A(p_, "any", "novelty")), ci(p_), n(A(p_, "any", "p_emb")), n(A(p_, "any", "D")), n(A(p_, "disorganized", "novelty")))))
if D and "dinov2_vits14" in SW:
    add(("p", dino_clause() + " With DINOv2 the top %s of the novelty scores contained no planted anomaly (%s) and %s artifact patches." % (pc(D["top_q"]["q"]), pc(D["top_q"]["precision_planted_anomaly"]), pc(D["top_q"]["precision_artifact"], 1))))
    rows2 = []
    for fr in SW["handcrafted"]["fractions"]:
        rows2.append([pc(fr, 1), n(sw("handcrafted", fr, "auroc_novelty")), n(sw("handcrafted", fr, "auroc_appearance")), n(sw("dinov2_vits14", fr, "auroc_novelty")), n(sw("dinov2_vits14", fr, "auroc_appearance"))])
    add(("table", "Table 2. AUROC for planted anomalous against normal patches as a function of the rarity scale (the fraction of the reference set used as the number of neighbours). The planted anomalies cover %s of the patches." % pc(D["n_anomalous"] / float(D["n_analysed"]), 1),
         ["Rarity scale", "handcrafted: novelty", "handcrafted: appearance", "DINOv2: novelty", "DINOv2: appearance"], rows2, [1.0, 1.2, 1.2, 1.2, 1.2], True, 16))
tq = p_["top_q"]
add(("p", "Among the top %s of the novelty scores (%d patches) %s lie in a planted anomaly (base rate %s) and %s within one lattice step of one (base rate %s); %s are artifacts. The strict patch-level precision is low because the score is highest on the rims of a region, where a patch overlaps the anomaly by less than half. "
          "Region level: a candidate patch fell inside %s of the planted disorganized regions and %s of the planted novel states; the remainder were missed at this threshold (Fig. @F:val@)."
     % (pc(tq["q"]), tq["k"], pc(tq["precision_planted_anomaly"]), pc(tq["base_rate_anomaly"], 1), pc(tq["precision_within_one_patch"]), pc(tq["base_rate_within_one_patch"], 1), pc(tq["precision_artifact"]),
        pc(p_["instance_recall"]["disorganized"]), pc(p_["instance_recall"]["novel_state"]))))
add(("fig", "fig_validation_example_%s.png" % p_["encoder"], "Fig. @F:val@. One synthetic slide (%s encoder): planted anomalies (green outlines), planted artifacts (red), and the three axes and the novelty score on the patch lattice." % p_["encoder"], 6.6))
q_ = p_["qc"]
ab = p_["ablation_no_qc"]
nl = p_["null_slides"]
add(("p", "Quality control flagged %s of the %d planted artifact patches (those overlapping an artifact by at least a quarter) and %s of the normal patches; the planted pen stroke and fold are caught more reliably than partial overlaps of the out-of-focus disc. "
          "Without QC the share of artifacts among the top novelty scores is %s, against %s with QC: in this benchmark artifacts rarely reach the top ranks even unfiltered, so the ablation shows a small effect and should not be read as proof that QC is unnecessary on real slides. "
          "On the two slides with nothing planted the pipeline produced %s candidate patches (%s expected from the threshold by construction) and no region that entered a finding."
     % (pc(q_["flagged_among_planted"]), q_["artifact_patches_planted"], pc(q_["false_flag_rate_normal"], 1), pc(ab["precision_artifact_in_top_q"], 1), pc(p_["top_q"]["precision_artifact"], 1),
        " and ".join(str(v["n_candidate_patches"]) for v in nl.values()), " and ".join(n(v["expected_by_construction"], 1) for v in nl.values()))))
add(("p", "The platform yielded %d findings on the synthetic cohort: %s. Their reproducibility under the two reference halves was %s."
     % (len(p_["findings"]), "; ".join("%s (%d patches, %d slide%s) with %s of its patches in a planted anomaly" % (f["id"], f["n_patches"], f["n_slides"], "s" if f["n_slides"] > 1 else "",
                                                                          pc(f["planted"]["disorganized"] + f["planted"]["novel_state"])) for f in p_["findings"]),
        ", ".join(pc(f["repro"]) for f in p_["findings"]))))
add(("h1", "5. Evaluation protocol for kidney whole-slide images (not yet run)"))
add(("p", "The following will be run on KPMP slides by anyone with access and will be reported with the findings: (i) the run manifest and the list of every image used; (ii) per slide, the fraction of patches excluded by QC and a review of a random sample of excluded and retained patches by a pathologist; "
          "(iii) the stability of the ranked findings under a different random seed, a different reference sample, and a different encoder; (iv) the recurrence of each finding across slides, patients and sites, with the site or scanner as the explanatory variable of a null model, since batch effects are the main alternative explanation for a recurring state; "
          "(v) expert review of the regions of the highest-ranked findings, blinded to the score, with a statement of which are known histology, which are artifacts and which, if any, are neither; (vi) the validation of the scenario analysis, if reported, against any longitudinal information that the open data contain."))
add(("h1", "6. Discussion"))
add(("p", "What the benchmark supports. With the handcrafted encoder the machinery does what it was built to do on data where the answer is known: it ranks planted structure above normal tissue, excludes the artifacts that its QC can see, and finds nothing coherent on slides with nothing planted. With DINOv2 at the default settings it does not (Section 4); the benchmark therefore does not validate the default configuration for real slides, and a real run should compare encoders, inspect the three axes separately and set the rarity scale deliberately (--weights, --rarity). "
          "The order and entropy definitions are MyUncle's, verified at d = 2, and extend to the embedding space in which tissue is represented. The explanation of a finding is a list of measured numbers that a reviewer can check against the image."))
add(("p", "What it does not support. Synthetic slides are not tissue: textures are simple, anomalies are large and sharply bounded, and the method's thresholds were set after seeing them. Nothing here shows that the method finds anything in a kidney, or that what it finds is biologically meaningful. "
          "Novelty is not pathology: a rare state may be a rare preparation, a scanner, a stain batch or a known entity that is merely uncommon in the cohort. The embedding is trained on natural images; on tissue it may group patches by stain or sharpness rather than by organization. "
          "The disorder measure compares a patch with only its four neighbours, which is noisy and responds to boundaries between compartments, which is why it is ranked within the tissue state. The probabilistic OR is not better than the best single axis here. The progression scenarios are model scenarios that no data have tested."))
add(("p", "Failure modes seen. A region of random texture is homogeneous at the patch scale and not disordered in this sense; the same holds for structure finer than a patch, and DINOv2 embeddings of the planted patchwork were more homogeneous than those of normal tissue, which inverted the disorder and contact axes; the recurring novel state is invisible to a nearest-neighbour distance at small $k$ and became visible only after rarity was judged at 1% of the cohort. "
          "Artifacts that overlap a patch only partly escape the QC. Improvements: a pathology-pretrained encoder (foundation models trained on large slide collections have been reported [@Chen2024;@Vorontsov2024], but were not available for this work), stain normalization [@Macenko2009], neighbourhoods larger than four patches, a calibrated null for the novelty score, and active expert feedback in the manner of anomaly-detection systems for astronomy [@Lochner2021]. "
          "Computing: the encoder dominates; a slide of ten thousand patches takes about ten minutes on one CPU core and minutes on a GPU, so the 2.4 TB of the first KPMP release is a cluster workload and the pipeline offers a patch budget per slide and a resumable cache."))
add(("h1", "7. Conclusions"))
add(("p", "MyUncleVision turns a whole-slide image into a lattice of local tissue states, measures its order and entropy with definitions that reproduce MyUncle's at d = 2, and ranks regions that are unusual in appearance, in disorder for their state, or in the contacts they make, with maps and explanations that an expert can check. "
          "On synthetic slides it works as designed with a handcrafted encoder and fails with DINOv2 at the default settings, for reasons the paper analyses. Its value for kidney pathology is untested and depends on an evaluation by experts that this paper specifies and does not report."))
add(("h1", "Declarations"))
add(("p0", "Funding: this work received no external funding."))
add(("p0", "Competing interests: the author declares no competing interests. [To be confirmed by the author.]"))
add(("p0", "Author contributions (CRediT): Leon Sandler, sole author: conceptualization, methodology, software, validation, formal analysis, investigation, visualization, writing – original draft, writing – review and editing."))
add(("p0", "Data and code availability: the code, the tests, the synthetic benchmark and its results are open at %s (MIT license) and archived on Zenodo (software: https://doi.org/10.5281/zenodo.24066260, manuscript: https://doi.org/10.5281/zenodo.24066256, all versions). The MyUncle core is vendored unmodified (doi:10.5281/zenodo.21223570). No patient data were used; the KPMP images are not part of this work." % GITHUB))
add(("p0", "Declaration of generative AI and AI-assisted technologies in the writing process: during the preparation of this work the author used Claude (Anthropic) to assist with writing and testing the code, running the analyses and drafting the text and figures. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the publication."))
add(("refs",))

# ================================================================================================ build
cit = E.Citations(LIB)
for b in B:
    for x in b[1:]:
        if isinstance(x, str):
            cit.scan(x)
FIGKEYS = {}
for b in B:
    if b[0] == "fig":
        pass
order = ["bridge", "pipe", "val"]
FIGNUM = {k: i + 1 for i, k in enumerate(order)}


def resolve(s):
    for k, v in FIGNUM.items():
        s = s.replace("@F:%s@" % k, str(v))
    return cit.resolve(s)


def build(path, blocks, journal):
    doc = E.Doc(cit, journal, True)
    # patch Doc.body_text so that figure and citation tokens are resolved
    num = 0
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
    exec(open(os.path.join(HERE, "build_herox.py"), encoding="utf-8").read())   # also defines DINO_SHORT used by the cover letter
    exec(open(os.path.join(HERE, "build_cover.py"), encoding="utf-8").read())
