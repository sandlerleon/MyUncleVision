# -*- coding: utf-8 -*-
"""ChemRxiv version of the manuscript: same evidence, framed around the statistical-mechanics content (order and entropy of lattices of unit vectors in d dimensions) and the simulated-lattice testbed.
Executed by build_documents.py (--variant chemrxiv) with the block list B, the results and the helper functions in scope.  No evidence is added or removed and no claim goes beyond the results:
no micrograph of a real material and no real tissue image was analysed, which the text says.  Whether ChemRxiv's moderators accept the scope is their decision."""

CHEM_TITLE = "Order and entropy of lattices of unit vectors in d dimensions: label-free discovery of unusual local organization in simulated maintenance lattices and synthetic images"
CHEM_KEYWORDS = "Keywords: lattice models; statistical mechanics; von Mises-Fisher distribution; orientational order; entropy; unsupervised anomaly detection; microscopy image analysis; open-source software"

import json as _json
_GS = _json.load(open(os.path.join(RES, "gaar_search_comparison.json")))
_GT = _json.load(open(os.path.join(RES, "gaar_test.json")))
_GL = _json.load(open(os.path.join(RES, "gaar_labelfree.json")))
_DEF, _NOS, _OPT = "frozen default", "no smoothing", "exhaustive dev optimum"
_RST, _GRD, _RND = "GaaR restart (seed 0, budget 40)", "GaaR greedy (seed 0, budget 40)", "GaaR random (seed 0, budget 40)"
_TF = lambda k: _GT["test"][k]["fitness"]
assert _TF(_RST) > _TF(_DEF) and _TF(_RND) > _TF(_DEF) and _TF(_GRD) > _TF(_DEF) and _TF(_OPT) > _TF(_DEF)
assert _TF(_RST) > _TF(_OPT) > _TF(_RND) and _GT["dev"][_OPT]["fitness"] > _GT["dev"][_RST]["fitness"] > _GT["dev"][_RND]["fitness"]
assert _GT["test"][_RST]["aps"][1] < _GT["test"][_DEF]["aps"][1] and _GT["test"][_RST]["aps"][0] > _GT["test"][_DEF]["aps"][0]
_st = _GS["strategies"]
_BT = _GT["bootstrap"]
_MS = _json.load(open(os.path.join(RES, "morphology_study.json")))
_mm = _MS["results"]
_sz = [str(x) for x in _MS["sizes"]]
assert _BT[_RST]["diff_vs_default_ci"][0] < 0 < _BT[_RST]["diff_vs_default_ci"][1]
assert all(_mm[b]["no smoothing"]["diff_vs_frozen_ci"][0] > 0 for b in _sz)
assert all(_mm[a]["frozen (mean 3)"]["ap"] < _mm[b]["frozen (mean 3)"]["ap"] for a, b in zip(_sz, _sz[1:]))
assert all(_mm[b]["mean 5"]["diff_vs_frozen_ci"][1] < 0 for b in _sz)
assert all(_mm[b]["GaaR restart champion"]["diff_vs_frozen_ci"][0] > 0 for b in ("1", "2", "3")) and _mm["6"]["GaaR restart champion"]["diff_vs_frozen_ci"][0] < 0 < _mm["6"]["GaaR restart champion"]["diff_vs_frozen_ci"][1]
assert all(_mm[b]["GaaR restart champion"]["ap"] < _mm[b]["no smoothing"]["ap"] for b in _sz)
_f2 = lambda x: ("-%.2f" % -x) if x < 0 else ("%.2f" % x)
_dci = lambda a: "[%s, %s]" % (("-%.2f" % -a[0]) if a[0] < 0 else ("+%.2f" % a[0]), ("-%.2f" % -a[1]) if a[1] < 0 else ("+%.2f" % a[1]))
_dd = lambda k: _GT["test"][k]["fitness"] - _GT["test"][_DEF]["fitness"]

assert _st["greedy"]["hit_top1pct"] > 2 * _st["random"]["hit_top1pct"] and _st["restart"]["hit_top1pct"] > 2 * _st["random"]["hit_top1pct"]
assert _st["random"]["hit_top5pct"] >= _st["population"]["hit_top5pct"] > _st["restart"]["hit_top5pct"] >= _st["greedy"]["hit_top5pct"]
_mor = {r["scenario"]: r["moran"] for r in _GL["rows"]}
_eff = {r["scenario"]: r["effect"] for r in _GL["rows"]}
_bad = [k for k in _eff if _eff[k] < -0.2]
assert sorted(_bad) == ["sim-dev-rotor", "sim-test-rotor"] and all(_mor[k] < 0.02 for k in _bad) and all(_mor[k] > 0.1 for k in _mor if k not in _bad)

CHEM_ABSTRACT = (
    "Averaged order parameters say how ordered a lattice of orientational states is, but not where its organization is unusual. "
    "We extend the local order and entropy quantities of the MyUncle maintenance-lattice model from the plane to unit vectors in d dimensions with the von Mises-Fisher distribution; both equal MyUncle's at d = 2 (tested against the unmodified code). "
    "On this basis we build a label-free score that judges how rare the state of a site is at several prevalence scales and smooths it over the lattice, and compare it with isolation forest, local outlier factor, nearest-neighbor and Mahalanobis baselines, with settings frozen before the held-out data were generated. "
    "On lattices simulated with MyUncle (damaged sites as ground truth; parameter sets labeled polymer photodegradation, liquid crystal and magnetic storage decay), smoothing lowered the average precision of every baseline because damaged sites are not compact regions. "
    "On synthetic stained-tissue images, smoothing raised spatial coherence for every scorer and the frozen score was the best method with DINOv2 embeddings and a poor one with handcrafted features. "
    "A controlled study on an independent generator found that smoothing hurt at every anomaly size, least for compact anomalies. A supervised search over 360 configurations with GaaR raised the lower-tail average precision on fresh test scenarios from %s to %s, but the paired interval of the gain included zero and random search also beat the default. "
    "All evidence is simulated; validation on experimental micrographs is future work. Code and results are open." % (n(_TF(_DEF)), n(_TF(_RST))))

CHEM_INTRO = [
    ("p", "Lattice systems whose sites carry an orientation or a state vector, such as liquid-crystal textures, degrading polymer films, magnetic media and self-assembled layers, are commonly summarized by averaged order parameters. These say how ordered a system is, not where something unusual is. "
          "Finding unusual local organization without labels, and saying how unusual it is in the language of order and entropy, is the problem addressed here. Unsupervised anomaly detection has been developed in other fields: in astronomy for large surveys [@Baron2017;@Lochner2021], and in medical imaging mostly as detection of deviation from a healthy appearance with generative models [@Schlegl2019]."),
    ("p", "MyUncle [@MyUncle2026] is a simulation framework for a maintenance lattice: rotors on a square lattice whose order is eroded by a state-dependent flux and restored by repair, with an order-entropy bridge that links the length of the neighbors' mean unit vector to a von Mises entropy. "
          "This paper makes five contributions. (1) It extends the local order and entropy quantities from the plane to unit vectors in d dimensions (the von Mises-Fisher distribution) and verifies that they equal MyUncle's at d = 2 against the unmodified code. "
          "(2) It builds a label-free score that judges the rarity of the state of a site at several prevalence scales and smooths it over the lattice. "
          "(3) It tests the score, with frozen settings and baselines, first on lattices simulated with MyUncle's kinetic Monte Carlo (damaged sites as ground truth; one development and three held-out model parameter sets) and second on synthetic stained-tissue images, where the lattice sites are image patches described by an embedding; the results are mixed and are reported as such. "
          "(4) It studies a supervised configuration search with GaaR [@GaaR2026] on fresh test scenarios that no search saw. (5) It provides an open, containerized pipeline for all of it, archived with the vendored MyUncle core [@MyUncleVision2026]. All evidence is computational and simulated: no micrograph of a real material and no real tissue image was analysed, and the application to experimental data is stated as future work (Section 5)."),
]


_ROWS6 = [[lab, n(_GT["dev"][k]["fitness"])] + [n(x) for x in _GT["test"][k]["aps"]] + ["%s [%s, %s]" % (n(_TF(k)), _f2(_BT[k]["fitness_ci"][0]), _f2(_BT[k]["fitness_ci"][1])), "-" if k == _DEF else "%+.2f %s" % (_dd(k), _dci(_BT[k]["diff_vs_default_ci"]))] for k, lab in ((_DEF, "Frozen default"), (_NOS, "No smoothing"), (_RST, "GaaR restart / population champion"), (_GRD, "GaaR greedy / random-search champion"), (_OPT, "Exhaustive development optimum"))]
CHEM_GAAR = [
    ("h2", "4.6 Configuration search with GaaR"),
    ("p", "The frozen configuration is one choice for all data, and the results above show that it is not best for every embedding. GaaR (Goal-Aware Adaptive Regulation) [@GaaR2026], an evolutionary search framework by the same author, was used to select a configuration of the score from %d candidates: smoothing type and window (none, mean 3, 5, 7, Gaussian 5, 7, median 3, 5), five sets of axis weights, the PCA dimension (8, 16, 32) and the largest rarity scale (2, 5, 10%%). "
          "This is supervised hyperparameter selection on labeled development scenarios, not label-free adaptation: the chosen configuration is fixed afterwards. The fitness is a lower-tail average precision, F = 0.5 mean(AP) + 0.5 min(AP), over four development scenarios (the development images with each embedding and the development lattices with each of their two embeddings). "
          "All %d configurations were evaluated once on these scenarios, so any search can be compared exactly with the optimum; each strategy of GaaR and plain random search then ran %d times with a budget of %d evaluations. The champions were evaluated once on fresh test scenarios that no search saw: 16 new synthetic images (new seeds, the held-out anomaly types) and 48 new lattice states, including a preset not used before (superconducting scaffold)."
          % (_GS["n_configs"], _GS["n_configs"], _GS["repeats"], _GS["budget"])),
    ("table", "Table 5. Fresh test scenarios: average precision (AP) of the frozen configuration, of no smoothing and of the champions found on the development scenarios (development fitness F in column 2; test AP for handcrafted-feature images, DINOv2 images, and lattices with the rotor and the rotor + D embedding in columns 3-6; test fitness with a 95% interval in column 7 and its paired difference from the default in column 8; intervals resample the 16 images and the 16 simulation runs of each scenario).",
     ["Configuration", "F dev", "Img. hand.", "Img. DINOv2", "Lat. rotor", "Lat. rotor+D", "F test [95% interval]", "Difference from default [95% interval]"], _ROWS6, [1.9, 0.5, 0.55, 0.65, 0.55, 0.65, 1.1, 1.35], True, 12),
    ("p", "The default configuration ranked %dth of %d on the development fitness (%s; optimum %s). With %d evaluations, the greedy and restart strategies found a configuration in the top 1%% in %s and %s of the runs and random search in %s (population: %s), whereas at the top 5%% level random search did as well or better (%s against %s, %s and %s). "
          "On the test scenarios (Table 5) the champion of the restart and population strategies raised the test fitness from %s (default) to %s, mainly by lifting the worst scenario (handcrafted-feature images, AP %s to %s), while the AP with DINOv2 fell from %s to %s. The paired 95%% interval of this gain (%+.2f) is %s and the champion was better than the default in %s of the resamples, so with 16 independent units per scenario it cannot be distinguished from zero. The champion of the greedy strategy and of random search (the same configuration here) reached %s, and the exhaustive development optimum %s: every development-optimized configuration beat the default, but their order did not follow the development fitness. "
          "The gain is therefore due to the selected configuration (a smaller PCA dimension and the disorder axis at weight 0.5, with 3 x 3 mean smoothing) and not to the search algorithm; each strategy contributed one champion, four scenarios are few, and the test data come from the same generators as the development data."
          % (int(round(_GS["default_rank_fraction"] * _GS["n_configs"])) + 1, _GS["n_configs"], n(_GS["default_fitness"]), n(_GS["optimum"]), _GS["budget"], pc(_st["greedy"]["hit_top1pct"]), pc(_st["restart"]["hit_top1pct"]), pc(_st["random"]["hit_top1pct"]), pc(_st["population"]["hit_top1pct"]),
             pc(_st["random"]["hit_top5pct"]), pc(_st["greedy"]["hit_top5pct"]), pc(_st["restart"]["hit_top5pct"]), pc(_st["population"]["hit_top5pct"]),
             n(_TF(_DEF)), n(_TF(_RST)), n(_GT["test"][_DEF]["aps"][0]), n(_GT["test"][_RST]["aps"][0]), n(_GT["test"][_DEF]["aps"][1]), n(_GT["test"][_RST]["aps"][1]), _dd(_RST), _dci(_BT[_RST]["diff_vs_default_ci"]), pc(_BT[_RST]["share_better_than_default"]), n(_TF(_GRD)), n(_TF(_OPT)))),
    ("h2", "4.7 Smoothing against anomaly morphology"),
    ("p", "The results above leave open when smoothing helps. A controlled study on an independent generator varied only the spatial size of the anomalies. Each of %d lattices (32 x 32 sites, 16-dimensional embeddings) has smooth background domains and 3%% anomalous sites with a fixed separability (a shift of 10 noise standard deviations, calibrated beforehand on the unsmoothed score only); the anomalies are b x b blocks, b = 1 being scattered single sites and b = 6 compact regions, and each block has its own direction, so no recurring state exists. "
          "Without smoothing the average precision was between %s and %s across sizes; the frozen 3 x 3 mean lowered it to %s at b = 1 and %s at b = 6 (Table 6). Smoothing hurt at every size, the loss shrinking as the anomalies grew, and a 5 x 5 mean was worse than the 3 x 3 mean at every size (paired 95%% intervals of all these differences exclude zero). "
          "The champion of the GaaR search was better than the frozen configuration for b up to 3 and not distinguishable from it for b = 4 and 6, but it stayed below no smoothing at every size: configuration selection on the development scenarios, in which smoothing helped the DINOv2 images, mitigated the loss without removing it."
          % (_MS["images_per_cohort"] * _MS["cohorts"], "%.2f" % min(_mm[b]["no smoothing"]["ap"] for b in _sz), "%.2f" % max(_mm[b]["no smoothing"]["ap"] for b in _sz), "%.2f" % _mm["1"]["frozen (mean 3)"]["ap"], "%.2f" % _mm["6"]["frozen (mean 3)"]["ap"])),
    ("table", "Table 6. Average precision against anomaly block size b (independent embedding-lattice generator; 3% anomalous sites, fixed separability).",
     ["Configuration", "b = 1", "b = 2", "b = 3", "b = 4", "b = 6"],
     [[lab] + ["%.2f" % _mm[b][cfg]["ap"] for b in _sz] for cfg, lab in (("no smoothing", "No smoothing"), ("frozen (mean 3)", "Frozen (3 x 3 mean)"), ("mean 5", "5 x 5 mean"), ("GaaR restart champion", "GaaR restart champion"), ("exhaustive development optimum", "Exhaustive development optimum"))],
     [2.6, 0.7, 0.7, 0.7, 0.7, 0.7], True, 14),
    ("p", "A label-free alternative was explored. Moran's I of the unsmoothed rarity score (spatial autocorrelation of the score field) was below 0.02 in exactly the two lattice scenarios, one development and one test, in which smoothing lost most of the AP, and above 0.1 in the other six. A rule that smooths only above 0.05 would have avoided the largest loss in both, but it would not have helped the handcrafted-feature test images (eight scenarios, one threshold, exploratory)."),
]

CHEM_FUTURE = [
    ("h1", "5. Future work: validation on experimental micrographs and histology"),
    ("p", "The evidence in this paper is simulated: the lattices were generated with MyUncle by the author of the method, and the images were generated by the author with simple textures and sharply bounded anomalies. They cannot show that the method finds anything of physical, chemical or biological meaning in an experimental system. "
          "Validation on experimental data is therefore the principal next step, and the study is specified here in advance so that its design cannot be shaped by its results. It has not been started."),
    ("h2", "5.1 Data and design"),
    ("p", "Two kinds of data are planned. The first is micrographs of ordered materials whose defect structure can be established independently, such as polarized-light textures of liquid crystals or microscopy of block-copolymer and self-assembled films, with defects annotated by at least two experts or by an independent measurement. "
          "The second is publicly released whole-slide images of tissue, for which the imaging pipeline of Section 2 was built. For each kind a pilot of 10 to 20 images will be used to inspect the quality-control exclusions and the maps and to fix the settings. "
          "The settings, the list of baselines and the analysis plan will then be frozen and published in the repository before an independent set of images from other samples, instruments and operators is analysed once, following the procedure used here for the held-out cohorts."),
    ("h2", "5.2 Comparators and endpoints"),
    ("p", "The frozen score will be compared with the baselines of Section 3 (isolation forest, local outlier factor, nearest-neighbor and Mahalanobis distances, raw and smoothed) and with other spatial postprocessing (Gaussian and median filters, other window sizes). "
          "Primary endpoint: the share of the top-ranked regions that at least two experts, blinded to the method and to the ranking, judge to be a defect or an unexplained feature rather than a known feature or an artifact, with interrater agreement (Cohen's kappa) and bootstrap confidence intervals over images, compared between methods by paired resampling. "
          "Secondary endpoints: the recall of annotated defects at a fixed review budget; the fraction of artifact regions among the top-ranked ones; the stability of the flagged set under another seed, reference sample, embedding and patch size (Jaccard overlap); computing time. "
          "The success criteria will be written into the analysis plan before the independent set is analysed."),
    ("h2", "5.3 Further planned work"),
    ("p", "A second synthetic generator with different textures, anomaly boundaries and artifacts, evaluated once with the frozen settings; bootstrap intervals for the average precision and for the differences from the Mahalanobis baseline, with more planted instances than the seven of the present held-out image cohort and more than four simulation runs per parameter set; "
          "a direct comparison of the lattice smoothing with Gaussian and median filtering and other neighborhood sizes; smoothing of the raw score instead of its rank; fitting the MyUncle parameter sets to measured data, which would turn the labels used here for the parameter sets into calibrated models; "
          "stain normalization [@Macenko2009] and a pathology-pretrained encoder [@Chen2024;@Vorontsov2024] for tissue images. If the experimental study shows no advantage over the baselines, that result will be reported."),
]

CHEM_DISCUSSION = [
    ("p", "What the evaluation supports. The order and entropy definitions are MyUncle's, verified at d = 2, and extend to unit vectors in d dimensions with the von Mises-Fisher distribution. The lattice smoothing gives spatially coherent candidate regions for any scorer when the anomalies form regions, "
          "and it is harmful when they do not: on the simulated lattices the damaged sites are scattered and smoothing lowered the average precision of every baseline, whereas on the synthetic images it raised the spatial coherence of every scorer. The multi-scale rarity removes the dependence on a prevalence chosen in advance, and a supervised search with GaaR improved the lower-tail average precision on fresh test scenarios, although random search also beat the default and the gain therefore belongs to the selected configuration, "
          "although the damaged share of three of the four parameter sets (about 10%, 30% and 26%; the fourth is 8%) is near or above the largest prevalence scale of the rarity score (10% of the reference set), a limit of the frozen settings that the simulated testbed exposes."),
    ("p", "What it does not support. The lattices and the images are simulated, written by the author of the method (the simulation is MyUncle's, with an easy kind of anomaly, a site that lost its coupling, and with model parameter sets that carry application names but are not fitted to measurements); nothing here shows that the method finds anything in an experimental material or in tissue, or that what it finds is meaningful. "
          "The held-out sets were used once but are of the same family as the development sets, and four simulation runs per parameter set are few. The two embeddings of the images disagree on whether smoothing helps and on whether the lattice axes help, and a single Gaussian model scored higher than the frozen score on the development cohort. "
          "Novelty is not a defect: a rare state may be a rare sample preparation, an instrument, a staining or imaging batch, or a known feature that is merely uncommon in the set. The image embedding is trained on natural images and may group patches by stain or sharpness, and quality control flagged only part of the planted artifact patches in the development runs (42%)."),
    ("p", "Improvements: neighborhoods larger than four sites, a calibrated null for the novelty score, smoothing of the raw score instead of its rank, an embedding trained on micrographs of the system at hand, and active expert feedback in the manner of anomaly detection systems for astronomy [@Lochner2021]. "
          "Computing: the encoder dominates; an image of ten thousand patches takes about ten minutes on one CPU core and minutes on a GPU, and the pipeline offers a patch budget per image and a resumable cache."),
]

CHEM_CONCLUSION = [
    ("p", "The local disorder and the von Mises-Fisher order information of MyUncle generalize to lattices of unit vectors in d dimensions and equal MyUncle's at d = 2. A label-free score built on them, rarity of the state of a site at several prevalence scales smoothed over the lattice, was tested on lattices simulated with MyUncle and on synthetic stained-tissue images, with frozen settings and four baselines. "
          "The results are mixed: smoothing helped only with the DINOv2 embeddings of the synthetic images, whose unsmoothed score is noisy, and hurt on simulated lattices and in a controlled study on an independent generator, least for compact anomalies; adding the local disorder as a feature helped on the simulated lattices, the frozen score was competitive but not uniformly best, and a simple Mahalanobis baseline was a strong competitor. "
          "A supervised configuration search with GaaR raised the lower-tail average precision on fresh test scenarios, but the paired interval of the gain included zero and random search also beat the default, so any gain belongs to the selected configuration. All evidence is simulated; validation on experimental micrographs and on histology, with expert review, is the planned next step (Section 5)."),
]


def chemrxiv_blocks(B):
    """return the ChemRxiv block list derived from the journal block list B"""
    def find(kind, prefix, start=0):
        for i in range(start, len(B)):
            if B[i][0] == kind and isinstance(B[i][1], str) and B[i][1].startswith(prefix):
                return i
        raise KeyError(prefix)

    out = list(B)
    # title, abstract, keywords
    out[find("title", "MyUncleVision")] = ("title", CHEM_TITLE)
    out[find("p0", "Unsupervised discovery in whole-slide")] = ("p0", CHEM_ABSTRACT)
    out[find("p0", "Keywords:")] = ("p0", CHEM_KEYWORDS)
    # introduction
    i = find("p", "Digital pathology produces")
    out[i:i + 2] = CHEM_INTRO
    # headings and small framing edits
    def sub(kind, old, new):
        for j in range(len(out)):
            if out[j][0] == kind and isinstance(out[j][1], str) and old in out[j][1]:
                out[j] = (kind,) + tuple(x.replace(old, new) if isinstance(x, str) else x for x in out[j][1:])
                return
        raise KeyError(old)

    sub("h1", "2. Method", "2. Theory and method")
    sub("h2", "2.1 Slides, patches and quality control", "2.1 Imaging application: images, patches and quality control")
    sub("h2", "2.3 Order and entropy of the tissue lattice", "2.3 Order and entropy of a lattice of unit vectors in d dimensions")
    sub("p", "No real kidney image was available for this work.", "For the imaging application no real image of tissue or of a material was available for this work; the lattice simulations of Section 4.5 are a separate testbed.")
    sub("h1", "4. Results on synthetic slides", "4. Results on synthetic images and simulated lattices")
    sub("h2", "4.5 A second testbed: lattices simulated with MyUncle", "4.5 Simulated maintenance lattices (MyUncle)")
    sub("p", "The development set was one preset", "The presets are model parameter sets that MyUncle labels with application names; they are not fitted to measurements of any material. The development set was one preset")
    # future work, discussion, conclusions
    i, j = find("h1", "5. Future work"), find("h1", "6. Discussion")
    out[i:j] = CHEM_GAAR + CHEM_FUTURE
    i, j = find("p", "What the evaluation supports."), find("h1", "7. Conclusions")
    out[i:j] = CHEM_DISCUSSION
    i = find("p", "MyUncleVision turns a whole-slide image")
    out[i:i + 1] = CHEM_CONCLUSION
    sub("p0", "No patient data were used; the KPMP images are not part of this work.", "No experimental, patient or tissue data were used.")
    ia = find("small", "Corresponding author")
    k = find("p", "With the rotor alone")
    out[k + 1:k + 1] = [("p", "Fig. @F:maps@ shows one simulated run (preset chiral scaffold, 32 x 32 sites, snapshots 15 steps apart): the rotor angle, the local disorder D, the order information H_chi and the novelty score of the frozen method. Novelty is a percentile rank relative to the snapshots analysed together, so it marks where a state is unusual among the snapshots shown; it has not been validated against known defect locations."),
                        ("fig", "fig_bridge_maps.png", "Fig. @F:maps@. A MyUncle simulation analysed with MyUncleVision (command python -m myuncle_vision bridge, preset chiral scaffold, 32 x 32 sites, seed 0). Rows: rotor angle, local disorder D, order information H_chi (von Mises-Fisher, d = 2) and novelty of the frozen method; columns: snapshots at steps 0, 15, 30, 45 and 60.", 6.4)]
    out.insert(ia + 1, ("small", "Under consideration at Pattern Recognition Letters (submission PRLETTERS-D-26-02906)"))
    return out
