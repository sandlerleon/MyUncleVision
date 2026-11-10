# -*- coding: utf-8 -*-
"""ChemRxiv version of the manuscript: same evidence, framed around the statistical-mechanics content (order and entropy of lattices of unit vectors in d dimensions) and the simulated-lattice testbed.
Executed by build_documents.py (--variant chemrxiv) with the block list B, the results and the helper functions in scope.  No evidence is added or removed and no claim goes beyond the results:
no micrograph of a real material and no real tissue image was analysed, which the text says.  Whether ChemRxiv's moderators accept the scope is their decision."""

CHEM_TITLE = "Order and entropy of lattices of unit vectors in d dimensions: label-free discovery of unusual local organization in simulated maintenance lattices and synthetic images"
CHEM_KEYWORDS = "Keywords: lattice models; statistical mechanics; von Mises-Fisher distribution; orientational order; entropy; unsupervised anomaly detection; microscopy image analysis; open-source software"

CHEM_ABSTRACT = (
    "Averaged order parameters say how ordered a lattice of orientational states is, but not where its organization is unusual. "
    "We extend the local order and entropy quantities of the MyUncle maintenance-lattice model from the plane to unit vectors in d dimensions: the local disorder is one minus the length of the mean of the neighbors' unit vectors, and the information held in the local order follows from the von Mises-Fisher distribution; both equal MyUncle's at d = 2 (tested against the unmodified code). "
    "On this basis we build a label-free score that judges how rare the state of a site is at several prevalence scales and smooths it over the lattice, and compare it with isolation forest, local outlier factor, nearest-neighbor and Mahalanobis baselines, with settings frozen before the held-out data were generated. "
    "On lattices simulated with MyUncle (damaged sites as ground truth; one development and three held-out parameter sets, labeled polymer photodegradation, liquid crystal and magnetic storage decay), smoothing lowered the average precision of every baseline because damaged sites are not compact regions, adding the local disorder as a feature raised it, and the frozen score was within 0.02 of the best on the held-out sets but clearly beaten on the development set. "
    "On synthetic stained-tissue images, smoothing raised the spatial coherence of every scorer; the frozen score was the best method with DINOv2 embeddings and a poor one with handcrafted features. "
    "All evidence is simulated; validation on experimental micrographs is future work. Code and results are open.")

CHEM_INTRO = [
    ("p", "Lattice systems whose sites carry an orientation or a state vector, such as liquid-crystal textures, degrading polymer films, magnetic media and self-assembled layers, are commonly summarized by averaged order parameters. These say how ordered a system is, not where something unusual is. "
          "Finding unusual local organization without labels, and saying how unusual it is in the language of order and entropy, is the problem addressed here. Unsupervised anomaly detection has been developed in other fields: in astronomy for large surveys [@Baron2017;@Lochner2021], and in medical imaging mostly as detection of deviation from a healthy appearance with generative models [@Schlegl2019]."),
    ("p", "MyUncle [@MyUncle2026] is a simulation framework for a maintenance lattice: rotors on a square lattice whose order is eroded by a state-dependent flux and restored by repair, with an order-entropy bridge that links the length of the neighbors' mean unit vector to a von Mises entropy. "
          "This paper makes four contributions. (1) It extends the local order and entropy quantities from the plane to unit vectors in d dimensions (the von Mises-Fisher distribution) and verifies that they equal MyUncle's at d = 2 against the unmodified code. "
          "(2) It builds a label-free score that judges the rarity of the state of a site at several prevalence scales and smooths it over the lattice. "
          "(3) It tests the score, with frozen settings and baselines, first on lattices simulated with MyUncle's kinetic Monte Carlo (damaged sites as ground truth; one development and three held-out model parameter sets) and second on synthetic stained-tissue images, where the lattice sites are image patches described by an embedding; the results are mixed and are reported as such. "
          "(4) It provides an open, containerized pipeline for both, archived with the vendored MyUncle core [@MyUncleVision2026]. All evidence is computational and simulated: no micrograph of a real material and no real tissue image was analysed, and the application to experimental data is stated as future work (Section 5)."),
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
          "and it is harmful when they do not: on the simulated lattices the damaged sites are scattered and smoothing lowered the average precision of every baseline, whereas on the synthetic images it raised the spatial coherence of every scorer. The multi-scale rarity removes the dependence on a prevalence chosen in advance, "
          "although the damaged share of three of the four parameter sets (about 10%, 30% and 26%; the fourth is 8%) is near or above the largest prevalence scale of the rarity score (10% of the reference set), a limit of the frozen settings that the simulated testbed exposes."),
    ("p", "What it does not support. The lattices and the images are simulated, written by the author of the method (the simulation is MyUncle's, with an easy kind of anomaly, a site that lost its coupling, and with model parameter sets that carry application names but are not fitted to measurements); nothing here shows that the method finds anything in an experimental material or in tissue, or that what it finds is meaningful. "
          "The held-out sets were used once but are of the same family as the development sets, and four simulation runs per parameter set are few. The two embeddings of the images disagree on whether smoothing helps and on whether the lattice axes help, and a single Gaussian model scored higher than the frozen score on the development cohort. "
          "Novelty is not a defect: a rare state may be a rare sample preparation, an instrument, a staining or imaging batch, or a known feature that is merely uncommon in the set. The image embedding is trained on natural images and may group patches by stain or sharpness, and quality control flagged only part of the planted artifact patches in the development runs (42%)."),
    ("p", "Improvements: neighborhoods larger than four sites, a calibrated null for the novelty score, smoothing of the raw score instead of its rank, an embedding trained on micrographs of the system at hand, and active expert feedback in the manner of anomaly detection systems for astronomy [@Lochner2021]. "
          "Computing: the encoder dominates; an image of ten thousand patches takes about ten minutes on one CPU core and minutes on a GPU, and the pipeline offers a patch budget per image and a resumable cache."),
]

CHEM_CONCLUSION = [
    ("p", "The local disorder and the von Mises-Fisher order information of MyUncle generalize to lattices of unit vectors in d dimensions and equal MyUncle's at d = 2. A label-free score built on them, rarity of the state of a site at several prevalence scales smoothed over the lattice, was tested on lattices simulated with MyUncle and on synthetic stained-tissue images, with frozen settings and four baselines. "
          "The results are mixed: smoothing helps when anomalies form regions and hurts when they do not, adding the local disorder as a feature helped on the simulated lattices, the frozen score was competitive but not uniformly best, and a simple Mahalanobis baseline was a strong competitor. "
          "All evidence is simulated; validation on experimental micrographs and on histology, with expert review, is the planned next step (Section 5)."),
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
    out[i:j] = CHEM_FUTURE
    i, j = find("p", "What the evaluation supports."), find("h1", "7. Conclusions")
    out[i:j] = CHEM_DISCUSSION
    i = find("p", "MyUncleVision turns a whole-slide image")
    out[i:i + 1] = CHEM_CONCLUSION
    sub("p0", "No patient data were used; the KPMP images are not part of this work.", "No experimental, patient or tissue data were used.")
    ia = find("small", "Corresponding author")
    out.insert(ia + 1, ("small", "Under consideration at Pattern Recognition Letters (submission PRLETTERS-D-26-02906)"))
    return out
