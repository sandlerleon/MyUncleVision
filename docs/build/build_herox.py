# -*- coding: utf-8 -*-
"""HeroX submission pack (executed by build_documents.py, which supplies H, D, PRIMARY, KP, n, pc, A, ci, LIB).

Writes docs/MyUncleVision_HeroX_Submission.docx and docs/HeroX_fields.md.  The portal limits narrative fields to about 3,000 characters per page, roughly 8.5 pages in all;
the character count of every field is printed in the pack.  Without a real KPMP run (--kpmp-run) the pack says so on its first page and in every narrative that depends on kidney results.
"""
import docx as _docx
from docx.shared import Pt, Inches, RGBColor

P_ = PRIMARY
DINO_SHORT = ("With the DINOv2 encoder the same synthetic benchmark gives an AUROC of %s for the novelty score, below chance: at the default rarity scale of 1%% the planted states (about %s of the patches) are not rare, and DINOv2 embeds them more homogeneously than normal tissue, "
              "so two of the three axes point the wrong way; the appearance axis alone recovers when the scale is raised (AUROC %s at 10%%). With DINOv2, %d region(s) on slides with nothing planted entered a finding. The default settings are therefore not validated for DINOv2 on this benchmark, and a real run should compare encoders and inspect the axes separately."
              % (n(A(D, "any", "novelty")), pc(D["n_anomalous"] / float(D["n_analysed"]), 1), n(sw("dinov2_vits14", 0.10, "auroc_appearance")), sum(v["n_regions_in_findings"] for v in D["null_slides"].values()))) if (D and "dinov2_vits14" in SW) else ""
tq, q_, ab, nl = P_["top_q"], P_["qc"], P_["ablation_no_qc"], P_["null_slides"]
PENDING = ("[KPMP RUN PENDING] No KPMP whole-slide image has been analysed with this platform yet. This paragraph is replaced automatically by the results of the run when "
           "`python docs/build/build_documents.py --kpmp-run <output directory of the KPMP run>` is executed. Do not submit the pack with this notice in it.")


def kp_block(top=5):
    if not KP:
        return PENDING
    f = KP["findings"]["findings"][:top]
    s = KP["summary"]
    man = KP["manifest"]
    lines = ["Run on %d KPMP slides (%d analysed; encoder %s; MyUncle core unmodified: %s). The run produced %d candidate states." % (man["n_slides_listed"], man["n_slides_analysed"], man["encoder"], man["myuncle_core_unmodified"], man["n_findings"])]
    for x in f:
        lines.append("%s (%s): %d patches, %.2f mm2, %d slides, novelty %.2f, reproducibility %.0f%%. %s" % (x["id"], x["type"], x["n_patches"], x["area_mm2"], x["n_slides"], x["novelty"], 100 * x["reproducibility"], x["explanation"]))
    return "\n".join(lines)


abstract = ("Kidney biopsies are photographed at very high resolution, but most of each image is never looked at closely, and computer programs are usually trained to find only what people have already named. "
            "MyUncleVision looks for the unnamed. It cuts each biopsy image into small tiles, describes every tile with a pretrained image model, and then measures how orderly the neighbourhood of every tile is, "
            "using a mathematical framework (MyUncle) developed for orderly and disorderly lattices. Places that look unusual, are unusually disordered for the kind of tissue they are, or make unusual contacts between kinds of tissue are ranked, "
            "grouped across biopsies, mapped on the slide and explained in plain numbers, so that a pathologist can check them. Artifacts such as blur, pen marks and folds are screened out first. "
            "The software runs from one command in a container and lists every image it used. It is tested on synthetic slides with planted answers, where, with a simple colour-and-texture description of the tiles, it ranks the planted structure well above normal tissue; with the pretrained image model it did not on that synthetic test, so its settings must be tuned and checked on real slides; "
            "it has not yet been run on kidney slides, and its candidates are for expert review, not diagnoses.")

pending_note = "" if KP else "\n\n" + PENDING
nar1 = ("What counts as a discovery here. A finding is a candidate organizational state of tissue: a group of regions, found in one or more slides, that is unusual in at least one of three measured ways and that survives four filters. "
        "The three measures are (1) appearance rarity: how far the tile's description is from the tiles of the rest of the cohort, judged at the scale of 1%% of the cohort so that a rare state that recurs is still rare; "
        "(2) disorder for the tissue type: how much the tile disagrees with its neighbours, compared only with tiles of the same prototype tissue state, so that the ordinary borders between glomeruli, tubules and interstitium do not count; "
        "(3) rare contacts: how seldom the tile's tissue state touches the states around it. The four filters are artifact screening, a minimum region size, recurrence across slides (a state that appears in many biopsies is less likely to be one bad slide), "
        "and reproducibility when the reference set is split into two halves.\n\n"
        "How a finding is separated from artifacts and routine histology. Artifact tiles (out of focus, pen, fold, glass) are removed before anything is ranked. Routine histology is handled by conditioning disorder on the tissue state and by the cohort-wide contact statistics. "
        "Each finding reports how many slides contain it and the entropy of that distribution, so a finding confined to one slide is labelled as such and a batch or scanner effect cannot be ruled out; the evaluation protocol compares recurrence with a null model in which the site or scanner is the explanatory variable.\n\n"
        "Evidence available now. On synthetic slides with planted answers (not kidney tissue), with the handcrafted colour-and-texture encoder, the novelty score ranks planted anomalous tiles above normal tiles with an AUROC of %s (slide-bootstrap interval %s), "
        "and %s of the highest-scoring tiles lie within one tile of a planted anomaly against a base rate of %s. These numbers say that the machinery works as designed; they do not say that the kidney contains anything novel.\n\n"
        "Kidney results.\n%s%s"
        % (n(A(P_, "any", "novelty")), ci(P_), pc(tq["precision_within_one_patch"]), pc(tq["base_rate_within_one_patch"], 1), kp_block(), ""))
nar1 = nar1.replace("Kidney results.", DINO_SHORT + "\n\nKidney results.", 1)
nar2 = ("Approach. Each slide is read (OpenSlide for SVS), a tissue mask is made, and the slide is cut into non-overlapping tiles of 224 pixels at 0.5 micrometres per pixel; the tile grid is a lattice. Tiles failing quality control are excluded. "
        "Every tile is described by DINOv2 ViT-S/14 (a pretrained image model; no training on KPMP data). The descriptions are centred, reduced to 16 dimensions and normalized to a sphere. "
        "Local disorder is MyUncle's D = 1 minus the length of the mean of the neighbours' unit vectors; the entropy of the neighbourhood is the von Mises-Fisher entropy in d dimensions, which equals MyUncle's own entropy exactly at d = 2 (checked against the unmodified MyUncle code by automated tests). "
        "A weighted probabilistic OR of three rank-based axes gives the novelty score; regions are clustered across slides and ranked by novelty, recurrence and reproducibility. MyUncle's simulation is reused to run model scenarios of tissue progression from an observed slide (scenarios, not prognosis).\n\n"
        "False-positive control. Artifact screening; state-wise disorder; the 1%% rarity scale; minimum region size; recurrence; reproducibility under split reference sets; a spatial-shuffle null of the slide's overall order; and a refusal to report a finding as more than a candidate. "
        "On synthetic slides, quality control flagged %s of the planted artifact tiles at a false-flag rate of %s on normal tiles; on the two slides with nothing planted (one with only artifacts) the pipeline produced %s candidate tiles (%s expected from the threshold by construction) and no finding.\n\n"
        "Performance and reliability. The reference set for rarity excludes the tile's own slide neighbourhood, so no tile is compared with itself. On the synthetic benchmark (%d slides, %s tiles; handcrafted encoder) the AUROC of the novelty score is %s (interval %s); the appearance axis alone reaches %s and disorder alone %s, "
        "so the combination is not better than its best axis there; it is kept because the axes see different things. These are development results: the thresholds and the rarity scale were adjusted after seeing synthetic data. "
        "Generalization to kidney tissue is untested.\n\n"
        "Computing. The encoder dominates: about 15-30 tiles per second per CPU core, so a slide of ten thousand tiles takes about ten minutes on a CPU and minutes on a GPU; a patch budget per slide (--max-patches) and a resumable cache bound the work. "
        "The benchmark of %d slides ran in %.0f s on one CPU."
        % (pc(q_["flagged_among_planted"]), pc(q_["false_flag_rate_normal"], 1), " and ".join(str(v["n_candidate_patches"]) for v in nl.values()), " and ".join(n(v["expected_by_construction"], 1) for v in nl.values()),
           P_["n_slides"], "{:,}".format(P_["n_patches"]), n(A(P_, "any", "novelty")), ci(P_), n(A(P_, "any", "p_emb")), n(A(P_, "any", "D")), P_["n_slides"], P_["runtime_seconds"]))
nar3 = ("What an expert receives. For every slide: a four-panel heat-map figure (candidate regions, novelty, local disorder D, order information), a binary candidate mask, the novelty lattice, region boxes in the slide's own pixel coordinates (so they can be opened in any viewer), "
        "and a table with one row per tile (position, QC result, tissue state, the three axes, D and the entropy). For every finding: its type, size in square millimetres, the slides that contain it, its recurrence and reproducibility, and an explanation made only of measured numbers "
        "(for example: local disorder against the cohort median, hematoxylin and eosin density percentiles, the most frequent prototype state and its share of the cohort). No pathological term is assigned, because the method has no labels; every explanation ends with the request for expert review, "
        "and a finding that occurs in one slide only says so.\n\n"
        "Connection to image features and patient stratification. Findings are defined on image features, and the per-slide summary (order against a spatial-shuffle null, disorder, damaged fraction, the model's rho-hat and whether it lies in MyUncle's bistable window) can be joined to patient-level data by the organizers; "
        "this submission makes no claim about patient groups.\n\n"
        "Domain experts. No pathologist or nephrologist has been involved so far, and no output has been expert-reviewed. This is the main weakness of the submission under this criterion. The protocol asks for blinded review of the highest-ranked findings, "
        "with each assigned to known histology, artifact, or neither, and arranging a collaborator with kidney pathology expertise is a priority before submission (not yet arranged).%s" % ("" if KP else "\n\n" + PENDING))
methods = ("Data selection: KPMP open (publicly released) whole-slide images only; controlled-access data are not used; the images actually used are listed with their file names, sizes and the way they were used in images_used.csv (written by every run). "
           "Preprocessing: tissue mask on a thumbnail; 224-pixel tiles at 0.5 micrometres per pixel (the grid is coarsened automatically when --max-patches is set); quality control (blur relative to the slide's median, pen ink, fold or debris, glass). "
           "Model design: DINOv2 ViT-S/14 (timm weights vit_small_patch14_dinov2.lvd142m) for tile embeddings; cohort PCA to 16 dimensions and k-means with 12 prototype states fitted at run time; the vendored, unmodified MyUncle core (third_party/myuncle/MyUncle_core.py, SHA-256 recorded) for the reference order-entropy definitions, the bistability analysis and the kinetic Monte Carlo of the progression scenarios. "
           "Training: none; everything is unsupervised and fitted on the input cohort at run time, so nothing is stored between runs and no label or clinical variable is used. "
           "Ranking: three cohort-rank axes (appearance rarity at the 1% scale, state-wise disorder, contact rarity), weighted probabilistic OR, regions of at least three tiles, clustering across slides, ranking by novelty, recurrence and reproducibility. "
           "Model files: the DINOv2 weights (embeddings; supports all findings), the MyUncle core (reference definitions and scenarios; supports the scenario analysis and the order/entropy checks), cohort PCA and k-means (cohort space and states; supports all findings).")
validation = ("Performance on unseen images: the pipeline has no training step, so every slide is unseen; on the synthetic benchmark the results are those above. On KPMP slides the study stage is not yet done and HeroX will run the code on unreleased slides. "
              "Uncertainty: slide-bootstrap interval of the AUROC (%s); reproducibility of each finding under two disjoint reference halves; a spatial-shuffle null of each slide's order. "
              "Failure modes seen: DINOv2 at the default settings did not separate the planted structure on the synthetic benchmark (see the Approach narrative); a region of random texture is not disordered at the tile scale; a recurring novel state is invisible to nearest-neighbour distance at small k (fixed by judging rarity at 1%% of the cohort); artifacts that overlap a tile only partly escape quality control; "
              "disorder is noisy with four neighbours and responds to compartment borders (hence the state-wise rank). Without quality control the share of artifacts among the top scores was %s against %s with it on the synthetic slides. "
              "Improvements: a pathology-pretrained encoder, stain normalization, larger neighbourhoods, a calibrated null, expert feedback. Computing resources: the benchmark ran on one CPU in %.0f s; a GPU is advised for the full release."
              % (ci(P_), pc(ab["precision_artifact_in_top_q"], 1), pc(tq["precision_artifact"], 1), P_["runtime_seconds"]))
nar2 = nar2.replace("Computing. The encoder dominates", DINO_SHORT + "\n\nComputing. The encoder dominates", 1)
team = ("Leon Sandler, independent researcher (computational), sole author of the code and of MyUncle, the framework this work builds on. No pathologist, nephrologist or other kidney expert is part of the team at the time of writing; "
        "a collaboration for expert review of the findings has not yet been arranged and is a priority. All members are listed with name, affiliation and contact: Leon Sandler, Independent Researcher, Northbrook, Illinois, USA, sandler.leon@gmail.com. "
        "A generative AI assistant (Claude, Anthropic) helped write and test the code and draft the documents; the author reviewed them and takes responsibility.")
visual = [("fig_validation_example_%s.png" % P_["encoder"], "Synthetic slide (not kidney): planted anomalies (green), planted artifacts (red), the appearance and disorder axes and the novelty score on the tile lattice. Supports the performance narrative."),
          ("fig_pipeline.png", "Pipeline from slides to ranked, mapped and explained candidate states. Supports the technical narrative."),
          ("fig_bridge.png", "Order-information bridge in d dimensions and its agreement at d = 2 with the unmodified MyUncle core. Supports the technical narrative.")]
cited = ["Oquab2023", "Goode2013", "Lake2023", "Banerjee2005", "Mardia2000", "Pedregosa2011", "Lochner2021", "Baron2017", "MyUncle2026", "Ruifrok2001"]
citations_text = "\n".join("[%d] %s" % (i + 1, LIB[k]) for i, k in enumerate(cited))
checklist = [
    "Register on HeroX (https://www.herox.com/KidneyAI) and obtain the personal download commands for the open KPMP images (Release 1: 6,256 SVS files, about 2.4 TB; start with a subset that fits your disk).",
    "Run the container on the downloaded slides, then rebuild this pack with the run: python docs/build/build_documents.py --kpmp-run <output directory>.",
    "Confirm eligibility (age 18 or older; U.S. citizen or permanent resident to receive a cash prize if you enter as an individual or team captain; not a federal employee acting in scope; no federal grant or contract funds used).",
    "Copy the exact rules statement from the portal onto the cover page (the portal requires it; submissions without it are not evaluated). This pack does not make that agreement for you.",
    "Push the repository, record the commit ID dated on or before the deadline (29 April 2027), and test the Docker image from a clean clone.",
    "Look for a renal pathologist or nephrologist willing to review the top findings; criterion 3 asks whether domain experts are involved.",
    "Check the IP terms: HeroX requires an irrevocable, nonexclusive licence to NIH; the repository is public under MIT.",
    "Do not use HHS, NIH or NIDDK logos or claim federal endorsement.",
]

SKIP_HEROX = ARGS.skip_herox
FIELDS = [("Public abstract (plain language; NIDDK may publish it)", abstract), ("Finding 1: Discovery and Novelty (50 points)", nar1), ("Finding 2: Technical Approach and Performance (30 points)", nar2),
          ("Finding 3: Scientific Interpretability and Expert Usability (20 points)", nar3), ("Methods (about one page)", methods), ("Validation", validation), ("Team description", team), ("Citations", citations_text)]

if SKIP_HEROX:
    print('HeroX pack skipped (--skip-herox)')
else:
    d = _docx.Document()
    st = d.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    for s in d.sections:
        s.left_margin = s.right_margin = Inches(1.0)
        s.top_margin = s.bottom_margin = Inches(0.9)
    h = d.add_paragraph()
    r = h.add_run("MyUncleVision: HeroX Kidney AI Discovery Challenge submission pack")
    r.bold = True
    r.font.size = Pt(15)
    d.add_paragraph("Leon Sandler, Independent Researcher, %s. Repository: %s. Prepared %s." % (EMAIL, GITHUB, datetime.date.today().strftime("%d %B %Y").lstrip("0")))
    if not KP:
        p = d.add_paragraph()
        rr = p.add_run("STATUS: DRAFT. NOT READY TO SUBMIT. " + PENDING)
        rr.bold = True
        rr.font.color.rgb = RGBColor(0xC0, 0x1D, 0x1D)
    d.add_paragraph().add_run("Before submitting").bold = True
    for c in checklist:
        d.add_paragraph(c, style="List Number")
    d.add_paragraph().add_run("Cover page statement (required by the portal)").bold = True
    d.add_paragraph("[To be pasted by the Team Captain from the portal's rules statement. Not completed by the assistant.]")
    d.add_paragraph().add_run("Eligibility information and confirmations").bold = True
    d.add_paragraph("[To be completed and confirmed by the Team Captain: name, age, U.S. citizenship or permanent residency, federal-employment status, funding source, registration type (individual / team / entity).]")
    md = ["# MyUncleVision: HeroX submission fields", "", "Character counts are shown for every field (the portal allows about 3,000 characters per page, roughly 8.5 pages in all).", ""]
    if not KP:
        md += ["> **DRAFT, NOT READY TO SUBMIT.** " + PENDING, ""]
    md += ["## Before submitting", ""] + ["%d. %s" % (i + 1, c) for i, c in enumerate(checklist)] + [""]
    total = 0
    for title, text in FIELDS[:-1]:
        total += len(text)
        d.add_paragraph().add_run("%s [%d characters]" % (title, len(text))).bold = True
        for para in text.split("\n"):
            if para.strip():
                d.add_paragraph(para)
        md += ["## %s (%d characters)" % (title, len(text)), "", text, ""]
    d.add_paragraph().add_run("Visual evidence (optional; captioned figures tied to findings)").bold = True
    md += ["## Visual evidence", ""]
    import os as _os
    for f, cap in visual:
        path = _os.path.join(FIGS, f)
        if _os.path.exists(path):
            d.add_picture(path, width=Inches(6.0))
        d.add_paragraph(cap)
        md += ["- `docs/figures/%s`: %s" % (f, cap)]
    d.add_paragraph().add_run("CSV of KPMP images used").bold = True
    d.add_paragraph("`images_used.csv`, written by every run into the output directory: one row per input image (file name, relative path, size, optional SHA-256, resolution, tile counts, status) and the way it was used; it also states that the input is KPMP open data supplied by the participant.")
    d.add_paragraph().add_run("Citations [%d characters]" % len(citations_text)).bold = True
    for line in citations_text.split("\n"):
        d.add_paragraph(line)
    d.add_paragraph().add_run("Optional appendix").bold = True
    d.add_paragraph("Limitations: the platform was validated on synthetic slides only; the vision encoder is not trained on tissue; novelty is not pathology; batch and scanner effects are the main alternative explanation for any recurring state; the scenario analysis is model-based. "
                    "The code and every number in this pack are generated from the repository; the validation results are in results/validation_*.json.")
    md += ["", "## CSV of KPMP images used", "", "`images_used.csv`, written by every run into the output directory.", "", "## Citations (%d characters)" % len(citations_text), "", citations_text, "",
           "Total characters in the narrative fields: %d (about %.1f pages at 3,000 characters per page)." % (total, total / 3000.0)]
    d.core_properties.title = "MyUncleVision HeroX submission pack"
    d.core_properties.author = "Leon Sandler"
    d.save(_os.path.join(OUT, "MyUncleVision_HeroX_Submission.docx"))
    open(_os.path.join(OUT, "HeroX_fields.md"), "w", encoding="utf-8").write("\n".join(md))
    print("HeroX pack built: %d narrative characters (%.1f pages)" % (total, total / 3000.0))
