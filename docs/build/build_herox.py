# -*- coding: utf-8 -*-
"""HeroX submission pack (executed by build_documents.py, which supplies the benchmark results, n, pc, met, FROZEN, MAHA_S, KP, LIB ...).

Writes docs/MyUncleVision_HeroX_Submission.docx and docs/HeroX_fields.md.  The portal limits narrative fields to about 3,000 characters per page, roughly 8.5 pages in all; the character count of every field is printed in the pack.
Without a real KPMP run (--kpmp-run) the pack says so on its first page and in every narrative that depends on kidney results.
"""
import docx as _docx
from docx.shared import Pt, Inches, RGBColor
import os as _os

PENDING = ("[KPMP RUN PENDING] No KPMP whole-slide image has been analysed with this platform yet. This paragraph is replaced automatically by the results of the run when "
           "`python docs/build/build_documents.py --kpmp-run <output directory of the KPMP run>` is executed. Do not submit the pack with this notice in it.")


def kp_block(top=5):
    if not KP:
        return PENDING
    man = KP["manifest"]
    lines = ["Run on %d KPMP slides (%d analysed; encoder %s; MyUncle core unmodified: %s). The run produced %d candidate states." % (man["n_slides_listed"], man["n_slides_analysed"], man["encoder"], man["myuncle_core_unmodified"], man["n_findings"])]
    for x in KP["findings"]["findings"][:top]:
        lines.append("%s (%s): %d patches, %.2f mm2, %d slides, novelty %.2f, reproducibility %.0f%%. %s" % (x["id"], x["type"], x["n_patches"], x["area_mm2"], x["n_slides"], x["novelty"], 100 * x["reproducibility"], x["explanation"]))
    return "\n".join(lines)


EVID = ("Evidence available now (synthetic slides, not kidney tissue). Settings were chosen on a development cohort and frozen; a held-out cohort of %d slides with three unseen anomaly types, about %s prevalence and stain shifts was evaluated once. "
        "With the DINOv2 encoder the frozen score was the best of all methods tried (average precision %s, AUROC %s, region recall %s); with handcrafted features it was poor (average precision %s, region recall %s). "
        "On the development cohort a plain Mahalanobis baseline with the same smoothing scored higher than the frozen score (average precision %s against %s with DINOv2). "
        "The disorder and contact axes of the lattice helped with handcrafted features and hurt with DINOv2, so they are optional diagnostics and not part of the default score. These numbers say that the machinery can work as designed and that its choices are not yet robust; they do not say that the kidney contains anything novel."
        % (BD["n_slides"], pc(BD["anomalous_patch_share"], 1), n(met(BD, FROZEN, "average_precision")), n(met(BD, FROZEN, "auroc")), n(met(BD, FROZEN, "region_recall")), n(met(BH, FROZEN, "average_precision")), n(met(BH, FROZEN, "region_recall")),
           n(met(DD, MAHA_S, "average_precision")), n(met(DD, FROZEN, "average_precision"))))

abstract = ("Kidney biopsies are photographed at very high resolution, but most of each image is never looked at closely, and computer programs are usually trained to find only what people have already named. "
            "MyUncleVision looks for the unnamed. It cuts each biopsy image into small tiles, describes every tile with a pretrained image model, scores how rare each tile's appearance is, and smooths that score over the grid of tiles so that unusual regions, not isolated tiles, stand out. "
            "Regions are grouped across biopsies, mapped on the slide and explained in plain numbers, so that a pathologist can check them; artifacts such as blur, pen marks and folds are screened out first, and a mathematical framework (MyUncle) for orderly and disorderly lattices supplies measures of local disorder. "
            "The software runs from one command in a container and lists every image it used. It has been tested only on synthetic slides with planted answers, where it works well with one image model and poorly with another; "
            "it has not yet been run on kidney slides, and its candidates are for expert review, not diagnoses.")
nar1 = ("What counts as a discovery here. A finding is a candidate organizational state of tissue: a group of regions, found in one or more slides, whose appearance is rare in the cohort and that survives four filters: artifact screening, a minimum region size, recurrence across slides (a state that appears in many biopsies is less likely to be one bad slide) and reproducibility when the reference set is split into two halves. "
        "Rarity is judged at several prevalence scales (0.5% to 10% of the cohort) because the frequency of an interesting state is not known in advance, and the score is smoothed over the tile grid so that compact regions are favoured.\n\n"
        "How a finding is separated from artifacts and routine histology. Artifact tiles (out of focus, pen, fold, glass) are removed before anything is ranked. Each finding reports how many slides contain it and the entropy of that distribution, so a finding confined to one slide is labelled as such and a batch or scanner effect cannot be ruled out; "
        "the evaluation protocol compares recurrence with a null model in which the site or scanner is the explanatory variable. Two further measures from the lattice framework, local disorder for the tissue state and the rarity of contacts between tissue states, are reported for every tile as diagnostics: on the synthetic benchmark they helped with one image model and hurt with the other, so they do not drive the ranking.\n\n"
        + EVID + "\n\nKidney results.\n" + kp_block())
nar2 = ("Approach. Each slide is read (OpenSlide for SVS), a tissue mask is made, and the slide is cut into non-overlapping tiles of 224 pixels at 0.5 micrometres per pixel; the tile grid is a lattice. Tiles failing quality control are excluded. "
        "Every tile is described by DINOv2 ViT-S/14 (a pretrained image model; no training on KPMP data), reduced to 16 dimensions and normalized to a sphere. Appearance rarity is the mean distance to the nearest tiles of a reference sample of the cohort, "
        "at five scales, taking the highest rank; the score is then averaged over each tile's 3 x 3 neighbourhood. Local disorder is MyUncle's D = 1 minus the length of the mean of the neighbours' unit vectors; the von Mises-Fisher entropy in d dimensions equals MyUncle's own entropy exactly at d = 2 (checked against the unmodified MyUncle code by automated tests). "
        "Regions are clustered across slides and ranked by novelty, recurrence and reproducibility.\n\n"
        "Comparison and false-positive control. On the same embeddings the score was compared with Isolation Forest, Local Outlier Factor, nearest-neighbour distance and Mahalanobis distance, each raw and smoothed in the same way. Smoothing raised the spatial coherence of every scorer. On the development cohort the frozen score beat the nonparametric baselines but not smoothed Mahalanobis (average precision %s and %s with DINOv2). "
        "On the held-out cohort (settings frozen beforehand, evaluated once) the frozen score was the best method with DINOv2 (average precision %s, %s false regions per slide) and a poor one with handcrafted features (%s); with handcrafted features it was also unstable to a coarser patch size. Seeds, embedding size and number of states changed the flagged set little.\n\n"
        "Computing. The encoder dominates: about 15 to 30 tiles per second per CPU core, so a slide of ten thousand tiles takes about ten minutes on a CPU and minutes on a GPU; a patch budget per slide (--max-patches) and a resumable cache bound the work. "
        "The Docker image was built and run end to end on three slides with its default command, offline, in about two minutes."
        % (n(met(DD, MAHA_S, "average_precision")), n(met(DD, FROZEN, "average_precision")), n(met(BD, FROZEN, "average_precision")), n(met(BD, FROZEN, "false_regions_per_slide")), n(met(BH, FROZEN, "average_precision"))))
nar3 = ("What an expert receives. For every slide: a four-panel heat-map figure (candidate regions, novelty, local disorder D, order information), a binary candidate mask, the novelty lattice, region boxes in the slide's own pixel coordinates (so they can be opened in any viewer), "
        "and a table with one row per tile (position, QC result, tissue state, the axes, D and the entropy). For every finding: its type, size in square millimetres, the slides that contain it, its recurrence and reproducibility, and an explanation made only of measured numbers "
        "(local disorder against the cohort median, hematoxylin and eosin density percentiles, the most frequent prototype state and its share of the cohort). No pathological term is assigned, because the method has no labels; every explanation ends with the request for expert review, "
        "and a finding that occurs in one slide only says so.\n\n"
        "Connection to image features and patient stratification. Findings are defined on image features, and the per-slide summary (order against a spatial-shuffle null, disorder, candidate counts) can be joined to patient-level data by the organizers; this submission makes no claim about patient groups.\n\n"
        "Domain experts. No pathologist or nephrologist has been involved so far, and no output has been expert-reviewed. This is the main weakness of the submission under this criterion. The protocol asks for blinded review of the highest-ranked findings by at least two experts, "
        "with each assigned to known histology, artifact, or neither, and with inter-rater agreement; arranging a collaborator with kidney pathology expertise is a priority before submission (not yet arranged).%s" % ("" if KP else "\n\n" + PENDING))
methods = ("Data selection: KPMP open (publicly released) whole-slide images only; controlled-access data are not used; the images actually used are listed with their file names, sizes and the way they were used in images_used.csv (written by every run). "
           "Preprocessing: tissue mask on a thumbnail; 224-pixel tiles at 0.5 micrometres per pixel (the grid is coarsened automatically when --max-patches is set); quality control (blur relative to the slide's median, pen ink, fold or debris, glass). "
           "Model design: DINOv2 ViT-S/14 (timm weights vit_small_patch14_dinov2.lvd142m) for tile embeddings; cohort PCA to 16 dimensions and k-means with 12 prototype states fitted at run time; the vendored, unmodified MyUncle core (third_party/myuncle/MyUncle_core.py, SHA-256 recorded) for the reference order-entropy definitions. "
           "Training: none; everything is unsupervised and fitted on the input cohort at run time, so nothing is stored between runs and no label or clinical variable is used. "
           "Ranking: multi-scale appearance rarity (nearest-neighbour distance to a cohort reference sample), 3 x 3 lattice smoothing, regions of at least three tiles, clustering across slides, ranking by novelty, recurrence and reproducibility. "
           "Model files: the DINOv2 weights (embeddings; supports all findings), cohort PCA, k-means and the reference sample (cohort space, states, rarity; support all findings), the MyUncle core (reference definitions; supports the order and entropy diagnostics).")
validation = ("Performance on unseen images: the pipeline has no training step, so every slide is unseen; settings were frozen on a development cohort and evaluated once on a held-out synthetic cohort (results above and in the repository). HeroX will run the code on unreleased KPMP slides. "
              "Uncertainty: slide-bootstrap intervals of the AUROC, reproducibility of each finding under two disjoint reference halves, robustness to seeds, embedding size, states, weights, smoothing and patch size. "
              "Failure modes seen: with the pretrained model the original three-axis score at a single rarity scale scored below chance because the planted states were more common than the scale and the encoder embedded them tightly (fixed by multi-scale rarity, found on the development cohort); "
              "with handcrafted features the frozen score diluted small anomalies on the held-out cohort and collapsed at a coarser patch size; partly blurred tiles escape quality control; a single-Gaussian baseline was competitive. "
              "Improvements: a pathology-pretrained encoder, stain normalization, smoothing of the raw score, a calibrated null, expert feedback. Computing resources: the benchmarks ran on one CPU; a GPU is advised for the full release.")
team = ("Leon Sandler, independent researcher (computational), sole author of the code and of MyUncle, the framework this work builds on. No pathologist, nephrologist or other kidney expert is part of the team at the time of writing; "
        "a collaboration for expert review of the findings has not yet been arranged and is a priority. All members are listed with name, affiliation and contact: Leon Sandler, Independent Researcher, Northbrook, Illinois, USA, sandler.leon@gmail.com. "
        "A generative AI assistant (Claude, Anthropic) helped write and test the code and draft the documents; the author reviewed them and takes responsibility.")
visual = [("fig_heldout_example.png", "Held-out synthetic slide (not kidney): planted anomalies (green), planted artifact (red), unsmoothed appearance score, the frozen score and a smoothed Mahalanobis baseline. Supports the performance narrative."),
          ("fig_heldout.png", "AUROC and average precision of the main methods on the held-out synthetic cohort, both encoders. Supports the performance narrative."),
          ("fig_pipeline.png", "Pipeline from slides to ranked, mapped and explained candidate states. Supports the technical narrative.")]
cited = ["Oquab2023", "Goode2013", "Lake2023", "Banerjee2005", "Mardia2000", "Pedregosa2011", "Lochner2021", "Baron2017", "MyUncle2026", "Ruifrok2001"]
citations_text = "\n".join("[%d] %s" % (i + 1, LIB[k]) for i, k in enumerate(cited))
checklist = [
    "Register on HeroX (https://www.herox.com/KidneyAI) and obtain the personal download commands for the open KPMP images (Release 1: 6,256 SVS files, about 2.4 TB; start with a subset that fits your disk).",
    "Run the container on the downloaded slides (docs/KPMP_RUN_GUIDE.md), then rebuild this pack with the run: python docs/build/build_documents.py --kpmp-run <output directory>.",
    "Confirm eligibility (age 18 or older; U.S. citizen or permanent resident to receive a cash prize if you enter as an individual or team captain; not a federal employee acting in scope; no federal grant or contract funds used).",
    "Copy the exact rules statement from the portal onto the cover page (the portal requires it; submissions without it are not evaluated). This pack does not make that agreement for you.",
    "Push the repository, record the commit ID dated on or before the deadline (29 April 2027), and test the Docker image from a clean clone.",
    "Look for a renal pathologist or nephrologist willing to review the top findings blinded to the score; criterion 3 asks whether domain experts are involved.",
    "Check the IP terms: HeroX requires an irrevocable, nonexclusive licence to NIH; the repository is public under MIT.",
    "Do not use HHS, NIH or NIDDK logos or claim federal endorsement.",
]
FIELDS = [("Public abstract (plain language; NIDDK may publish it)", abstract), ("Finding 1: Discovery and Novelty (50 points)", nar1), ("Finding 2: Technical Approach and Performance (30 points)", nar2),
          ("Finding 3: Scientific Interpretability and Expert Usability (20 points)", nar3), ("Methods (about one page)", methods), ("Validation", validation), ("Team description", team), ("Citations", citations_text)]

if ARGS.skip_herox:
    print("HeroX pack skipped (--skip-herox)")
else:
    d = _docx.Document()
    st = d.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    for s in d.sections:
        s.left_margin = s.right_margin = Inches(1.0)
        s.top_margin = s.bottom_margin = Inches(0.9)
    r = d.add_paragraph().add_run("MyUncleVision: HeroX Kidney AI Discovery Challenge submission pack")
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
    d.add_paragraph("Limitations: validated on synthetic slides only (a development and a held-out cohort, written by the author of the method); the vision encoder is not trained on tissue; novelty is not pathology; batch and scanner effects are the main alternative explanation for any recurring state; "
                    "the frozen settings were not robust to every change (coarser patches, handcrafted features); the exploratory scenario analysis (--scenarios) is a modelling analogy and is off by default. Every number in this pack is generated from the repository (results/*.json).")
    md += ["", "## CSV of KPMP images used", "", "`images_used.csv`, written by every run into the output directory.", "", "## Citations (%d characters)" % len(citations_text), "", citations_text, "",
           "Total characters in the narrative fields: %d (about %.1f pages at 3,000 characters per page)." % (total, total / 3000.0)]
    d.core_properties.title = "MyUncleVision HeroX submission pack"
    d.core_properties.author = "Leon Sandler"
    d.save(_os.path.join(OUT, "MyUncleVision_HeroX_Submission.docx"))
    open(_os.path.join(OUT, "HeroX_fields.md"), "w", encoding="utf-8").write("\n".join(md))
    print("HeroX pack built: %d narrative characters (%.1f pages)" % (total, total / 3000.0))
