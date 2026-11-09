# -*- coding: utf-8 -*-
"""Cover letter and highlights for the target Elsevier journal (--journal); executed by build_documents.py."""
import docx as _d
from docx.shared import Pt as _Pt, Inches as _In

c = _d.Document()
c.styles["Normal"].font.name = "Calibri"
c.styles["Normal"].font.size = _Pt(11)
c.styles["Normal"].paragraph_format.space_after = _Pt(8)
for s_ in c.sections:
    s_.left_margin = s_.right_margin = _In(1.0)


def para(t, bold=False):
    p = c.add_paragraph()
    r = p.add_run(t)
    r.bold = bold
    return p


for t in ("Leon Sandler", "Independent Researcher", "Northbrook, Illinois, USA", EMAIL, "ORCID 0009-0007-4584-808X", "", datetime.date.today().strftime("%d %B %Y").lstrip("0"), "The Editor-in-Chief", JOURNAL, "Elsevier", "", "Dear Editor,"):
    para(t, t == "Leon Sandler")
para("I submit the manuscript “%s” for consideration as a research report in %s." % (TITLE, JOURNAL))
para("Label-free anomaly discovery in large image collections needs to keep artifacts and routine structure out of its findings and to hand experts places to look at; the application here is whole-slide histopathology. The manuscript presents MyUncleVision, an open pipeline that places patch embeddings on a tissue lattice, judges appearance rarity at several prevalence scales, "
     "smooths the score over the lattice, and ranks, maps and explains regions across slides. Its order and entropy quantities generalize those of the maintenance-lattice framework MyUncle from two to d dimensions and equal them exactly at d = 2 (tested against the unmodified core). It runs from one command in a Docker container, which I built and ran end to end.")
para("Fit with the journal: the contribution is a pattern recognition method and its evaluation: unsupervised (label-free) anomaly scoring on patch embeddings, spatial smoothing of the score on a lattice, and a controlled comparison with Isolation Forest, Local Outlier Factor, nearest-neighbor and Mahalanobis scorers on a development cohort and a held-out cohort whose settings were frozen beforehand. The application area is histopathology, but nothing in the method or code is specific to it: any image collection can be processed, with the pipeline's own or a user-supplied embedding model.")
para("The study is explicit about what it shows and does not show. It is evaluated only on synthetic slides with planted anomalies and artifacts, written by me: a development cohort used to choose the settings, and a held-out cohort of 16 slides with three unseen anomaly types, about 1%% prevalence and stain shifts, evaluated once with frozen settings. "
     "The results are mixed and reported as such. Lattice smoothing raised the spatial coherence of every scorer; the frozen score was the best of all methods on the held-out cohort with DINOv2 (average precision %s) but poor with handcrafted features (%s); "
     "a Mahalanobis baseline smoothed in the same way scored higher on the development cohort; and the disorder and contact axes of the lattice helped with handcrafted features and hurt with DINOv2, so they are optional diagnostics. The platform has not been applied to real tissue and no pathologist has reviewed any output; validation on real histopathology is stated as future work. "
     "Section 5 sets out, as future work, a prespecified plan for validation on real histopathology (public Kidney Precision Medicine Project slides, blinded expert review by at least two pathologists, frozen settings and the same baselines); it has not been started. I submit the paper as a methods and software contribution whose evidence is synthetic, and I would welcome the editor's view on whether that scope is suitable for the journal."
     % (n(met(BD, FROZEN, "average_precision")), n(met(BH, FROZEN, "average_precision"))))
para("The software was developed for the NIDDK/HeroX Kidney AI Discovery Challenge; no entry has been submitted at the time of writing, and challenge participation would not constitute independent validation. The code, tests, benchmarks and results are open at %s and archived at https://doi.org/%s (software) and https://doi.org/%s (manuscript preprint)." % (GITHUB, ZEN_CODE, ZEN_PAPER))
para("A generative AI tool (Claude, Anthropic) assisted with writing and testing the code, running the analyses and drafting the text and figures; I reviewed and edited all of it and take full responsibility, as disclosed in the manuscript. "
     "The manuscript is original and is not under consideration at any other journal. I am the sole author. I have no competing interests to declare and the work received no funding.")
for t in ("Thank you for considering this submission.", "Sincerely,", "Leon Sandler"):
    para(t, t == "Leon Sandler")
c.core_properties.title = "Cover letter: " + TITLE
c.core_properties.author = "Leon Sandler"
c.save(os.path.join(OUT, "Cover_Letter_%s.docx" % SLUG))
print("cover letter built")

hd = _d.Document()
hd.styles["Normal"].font.name = "Calibri"
hd.styles["Normal"].font.size = _Pt(11)
hd.add_paragraph().add_run("Highlights").bold = True
hd.add_paragraph(TITLE)
for t in HIGHLIGHTS:
    hd.add_paragraph(t, style="List Bullet")
hd.core_properties.title = "Highlights: " + TITLE
hd.core_properties.author = "Leon Sandler"
hd.save(os.path.join(OUT, "Highlights_%s.docx" % SLUG))
print("highlights built (%s): %s" % (SLUG, ", ".join(str(len(t)) for t in HIGHLIGHTS)))
