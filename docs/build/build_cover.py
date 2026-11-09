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
para("I submit the manuscript “%s” for consideration as a research article in %s." % (TITLE, JOURNAL))
para("Unsupervised discovery in whole-slide images needs to keep artifacts and routine histology out of its findings and to hand experts something they can check. The manuscript presents MyUncleVision, an open pipeline that places patch embeddings on a tissue lattice and "
     "measures order, entropy, state transitions and a three-axis novelty score, with definitions that generalize those of the maintenance-lattice framework MyUncle from two to d dimensions and reproduce them exactly at d = 2 (tested against the unmodified core). "
     "It ranks regions across slides, maps them, explains them with measured numbers, and runs from one command in a container.")
para("The study is explicit about what it is. It is validated on synthetic slides with planted anomalies and artifacts, which are not tissue; the novelty score reaches an AUROC of %s on them (slide-bootstrap interval %s) but the combination is not better than its best single axis, "
     "and the thresholds were adjusted after seeing synthetic results, so these are development results. The platform has not been run on kidney or any other real tissue beyond a file-format test, no pathologist has reviewed any output, and the manuscript specifies, "
     "without reporting, the evaluation protocol for the Kidney Precision Medicine Project images. I offer it as a methods and software paper whose real-tissue evaluation is the next step, and I would be glad to know if the journal prefers that evaluation to be part of the paper."
     % (n(A(PRIMARY, "any", "novelty")), ci(PRIMARY)))
if DINO_SHORT:
    para(DINO_SHORT)
para("The code, tests and results are open at %s. The manuscript is original, has not been published elsewhere and is not under consideration at any other journal. I am the sole author. "
     "A generative AI tool (Claude, Anthropic) assisted with writing and testing the code, running the analyses and drafting the text and figures; I reviewed and edited all of it and take full responsibility, as declared in the manuscript. I have no competing interests to declare and the work received no funding." % GITHUB)
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
