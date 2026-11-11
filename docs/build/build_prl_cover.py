# -*- coding: utf-8 -*-
"""Cover letter, highlights and Editorial Manager field values for the Pattern Recognition Letters letter (numbers from results/og_test.json).
    python docs/build/build_prl_cover.py"""
import json
import os

import docx
from docx.shared import Pt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "prl")
RES = os.path.join(HERE, "..", "..", "results")
OT = json.load(open(os.path.join(RES, "og_test.json")))
OS = json.load(open(os.path.join(RES, "og_search.json")))
R = OT["results"]
F = lambda c: R[c]["_fitness"]
BT = OT["bootstrap"]
NONE, MEAN, SIMW = "none", "frozen (mean 3)", "similarity-weighted (3), untuned"
TITLE = "When does spatial smoothing help anomaly scoring on lattices of embeddings? Fixed windows, similarity-weighted aggregation and order gates"
H = ["A fixed smoothing window can destroy anomaly signals, depending on their shape",
     "Similarity-weighted aggregation beat fixed windows and no smoothing on fresh tests",
     "Order and von Mises-Fisher gates did not beat plain embedding similarity",
     "GaaR rule search was not better than random search at equal budget",
     "d-dimensional MyUncle entropy extension verified; it does not drive detection"]
assert all(len(h) <= 85 for h in H) and 3 <= len(H) <= 5, [len(h) for h in H]

d = docx.Document()
d.styles["Normal"].font.name = "Calibri"
d.styles["Normal"].font.size = Pt(11)
d.add_paragraph().add_run("Highlights").bold = True
d.add_paragraph(TITLE)
for h in H:
    d.add_paragraph(h, style="List Bullet")
d.save(os.path.join(OUT, "Highlights_PRL_v2.docx"))

c = docx.Document()
c.styles["Normal"].font.name = "Calibri"
c.styles["Normal"].font.size = Pt(11)


def P(t, b=False):
    p = c.add_paragraph()
    r = p.add_run(t)
    r.bold = b


for t, b in (("Leon Sandler", True), ("Independent Researcher", False), ("Northbrook, Illinois, USA", False), ("sandler.leon@gmail.com", False), ("ORCID 0009-0007-4584-808X", False), ("", False), ("10 November 2026", False), ("The Editor-in-Chief", False), ("Pattern Recognition Letters", False), ("", False)):
    P(t, b)
P("Re: PRLETTERS-D-26-02906, revised submission", True)
P("Dear Editor,")
P("Thank you for the editorial office's message. In response, I have uploaded the Authorship Confirmation form as a Cover Letter item and have reformatted the manuscript to the journal's requirements (elsarticle layout, a length inside the seven-page limit including figures, tables and references, an abstract of fewer than 200 words, five highlights of at most 85 characters, author-year references, high-resolution figure files).")
P("I have also changed the manuscript and its title. The first version, \"MyUncleVision: An Interpretable Lattice-Based Framework for Unsupervised Anomaly Discovery in Whole-Slide Histopathology Images\", was written around one application and was far over the page limit. The revised manuscript, \"%s\", asks a narrower question that applies to any lattice of embeddings: when does smoothing an anomaly score over a spatial window help?" % TITLE)
P("The study is entirely computational. On lattices simulated with MyUncle, on synthetic images and on a controlled generator that varies only the size and shape of the anomalies, a fixed 3x3 mean destroyed the signal for scattered and bar-shaped anomalies and helped only with DINOv2 embeddings. "
  "Aggregation weighted by the similarity of the embeddings of neighboring sites, with a label-free scale, preserved those anomalies: on ten independent test scenarios the lower-tail average precision rose from %.2f (fixed mean) and %.2f (no smoothing) to %.2f, and the paired 95%% intervals of the differences exclude zero. "
  "I state that similarity-weighted filtering is an established kind of edge-preserving filter; the contribution is the controlled failure analysis and the independent test, not the filter." % (F(MEAN), F(NONE), F(SIMW)))
P("Two further results are negative and are reported as such. Gates based on local order and on the von Mises-Fisher information of the neighborhood (the d-dimensional extension of the entropy quantities of the MyUncle maintenance-lattice framework, which I verified against the unmodified code) did not beat ordinary similarity, and the manuscript says that the entropy quantities characterize local order but do not drive the detection. "
  "A search over %d rule configurations with GaaR, an evolutionary search framework developed by the author, was not better than random search at equal budget; selecting rule parameters uses development labels, which the manuscript states. All evidence is simulated and was written by me, and experimental validation is listed as future work. I would welcome the editor's view on whether this scope suits the journal." % OS["n_unique"])
P("Earlier versions of this work are openly archived on Zenodo (https://doi.org/10.5281/zenodo.24066256, all versions) and the code, tests, search tables, seeds and results are open at https://github.com/sandlerleon/MyUncleVision (archived at https://doi.org/10.5281/zenodo.24066260). A ChemRxiv posting of a longer version is planned and will state that the paper is under consideration at Pattern Recognition Letters; I will tell the office if it is posted. The manuscript is not published and is not under consideration elsewhere, and it is not related to any conference paper.")
P("A generative AI tool (Claude, Anthropic) assisted with writing and testing the code, running the analyses and drafting the text and figures; I reviewed and edited all of it and take full responsibility, as disclosed in the manuscript. I am the sole author. I have no competing interests to declare and the work received no funding.")
P("Thank you for considering this revised submission.")
P("Sincerely,")
P("Leon Sandler", True)
c.core_properties.title = "Cover letter: " + TITLE
c.core_properties.author = "Leon Sandler"
c.save(os.path.join(OUT, "Cover_Letter_PRL_v2.docx"))

s = open(os.path.join(OUT, "main.tex"), encoding="utf-8").read()
BS = chr(92)
a = s.split(BS + "begin{abstract}")[1].split(BS + "end{abstract}")[0].strip().replace("--", "-").replace("$", "").replace(BS + "%", "%").replace(BS + "times", "x")
open(os.path.join(OUT, "PRL_EM_fields.txt"), "w", encoding="utf-8").write(
    "PRLETTERS-D-26-02906 - edit submission (Editorial Manager) - values to enter\n\nTITLE (sentence case, no abbreviations):\n" + TITLE + "\n\nABSTRACT (%d words, limit 200):\n%s\n\nKEYWORDS:\nanomaly detection; spatial aggregation; embedding lattices; von Mises-Fisher distribution; orientational order; configuration search\n\nHIGHLIGHTS (3-5, <=85 characters each):\n%s\n" % (len(a.split()), a, "\n".join("- " + h for h in H)))
print("cover letter, highlights and EM fields written; highlights", [len(h) for h in H])
