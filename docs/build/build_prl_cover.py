# -*- coding: utf-8 -*-
"""Cover letter, highlights and Editorial Manager field values for the Pattern Recognition Letters letter (numbers from results/og_test.json and results/two_channel.json).
    python docs/build/build_prl_cover.py"""
import json
import os

import docx
import numpy as np
from docx.shared import Pt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "prl")
RES = os.path.join(HERE, "..", "..", "results")
OT = json.load(open(os.path.join(RES, "og_test.json")))
OS = json.load(open(os.path.join(RES, "og_search.json")))
TC = json.load(open(os.path.join(RES, "two_channel.json")))
R = OT["results"]
F = lambda c: R[c]["_fitness"]
NONE, MEAN, SIMW = "none", "frozen (mean 3)", "similarity-weighted (3), untuned"
RAR, CA, CB, FUS = "rarity", "rarity, similarity-weighted (channel A)", "directional disorder (channel B)", "fusion select: A + directional disorder"
_arr = [k for k, nm in enumerate(TC["scenarios"]) if nm.startswith("arr-")]
gm = lambda c: float(np.mean([TC["ap"][c][k] for k in _arr]))
TF, TB = TC["fitness"], TC["bootstrap"]
assert TB[FUS]["diff_vs_A"]["ci"][0] > 0 and gm(RAR) < 0.08 and gm(CB) > 0.8 and TB[FUS]["diff_old_vs_A"]["ci"][0] > -0.01
TITLE = "Rarity and local order: two complementary label-free channels for anomaly detection on lattices of embeddings"
KEYWORDS = "anomaly detection; embedding lattices; spatial arrangement; orientational order; similarity-weighted aggregation; label-free fusion"
H = ["Rarity scores are blind to anomalies of arrangement; local order detects them",
     "Directional local disorder ignores domain walls and marks disorganized regions",
     "A label-free tail index chooses the channel per cohort, with no loss elsewhere",
     "Similarity-weighted aggregation keeps anomalies that a fixed window destroys",
     "Order gates and evolutionary rule search did not help; reported as negatives"]
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


for t, b in (("Leon Sandler", True), ("Independent Researcher", False), ("Northbrook, Illinois, USA", False), ("sandler.leon@gmail.com", False), ("ORCID 0009-0007-4584-808X", False), ("", False), ("11 November 2026", False), ("The Editor-in-Chief", False), ("Pattern Recognition Letters", False), ("", False)):
    P(t, b)
P("Re: PRLETTERS-D-26-02906, revised submission", True)
P("Dear Editor,")
P("Thank you for the editorial office's message. In response, I have uploaded the Authorship Confirmation form as a Cover Letter item and have reformatted the manuscript to the journal's requirements (elsarticle layout, a length inside the seven-page limit including figures, tables and references, an abstract of fewer than 200 words, five highlights of at most 85 characters, author-year references, high-resolution figure files).")
P("I have also changed the manuscript and its title. The first version, \"MyUncleVision: An Interpretable Lattice-Based Framework for Unsupervised Anomaly Discovery in Whole-Slide Histopathology Images\", was written around one application and was far over the page limit. The revised manuscript, \"%s\", makes one general point about any lattice of embeddings: a rarity score sees what is at a site but not how the sites are arranged, and a second label-free channel based on local order closes that gap." % TITLE)
P("The study is entirely computational. The main results are these. (1) Arrangement anomalies, regions in which the same embeddings sit in the wrong places, leave the distribution of the embeddings unchanged; on independent test scenarios the rarity score stayed at chance (average precision %.2f) while a directional form of the local disorder of the MyUncle lattice model reached %.2f. "
  "(2) A label-free tail index chooses the channel per cohort; on fifteen independent test scenarios this raised the lower-tail average precision from %.2f to %.2f, with a paired 95%% interval of the gain of %+.2f to %+.2f, and without loss on the scenarios that contain only feature anomalies. "
  "(3) For the rarity channel itself, aggregation weighted by the similarity of neighboring embeddings preserved the scattered and bar-shaped anomalies that a fixed 3x3 mean destroyed (lower-tail average precision %.2f against %.2f for the fixed mean and %.2f without smoothing on ten independent test scenarios, paired intervals above zero), and a contrast-survival model explains the loss."
  % (gm(RAR), gm(CB), TF[CA], TF[FUS], TB[FUS]["diff_vs_A"]["ci"][0], TB[FUS]["diff_vs_A"]["ci"][1], F(SIMW), F(MEAN), F(NONE)))
P("The manuscript states its limits plainly. The simple distance of a site to its neighbors detects rearranged regions as well as the directional disorder does; the directional form matters because it does not respond to walls between ordered domains, which is what makes the label-free choice of the channel work. A cohort that contains both kinds of anomaly still needs both maps, because the choice is made per cohort. "
  "Similarity-weighted filtering is an established kind of edge-preserving filter, and I make no claim to its novelty. Gates on local order and on the von Mises-Fisher information, a nucleation-and-growth rule and a search over %d rule configurations with GaaR (an evolutionary search framework developed by the author, which did not beat random search) did not help and are reported as negative results. "
  "A first fusion attempt failed on the test scenarios and is reported, so thirteen of the fifteen test scenarios had been used once before the final rule was designed. All evidence is simulated and the generators were written by me; experimental validation is listed as future work. I would welcome the editor's view on whether this scope suits the journal." % OS["n_unique"])
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
assert BS + "title{" + TITLE + "}" in s, "the title of the letter and of the cover letter differ"
a = s.split(BS + "begin{abstract}")[1].split(BS + "end{abstract}")[0].strip().replace("--", "-").replace("$", "").replace(BS + "%", "%").replace(BS + "times", "x")
open(os.path.join(OUT, "PRL_EM_fields.txt"), "w", encoding="utf-8").write(
    "PRLETTERS-D-26-02906 - edit submission (Editorial Manager) - values to enter\n\nTITLE (sentence case, no abbreviations):\n" + TITLE + "\n\nABSTRACT (%d words, limit 200):\n%s\n\nKEYWORDS:\n%s\n\nHIGHLIGHTS (3-5, <=85 characters each):\n%s\n" % (len(a.split()), a, KEYWORDS, "\n".join("- " + h for h in H)))
print("cover letter, highlights and EM fields written; highlights", [len(h) for h in H])
