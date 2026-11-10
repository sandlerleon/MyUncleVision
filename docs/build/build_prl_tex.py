# -*- coding: utf-8 -*-
"""Pattern Recognition Letters version (LaTeX, elsarticle; prletter.sty is loaded when present): fills prl_template.tex.in with numbers from results/*.json.
    python docs/build/build_prl_tex.py      -> docs/prl/main.tex (+ figures), compiled with pdflatex when available
The guards of build_documents.py run on import, so every qualitative sentence is checked against the results."""
import os
import re
import shutil
import subprocess
import sys

sys.argv = ["build_prl_tex.py", "--skip-herox"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_documents as bd                                   # noqa: E402

n, pc, met = bd.n, bd.pc, bd.met
FROZEN, MAHA, MAHA_S, T3, ORIG = bd.FROZEN, bd.MAHA, bd.MAHA_S, bd.T3, bd.ORIG
BD, BH, DD, DH, BRD = bd.BD, bd.BH, bd.DD, bd.DH, bd.BRD
bm = bd._bm
OUT = os.path.join(HERE, "..", "prl")
os.makedirs(OUT, exist_ok=True)
for f in ("fig_bridge.png", "fig_heldout_example.png", "fig_bridge_maps.png"):
    shutil.copy(os.path.join(HERE, "..", "figures", f), os.path.join(OUT, f))

ABSTRACT = (
    "Averaged order parameters say how ordered a lattice of orientational states is, not where its organization is unusual. "
    "We extend the local order and entropy quantities of the MyUncle maintenance-lattice model from the plane to unit vectors in $d$ dimensions with the von Mises--Fisher distribution; they equal MyUncle's at $d=2$ (tested against the unmodified code). "
    "On this basis we build a label-free score that judges how rare the state of a site is at several prevalence scales and smooths it over the lattice, and compare it with isolation forest, local outlier factor, nearest-neighbor and Mahalanobis baselines, with settings frozen before the held-out data. "
    "On lattices simulated with MyUncle (damaged sites as ground truth), smoothing lowered the average precision of every baseline because damaged sites are not compact regions, adding the local disorder as a feature raised it, and the frozen score was within 0.02 of the best on the held-out sets but clearly beaten on the development set. "
    "On synthetic stained-tissue images, smoothing raised spatial coherence for every scorer, and the frozen score was the best method with DINOv2 embeddings and a poor one with handcrafted features. "
    "All evidence is simulated; code and results are open.")
assert len(ABSTRACT.split()) <= 200, len(ABSTRACT.split())

tex = lambda s: s.replace("&", r"\&").replace("%", r"\%")
pf = BRD[("held", "rotor")]["damaged_fraction_by_preset"]
sim_rows = [(FROZEN, "MyUncleVision (frozen, smoothed)"), ("Isolation Forest", "Isolation forest"), ("Isolation Forest, lattice-smoothed (rank)", "\\quad + smoothing"), ("Mahalanobis distance", "Mahalanobis distance"),
            (MAHA_S, "\\quad + smoothing"), ("kNN distance, k = 10", "Nearest-neighbor distance"), ("kNN distance, k = 10, lattice-smoothed (rank)", "\\quad + smoothing"), ("local disorder D alone", "Local disorder $D$ alone")]
tabsim = " \\\\\n".join(" & ".join([lab] + [n(bm("held", e, m, k)) for e in ("rotor", "rotor_D") for k in ("auroc", "average_precision")] + [n(bm("dev", e, m)) for e in ("rotor", "rotor_D")]) for m, lab in sim_rows) + " \\\\"
img_rows = [(FROZEN, "MyUncleVision (frozen, smoothed)"), ("appearance, multi-scale", "Appearance, multi-scale, unsmoothed"), (T3, "Three-axis OR, smoothed"), ("Isolation Forest, lattice-smoothed (rank)", "Isolation forest, smoothed"),
            ("Mahalanobis distance", "Mahalanobis distance"), (MAHA_S, "\\quad + smoothing"), ("kNN distance, k = 10", "Nearest-neighbor distance"), ("kNN distance, k = 10, lattice-smoothed (rank)", "\\quad + smoothing")]
keys = ("auroc", "average_precision", "spatial_coherence", "region_recall")
tabimg = " \\\\\n".join(" & ".join([lab] + [n(met(R, m, k)) for R in (BH, BD) for k in keys]) for m, lab in img_rows) + " \\\\"

V = dict(
    ABSTRACT=ABSTRACT, TABSIM=tabsim, TABIMG=tabimg,
    DEVPREV=pc(BRD[("dev", "rotor")]["prevalence_all"]),
    HELDPREV="%s, %s and %s" % (pc(pf["polymer_photodegradation"]), pc(pf["liquid_crystal"]), pc(pf["magnetic_storage_decay"])),
    DEVIMGPREV=pc(DH["anomalous_patch_share"], 1), HELDIMGPREV=pc(BD["anomalous_patch_share"], 1),
    S_AP=n(bm("held", "rotor", FROZEN)), S_AUROC=n(bm("held", "rotor", FROZEN, "auroc")), S_MAHA=n(bm("held", "rotor", "Mahalanobis distance")), S_IF=n(bm("held", "rotor", "Isolation Forest")),
    S_IFSM=n(bm("held", "rotor", "Isolation Forest, lattice-smoothed (rank)")), S_DEV_R=n(bm("dev", "rotor", FROZEN)), S_DEV_RD=n(bm("dev", "rotor_D", FROZEN)), S_HELD_RD=n(bm("held", "rotor_D", FROZEN)),
    S_DEV_IF=n(bm("dev", "rotor_D", "Isolation Forest")), S_DALONE=n(bm("held", "rotor_D", "local disorder D alone")),
    OLD_HC=n(met(DH, ORIG, "auroc")), OLD_DINO=n(met(DD, ORIG, "auroc")),
    COH0_HC=n(met(DH, "appearance, multi-scale", "spatial_coherence")), COH1_HC=n(met(DH, FROZEN, "spatial_coherence")),
    COH0_D=n(met(DD, "appearance, multi-scale", "spatial_coherence")), COH1_D=n(met(DD, FROZEN, "spatial_coherence")),
    DEV_MS_HC=n(met(DH, MAHA_S, "average_precision")), DEV_MS_D=n(met(DD, MAHA_S, "average_precision")), DEV_F_HC=n(met(DH, FROZEN, "average_precision")), DEV_F_D=n(met(DD, FROZEN, "average_precision")),
    H_AP_D=n(met(BD, FROZEN, "average_precision")), H_AUROC_D=n(met(BD, FROZEN, "auroc")), H_COH_D=n(met(BD, FROZEN, "spatial_coherence")), H_REC_D=n(met(BD, FROZEN, "region_recall")),
    H_MS_D=n(met(BD, MAHA_S, "average_precision")), H_U_D=n(met(BD, "appearance, multi-scale", "average_precision")),
    H_AP_HC=n(met(BH, FROZEN, "average_precision")), H_REC_HC=n(met(BH, FROZEN, "region_recall")), H_AUROC_HC=n(met(BH, FROZEN, "auroc")), H_COH_HC=n(met(BH, FROZEN, "spatial_coherence")),
    H_T3_HC=n(met(BH, T3, "average_precision")), H_T3_D=n(met(BD, T3, "average_precision")),
)
s = open(os.path.join(HERE, "prl_template.tex.in"), encoding="utf-8").read()
for k, v in V.items():
    s = s.replace("@%s@" % k, v.replace("%", r"\%") if k not in ("ABSTRACT", "TABSIM", "TABIMG") else v.replace("%", r"\%"))
left = re.findall(r"@[A-Z_0-9]+@", s)
assert not left, left
open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8").write(s)
words = len(re.sub(r"\\[a-zA-Z]+|[{}$&\\]", " ", s.split("\\begin{thebibliography}")[0]).split())
print("main.tex written; abstract %d words; ~%d words in the body" % (len(ABSTRACT.split()), words))
if shutil.which("pdflatex"):
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=OUT, capture_output=True, text=True)
    print("pdflatex exit", r.returncode)
    if r.returncode:
        print(r.stdout[-2500:])
