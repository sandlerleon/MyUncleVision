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

import json
RES = os.path.join(HERE, "..", "..", "results")
GS = json.load(open(os.path.join(RES, "gaar_search_comparison.json")))
GT = json.load(open(os.path.join(RES, "gaar_test.json")))
GL = json.load(open(os.path.join(RES, "gaar_labelfree.json")))
TF = lambda name: GT["test"][name]["fitness"]
DEF, NOS, OPT = "frozen default", "no smoothing", "exhaustive dev optimum"
RST, GRD, RND = "GaaR restart (seed 0, budget 40)", "GaaR greedy (seed 0, budget 40)", "GaaR random (seed 0, budget 40)"
assert TF(RST) > TF(DEF) and TF(RND) > TF(DEF) and TF(GRD) > TF(DEF) and TF(OPT) > TF(DEF), "text says every development-optimized configuration beat the default on the test scenarios"
assert TF(RST) > TF(OPT) > TF(RND), "text says the test ordering did not follow the development fitness"
assert GT["dev"][OPT]["fitness"] > GT["dev"][RST]["fitness"] > GT["dev"][RND]["fitness"], "development order"
assert GT["test"][RST]["aps"][1] < GT["test"][DEF]["aps"][1], "text says the DINOv2 test AP fell"
assert GT["test"][RST]["aps"][0] > GT["test"][DEF]["aps"][0], "text says the handcrafted test AP rose"
st = GS["strategies"]
assert st["greedy"]["hit_top1pct"] > 2 * st["random"]["hit_top1pct"] and st["restart"]["hit_top1pct"] > 2 * st["random"]["hit_top1pct"], "text says greedy/restart reach the top 1% more than twice as often as random"
assert st["random"]["hit_top5pct"] >= st["population"]["hit_top5pct"] > st["restart"]["hit_top5pct"] >= st["greedy"]["hit_top5pct"], "text says random matched or beat GaaR at the top 5% level"
mor = {r["scenario"]: r["moran"] for r in GL["rows"]}
eff = {r["scenario"]: r["effect"] for r in GL["rows"]}
bad = [k for k in eff if eff[k] < -0.2]
assert sorted(bad) == ["sim-dev-rotor", "sim-test-rotor"] and all(mor[k] < 0.02 for k in bad) and all(mor[k] > 0.1 for k in mor if k not in bad), "text says Moran's I separates exactly the two scenarios where smoothing destroyed the signal"
assert eff["images-test-handcrafted"] < -0.1 and mor["images-test-handcrafted"] > 0.05, "text says the rule would not help the handcrafted test images"
_row = lambda name, lab: [lab, n(GT["dev"][name]["fitness"])] + [n(x) for x in GT["test"][name]["aps"]] + [n(GT["test"][name]["fitness"])]
TABGAAR = " \\\\\n".join(" & ".join(_row(nm, lab)) for nm, lab in ((DEF, "Frozen default"), (NOS, "No smoothing"), (RST, "GaaR restart / population champion"), (GRD, "GaaR greedy / random-search champion"), (OPT, "Exhaustive development optimum"))) + " \\\\"
ABSTRACT = (
    "Averaged order parameters say how ordered a lattice of orientational states is, not where its organization is unusual. "
    "We extend the local order and entropy quantities of the MyUncle maintenance-lattice model to unit vectors in $d$ dimensions (von Mises--Fisher distribution); they equal MyUncle's at $d=2$. "
    "On this basis we build a label-free score that judges how rare the state of a site is at several prevalence scales and smooths it over the lattice, and compare it with four baselines, with settings frozen before the held-out data. "
    "On lattices simulated with MyUncle, smoothing lowered the average precision of every baseline because the damaged sites are not compact regions. "
    "On synthetic stained-tissue images, smoothing raised spatial coherence for every scorer, and the frozen score was the best method with DINOv2 embeddings and a poor one with handcrafted features. "
    "A supervised search over 360 configurations with GaaR, an evolutionary search framework, raised the lower-tail average precision on fresh test scenarios from %s to %s; random search also beat the default, so the gain belongs to the selected configuration, not to the search. "
    "All evidence is simulated; code and results are open." % (n(TF(DEF)), n(TF(RST))))
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
    ABSTRACT=ABSTRACT, TABSIM=tabsim, TABIMG=tabimg, TABGAAR=TABGAAR,
    G_RANK=str(int(round(GS["default_rank_fraction"] * GS["n_configs"])) + 1), G_NC=str(GS["n_configs"]), G_BUDGET=str(GS["budget"]), G_REP=str(GS["repeats"]), G_DEFDEV=n(GS["default_fitness"]), G_OPTDEV=n(GS["optimum"]),
    G_G1=pc(st["greedy"]["hit_top1pct"]), G_R1=pc(st["restart"]["hit_top1pct"]), G_P1=pc(st["population"]["hit_top1pct"]), G_N1=pc(st["random"]["hit_top1pct"]),
    G_G5=pc(st["greedy"]["hit_top5pct"]), G_R5=pc(st["restart"]["hit_top5pct"]), G_P5=pc(st["population"]["hit_top5pct"]), G_N5=pc(st["random"]["hit_top5pct"]),
    G_MB_R=n(st["restart"]["median_best"]), G_MB_N=n(st["random"]["median_best"]),
    G_DEF_F=n(TF(DEF)), G_RST_F=n(TF(RST)), G_GRD_F=n(TF(GRD)), G_OPT_F=n(TF(OPT)), G_NOS_F=n(TF(NOS)),
    G_DEF_HC=n(GT["test"][DEF]["aps"][0]), G_RST_HC=n(GT["test"][RST]["aps"][0]), G_DEF_D=n(GT["test"][DEF]["aps"][1]), G_RST_D=n(GT["test"][RST]["aps"][1]),
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
