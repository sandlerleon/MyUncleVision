# -*- coding: utf-8 -*-
"""Pattern Recognition Letters version (LaTeX, elsarticle; prletter.sty is loaded when present): fills prl_template.tex.in with numbers from results/*.json.
    python docs/build/build_prl_tex.py      -> docs/prl/main.tex (+ figures), compiled with pdflatex when available
Every qualitative sentence of the results section is guarded by an assert on the result files."""
import json
import os
import re
import shutil
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
OUT = os.path.join(HERE, "..", "prl")
os.makedirs(OUT, exist_ok=True)
for f in ("fig_bridge.png", "fig_bridge_maps.png"):
    shutil.copy(os.path.join(HERE, "..", "figures", f), os.path.join(OUT, f))
J = lambda name: json.load(open(os.path.join(RES, name)))
OT, OS, OG = J("og_test.json"), J("og_search.json"), J("og_grid_dev.json")
GT = J("gaar_test.json")
NG, CG = J("og_ngh.json"), J("og_closure.json")
SV, SV5 = J("og_survival.json"), J("og_survival_w5.json")
LF = J("og_lfsurv.json")
R = OT["results"]
BT = OT["bootstrap"]
n2 = lambda x: "%.2f" % x
pc = lambda x: "%d\\%%" % round(100 * x)
sg = lambda x: ("$-$%.2f" % -x) if x < -0.0049 else ("+%.2f" % x)
cif = lambda c: "[%s, %s]" % (sg(c[0]), sg(c[1]))
dif = lambda d, c: "%s %s" % (sg(d), cif(c))

NONE, MEAN, SIMW = "none", "frozen (mean 3)", "similarity-weighted (3), untuned"
ONE, OGC, OGV, OGD, GAR, GAR_R = "best onesided (dev)", "best og_cos (dev)", "best og_vmf (dev)", "best og_disorder (dev)", "GaaR restart champion", "GaaR random-search champion"
SC = dict(H="images-test-handcrafted", D="images-test-dinov2_vits14", R="sim-test-rotor", RD="sim-test-rotor_D", SC="morph-test-scattered", B3="morph-test-block3", B6="morph-test-block6", BAR="morph-test-bar8", IRR="morph-test-irregular12", BND="morph-test-boundary3")
ap = lambda c, k: R[c][SC[k]]["ap"]
F = lambda c: R[c]["_fitness"]
rec = lambda c, k: R[c][SC[k]]["region_recall"]
morph = [SC[k] for k in ("SC", "B3", "B6", "BAR", "IRR", "BND")]
fr = lambda c: float(np.mean([R[c][s]["false_regions_per_image"] for s in morph]))
diff = lambda c, ref: (F(c) - F(ref), BT[c]["diff_vs_" + ref.split()[0]]["ci"])

# ---- guards: qualitative claims made in the text
assert all(ap(MEAN, k) < ap(NONE, k) for k in ("SC", "BAR", "R")), "text: the fixed mean lowered AP for scattered, bar and rotor-lattice scenarios"
assert ap(MEAN, "D") > ap(NONE, "D"), "text: the mean helped the DINOv2 images"
assert rec(MEAN, "SC") < rec(NONE, "SC") and rec(MEAN, "BAR") < rec(NONE, "BAR")
assert diff(MEAN, NONE)[1][1] < 0 and F(MEAN) < F(NONE), "text: the fixed mean was below no smoothing in F"
assert diff(SIMW, MEAN)[1][0] > 0 and diff(SIMW, NONE)[1][0] > 0, "text: the untuned similarity-weighted rule beat the fixed mean and no smoothing with paired intervals above zero"
assert rec(SIMW, "SC") > rec(MEAN, "SC") and rec(SIMW, "BAR") > rec(MEAN, "BAR") and fr(SIMW) < fr(NONE)
assert ap(SIMW, "D") > ap(MEAN, "D") and ap(SIMW, "H") <= ap(NONE, "H") + 0.01 and ap(SIMW, "R") < ap(NONE, "R")
assert BT[ONE]["diff_vs_none"]["ci"][0] < 0 < BT[ONE]["diff_vs_none"]["ci"][1], "text: the one-sided rule matched no smoothing"
assert F(OGC) < F(SIMW) and F(OGV) < F(SIMW) and np.allclose([R[OGV][s]["ap"] for s in SC.values()], [R[OGD][s]["ap"] for s in SC.values()], atol=0.01), "text: gates did not beat similarity; vMF and disorder gates coincide"
_st = OS["strategies"]
assert all(_st["random"]["hit_top5pct"] > _st[k]["hit_top5pct"] and _st["random"]["hit_top1pct"] > _st[k]["hit_top1pct"] for k in ("greedy", "restart", "population")), "text: random search was better at the top-5% and top-1% hit rates"
assert OT["candidates"][GAR] == OT["candidates"][GAR_R] and F(GAR) < F(SIMW) + 0.01
_old = GT["test"]["GaaR restart (seed 0, budget 40)"]["fitness"] - GT["test"]["frozen default"]["fitness"]
_oci = GT["bootstrap"]["GaaR restart (seed 0, budget 40)"]["diff_vs_default_ci"]
assert _oci[0] < 0 < _oci[1]

NGH = "nucleation-growth (dev-selected)"
assert abs(NG["results"][NGH]["_fitness"] - NG["results"][NONE]["_fitness"]) < 0.01 and NG["bootstrap"][NGH]["diff_vs_none"]["ci"][0] < 0 < NG["bootstrap"][NGH]["diff_vs_none"]["ci"][1], "text: the nucleation-growth rule matched no smoothing"
assert NG["bootstrap"][NGH]["diff_vs_similarity-weighted"]["ci"][1] < 0, "text: the nucleation-growth rule was below the similarity-weighted rule"
assert CG["spearman_retention_vs_mean_effect"]["rho"] > 0.3 and CG["spearman_retention_vs_mean_effect"]["p"] < 0.01
assert CG["test_mean_ap"]["retention_choice"] <= CG["test_mean_ap"]["always_none"] + 0.01 and CG["test_mean_ap"]["always_simw"] > CG["test_mean_ap"]["retention_choice"], "text: the retention-based choice did not beat no smoothing; similarity weighting made it unnecessary"
_rng = lambda c: max(ap(c, k) for k in ("SC", "B3", "B6", "BAR", "IRR", "BND")) - min(ap(c, k) for k in ("SC", "B3", "B6", "BAR", "IRR", "BND"))
assert _rng(MEAN) > 3 * _rng(SIMW) and _rng(SIMW) <= _rng(NONE) + 0.02, "text: shape ranges"
import math
_th = float(np.mean([r["theta"] for r in SV["rows"]]))
_kstar = 9 * (_th - SV["r_n"]) / (SV["r_a"] - SV["r_n"])
_bs = SV["by_shape"]
_apn = {}
for _r in SV["rows"]:
    _apn.setdefault(_r["scenario"].split("-")[-1], []).append(_r["ap_none"])
assert SV["spearman_all"]["rho"] > 0.7 and SV["spearman_test"]["rho"] > 0.7 and SV5["spearman_all"]["rho"] > 0.5, "text: the contrast-survival model correlated with the measured AP of the fixed mean"
assert _kstar > 8 and _bs["scattered"]["phi"] == 0 and _bs["bar8"]["phi"] == 0 and _bs["scattered"]["ap_mean"] < 0.06 and _bs["bar8"]["ap_mean"] < 0.06, "text: k* is almost the whole window; scattered sites and bars are erased"
assert max(_bs, key=lambda k: _bs[k]["phi"]) == "block6" and max(_bs, key=lambda k: _bs[k]["ap_mean"]) == "block6", "text: the 6 x 6 blocks survive best"
_shape_lab = [("scattered", "Scattered sites"), ("block3", "$3\\times3$ blocks"), ("block6", "$6\\times6$ blocks"), ("bar8", "Bars of 8"), ("irregular12", "Irregular clusters (12)"), ("boundary3", "Boundary blocks $3\\times3$")]
tabcs = (" \\\\\n").join(" & ".join([lab, n2(_bs[k]["phi"]), n2(_bs[k]["ap_mean"]), n2(float(np.mean(_apn[k])))]) for k, lab in _shape_lab) + " \\\\"
_lr = LF["rows"]
_dv = [x for x in _lr if x["split"] == "dev"]
_ts = [x for x in _lr if x["split"] == "test"]
_best = max(sorted({round(x["phi_lf"], 3) for x in _dv}), key=lambda t: np.mean([x["ap_mean"] if x["phi_lf"] >= t else x["ap_none"] for x in _dv]))
_choice = float(np.mean([x["ap_mean"] if x["phi_lf"] >= _best else x["ap_none"] for x in _ts]))
_apnt = float(np.mean([x["ap_none"] for x in _ts]))
assert LF["lf_vs_ap_mean"]["rho"] > 0.5 and LF["lf_vs_ap_mean_test"]["rho"] > 0.45 and LF["lf_vs_ap_simw"]["rho"] < 0 and LF["lf_vs_ap_none"]["rho"] < 0, "text: the label-free predictor tracks the fixed mean and correlates negatively with the AP of none and simw"
assert sum(x["ap_mean"] > x["ap_none"] for x in _lr) == 0 and all(x["ap_simw"] > x["ap_mean"] and x["ap_simw"] > x["ap_none"] for x in _lr), "text: the fixed mean never beat no smoothing; simw beat both in every cohort"
assert _choice <= _apnt + 0.01, "text: the predictor-based choice did not beat no smoothing"
rows = [(NONE, "No smoothing"), (MEAN, "Fixed $3\\times3$ mean (frozen)"), (SIMW, "Similarity-weighted $3\\times3$ (untuned)"), (GAR, "GaaR / random-search champion"), (ONE, "One-sided (dev-selected)"), (OGC, "Order gate, compatibility (dev)"), (OGV, "Order gate, $R_i$ or $H_\\chi$ (dev)")]
tabap = (" \\\\\n").join(" & ".join([lab] + [n2(ap(c, k)) for k in ("H", "D", "R", "RD", "SC", "B3", "B6", "BAR", "IRR", "BND")] + ["%s [%s, %s]" % (n2(F(c)), n2(BT[c]["fitness_ci"][0]), n2(BT[c]["fitness_ci"][1]))]) for c, lab in rows) + " \\\\"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_names = OT["scenarios"]
_lab = ["images\nhand.", "images\nDINOv2", "lattice\nrotor", "lattice\nrotor+D", "scattered", "blocks 3", "blocks 6", "bars 8", "irregular", "boundary"]
_series = [(NONE, "no smoothing", "#8c8c8c"), (MEAN, "fixed 3x3 mean", "#c0392b"), (SIMW, "similarity-weighted 3x3", "#2c7fb8"), (OGC, "order gate (dev-selected)", "#e08e0b")]
fig, ax = plt.subplots(figsize=(7.2, 2.5))
w = 0.2
for j, (c, lab, col) in enumerate(_series):
    xs = np.arange(len(_names)) + (j - 1.5) * w
    y = [R[c][nm]["ap"] for nm in _names]
    lo = [R[c][nm]["ap"] - BT[c]["ap_ci"][k][0] for k, nm in enumerate(_names)]
    hi = [BT[c]["ap_ci"][k][1] - R[c][nm]["ap"] for k, nm in enumerate(_names)]
    ax.bar(xs, y, w, color=col, label=lab, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], error_kw=dict(lw=0.6, capsize=1))
ax.set_xticks(range(len(_names)))
ax.set_xticklabels(_lab, fontsize=6.5)
ax.set_ylabel("average precision", fontsize=8)
ax.tick_params(axis="y", labelsize=7)
ax.legend(fontsize=6.5, ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.18))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_rules.png"), dpi=200)
plt.close(fig)
_fam = OS["family_champions"]
_famlab = [("none", "No smoothing"), ("mean", "Fixed mean"), ("simw", "Similarity-weighted"), ("onesided", "One-sided"), ("og_cos", "Order gate, compatibility"), ("og_disorder", "Order gate, $R_i$"), ("og_vmf", r"Order gate, $H_\chi$")]
_par = lambda g: ("window %d" % g["window"] if g["rule"] != "none" else "-") + ((r", $\vartheta=%.1f$, $\alpha=%d$" % (g["theta"], g["alpha"])) if g["rule"].startswith("og_") or g["rule"] == "onesided_gate" else "") + ((r", $\lambda=%.2f$" % g["lam"]) if g["rule"] != "none" and g["lam"] > 0 else "")
tabdev = (" \\\\\n").join(" & ".join([lab, n2(_fam[r]["dev_fitness"]), _par(_fam[r]["config"])]) for r, lab in _famlab) + " \\\\"
ABSTRACT = (
    "Anomaly scores on lattices of embeddings, such as the patches of an image, are often smoothed over a fixed spatial window. "
    "On lattices simulated with MyUncle, on synthetic images and on a controlled generator that varies only the size and shape of the anomalies, a fixed $3\\times3$ mean destroyed the signal for scattered and bar-shaped anomalies and helped only with DINOv2 embeddings. "
    "Aggregation weighted by the cosine similarity of the embeddings of neighboring sites, with a label-free scale, preserved scattered, bar-shaped, irregular and boundary anomalies: on independent test scenarios it raised the lower-tail average precision from %s (fixed mean) and %s (no smoothing) to %s, with paired 95\\%% intervals above zero. "
    "Gates based on local order and on the von Mises--Fisher information of the neighborhood, a $d$-dimensional extension of MyUncle's entropy quantities, did not beat ordinary similarity; rank gates on that information coincide with gates on the resultant length. "
    "A contrast-survival model, in which a fixed window keeps only the interior of an anomaly, predicted the loss (Spearman %s). A GaaR search over %d rule configurations did not beat random search. All evidence is simulated; code and results are open." % (n2(F(MEAN)), n2(F(NONE)), n2(F(SIMW)), n2(SV["spearman_all"]["rho"]), OS["n_unique"]))
assert len(ABSTRACT.split()) <= 200, len(ABSTRACT.split())

V = dict(
    ABSTRACT=ABSTRACT, TABAP=tabap, TABDEV=tabdev, OG_REP=str(OS["repeats"]), OG_BUDGET=str(OS["budget"]), OG_N=str(OS["n_unique"]), OG_FRANK=str(OS["frozen_rank"]), OG_FF=n2(OS["frozen_fitness"]), OG_OPT=n2(OS["optimum"]),
    OG_RN5=pc(_st["random"]["hit_top5pct"]), OG_G5=pc(_st["greedy"]["hit_top5pct"]), OG_R5=pc(_st["restart"]["hit_top5pct"]), OG_P5=pc(_st["population"]["hit_top5pct"]),
    OG_RN1=pc(_st["random"]["hit_top1pct"]), OG_G1=pc(_st["greedy"]["hit_top1pct"]), OG_R1=pc(_st["restart"]["hit_top1pct"]), OG_P1=pc(_st["population"]["hit_top1pct"]),
    T_NONE_SC=n2(ap(NONE, "SC")), T_MEAN_SC=n2(ap(MEAN, "SC")), T_NONE_BAR=n2(ap(NONE, "BAR")), T_MEAN_BAR=n2(ap(MEAN, "BAR")), T_NONE_ROT=n2(ap(NONE, "R")), T_MEAN_ROT=n2(ap(MEAN, "R")),
    R_NONE_SC=n2(rec(NONE, "SC")), R_MEAN_SC=n2(rec(MEAN, "SC")), R_NONE_BAR=n2(rec(NONE, "BAR")), R_MEAN_BAR=n2(rec(MEAN, "BAR")), T_NONE_D=n2(ap(NONE, "D")), T_MEAN_D=n2(ap(MEAN, "D")),
    F_MEAN=n2(F(MEAN)), F_NONE=n2(F(NONE)), F_SIMW=n2(F(SIMW)), F_OGCOS=n2(F(OGC)), F_OGVMF=n2(F(OGV)), F_GAAR=n2(F(GAR)),
    D_MEAN_NONE=dif(*diff(MEAN, NONE)), D_SIMW_MEAN=dif(*diff(SIMW, MEAN)), D_SIMW_NONE=dif(*diff(SIMW, NONE)), D_ONE_NONE=dif(*diff(ONE, NONE)), D_OGCOS_SIMW=dif(*diff(OGC, SIMW)), D_OGVMF_SIMW=dif(*diff(OGV, SIMW)),
    R_SIMW_SC=n2(rec(SIMW, "SC")), R_SIMW_BAR=n2(rec(SIMW, "BAR")), FR_NONE=n2(fr(NONE)), FR_SIMW=n2(fr(SIMW)), T_SIMW_D=n2(ap(SIMW, "D")), T_SIMW_H=n2(ap(SIMW, "H")), T_NONE_H=n2(ap(NONE, "H")), T_SIMW_ROT=n2(ap(SIMW, "R")),
    F_NGH=n2(NG["results"][NGH]["_fitness"]), D_NGH_NONE=dif(NG["results"][NGH]["_fitness"] - NG["results"][NONE]["_fitness"], NG["bootstrap"][NGH]["diff_vs_none"]["ci"]), D_NGH_SIMW=dif(NG["results"][NGH]["_fitness"] - NG["results"][SIMW]["_fitness"], NG["bootstrap"][NGH]["diff_vs_similarity-weighted"]["ci"]),
    N_COH=str(CG["spearman_retention_vs_mean_effect"]["n"]), CG_RHO=n2(CG["spearman_retention_vs_mean_effect"]["rho"]), CG_P="%.4f" % CG["spearman_retention_vs_mean_effect"]["p"], CG_SHARE=pc(CG["share_chose_mean_test"]),
    CG_CHOICE=n2(CG["test_mean_ap"]["retention_choice"]), CG_NONE=n2(CG["test_mean_ap"]["always_none"]), CG_ORACLE=n2(CG["test_mean_ap"]["oracle"]), CG_SIMW=n2(CG["test_mean_ap"]["always_simw"]),
    RG_NONE=n2(_rng(NONE)), RG_MEAN=n2(_rng(MEAN)), RG_SIMW=n2(_rng(SIMW)),
    CS_RA=n2(SV["r_a"]), CS_RN=n2(SV["r_n"]), CS_TH=n2(_th), CS_K="%.1f" % _kstar, CS_N=str(SV["spearman_all"]["n"]), CS_RHO=n2(SV["spearman_all"]["rho"]), CS_RHOT=n2(SV["spearman_test"]["rho"]), CS_RHO5=n2(SV5["spearman_all"]["rho"]), TABCS=tabcs,
    LF_RHO=n2(LF["lf_vs_ap_mean"]["rho"]), LF_RHOT=n2(LF["lf_vs_ap_mean_test"]["rho"]), LF_RHO2=n2(LF["sensitivity"]["2.0"]), LF_RHO5=n2(LF["sensitivity"]["5.0"]), LF_LAB=n2(LF["lf_vs_label_phi"]["rho"]),
    LF_NONE=sg(LF["lf_vs_ap_none"]["rho"]), LF_SIMW=sg(LF["lf_vs_ap_simw"]["rho"]), LF_APN=n2(LF["mean_ap"]["none"]), LF_APM=n2(LF["mean_ap"]["mean"]), LF_APS=n2(LF["mean_ap"]["simw"]), LF_CH=n2(_choice), LF_APNT=n2(_apnt),
    OLD_DIFF=sg(_old), OLD_CI=cif(_oci), TEST_UNITS="16 to 50",
)
s = open(os.path.join(HERE, "prl_template.tex.in"), encoding="utf-8").read()
for k, v in V.items():
    s = s.replace("@%s@" % k, v)
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
