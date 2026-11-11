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
for f in ("fig_bridge.png", "fig_bridge_maps.png", "fig_channels.png"):
    shutil.copy(os.path.join(HERE, "..", "figures", f), os.path.join(OUT, f))
J = lambda name: json.load(open(os.path.join(RES, name)))
OT, OS, OG = J("og_test.json"), J("og_search.json"), J("og_grid_dev.json")
GT = J("gaar_test.json")
NG, CG = J("og_ngh.json"), J("og_closure.json")
SV, SV5 = J("og_survival.json"), J("og_survival_w5.json")
LF = J("og_lfsurv.json")
TC, TD, G1 = J("two_channel.json"), J("two_channel_dev.json"), J("og_geometry_first_attempt.json")
R = OT["results"]
BT = OT["bootstrap"]
n2 = lambda x: "%.2f" % x
pc = lambda x: "%d\\%%" % round(100 * x)
sg = lambda x: ("$-$%.2f" % -x) if x <= -0.005 else (("+%.2f" % x) if x >= 0.005 else "0.00")
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
# ---- two channels (tc_study.py): values and guards
RAR, MEAN3, CA, CB = "rarity", "rarity, fixed 3x3 mean", "rarity, similarity-weighted (channel A)", "directional disorder (channel B)"
DISO, CNB, MAN = "isotropic local disorder D", "center-to-neighbor distance", "local manifold inconsistency (m=2)"
FUS, FUS_CNB, FUS_D = "fusion select: A + directional disorder", "fusion select: A + center-to-neighbor distance", "fusion select: A + isotropic disorder D"
ZMAX, RMAX = "fusion zmax: A + directional disorder", "fusion rankmax: A + directional disorder"
TWS = ["fusion tailw p=%d: A + directional disorder" % p_ for p_ in (1, 2, 4)]
_sn = TC["scenarios"]
_ix = lambda key: [k for k, nm in enumerate(_sn) if key in nm][0]
tap = lambda c, key: TC["ap"][c][_ix(key)]
_arr = [k for k, nm in enumerate(_sn) if nm.startswith("arr-")]
_mix = [k for k, nm in enumerate(_sn) if nm.startswith("mixed-")]
_ten = [k for k in range(len(_sn)) if k not in _arr and k not in _mix]
assert len(_sn) == 15 and len(_arr) == 3 and len(_mix) == 2 and len(_ten) == 10
gm = lambda c, idx: float(np.mean([TC["ap"][c][k] for k in idx]))
TF, TB = TC["fitness"], TC["bootstrap"]
ci2 = lambda c: "[%s, %s]" % (n2(c[0]), n2(c[1]))
_prev = 0.07
assert TC["selected_on_dev"] == FUS and TD["fitness"][FUS] == max(v for k, v in TD["fitness"].items() if k.startswith("fusion") and k.endswith("A + directional disorder")), "text: the selection was the best fusion rule on the development scenarios"
assert gm(RAR, _arr) < 0.08 and gm(CA, _arr) < 0.08, "text: the rarity is at chance on rearranged regions, with and without similarity weighting"
assert gm(CB, _arr) > 0.8 and TB[CB]["diff_arr_vs_A"]["ci"][0] > 0.7, "text: the directional disorder finds rearranged regions"
assert abs(gm(CNB, _arr) - gm(CB, _arr)) < 0.02 and gm(DISO, _arr) < gm(CB, _arr) - 0.2 and gm(MAN, _arr) < gm(CB, _arr) - 0.2, "text: the neighbor distance is equally good; D and the manifold statistic are below"
assert tap(CB, "scattered") > 0.5 and tap(CB, "bar8") > 0.5 and tap(CB, "block3") < 0.05 and tap(CB, "block6") < 0.05, "text: channel B alone finds thin anomalies and ignores compact blocks"
assert max(tap(CB, "handcrafted"), tap(CB, "dinov2")) < 0.05 and max(tap(CB, "sim-test-rotor"), tap(CB, "rotor_D")) < min(tap(CA, "sim-test-rotor"), tap(CA, "rotor_D")), "text: channel B alone is weak on images and MyUncle lattices"
_tail = lambda key, scs: float(np.mean([x[key] for sc in scs for x in TC["tails"][_ix(sc)]]))
_blk = ("morph-test-block3", "morph-test-block6", "morph-test-boundary3")
assert _tail("B", _blk) < 5 < 15 < _tail("B_cnb", _blk) and _tail("B_D", _blk) > 15, "text: tail indices in the block scenarios"
assert TB[FUS]["diff_vs_A"]["ci"][0] > 0 and TB[FUS]["diff_vs_rarity"]["ci"][0] > 0 and TF[FUS] > TF[CA] > TF[RAR] > TF[MEAN3], "text: the fused detector beat channel A and the rarity with paired intervals above zero"
assert abs(gm(FUS, _ten) - gm(CA, _ten)) < 0.01 and TB[FUS]["diff_old_vs_A"]["ci"][0] > -0.01, "text: nothing was lost on the ten earlier scenarios"
_feat = [x["channel"] for k in _ten for x in TC["tails"][k]]
_rear = [x["channel"] for k in _arr + _mix for x in TC["tails"][k]]
assert all(c == "B" for c in _rear) and _feat.count("B") == 1 and [x["channel"] for x in TC["tails"][_ix("bar8")]].count("B") == 1, "text: the channel choices"
assert TC["ap"]["fusion select0: A + directional disorder"] == TC["ap"][FUS], "text: the guard was never the deciding condition"
assert TB[FUS]["diff_vs_select_cnb"]["ci"][0] > 0 and TB[FUS]["diff_vs_select_D"]["ci"][0] > 0 and tap(FUS_CNB, "block3") < tap(FUS, "block3") - 0.1 and tap(FUS_CNB, "boundary3") < tap(FUS, "boundary3") - 0.1, "text: the directional form mattered for the choice"
assert max(TF[ZMAX], TF[RMAX], *[TF[t] for t in TWS]) < TF[FUS] and tap(ZMAX, "dinov2") < 0.05, "text: the site-wise rules were below the selection"
_ty = TC["typed"]
_m1, _m2 = "mixed-test-block3", "mixed-test-scattered"
assert all(tap(FUS, m) > tap(CA, m) for m in (_m1, _m2)) and all(_ty[m][CB]["rearranged"] > 0.5 and _ty[m][RAR]["outliers"] > 0.4 and _ty[m][CB]["outliers"] < 0.1 and _ty[m][RAR]["rearranged"] < 0.1 for m in (_m1, _m2)), "text: mixed cohorts by kind"
assert _ty[_m2][CA]["outliers"] < _ty[_m2][RAR]["outliers"] - 0.2, "text: similarity weighting lowered the AP of scattered outliers in the smooth background"
_g1b, _g1a = G1["best_dev_config"], "similarity-weighted rarity"
assert _g1b.startswith("simw + disorder") and G1["dev"]["fitness"][_g1b] > G1["dev"]["fitness"][_g1a] and G1["test"]["fitness"][_g1b] < G1["test"]["fitness"][_g1a], "text: the first attempt did not beat channel A on the test"
_dino = TC["tails"][_ix("dinov2")][0]
assert _dino["A"] < 3 and _dino["B"] < 3
_row = lambda c, lab, bold=False: " & ".join([lab, n2(gm(c, _ten)), n2(gm(c, _arr)), n2(gm(c, _mix)), "%s %s" % (n2(TF[c]), ci2(TB[c]["fitness_ci"]))])
tabtc = (" \\\\\n").join([_row(RAR, "Rarity, no smoothing"), _row(MEAN3, "Rarity, fixed $3\\times3$ mean"), _row(CA, "Channel A (similarity-weighted rarity)"), _row(DISO, "Isotropic disorder $D$"),
                          _row(MAN, "Local manifold statistic"), _row(CNB, "Distance to the 8 neighbors"), _row(CB, "Channel B (directional disorder)"), "\\midrule\n" + _row(FUS, "Channel chosen by tail index")]) + " \\\\"
TCV = dict(
    TABTC=tabtc, TC_ARR_R=n2(gm(RAR, _arr)), TC_PREV="0.06 to 0.07", TC_ARR_A=n2(gm(CA, _arr)), TC_ARR_B=n2(gm(CB, _arr)), TC_ARR_B_CI=ci2(TB[CB]["mean_arr_ci"]), TC_ARR_GAIN=cif(TB[CB]["diff_arr_vs_A"]["ci"]),
    TC_ARR_CNB=n2(gm(CNB, _arr)), TC_ARR_D=n2(gm(DISO, _arr)), TC_ARR_MAN=n2(gm(MAN, _arr)), TC_B_SC=n2(tap(CB, "scattered")), TC_B_BAR=n2(tap(CB, "bar8")), TC_B_B3=n2(tap(CB, "block3")), TC_B_B6=n2(tap(CB, "block6")),
    TC_TAIL_B_BLK="%.1f" % _tail("B", _blk), TC_TAIL_CNB_BLK="%.0f" % _tail("B_cnb", _blk), TC_TAIL_D_BLK="%.0f" % _tail("B_D", _blk),
    TC_B_H=n2(tap(CB, "handcrafted")), TC_B_D=n2(tap(CB, "dinov2")), TC_B_R=n2(tap(CB, "sim-test-rotor")), TC_B_RD=n2(tap(CB, "rotor_D")),
    TC_DEV_FUS=n2(TD["fitness"][FUS]), TC_DEV_A=n2(TD["fitness"][CA]), TC_F_FUS=n2(TF[FUS]), TC_F_FUS_CI=ci2(TB[FUS]["fitness_ci"]), TC_F_A=n2(TF[CA]), TC_F_R=n2(TF[RAR]), TC_F_M=n2(TF[MEAN3]),
    TC_GAIN_A=dif(TF[FUS] - TF[CA], TB[FUS]["diff_vs_A"]["ci"]), TC_GAIN_R=dif(TF[FUS] - TF[RAR], TB[FUS]["diff_vs_rarity"]["ci"]), TC_GAIN_A_CI="%s to %s" % (sg(TB[FUS]["diff_vs_A"]["ci"][0]), sg(TB[FUS]["diff_vs_A"]["ci"][1])),
    TC_OLD_FUS=n2(gm(FUS, _ten)), TC_OLD_A=n2(gm(CA, _ten)), TC_OLD_DIFF=cif(TB[FUS]["diff_old_vs_A"]["ci"]), TC_NA=str(_feat.count("A")), TC_NF=str(len(_feat)), TC_NB=str(len(_rear)),
    TC_F_FCNB=n2(TF[FUS_CNB]), TC_D_FCNB=dif(TF[FUS] - TF[FUS_CNB], TB[FUS]["diff_vs_select_cnb"]["ci"]), TC_CNB_B3=n2(tap(FUS_CNB, "block3")), TC_CNB_BND=n2(tap(FUS_CNB, "boundary3")), TC_FUS_B3=n2(tap(FUS, "block3")),
    TC_FUS_BND=n2(tap(FUS, "boundary3")), TC_F_FD=n2(TF[FUS_D]), TC_D_FD=dif(TF[FUS] - TF[FUS_D], TB[FUS]["diff_vs_select_D"]["ci"]), TC_F_ZMAX=n2(TF[ZMAX]), TC_ZMAX_D=n2(tap(ZMAX, "dinov2")), TC_F_RMAX=n2(TF[RMAX]),
    TC_F_TW=n2(min(TF[t] for t in TWS)) if n2(min(TF[t] for t in TWS)) == n2(max(TF[t] for t in TWS)) else "%s to %s" % (n2(min(TF[t] for t in TWS)), n2(max(TF[t] for t in TWS))),
    TC_MIX_FUS1=n2(tap(FUS, _m1)), TC_MIX_FUS2=n2(tap(FUS, _m2)), TC_MIX_A1=n2(tap(CA, _m1)), TC_MIX_A2=n2(tap(CA, _m2)), TC_MIX_B_ARR1=n2(_ty[_m1][CB]["rearranged"]), TC_MIX_B_ARR2=n2(_ty[_m2][CB]["rearranged"]),
    TC_MIX_R_OUT1=n2(_ty[_m1][RAR]["outliers"]), TC_MIX_R_OUT2=n2(_ty[_m2][RAR]["outliers"]), TC_MIX_B_OUT1=n2(_ty[_m1][CB]["outliers"]), TC_MIX_B_OUT2=n2(_ty[_m2][CB]["outliers"]), TC_MIX_A_OUT2=n2(_ty[_m2][CA]["outliers"]),
    G1_DEV_BLEND="%.3f" % G1["dev"]["fitness"][_g1b], G1_DEV_A="%.3f" % G1["dev"]["fitness"][_g1a], G1_TEST_BLEND="%.3f" % G1["test"]["fitness"][_g1b], G1_TEST_A="%.3f" % G1["test"]["fitness"][_g1a],
    TC_TAIL_A_DINO="%.1f" % _dino["A"], TC_TAIL_B_DINO="%.1f" % _dino["B"],
)
ABSTRACT = (
    "Anomaly scores on lattices of embeddings, such as image patches, usually measure how rare the embedding of a site is. We show on simulated data that rarity is one of two complementary label-free channels. "
    "Channel A is the rarity aggregated with weights given by the similarity of neighboring embeddings; it preserved scattered and bar-shaped anomalies that a fixed $3\\times3$ mean destroyed "
    "(lower-tail average precision %s against %s on ten independent test scenarios), and a contrast-survival model explains the loss. "
    "Channel B is a directional form of the local disorder of the MyUncle lattice model, the minimum of the disorder over the four quadrants of a site. It detects arrangement anomalies, the same embeddings in the wrong places: "
    "there the rarity stayed at chance (average precision %s) and channel B reached %s. A label-free tail index chooses the channel per cohort; on fifteen independent test scenarios this raised the lower-tail average precision "
    "from %s to %s (paired 95\\%% interval of the gain %s to %s) without loss on the scenarios with feature anomalies only. "
    "Order gates and an evolutionary configuration search did not help, and cohorts with both kinds of anomaly need both maps. Code and results are open."
    % (n2(F(SIMW)), n2(F(MEAN)), n2(gm(RAR, _arr)), n2(gm(CB, _arr)), n2(TF[CA]), n2(TF[FUS]), sg(TB[FUS]["diff_vs_A"]["ci"][0]), sg(TB[FUS]["diff_vs_A"]["ci"][1])))
assert len(ABSTRACT.split()) <= 200, len(ABSTRACT.split())

V = dict(
    ABSTRACT=ABSTRACT, TABAP=tabap, OG_REP=str(OS["repeats"]), OG_BUDGET=str(OS["budget"]), OG_N=str(OS["n_unique"]), OG_FRANK=str(OS["frozen_rank"]), OG_FF=n2(OS["frozen_fitness"]), OG_OPT=n2(OS["optimum"]),
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
V.update(TCV)
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
