# -*- coding: utf-8 -*-
"""ChemRxiv version of the manuscript: same evidence, framed around the statistical-mechanics content (order and entropy of lattices of unit vectors in d dimensions) and the simulated-lattice testbed.
Executed by build_documents.py (--variant chemrxiv) with the block list B, the results and the helper functions in scope.  No evidence is added or removed and no claim goes beyond the results:
no micrograph of a real material and no real tissue image was analysed, which the text says.  Whether ChemRxiv's moderators accept the scope is their decision."""

CHEM_TITLE = "Rarity and local order on lattices of unit vectors in d dimensions: two complementary label-free channels for unusual local organization in simulated maintenance lattices and synthetic images"
CHEM_KEYWORDS = "Keywords: lattice models; statistical mechanics; von Mises-Fisher distribution; orientational order; entropy; spatial arrangement; unsupervised anomaly detection; microscopy image analysis; open-source software"

import json as _json
_GS = _json.load(open(os.path.join(RES, "gaar_search_comparison.json")))
_GT = _json.load(open(os.path.join(RES, "gaar_test.json")))
_GL = _json.load(open(os.path.join(RES, "gaar_labelfree.json")))
_DEF, _NOS, _OPT = "frozen default", "no smoothing", "exhaustive dev optimum"
_RST, _GRD, _RND = "GaaR restart (seed 0, budget 40)", "GaaR greedy (seed 0, budget 40)", "GaaR random (seed 0, budget 40)"
_TF = lambda k: _GT["test"][k]["fitness"]
assert _TF(_RST) > _TF(_DEF) and _TF(_RND) > _TF(_DEF) and _TF(_GRD) > _TF(_DEF) and _TF(_OPT) > _TF(_DEF)
assert _TF(_RST) > _TF(_OPT) > _TF(_RND) and _GT["dev"][_OPT]["fitness"] > _GT["dev"][_RST]["fitness"] > _GT["dev"][_RND]["fitness"]
assert _GT["test"][_RST]["aps"][1] < _GT["test"][_DEF]["aps"][1] and _GT["test"][_RST]["aps"][0] > _GT["test"][_DEF]["aps"][0]
_st = _GS["strategies"]
_BT = _GT["bootstrap"]
_MS = _json.load(open(os.path.join(RES, "morphology_study.json")))
_mm = _MS["results"]
_sz = [str(x) for x in _MS["sizes"]]
assert _BT[_RST]["diff_vs_default_ci"][0] < 0 < _BT[_RST]["diff_vs_default_ci"][1]
assert all(_mm[b]["no smoothing"]["diff_vs_frozen_ci"][0] > 0 for b in _sz)
assert all(_mm[a]["frozen (mean 3)"]["ap"] < _mm[b]["frozen (mean 3)"]["ap"] for a, b in zip(_sz, _sz[1:]))
assert all(_mm[b]["mean 5"]["diff_vs_frozen_ci"][1] < 0 for b in _sz)
assert all(_mm[b]["GaaR restart champion"]["diff_vs_frozen_ci"][0] > 0 for b in ("1", "2", "3")) and _mm["6"]["GaaR restart champion"]["diff_vs_frozen_ci"][0] < 0 < _mm["6"]["GaaR restart champion"]["diff_vs_frozen_ci"][1]
assert all(_mm[b]["GaaR restart champion"]["ap"] < _mm[b]["no smoothing"]["ap"] for b in _sz)
_f2 = lambda x: ("-%.2f" % -x) if x < 0 else ("%.2f" % x)
_dci = lambda a: "[%s, %s]" % (("-%.2f" % -a[0]) if a[0] < 0 else ("+%.2f" % a[0]), ("-%.2f" % -a[1]) if a[1] < 0 else ("+%.2f" % a[1]))
_dd = lambda k: _GT["test"][k]["fitness"] - _GT["test"][_DEF]["fitness"]

assert _st["greedy"]["hit_top1pct"] > 2 * _st["random"]["hit_top1pct"] and _st["restart"]["hit_top1pct"] > 2 * _st["random"]["hit_top1pct"]
assert _st["random"]["hit_top5pct"] >= _st["population"]["hit_top5pct"] > _st["restart"]["hit_top5pct"] >= _st["greedy"]["hit_top5pct"]
_mor = {r["scenario"]: r["moran"] for r in _GL["rows"]}
_eff = {r["scenario"]: r["effect"] for r in _GL["rows"]}
_bad = [k for k in _eff if _eff[k] < -0.2]
assert sorted(_bad) == ["sim-dev-rotor", "sim-test-rotor"] and all(_mor[k] < 0.02 for k in _bad) and all(_mor[k] > 0.1 for k in _mor if k not in _bad)

_OT = _json.load(open(os.path.join(RES, "og_test.json")))
_OS = _json.load(open(os.path.join(RES, "og_search.json")))
_NG = _json.load(open(os.path.join(RES, "og_ngh.json")))
_CG = _json.load(open(os.path.join(RES, "og_closure.json")))
_R, _BTO = _OT["results"], _OT["bootstrap"]
_NONE, _MEAN, _SIMW = "none", "frozen (mean 3)", "similarity-weighted (3), untuned"
_ONE, _OGC, _OGV, _GAR = "best onesided (dev)", "best og_cos (dev)", "best og_vmf (dev)", "GaaR restart champion"
_SCN = dict(H="images-test-handcrafted", D="images-test-dinov2_vits14", R="sim-test-rotor", RD="sim-test-rotor_D", SC="morph-test-scattered", B3="morph-test-block3", B6="morph-test-block6", BAR="morph-test-bar8", IRR="morph-test-irregular12", BND="morph-test-boundary3")
_ap = lambda c, k: _R[c][_SCN[k]]["ap"]
_F = lambda c: _R[c]["_fitness"]
_rec = lambda c, k: _R[c][_SCN[k]]["region_recall"]
_sg = lambda x: ("-%.2f" % -x) if x <= -0.005 else (("+%.2f" % x) if x >= 0.005 else "0.00")
_ci2 = lambda c: "[%s, %s]" % (_sg(c[0]), _sg(c[1]))
_dd2 = lambda c, ref: "%s %s" % (_sg(_F(c) - _F(ref)), _ci2(_BTO[c]["diff_vs_" + ref.split()[0]]["ci"]))
assert all(_ap(_MEAN, k) < _ap(_NONE, k) for k in ("SC", "BAR", "R")) and _ap(_MEAN, "D") > _ap(_NONE, "D")
assert _BTO[_MEAN]["diff_vs_none"]["ci"][1] < 0 and _BTO[_SIMW]["diff_vs_frozen"]["ci"][0] > 0 and _BTO[_SIMW]["diff_vs_none"]["ci"][0] > 0
assert _F(_OGC) < _F(_SIMW) and _F(_OGV) < _F(_SIMW)
assert _BTO[_ONE]["diff_vs_none"]["ci"][0] < 0 < _BTO[_ONE]["diff_vs_none"]["ci"][1]
_so = _OS["strategies"]
assert all(_so["random"]["hit_top5pct"] > _so[k]["hit_top5pct"] and _so["random"]["hit_top1pct"] > _so[k]["hit_top1pct"] for k in ("greedy", "restart", "population"))
_pcx = lambda x: "%d%%" % round(100 * x)

_TC = _json.load(open(os.path.join(RES, "two_channel.json")))
_TD = _json.load(open(os.path.join(RES, "two_channel_dev.json")))
_G1 = _json.load(open(os.path.join(RES, "og_geometry_first_attempt.json")))
_RAR, _M3, _CA, _CB = "rarity", "rarity, fixed 3x3 mean", "rarity, similarity-weighted (channel A)", "directional disorder (channel B)"
_DISO, _CNB, _MAN = "isotropic local disorder D", "center-to-neighbor distance", "local manifold inconsistency (m=2)"
_FUS, _FCNB, _FD = "fusion select: A + directional disorder", "fusion select: A + center-to-neighbor distance", "fusion select: A + isotropic disorder D"
_ZMAX, _RMAX = "fusion zmax: A + directional disorder", "fusion rankmax: A + directional disorder"
_TWS = ["fusion tailw p=%d: A + directional disorder" % _p for _p in (1, 2, 4)]
_tsn = _TC["scenarios"]
_tix = lambda key: [k for k, nm in enumerate(_tsn) if key in nm][0]
_tap = lambda c, key: _TC["ap"][c][_tix(key)]
_iarr = [k for k, nm in enumerate(_tsn) if nm.startswith("arr-")]
_imix = [k for k, nm in enumerate(_tsn) if nm.startswith("mixed-")]
_iten = [k for k in range(len(_tsn)) if k not in _iarr and k not in _imix]
_gm = lambda c, idx: sum(_TC["ap"][c][k] for k in idx) / len(idx)
_TFt, _TBt = _TC["fitness"], _TC["bootstrap"]
_cip = lambda c: "[%.2f, %.2f]" % (c[0], c[1])
_ttail = lambda key, scs: sum(x[key] for sc in scs for x in _TC["tails"][_tix(sc)]) / sum(len(_TC["tails"][_tix(sc)]) for sc in scs)
_blk = ("morph-test-block3", "morph-test-block6", "morph-test-boundary3")
_feat = [x["channel"] for k in _iten for x in _TC["tails"][k]]
_rear = [x["channel"] for k in _iarr + _imix for x in _TC["tails"][k]]
_ty = _TC["typed"]
_m1, _m2 = "mixed-test-block3", "mixed-test-scattered"
_g1b, _g1a = _G1["best_dev_config"], "similarity-weighted rarity"
assert _TC["selected_on_dev"] == _FUS and len(_tsn) == 15 and len(_iten) == 10
assert _gm(_RAR, _iarr) < 0.08 and _gm(_CA, _iarr) < 0.08 and _gm(_CB, _iarr) > 0.8 and _TBt[_CB]["diff_arr_vs_A"]["ci"][0] > 0.7
assert abs(_gm(_CNB, _iarr) - _gm(_CB, _iarr)) < 0.02 and _gm(_DISO, _iarr) < _gm(_CB, _iarr) - 0.2 and _gm(_MAN, _iarr) < _gm(_CB, _iarr) - 0.2
assert _tap(_CB, "scattered") > 0.5 and _tap(_CB, "bar8") > 0.5 and _tap(_CB, "block3") < 0.05 and _tap(_CB, "block6") < 0.05
assert _ttail("B", _blk) < 5 < 15 < _ttail("B_cnb", _blk) and _ttail("B_D", _blk) > 15
assert _TBt[_FUS]["diff_vs_A"]["ci"][0] > 0 and _TBt[_FUS]["diff_vs_rarity"]["ci"][0] > 0 and _TBt[_FUS]["diff_old_vs_A"]["ci"][0] > -0.01
assert all(c == "B" for c in _rear) and _feat.count("B") == 1 and _TC["ap"]["fusion select0: A + directional disorder"] == _TC["ap"][_FUS]
assert _TBt[_FUS]["diff_vs_select_cnb"]["ci"][0] > 0 and _TBt[_FUS]["diff_vs_select_D"]["ci"][0] > 0
assert max(_TFt[_ZMAX], _TFt[_RMAX], *[_TFt[t] for t in _TWS]) < _TFt[_FUS] and _tap(_ZMAX, "dinov2") < 0.05
assert all(_ty[m][_CB]["rearranged"] > 0.5 and _ty[m][_RAR]["outliers"] > 0.4 and _ty[m][_CB]["outliers"] < 0.1 and _ty[m][_RAR]["rearranged"] < 0.1 for m in (_m1, _m2)) and _ty[_m2][_CA]["outliers"] < _ty[_m2][_RAR]["outliers"] - 0.2
assert _G1["dev"]["fitness"][_g1b] > _G1["dev"]["fitness"][_g1a] and _G1["test"]["fitness"][_g1b] < _G1["test"]["fitness"][_g1a]
_tcrow = lambda c, lab: [lab, "%.2f" % _gm(c, _iten), "%.2f" % _gm(c, _iarr), "%.2f" % _gm(c, _imix), "%.2f %s" % (_TFt[c], _cip(_TBt[c]["fitness_ci"]))]
_twr = n(min(_TFt[t] for t in _TWS)) if n(min(_TFt[t] for t in _TWS)) == n(max(_TFt[t] for t in _TWS)) else "%s to %s" % (n(min(_TFt[t] for t in _TWS)), n(max(_TFt[t] for t in _TWS)))
assert _tap(_MAN, "scattered") > 0.95 and max(_tap(_MAN, k) for k in ("block3", "block6", "bar8", "irregular12", "boundary3")) < 0.1
CHEM_TC = [
    ("h2", "4.10 Arrangement anomalies: local order as a second channel"),
    ("p", "A rarity score, however it is aggregated, sees only what is at a site. An anomaly that consists of ordinary elements in the wrong arrangement, a contextual anomaly [@Chandola2009] or, in industrial inspection, a logical anomaly [@Bergmann2022], leaves the distribution of the embeddings unchanged. "
          "To test this, the generator was given a spatially correlated background (a smooth random field) in which the embeddings of a region are permuted among its sites, so that every embedding keeps its value and only the arrangement is destroyed (6 x 6 and 8 x 8 blocks and irregular 24-site clusters; 7% of the sites), and mixed cohorts with rearranged 6 x 6 blocks (3.5%) together with feature outliers that leave the subspace of the normal sites (3 x 3 blocks or scattered sites, 1.5%). "
          "Development and test versions use disjoint seeds; the separability of the outliers was calibrated beforehand on the unsmoothed rarity only."),
    ("p", "The isotropic disorder D_i of Section 2.3 is large wherever neighboring vectors differ, which includes every wall between two ordered domains. The directional disorder B_i is the minimum, over the four 2 x 2 quadrant windows that contain site i, of one minus the length of the mean unit vector of the quadrant. A wall is ordered on at least one side, so one quadrant lies inside one domain and B_i stays near zero, while a disorganized region is disordered in every direction; taking the minimum over sub-windows is the idea of the edge-preserving filter of Kuwahara et al. [@Kuwahara1976], applied to the order parameter instead of the variance. "
          "Two channels result: channel A, the similarity-weighted rarity of Section 4.8, and channel B, the directional disorder. They have different units and are compared through a label-free tail index: the raw score of a channel (the nearest-neighbor distance at k = 1% for channel A, B_i for channel B) is standardized over the cohort with the median and the median absolute deviation [@Rousseeuw1993], and the tail index T is the 97th percentile of the result (1.88 for a Gaussian bulk without anomalies). "
          "The fused detector uses channel B for a cohort when T_B > T_A and T_B > 2 x 1.88, and channel A otherwise: one choice per cohort, without labels. It was chosen among seven fusion rules (the unguarded selection, the site-wise maximum of the standardized scores, the maximum of the cohort ranks, and maxima of the ranks weighted by T^p, p = 1, 2, 4) on fifteen development scenarios and evaluated once on fifteen test scenarios."),
    ("table", "Table 9. Two channels on the fifteen test scenarios: mean average precision over the ten scenarios of Table 7, the three arrangement scenarios and the two mixed scenarios, and the lower-tail fitness F over all fifteen with its 95% interval.",
     ["Detector", "Ten scenarios of Table 7", "Arrangement", "Mixed", "F [95% interval]"],
     [_tcrow(_RAR, "Rarity, no smoothing"), _tcrow(_M3, "Rarity, fixed 3 x 3 mean"), _tcrow(_CA, "Channel A (similarity-weighted rarity)"), _tcrow(_DISO, "Isotropic disorder D"), _tcrow(_MAN, "Local manifold statistic"),
      _tcrow(_CNB, "Distance to the 8 neighbors"), _tcrow(_CB, "Channel B (directional disorder)"), _tcrow(_FUS, "Channel chosen by tail index")],
     [2.5, 1.2, 0.9, 0.7, 1.3], True, 12),
    ("p", "On the three test scenarios with rearranged regions the rarity was at chance: its average precision was %s at a prevalence of 0.06 to 0.07, and similarity weighting did not change that (%s). The directional disorder reached %s (95%% interval %s; paired gain over channel A %s; Table 9). "
          "The simple distance of a site to its eight neighbors was equally good (%s), the isotropic disorder D reached %s and a local manifold statistic (the distance of a site from the plane spanned by the two leading principal directions of its 5 x 5 neighborhood) %s. The directional form is thus not a better detector of rearranged regions than a simple baseline; the difference lies in what else the statistics respond to. "
          "In the feature-outlier scenarios, whose background has walls between domains, channel B alone found thin anomalies (scattered sites %s, bars %s) and ignored compact blocks (%s and %s), whose interior is ordered; its tail index there was %.1f, against %.0f for the neighbor distance and %.0f for the isotropic disorder, which respond to every wall. On the images and on the MyUncle lattices channel B alone was weak (%s, %s, %s and %s)."
          % (n(_gm(_RAR, _iarr)), n(_gm(_CA, _iarr)), n(_gm(_CB, _iarr)), _cip(_TBt[_CB]["mean_arr_ci"]), _ci2(_TBt[_CB]["diff_arr_vs_A"]["ci"]), n(_gm(_CNB, _iarr)), n(_gm(_DISO, _iarr)), n(_gm(_MAN, _iarr)),
             n(_tap(_CB, "scattered")), n(_tap(_CB, "bar8")), n(_tap(_CB, "block3")), n(_tap(_CB, "block6")), _ttail("B", _blk), _ttail("B_cnb", _blk), _ttail("B_D", _blk),
             n(_tap(_CB, "handcrafted")), n(_tap(_CB, "dinov2")), n(_tap(_CB, "sim-test-rotor")), n(_tap(_CB, "rotor_D")))),
    ("p", "On the development scenarios the cohort-level selection had the highest fitness of the seven fusion rules (F = %s, against %s for channel A). On the fifteen test scenarios it reached F = %s (95%% interval %s), against %s for channel A, %s for the unsmoothed rarity and %s for the fixed mean; the paired gain over channel A was %s %s and over the unsmoothed rarity %s %s. "
          "The gain came from the scenarios with rearranged regions, and nothing was lost on the ten earlier scenarios (mean AP %s against %s; paired difference %s). The tail index chose channel A in %d of the %d test cohorts that contain only feature anomalies (the exception was one cohort of bars, where both channels work) and channel B in all %d cohorts with rearranged regions; the requirement T_B > 2 x 1.88 was never the deciding condition. "
          "The directional form mattered for the choice: with the neighbor distance in place of B_i the selection reached F = %s (paired difference to the directional version %s %s), because the walls of the block scenarios gave it a heavy tail and it was chosen wrongly there, and with the isotropic disorder F = %s (%s %s). "
          "The site-wise rules were below the selection: the maximum of the standardized scores %s (it failed on the DINOv2 images, AP %s), the maximum of the ranks %s, and the tail-weighted maxima %s."
          % (n(_TD["fitness"][_FUS]), n(_TD["fitness"][_CA]), n(_TFt[_FUS]), _cip(_TBt[_FUS]["fitness_ci"]), n(_TFt[_CA]), n(_TFt[_RAR]), n(_TFt[_M3]),
             _sg(_TFt[_FUS] - _TFt[_CA]), _ci2(_TBt[_FUS]["diff_vs_A"]["ci"]), _sg(_TFt[_FUS] - _TFt[_RAR]), _ci2(_TBt[_FUS]["diff_vs_rarity"]["ci"]),
             n(_gm(_FUS, _iten)), n(_gm(_CA, _iten)), _ci2(_TBt[_FUS]["diff_old_vs_A"]["ci"]), _feat.count("A"), len(_feat), len(_rear),
             n(_TFt[_FCNB]), _sg(_TFt[_FUS] - _TFt[_FCNB]), _ci2(_TBt[_FUS]["diff_vs_select_cnb"]["ci"]), n(_TFt[_FD]), _sg(_TFt[_FUS] - _TFt[_FD]), _ci2(_TBt[_FUS]["diff_vs_select_D"]["ci"]),
             n(_TFt[_ZMAX]), n(_tap(_ZMAX, "dinov2")), n(_TFt[_RMAX]), _twr)),
    ("fig", "fig_channels.png", "Fig. @F:channels@. The two channels on one test lattice that contains both kinds of anomaly (mixed scenario, seed 121). (a) Feature outliers and a rearranged block; (b) channel A, the rarity; (c) channel B, the directional disorder; (d) the top 3% of each channel. Rarity marks the feature outliers and is blind to the rearranged block; the directional disorder marks the rearranged block and is blind to the compact outliers.", 6.4),
    ("p", "Three limits were measured. First, a cohort-level choice follows one channel. In the mixed cohorts it chose channel B and reached an AP of %s and %s over both kinds of anomaly (channel A: %s and %s), but taken by kind, the rearranged sites were found by channel B (%s and %s) and the feature outliers only by the rarity (%s and %s unsmoothed; channel B %s and %s): when both kinds occur together, the two maps must be read side by side (Fig. @F:channels@). "
          "Second, in the smooth background of the mixed cohorts similarity weighting lowered the AP of scattered feature outliers from %s to %s, because coherent patches of mildly rare normal sites are reinforced; channel A is not uniformly better than the unsmoothed rarity. "
          "Third, a first attempt, fixed blends of the ranks of channel A and of the isotropic disorder, had been selected on thirteen of the development scenarios (F = %.3f against %.3f) and did not beat channel A on the corresponding test scenarios (%.3f against %.3f); the directional statistic and the tail index were designed after that, so thirteen of the fifteen test scenarios had been used once before. "
          "The local manifold statistic detected scattered single outliers almost perfectly (AP %s) but none of the other outlier shapes (at most %s), and it was weaker than the neighbor distance on rearranged regions, which is why it is not used as a channel."
          % (n(_tap(_FUS, _m1)), n(_tap(_FUS, _m2)), n(_tap(_CA, _m1)), n(_tap(_CA, _m2)), n(_ty[_m1][_CB]["rearranged"]), n(_ty[_m2][_CB]["rearranged"]), n(_ty[_m1][_RAR]["outliers"]), n(_ty[_m2][_RAR]["outliers"]),
             n(_ty[_m1][_CB]["outliers"]), n(_ty[_m2][_CB]["outliers"]), n(_ty[_m2][_RAR]["outliers"]), n(_ty[_m2][_CA]["outliers"]),
             _G1["dev"]["fitness"][_g1b], _G1["dev"]["fitness"][_g1a], _G1["test"]["fitness"][_g1b], _G1["test"]["fitness"][_g1a], n(_tap(_MAN, "scattered")), n(max(_tap(_MAN, k) for k in ("block3", "block6", "bar8", "irregular12", "boundary3"))))),
]

CHEM_ABSTRACT = (
    "Averaged order parameters say how ordered a lattice of orientational states is, but not where its organization is unusual. "
    "We extend the local order and entropy quantities of the MyUncle maintenance-lattice model to unit vectors in d dimensions with the von Mises-Fisher distribution (equal to MyUncle's at d = 2, tested against the unmodified code) "
    "and show, on simulated data, that two complementary label-free channels are needed to find unusual sites. "
    "Channel A is the rarity of a site, aggregated with weights given by the similarity of neighboring embeddings: on simulated lattices, synthetic images and a controlled generator it preserved scattered and bar-shaped anomalies that a fixed 3 x 3 mean destroyed (lower-tail average precision %s against %s on ten independent test scenarios, paired 95%% interval above zero). "
    "Channel B is a directional form of MyUncle's local disorder, the minimum of the disorder over the four quadrants of a site. It detects arrangement anomalies, the same embeddings in the wrong places: there the rarity stayed at chance (average precision %s) and channel B reached %s. "
    "A label-free tail index chooses the channel per cohort; on fifteen independent test scenarios this raised the lower-tail average precision from %s to %s (paired 95%% interval of the gain %s to %s) without loss on the scenarios with feature anomalies only. "
    "Order gates and a GaaR search over %d rule configurations did not help, and cohorts with both kinds of anomaly need both maps. All evidence is simulated; code and results are open."
    % (n(_F(_SIMW)), n(_F(_MEAN)), n(_gm(_RAR, _iarr)), n(_gm(_CB, _iarr)), n(_TFt[_CA]), n(_TFt[_FUS]), _sg(_TBt[_FUS]["diff_vs_A"]["ci"][0]), _sg(_TBt[_FUS]["diff_vs_A"]["ci"][1]), _OS["n_unique"]))

CHEM_INTRO = [
    ("p", "Lattice systems whose sites carry an orientation or a state vector, such as liquid-crystal textures, degrading polymer films, magnetic media and self-assembled layers, are commonly summarized by averaged order parameters. These say how ordered a system is, not where something unusual is. "
          "Finding unusual local organization without labels, and saying how unusual it is in the language of order and entropy, is the problem addressed here. Unsupervised anomaly detection has been developed in other fields: in astronomy for large surveys [@Baron2017;@Lochner2021], and in medical imaging mostly as detection of deviation from a healthy appearance with generative models [@Schlegl2019]."),
    ("p", "MyUncle [@MyUncle2026] is a simulation framework for a maintenance lattice: rotors on a square lattice whose order is eroded by a state-dependent flux and restored by repair, with an order-entropy bridge that links the length of the neighbors' mean unit vector to a von Mises entropy. "
          "This paper makes seven contributions. (1) It extends the local order and entropy quantities from the plane to unit vectors in d dimensions (the von Mises-Fisher distribution) and verifies that they equal MyUncle's at d = 2 against the unmodified code. "
          "(2) It builds a label-free score that judges the rarity of the state of a site at several prevalence scales and smooths it over the lattice. "
          "(3) It tests the score, with frozen settings and baselines, first on lattices simulated with MyUncle's kinetic Monte Carlo (damaged sites as ground truth; one development and three held-out model parameter sets) and second on synthetic stained-tissue images, where the lattice sites are image patches described by an embedding; the results are mixed and are reported as such. "
          "(4) It studies a supervised configuration search with GaaR [@GaaR2026] on fresh test scenarios that no search saw. (5) It tests, on independent scenarios with paired intervals, aggregation of the scores weighted by embedding similarity and gates on local order and on the von Mises-Fisher information (Section 4.8), which preserved scattered and bar-shaped anomalies that a fixed window destroys, while the entropy gate added nothing. (6) It shows that anomalies of arrangement, to which a rarity score is blind, are detected by a directional form of the local disorder, and that a label-free tail index chooses between the two channels per cohort (Section 4.10). (7) It provides an open, containerized pipeline for all of it, archived with the vendored MyUncle core [@MyUncleVision2026]. All evidence is computational and simulated: no micrograph of a real material and no real tissue image was analysed, and the application to experimental data is stated as future work (Section 5)."),
]


_ROWS6 = [[lab, n(_GT["dev"][k]["fitness"])] + [n(x) for x in _GT["test"][k]["aps"]] + ["%s [%s, %s]" % (n(_TF(k)), _f2(_BT[k]["fitness_ci"][0]), _f2(_BT[k]["fitness_ci"][1])), "-" if k == _DEF else "%+.2f %s" % (_dd(k), _dci(_BT[k]["diff_vs_default_ci"]))] for k, lab in ((_DEF, "Frozen default"), (_NOS, "No smoothing"), (_RST, "GaaR restart / population champion"), (_GRD, "GaaR greedy / random-search champion"), (_OPT, "Exhaustive development optimum"))]
CHEM_GAAR = [
    ("h2", "4.6 Configuration search with GaaR"),
    ("p", "The frozen configuration is one choice for all data, and the results above show that it is not best for every embedding. GaaR (Goal-Aware Adaptive Regulation) [@GaaR2026], an evolutionary search framework by the same author, was used to select a configuration of the score from %d candidates: smoothing type and window (none, mean 3, 5, 7, Gaussian 5, 7, median 3, 5), five sets of axis weights, the PCA dimension (8, 16, 32) and the largest rarity scale (2, 5, 10%%). "
          "This is supervised hyperparameter selection on labeled development scenarios, not label-free adaptation: the chosen configuration is fixed afterwards. The fitness is a lower-tail average precision, F = 0.5 mean(AP) + 0.5 min(AP), over four development scenarios (the development images with each embedding and the development lattices with each of their two embeddings). "
          "All %d configurations were evaluated once on these scenarios, so any search can be compared exactly with the optimum; each strategy of GaaR and plain random search then ran %d times with a budget of %d evaluations. The champions were evaluated once on fresh test scenarios that no search saw: 16 new synthetic images (new seeds, the held-out anomaly types) and 48 new lattice states, including a preset not used before (superconducting scaffold)."
          % (_GS["n_configs"], _GS["n_configs"], _GS["repeats"], _GS["budget"])),
    ("table", "Table 5. Fresh test scenarios: average precision (AP) of the frozen configuration, of no smoothing and of the champions found on the development scenarios (development fitness F in column 2; test AP for handcrafted-feature images, DINOv2 images, and lattices with the rotor and the rotor + D embedding in columns 3-6; test fitness with a 95% interval in column 7 and its paired difference from the default in column 8; intervals resample the 16 images and the 16 simulation runs of each scenario).",
     ["Configuration", "F dev", "Img. hand.", "Img. DINOv2", "Lat. rotor", "Lat. rotor+D", "F test [95% interval]", "Difference from default [95% interval]"], _ROWS6, [1.9, 0.5, 0.55, 0.65, 0.55, 0.65, 1.1, 1.35], True, 12),
    ("p", "The default configuration ranked %dth of %d on the development fitness (%s; optimum %s). With %d evaluations, the greedy and restart strategies found a configuration in the top 1%% in %s and %s of the runs and random search in %s (population: %s), whereas at the top 5%% level random search did as well or better (%s against %s, %s and %s). "
          "On the test scenarios (Table 5) the champion of the restart and population strategies raised the test fitness from %s (default) to %s, mainly by lifting the worst scenario (handcrafted-feature images, AP %s to %s), while the AP with DINOv2 fell from %s to %s. The paired 95%% interval of this gain (%+.2f) is %s and the champion was better than the default in %s of the resamples, so with 16 independent units per scenario it cannot be distinguished from zero. The champion of the greedy strategy and of random search (the same configuration here) reached %s, and the exhaustive development optimum %s: every development-optimized configuration beat the default, but their order did not follow the development fitness. "
          "The gain is therefore due to the selected configuration (a smaller PCA dimension and the disorder axis at weight 0.5, with 3 x 3 mean smoothing) and not to the search algorithm; each strategy contributed one champion, four scenarios are few, and the test data come from the same generators as the development data."
          % (int(round(_GS["default_rank_fraction"] * _GS["n_configs"])) + 1, _GS["n_configs"], n(_GS["default_fitness"]), n(_GS["optimum"]), _GS["budget"], pc(_st["greedy"]["hit_top1pct"]), pc(_st["restart"]["hit_top1pct"]), pc(_st["random"]["hit_top1pct"]), pc(_st["population"]["hit_top1pct"]),
             pc(_st["random"]["hit_top5pct"]), pc(_st["greedy"]["hit_top5pct"]), pc(_st["restart"]["hit_top5pct"]), pc(_st["population"]["hit_top5pct"]),
             n(_TF(_DEF)), n(_TF(_RST)), n(_GT["test"][_DEF]["aps"][0]), n(_GT["test"][_RST]["aps"][0]), n(_GT["test"][_DEF]["aps"][1]), n(_GT["test"][_RST]["aps"][1]), _dd(_RST), _dci(_BT[_RST]["diff_vs_default_ci"]), pc(_BT[_RST]["share_better_than_default"]), n(_TF(_GRD)), n(_TF(_OPT)))),
    ("h2", "4.7 Smoothing against anomaly morphology"),
    ("p", "The results above leave open when smoothing helps. A controlled study on an independent generator varied only the spatial size of the anomalies. Each of %d lattices (32 x 32 sites, 16-dimensional embeddings) has smooth background domains and 3%% anomalous sites with a fixed separability (a shift of 10 noise standard deviations, calibrated beforehand on the unsmoothed score only); the anomalies are b x b blocks, b = 1 being scattered single sites and b = 6 compact regions, and each block has its own direction, so no recurring state exists. "
          "Without smoothing the average precision was between %s and %s across sizes; the frozen 3 x 3 mean lowered it to %s at b = 1 and %s at b = 6 (Table 6). Smoothing hurt at every size, the loss shrinking as the anomalies grew, and a 5 x 5 mean was worse than the 3 x 3 mean at every size (paired 95%% intervals of all these differences exclude zero). "
          "The champion of the GaaR search was better than the frozen configuration for b up to 3 and not distinguishable from it for b = 4 and 6, but it stayed below no smoothing at every size: configuration selection on the development scenarios, in which smoothing helped the DINOv2 images, mitigated the loss without removing it."
          % (_MS["images_per_cohort"] * _MS["cohorts"], "%.2f" % min(_mm[b]["no smoothing"]["ap"] for b in _sz), "%.2f" % max(_mm[b]["no smoothing"]["ap"] for b in _sz), "%.2f" % _mm["1"]["frozen (mean 3)"]["ap"], "%.2f" % _mm["6"]["frozen (mean 3)"]["ap"])),
    ("table", "Table 6. Average precision against anomaly block size b (independent embedding-lattice generator; 3% anomalous sites, fixed separability).",
     ["Configuration", "b = 1", "b = 2", "b = 3", "b = 4", "b = 6"],
     [[lab] + ["%.2f" % _mm[b][cfg]["ap"] for b in _sz] for cfg, lab in (("no smoothing", "No smoothing"), ("frozen (mean 3)", "Frozen (3 x 3 mean)"), ("mean 5", "5 x 5 mean"), ("GaaR restart champion", "GaaR restart champion"), ("exhaustive development optimum", "Exhaustive development optimum"))],
     [2.6, 0.7, 0.7, 0.7, 0.7, 0.7], True, 14),
    ("p", "A label-free alternative was explored. Moran's I of the unsmoothed rarity score (spatial autocorrelation of the score field) was below 0.02 in exactly the two lattice scenarios, one development and one test, in which smoothing lost most of the AP, and above 0.1 in the other six. A rule that smooths only above 0.05 would have avoided the largest loss in both, but it would not have helped the handcrafted-feature test images (eight scenarios, one threshold, exploratory)."),
]


CHEM_OG = [
    ("h2", "4.8 Spatial aggregation rules and an independent test"),
    ("p", "Six aggregation rules turn the rarity score r into the final score s on the lattice (none; fixed w x w mean; similarity-weighted aggregation, which weights each neighbor by exp(-(1 - u_i . u_j)/tau) with tau the median of 1 - cos over lattice-adjacent pairs of the cohort, a label-free scale; a one-sided rule, which lets neighbors add evidence but never remove it; a nucleation-and-growth rule with hysteresis, in which a site whose rarity exceeds a nucleation threshold seeds and weaker neighbors are raised to the seed's score weighted by their compatibility, never lowered; and gates on the compatibility, on the resultant length R_i and on the von Mises-Fisher information H_chi of the neighbors, with an optional persistence factor over the windows 1, 3, 5 and 7). "
          "Similarity-weighted filtering is an established kind of edge-preserving filter; no novelty is claimed for it. H_chi is a monotone function of R_i, so a gate on the rank of H_chi is the same as a gate on the rank of R_i. "
          "The rarity score is computed once per scenario; the rule parameters (460 distinct configurations) were selected on ten development scenarios (the four of Sections 4.1-4.6 and six shapes of the controlled generator: scattered sites, 3 x 3 and 6 x 6 blocks, bars of 8, irregular 12-site clusters and 3 x 3 blocks on domain boundaries) by F = 0.5 mean(AP) + 0.5 min(AP), and the champions were evaluated once on ten independent test scenarios (new seeds, new images and lattices, a MyUncle preset not used before). Selecting rule parameters uses development labels; the rules are label-free."),
    ("table", "Table 7. Independent test scenarios: average precision of aggregation rules (images with handcrafted features and with DINOv2; MyUncle lattices with the rotor and the rotor + D embedding; controlled generator with scattered sites, 3 x 3 blocks, 6 x 6 blocks, bars, irregular clusters and boundary blocks) and the lower-tail fitness F with its 95% interval.",
     ["Rule", "Hand.", "DINO", "Rot.", "Rot.+D", "Scat.", "Blk3", "Blk6", "Bar", "Irr.", "Bnd.", "F [95% interval]"],
     [[lab] + ["%.2f" % _ap(c, k) for k in ("H", "D", "R", "RD", "SC", "B3", "B6", "BAR", "IRR", "BND")] + ["%.2f [%.2f, %.2f]" % (_F(c), _BTO[c]["fitness_ci"][0], _BTO[c]["fitness_ci"][1])]
      for c, lab in ((_NONE, "No smoothing"), (_MEAN, "Fixed 3 x 3 mean (frozen)"), (_SIMW, "Similarity-weighted 3 x 3 (untuned)"), (_GAR, "GaaR / random-search champion"), (_ONE, "One-sided (dev-selected)"), (_OGC, "Order gate, compatibility"), (_OGV, "Order gate, R_i or H_chi"))],
     [1.9, 0.45, 0.45, 0.45, 0.5, 0.45, 0.45, 0.45, 0.4, 0.4, 0.4, 1.2], True, 10),
    ("p", "The fixed 3 x 3 mean lowered the average precision of scattered anomalies from %s to %s, of bars from %s to %s and of the rotor-embedding lattices from %s to %s, and helped only the DINOv2 images (%s to %s); in the lower-tail fitness it was below no smoothing (%s against %s; paired difference %s). "
          "The untuned similarity-weighted rule reached F = %s, %s above the fixed mean and %s above no smoothing; it kept the region recall of scattered anomalies at %s (fixed mean: %s) and of bars at %s (%s), but did not help the handcrafted-feature images (%s against %s without smoothing). "
          "The one-sided rule matched no smoothing (%s). The development-selected order gates reached F = %s (compatibility) and %s (resultant length and H_chi, which gave the same scores because H_chi is monotone in R_i), below the untuned similarity-weighted rule; the entropy extension is therefore a formal result that did not improve detection. "
          "Among the %d distinct configurations, random search found a configuration in the top 5%% in %s of %d runs of %d evaluations, the GaaR strategies in %s (greedy), %s (restart) and %s (population): GaaR was not better than random search here, because many configurations of the similarity-weighted family are nearly equivalent."
          % (n(_ap(_NONE, "SC")), n(_ap(_MEAN, "SC")), n(_ap(_NONE, "BAR")), n(_ap(_MEAN, "BAR")), n(_ap(_NONE, "R")), n(_ap(_MEAN, "R")), n(_ap(_NONE, "D")), n(_ap(_MEAN, "D")), n(_F(_MEAN)), n(_F(_NONE)), _dd2(_MEAN, _NONE),
             n(_F(_SIMW)), _dd2(_SIMW, _MEAN), _dd2(_SIMW, _NONE), n(_rec(_SIMW, "SC")), n(_rec(_MEAN, "SC")), n(_rec(_SIMW, "BAR")), n(_rec(_MEAN, "BAR")), n(_ap(_SIMW, "H")), n(_ap(_NONE, "H")),
             _dd2(_ONE, _NONE), n(_F(_OGC)), n(_F(_OGV)), _OS["n_unique"], _pcx(_so["random"]["hit_top5pct"]), _OS["repeats"], _OS["budget"], _pcx(_so["greedy"]["hit_top5pct"]), _pcx(_so["restart"]["hit_top5pct"]), _pcx(_so["population"]["hit_top5pct"]))),
]

_NGHK = "nucleation-growth (dev-selected)"
_rng = lambda c: max(_ap(c, k) for k in ("SC", "B3", "B6", "BAR", "IRR", "BND")) - min(_ap(c, k) for k in ("SC", "B3", "B6", "BAR", "IRR", "BND"))
assert abs(_NG["results"][_NGHK]["_fitness"] - _NG["results"][_NONE]["_fitness"]) < 0.01 and _NG["bootstrap"][_NGHK]["diff_vs_similarity-weighted"]["ci"][1] < 0
assert _CG["spearman_retention_vs_mean_effect"]["rho"] > 0.3 and _CG["test_mean_ap"]["retention_choice"] <= _CG["test_mean_ap"]["always_none"] + 0.01
assert _rng(_MEAN) > 3 * _rng(_SIMW)
CHEM_OG2 = [
    ("p", "Two further ideas were tested. The nucleation-and-growth rule, with its parameters selected on the development scenarios, reached F = %s on the test scenarios, the same as no smoothing (paired difference %s) and below the similarity-weighted rule (%s). "
          "A label-free closure gap, motivated by the failure of mean-field closures that average a steep function over a heterogeneous neighborhood, is the share of the cohort's top 3%% sites that are still in the top 3%% after the fixed mean (the retention); it correlated with the effect of the fixed mean on the average precision over the %d development and test cohorts (Spearman rho = %s, p = %s), but a choice between the fixed mean and no smoothing based on it, fitted on the development cohorts, selected the mean for %s of the test cohorts and reached a mean AP of %s against %s for no smoothing (oracle choice %s) and %s for similarity weighting, which makes the choice unnecessary. "
          "As a robustness criterion in the spirit of grid convergence, the range of the average precision over the six anomaly shapes was %s without smoothing, %s for the fixed mean and %s for the similarity-weighted rule."
          % (n(_NG["results"][_NGHK]["_fitness"]), _sg(_NG["results"][_NGHK]["_fitness"] - _NG["results"][_NONE]["_fitness"]) + " " + _ci2(_NG["bootstrap"][_NGHK]["diff_vs_none"]["ci"]), _sg(_NG["results"][_NGHK]["_fitness"] - _NG["results"][_SIMW]["_fitness"]) + " " + _ci2(_NG["bootstrap"][_NGHK]["diff_vs_similarity-weighted"]["ci"]),
             _CG["spearman_retention_vs_mean_effect"]["n"], n(_CG["spearman_retention_vs_mean_effect"]["rho"]), "%.4f" % _CG["spearman_retention_vs_mean_effect"]["p"], _pcx(_CG["share_chose_mean_test"]), n(_CG["test_mean_ap"]["retention_choice"]), n(_CG["test_mean_ap"]["always_none"]), n(_CG["test_mean_ap"]["oracle"]), n(_CG["test_mean_ap"]["always_simw"]),
             n(_rng(_NONE)), n(_rng(_MEAN)), n(_rng(_SIMW)))),
]

_SV = _json.load(open(os.path.join(RES, "og_survival.json")))
_SV5 = _json.load(open(os.path.join(RES, "og_survival_w5.json")))
_LF = _json.load(open(os.path.join(RES, "og_lfsurv.json")))
_lr = _LF["rows"]
assert _LF["lf_vs_ap_mean"]["rho"] > 0.5 and sum(x["ap_mean"] > x["ap_none"] for x in _lr) == 0 and all(x["ap_simw"] > x["ap_mean"] and x["ap_simw"] > x["ap_none"] for x in _lr)
_thv = sum(r["theta"] for r in _SV["rows"]) / len(_SV["rows"])
_kv = 9 * (_thv - _SV["r_n"]) / (_SV["r_a"] - _SV["r_n"])
assert _SV["spearman_all"]["rho"] > 0.7 and _SV["spearman_test"]["rho"] > 0.7 and _kv > 8 and _SV["by_shape"]["scattered"]["phi"] == 0 and _SV["by_shape"]["bar8"]["phi"] == 0
CHEM_SURV = [
    ("h2", "4.9 Contrast survival of the fixed window"),
    ("p", "A simple model explains why the fixed mean fails. Let anomalous sites have the mean rarity rank r_a and normal sites r_n (calibrated on the development cohorts: %s and %s; the ranks saturate near 1 because the highest of five scales counts). After the w x w mean, a site with k anomalous among its n window sites scores (k r_a + (n - k) r_n)/n, and it stays in the top 3%% of the cohort (threshold theta read from the smoothed scores, label-free, %s on average) only if k >= k* = n (theta - r_n)/(r_a - r_n); here k* is about %.1f of 9 sites, so practically the whole window must be anomalous. "
          "The fixed mean therefore preserves only the interior of an anomaly and erases anomalies thinner than the window; for a square block of side b the surviving share is ((b - w + 1)/b)^2, and for scattered sites and bars it is zero. The predicted share of surviving anomalous sites, computed from the true masks, correlated with the measured average precision of the fixed mean over the %d cohorts of the controlled generator (Spearman rho = %s; test cohorts only %s; 5 x 5 window %s). "
          "The model is explanatory (r_a and r_n were calibrated with labels on the development cohorts and it describes the fixed mean only); it also explains why similarity weighting works, since neighbors with dissimilar embeddings receive small weights."
          % (n(_SV["r_a"]), n(_SV["r_n"]), n(_thv), _kv, _SV["spearman_all"]["n"], n(_SV["spearman_all"]["rho"]), n(_SV["spearman_test"]["rho"]), n(_SV5["spearman_all"]["rho"]))),
    ("p", "A label-free version replaces the masks by the score field itself: the candidates are the sites of the top 3%% of the unsmoothed rarity, and a candidate survives when all sites of its 3 x 3 window lie in the top 9%%. It correlated with the measured average precision of the fixed mean over the %d cohorts (Spearman rho = %s; test cohorts only %s) and with the label-based prediction (rho = %s). "
          "It tracks the loss of the fixed mean only: it correlated negatively with the average precision of no smoothing and of similarity weighting (%s and %s). In this generator the fixed mean did not beat no smoothing in any cohort, while similarity weighting beat both in all %d (mean AP %s without smoothing, %s for the fixed mean, %s for similarity weighting), so a choice between the fixed mean and no smoothing based on the predictor is unnecessary once similarity weighting is used."
          % (_SV["spearman_all"]["n"], n(_LF["lf_vs_ap_mean"]["rho"]), n(_LF["lf_vs_ap_mean_test"]["rho"]), n(_LF["lf_vs_label_phi"]["rho"]), n(_LF["lf_vs_ap_none"]["rho"]), n(_LF["lf_vs_ap_simw"]["rho"]), _SV["spearman_all"]["n"], n(_LF["mean_ap"]["none"]), n(_LF["mean_ap"]["mean"]), n(_LF["mean_ap"]["simw"]))),
    ("table", "Table 8. Contrast survival (window 3 x 3): predicted share of anomalous sites that survive the fixed mean and the measured average precision (AP) of the fixed mean (development and test cohorts).",
     ["Anomaly", "Predicted survival", "AP, fixed mean"],
     [[lab, "%.2f" % _SV["by_shape"][k2]["phi"], "%.2f" % _SV["by_shape"][k2]["ap_mean"]] for k2, lab in (("scattered", "Scattered sites"), ("block3", "3 x 3 blocks"), ("block6", "6 x 6 blocks"), ("bar8", "Bars of 8"), ("irregular12", "Irregular clusters (12)"), ("boundary3", "Boundary blocks 3 x 3"))],
     [2.6, 1.3, 1.3], True, 14),
]

CHEM_FUTURE = [
    ("h1", "5. Future work: validation on experimental micrographs and histology"),
    ("p", "The evidence in this paper is simulated: the lattices were generated with MyUncle by the author of the method, and the images were generated by the author with simple textures and sharply bounded anomalies. They cannot show that the method finds anything of physical, chemical or biological meaning in an experimental system. "
          "Validation on experimental data is therefore the principal next step, and the study is specified here in advance so that its design cannot be shaped by its results. It has not been started."),
    ("h2", "5.1 Data and design"),
    ("p", "Two kinds of data are planned. The first is micrographs of ordered materials whose defect structure can be established independently, such as polarized-light textures of liquid crystals or microscopy of block-copolymer and self-assembled films, with defects annotated by at least two experts or by an independent measurement. "
          "The second is publicly released whole-slide images of tissue, for which the imaging pipeline of Section 2 was built. For each kind a pilot of 10 to 20 images will be used to inspect the quality-control exclusions and the maps and to fix the settings. "
          "The settings, the list of baselines and the analysis plan will then be frozen and published in the repository before an independent set of images from other samples, instruments and operators is analysed once, following the procedure used here for the held-out cohorts."),
    ("h2", "5.2 Comparators and endpoints"),
    ("p", "The frozen score will be compared with the baselines of Section 3 (isolation forest, local outlier factor, nearest-neighbor and Mahalanobis distances, raw and smoothed) and with other spatial postprocessing (Gaussian and median filters, other window sizes). "
          "Primary endpoint: the share of the top-ranked regions that at least two experts, blinded to the method and to the ranking, judge to be a defect or an unexplained feature rather than a known feature or an artifact, with interrater agreement (Cohen's kappa) and bootstrap confidence intervals over images, compared between methods by paired resampling. "
          "Secondary endpoints: the recall of annotated defects at a fixed review budget; the fraction of artifact regions among the top-ranked ones; the stability of the flagged set under another seed, reference sample, embedding and patch size (Jaccard overlap); computing time. "
          "The success criteria will be written into the analysis plan before the independent set is analysed."),
    ("h2", "5.3 Further planned work"),
    ("p", "A second synthetic generator with different textures, anomaly boundaries and artifacts, evaluated once with the frozen settings; bootstrap intervals for the average precision and for the differences from the Mahalanobis baseline, with more planted instances than the seven of the present held-out image cohort and more than four simulation runs per parameter set; "
          "a direct comparison of the lattice smoothing with Gaussian and median filtering and other neighborhood sizes; smoothing of the raw score instead of its rank; fitting the MyUncle parameter sets to measured data, which would turn the labels used here for the parameter sets into calibrated models; "
          "stain normalization [@Macenko2009] and a pathology-pretrained encoder [@Chen2024;@Vorontsov2024] for tissue images. If the experimental study shows no advantage over the baselines, that result will be reported."),
]

CHEM_DISCUSSION = [
    ("p", "What the evaluation supports. The order and entropy definitions are MyUncle's, verified at d = 2, and extend to unit vectors in d dimensions with the von Mises-Fisher distribution. The lattice smoothing gives spatially coherent candidate regions for any scorer when the anomalies form regions, "
          "and it is harmful when they do not: on the simulated lattices the damaged sites are scattered and smoothing lowered the average precision of every baseline, whereas on the synthetic images it raised the spatial coherence of every scorer. The multi-scale rarity removes the dependence on a prevalence chosen in advance, and a supervised search with GaaR improved the lower-tail average precision on fresh test scenarios, although random search also beat the default and the gain therefore belongs to the selected configuration, "
          "although the damaged share of three of the four parameter sets (about 10%, 30% and 26%; the fourth is 8%) is near or above the largest prevalence scale of the rarity score (10% of the reference set), a limit of the frozen settings that the simulated testbed exposes."),
    ("p", "Local order turned out to be useful as a channel of its own rather than as a gate on the rarity: regions in which ordinary embeddings are rearranged are invisible to the rarity and are found by the directional disorder, and a label-free tail index chose the appropriate channel in the test cohorts (Section 4.10). "
          "The simple distance of a site to its neighbors finds such regions equally well; the directional form matters because it does not respond to walls between ordered domains, which is what lets the two channels be told apart without labels."),
    ("p", "What it does not support. The lattices and the images are simulated, written by the author of the method (the simulation is MyUncle's, with an easy kind of anomaly, a site that lost its coupling, and with model parameter sets that carry application names but are not fitted to measurements); nothing here shows that the method finds anything in an experimental material or in tissue, or that what it finds is meaningful. "
          "The held-out sets were used once but are of the same family as the development sets, and four simulation runs per parameter set are few. The two embeddings of the images disagree on whether smoothing helps and on whether the lattice axes help, and a single Gaussian model scored higher than the frozen score on the development cohort. "
          "A permutation is an idealized arrangement anomaly that changes nothing but the arrangement, the choice of the channel is made per cohort and cannot serve a cohort that contains both kinds of anomaly, and channel B needs a background that is locally ordered. Novelty is not a defect: a rare state may be a rare sample preparation, an instrument, a staining or imaging batch, or a known feature that is merely uncommon in the set. The image embedding is trained on natural images and may group patches by stain or sharpness, and quality control flagged only part of the planted artifact patches in the development runs (42%)."),
    ("p", "Improvements: neighborhoods larger than four sites, a calibrated null for the novelty score, smoothing of the raw score instead of its rank, an embedding trained on micrographs of the system at hand, and active expert feedback in the manner of anomaly detection systems for astronomy [@Lochner2021]. "
          "Computing: the encoder dominates; an image of ten thousand patches takes about ten minutes on one CPU core and minutes on a GPU, and the pipeline offers a patch budget per image and a resumable cache."),
]

CHEM_CONCLUSION = [
    ("p", "The local disorder and the von Mises-Fisher order information of MyUncle generalize to lattices of unit vectors in d dimensions and equal MyUncle's at d = 2. A label-free score built on them, rarity of the state of a site at several prevalence scales smoothed over the lattice, was tested on lattices simulated with MyUncle and on synthetic stained-tissue images, with frozen settings and four baselines. "
          "The results are mixed: smoothing helped only with the DINOv2 embeddings of the synthetic images, whose unsmoothed score is noisy, and hurt on simulated lattices and in a controlled study on an independent generator, least for compact anomalies; adding the local disorder as a feature helped on the simulated lattices, the frozen score was competitive but not uniformly best, and a simple Mahalanobis baseline was a strong competitor. "
          "A supervised search of the scoring configuration with GaaR gave a gain whose paired interval included zero. Similarity-weighted aggregation of the scores raised the lower-tail average precision on independent test scenarios above both the fixed window and no smoothing (paired intervals above zero), gates on local order and on the von Mises-Fisher information did not beat ordinary similarity, and a GaaR search over rule configurations was not better than random search. Rarity and local order are complementary channels: the directional local disorder found regions of rearranged embeddings to which the rarity was blind, and a label-free tail index that chooses the channel per cohort raised the lower-tail average precision over fifteen independent test scenarios without loss on the scenarios that contain only feature anomalies; cohorts with both kinds of anomaly still need both maps. All evidence is simulated; validation on experimental micrographs and on histology, with expert review, is the planned next step (Section 5)."),
]


def chemrxiv_blocks(B):
    """return the ChemRxiv block list derived from the journal block list B"""
    out = list(B)

    def find(kind, prefix, start=0):
        """index in the CURRENT block list (every replacement below changes its length, so positions must not be taken from the original list)"""
        hits = [i for i in range(start, len(out)) if out[i][0] == kind and isinstance(out[i][1], str) and out[i][1].startswith(prefix)]
        if len(hits) != 1:
            raise KeyError("%s: %d matches" % (prefix, len(hits)))
        return hits[0]

    # title, abstract, keywords
    out[find("title", "MyUncleVision")] = ("title", CHEM_TITLE)
    out[find("p0", "Unsupervised discovery in whole-slide")] = ("p0", CHEM_ABSTRACT)
    out[find("p0", "Keywords:")] = ("p0", CHEM_KEYWORDS)
    # introduction
    i = find("p", "Digital pathology produces")
    out[i:i + 2] = CHEM_INTRO
    # headings and small framing edits
    def sub(kind, old, new):
        for j in range(len(out)):
            if out[j][0] == kind and isinstance(out[j][1], str) and old in out[j][1]:
                out[j] = (kind,) + tuple(x.replace(old, new) if isinstance(x, str) else x for x in out[j][1:])
                return
        raise KeyError(old)

    sub("h1", "2. Method", "2. Theory and method")
    sub("h2", "2.1 Slides, patches and quality control", "2.1 Imaging application: images, patches and quality control")
    sub("h2", "2.3 Order and entropy of the tissue lattice", "2.3 Order and entropy of a lattice of unit vectors in d dimensions")
    sub("p", "No real kidney image was available for this work.", "For the imaging application no real image of tissue or of a material was available for this work; the lattice simulations of Section 4.5 are a separate testbed.")
    sub("h1", "4. Results on synthetic slides", "4. Results on synthetic images and simulated lattices")
    sub("h2", "4.5 A second testbed: lattices simulated with MyUncle", "4.5 Simulated maintenance lattices (MyUncle)")
    sub("p", "The development set was one preset", "The presets are model parameter sets that MyUncle labels with application names; they are not fitted to measurements of any material. The development set was one preset")
    # future work, discussion, conclusions
    i, j = find("h1", "5. Future work"), find("h1", "6. Discussion")
    out[i:j] = CHEM_GAAR + CHEM_OG + CHEM_OG2 + CHEM_SURV + CHEM_TC + CHEM_FUTURE
    i, j = find("p", "What the evaluation supports."), find("h1", "7. Conclusions")
    out[i:j] = CHEM_DISCUSSION
    i = find("p", "MyUncleVision turns a whole-slide image")
    out[i:i + 1] = CHEM_CONCLUSION
    sub("p0", "No patient data were used; the KPMP images are not part of this work.", "No experimental, patient or tissue data were used.")
    ia = find("small", "Corresponding author")
    k = find("p", "With the rotor alone")
    out[k + 1:k + 1] = [("p", "Fig. @F:maps@ shows one simulated run (preset chiral scaffold, 32 x 32 sites, snapshots 15 steps apart): the rotor angle, the local disorder D, the order information H_chi and the novelty score of the frozen method. Novelty is a percentile rank relative to the snapshots analysed together, so it marks where a state is unusual among the snapshots shown; it has not been validated against known defect locations."),
                        ("fig", "fig_bridge_maps.png", "Fig. @F:maps@. A MyUncle simulation analysed with MyUncleVision (command python -m myuncle_vision bridge, preset chiral scaffold, 32 x 32 sites, seed 0). Rows: rotor angle, local disorder D, order information H_chi (von Mises-Fisher, d = 2) and novelty of the frozen method; columns: snapshots at steps 0, 15, 30, 45 and 60.", 6.4)]
    out.insert(ia + 1, ("small", "Under consideration at Pattern Recognition Letters (submission PRLETTERS-D-26-02906)"))
    heads = [x[1] for x in out if x[0] in ("h1", "h2")]
    order = ["4.6 Configuration search with GaaR", "4.7 Smoothing against anomaly morphology", "4.8 Spatial aggregation rules and an independent test", "4.9 Contrast survival of the fixed window",
             "4.10 Arrangement anomalies: local order as a second channel", "5. Future work: validation on experimental micrographs and histology"]
    pos = [heads.index(h) for h in order]
    assert pos == sorted(pos) and pos[-1] - pos[0] == len(order) - 1, "section order"
    k6, k7 = find("h1", "6. Discussion"), find("h1", "7. Conclusions")
    assert out[k6 + 1][1].startswith("What the evaluation supports.") and out[k7 + 1][1].startswith("The local disorder and the von Mises-Fisher"), "discussion and conclusions in place"
    return out
