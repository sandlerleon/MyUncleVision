# -*- coding: utf-8 -*-
"""Knowledge-Based Systems manuscript: main.tex from intro.tex, framework.tex and the results files (every number is read from results/kbs*.json; qualitative claims are guarded).

    python docs/kbs/build_kbs_tex.py    -> docs/kbs/main.tex, compiled with pdflatex (+ bibtex-free: the bibliography is written inline)
"""
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
J = lambda n: json.load(open(os.path.join(RES, n)))
PH, DV, DV1, FZ, C1 = J("kbs_phase.json"), J("kbs_dev.json"), J("kbs_dev_run1.json"), J("kbs_frozen.json"), J("kbs_confirm.json")
EX, FZ2, C2, PR, PR2, RB = J("kbs2_explore.json"), J("kbs2_frozen.json"), J("kbs2_confirm.json"), J("kbs_props.json"), J("kbs_props2.json"), J("kbs_robust.json")
f2 = lambda x: "%.2f" % x
f3 = lambda x: "%.3f" % x
sg = lambda x: ("$-$%.2f" % -x) if x < -0.005 else (("+%.2f" % x) if x >= 0.005 else "0.00")
ci = lambda t: "%s [%s, %s]" % (sg(t["mean"]), sg(t["ci"][0]), sg(t["ci"][1]))
A_, B_ = "channel A (similarity-weighted rarity)", "channel B (directional disorder)"
MX, MXRB, SEL = "fusion: max", "fusion: max (rarity, B)", "fusion: cohort selection (tail index)"
IFS = "fusion: max (isolation forest, Shekhar)"
CH2 = FZ2["chosen"]
SH, LOF, IF, CLOF, ORLS = "spatial outlier statistic (Shekhar)", "LOF (embeddings)", "isolation forest (embeddings)", "contextual LOF ([u, neighbor mean])", "OR(LOF, Shekhar)"

# ------------------------------------------------------------------------------------------------ numbers and guards
grid = PH["grid"]
singles_worst_max = max(grid[k][d]["worst"] for k in grid for d in (A_, B_, "rarity", "center-to-neighbor distance", SEL))
fused_worst = [grid[k][MX]["worst"] for k in grid]
assert singles_worst_max < 0.2 and max(fused_worst) > 0.5
coh = PH["coherence"]
assert coh["corr=1"][B_]["arrangement"] > coh["corr=6"][B_]["arrangement"] and coh["corr=1"][A_]["feature"] > coh["corr=6"][A_]["feature"]
dc = DV["conditions"]
crit = DV["criterion_mean_worst_type_AP"]
assert DV["chosen"] == MXRB and crit[MX] > crit["fusion: OR"] and crit[MX] > crit["fusion: lcOR (locally calibrated B)"] and crit[MX] > crit["fusion: relmax (reliability-weighted B)"]
assert all(DV["tests_vs_best_simple"][k]["ci"][1] < 0 for k in DV["tests_vs_best_simple"])
dev_single = {k: float(np.mean([dc[c]["summary"][k]["worst"] for c in dc])) for k in (A_, B_, "rarity", SEL, IF, SH, LOF, CLOF, ORLS)}
dev_if = float(np.mean([dc[c]["summary"][IF]["feature"] for c in dc]))
dev_A = float(np.mean([dc[c]["summary"][A_]["feature"] for c in dc]))
dev_r = float(np.mean([dc[c]["summary"]["rarity"]["feature"] for c in dc]))
dev_sh = float(np.mean([dc[c]["summary"][SH]["arrangement"] for c in dc]))
dev_B = float(np.mean([dc[c]["summary"][B_]["arrangement"] for c in dc]))
assert dev_if > dev_A and dev_r > dev_A and dev_sh > dev_B
dev_ifs = float(np.mean([dc[c]["summary"][IFS]["worst"] for c in dc]))
assert dev_ifs > crit[MXRB]
P1s = C1["pooled_summary"]
t1 = C1["pooled_tests"]
assert P1s[MXRB]["worst"] < 0.06 and P1s["rarity"]["feature"] < 0.05 and P1s[LOF]["feature"] > 0.5
assert t1[ORLS]["worst"]["ci"][1] < 0 and t1[IFS]["worst"]["ci"][1] < 0 and t1[CLOF]["worst"]["ci"][1] < 0
cr2 = EX["criterion"]
assert EX["chosen"] == CH2 and CH2.startswith("max(isolation forest") and "Shekhar" in CH2
assert cr2[CH2] > cr2["max(isolation forest (embeddings) | channel B (directional disorder))"] and cr2[CH2] > cr2["OR(isolation forest (embeddings) | spatial outlier statistic (Shekhar))"]
c2 = C2["conditions"]
g3s, g3r = c2["G3 oriented texture, shuffled regions"]["summary"], c2["G3 oriented texture, rotated patches"]["summary"]
g3t = C2["g3_pooled_tests"]
g2r, g1r = c2["G2 replication (fresh seeds), phi=40, delta=1"]["summary"], c2["G1 replication (fresh seeds), blocks 3x3, s=2"]["summary"]
G3P = C2["g3_pooled_summary"]
singles = ["cohort kNN rarity", "within-lattice kNN rarity", LOF, IF, B_, SH, "center-to-neighbor distance"]
assert all(g3t[s]["worst"]["ci"][0] > 0 for s in singles), "text: the frozen fusion beat every single channel on G3"
assert g3t[MXRB]["worst"]["ci"][0] > 0 and g3t[ORLS]["worst"]["ci"][0] > 0 and g3t[CLOF]["worst"]["ci"][0] < 0 < g3t[CLOF]["worst"]["ci"][1]
assert max(g3r[k]["arrangement"] for k in g3r) < 0.25
assert g2r[ORLS]["worst"] > g2r[CH2]["worst"]
p3 = PR["P3"]["share_of_top3pct_on_walls"]
assert p3[B_] < p3[SH] and p3[B_] < p3["center-to-neighbor distance"]
p5 = PR2["P5"]
assert all(p5[k]["interior"]["mean"] < 0.5 for k in p5) and p5[SH]["border"]["mean"] > 0.8 and p5[B_]["border"]["mean"] < 0.6
_p2 = PR2["P2b"]
_o = ["block 1", "blob 6", "blob 12", "block 3", "block 6"]
assert all(_p2[a]["measured_survival_mean"] <= _p2[b]["measured_survival_mean"] for a, b in zip(_o, _o[1:])) and all(_p2[k]["predicted_survival"] <= _p2[k]["measured_survival_mean"] for k in _p2)
p4 = PR2["P4"]
assert p4["max fusion"]["TPR_f"] >= max(p4["isolation forest"]["TPR_f"], p4["Shekhar"]["TPR_f"]) - 1e-9 and p4["max fusion"]["FPR"] <= p4["isolation forest"]["FPR"] + p4["Shekhar"]["FPR"] + 1e-9

# ------------------------------------------------------------------------------------------------ tables
def row(name, s, keys=("feature", "arrangement", "all", "worst")):
    return name + " & " + " & ".join(f2(s[k]) for k in keys) + r" \\"


def table_dev():
    names = [("Rarity (cohort, unsmoothed)", "rarity"), ("Channel A (similarity-weighted rarity)", A_), ("Isolation forest", IF), ("LOF", LOF),
             ("Channel B (directional disorder)", B_), ("Shekhar statistic", SH), ("Contextual LOF", CLOF), ("OR(LOF, Shekhar)", ORLS),
             ("Cohort-level selection (tail index)", SEL), ("zmax of robust $z$", "fusion: zmax"), ("lcOR (locally calibrated $B$)", "fusion: lcOR (locally calibrated B)"),
             ("relmax (reliability-weighted $B$)", "fusion: relmax (reliability-weighted B)"), ("OR$(A,B)$", "fusion: OR"), ("max$(A,B)$", MX), ("max(rarity, $B$) [selected]", MXRB),
             ("max(isolation forest, Shekhar) [comparator]", IFS)]
    pooled = {k: {kk: float(np.mean([dc[c]["summary"][k][kk] for c in dc])) for kk in ("feature", "arrangement", "all", "worst")} for _, k in names}
    lines = [row(n, pooled[k]) for n, k in names]
    lines.insert(8, r"\midrule")
    lines.insert(15, r"\midrule")
    return "\n".join(lines)


def table_stage1():
    names = [("Rarity (cohort)", "rarity"), ("LOF", LOF), ("Isolation forest", IF), ("Channel B", B_), ("Shekhar statistic", SH), ("Contextual LOF", CLOF), ("OR(LOF, Shekhar)", ORLS),
             ("Cohort-level selection", SEL), ("max(isolation forest, Shekhar)", IFS), ("max(rarity, $B$) [frozen]", MXRB)]
    out = []
    for n, k in names:
        t = ("" if k == MXRB else (ci(t1[k]["worst"]) if k in t1 else "--"))
        out.append(n + " & " + " & ".join(f2(P1s[k][kk]) for kk in ("feature", "arrangement", "worst")) + " & " + t + r" \\")
    return "\n".join(out)


def table_stage2():
    names = [("Cohort kNN rarity", "cohort kNN rarity"), ("Within-lattice kNN rarity", "within-lattice kNN rarity"), ("LOF", LOF), ("Isolation forest", IF), ("Channel B", B_),
             ("Shekhar statistic", SH), ("Neighbor distance", "center-to-neighbor distance"), ("Contextual LOF", CLOF), ("OR(LOF, Shekhar)", ORLS), ("Cohort-level selection", SEL),
             ("max(rarity, $B$) [stage-1 rule]", MXRB), ("max(isolation forest, Shekhar) [frozen]", CH2)]
    out = []
    for n, k in names:
        cells = []
        for sm in (g3s, g3r):
            cells += [f2(sm[k]["feature"]), f2(sm[k]["arrangement"])]
        cells.append(f2(G3P[k]["worst"]))
        cells.append("" if k == CH2 else ci(g3t[k]["worst"]))
        out.append(n + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def table_robust():
    S = RB["settings"]
    out = []
    for lab, key in (("dimension", "dimension"), ("lattice size", "lattice size"), ("feature prevalence", "feature prevalence"), ("window", "window"),
                     ("coherence ($\\kappa$ scale, G2)", "coherence (G2 kappa scale)"), ("noise (G3)", "noise (G3)")):
        for v, d in S[key].items():
            sm = d["summary"]
            out.append("%s & %s & %s & %s & %s & %s & %.2f \\\\" % (lab if v == list(S[key])[0] else "", v, f2(sm[CH2]["worst"]), f2(sm[MXRB]["worst"]), f2(sm[ORLS]["worst"]),
                                                                     f2(sm[CLOF]["worst"]), d["timing"]["all_scores"]))
        out.append(r"\addlinespace")
    return "\n".join(out[:-1])


def table_p2():
    out = []
    for k, v in PR2["P2b"].items():
        nm = {"block 1": "single sites", "block 3": r"$3\times3$ blocks", "block 6": r"$6\times6$ blocks", "blob 6": "blobs of 6", "blob 12": "blobs of 12"}[k]
        out.append("%s & %s & %s & %s & %s & %s \\\\" % (nm, f2(v["p_star"]), f2(v["predicted_survival"]), f2(v["measured_survival_mean"]), f2(v["measured_survival_unsmoothed"]),
                                                        f2(v["measured_survival_similarity_weighted"])))
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ text
mean_g = lambda key, k: float(np.mean([dc[c]["summary"][k][key] for c in dc]))
rb = RB["settings"]
t_dim = {v: rb["dimension"][v]["timing"]["all_scores"] for v in rb["dimension"]}
t_n = {v: rb["lattice size"][v]["timing"]["all_scores"] for v in rb["lattice size"]}
_four = [CH2, MXRB, ORLS, CLOF]
_sets = [("dimension", v) for v in ("16", "64", "384")] + [(k, v) for k in ("feature prevalence", "window", "noise (G3)") for v in rb[k]]
_nbest = sum(rb[k][v]["summary"][CH2]["worst"] >= max(rb[k][v]["summary"][x]["worst"] for x in _four) - 0.01 for k, v in _sets)
_ntot = len(_sets)
assert all(rb["coherence (G2 kappa scale)"][v]["summary"][ORLS]["worst"] > rb["coherence (G2 kappa scale)"][v]["summary"][CH2]["worst"] for v in rb["coherence (G2 kappa scale)"])
assert rb["lattice size"]["32"]["summary"][CH2]["worst"] > rb["lattice size"]["64"]["summary"][CH2]["worst"] > rb["lattice size"]["128"]["summary"][CH2]["worst"]
assert rb["noise (G3)"]["0.16"]["summary"][CLOF]["worst"] > rb["noise (G3)"]["0.16"]["summary"][CH2]["worst"] + 0.01

ABSTRACT = (
    "Anomaly scores on lattices of embeddings, such as image patch embeddings, usually measure how rare the vector of a site is and cannot see ordinary vectors in the wrong "
    "places. We study when feature scores and spatial-order scores are complementary and how to combine them without labels. Five elementary propositions make the limits explicit: "
    "position-free scores are blind to rearrangements; a fixed aggregation window keeps an anomalous site only if its local anomaly occupancy exceeds a computable threshold; a "
    "directional dispersion statistic ignores straight walls between ordered domains; a site-wise maximum of calibrated ranks keeps every detection of either score at the cost of at "
    "most their combined false positives; and rotation-invariant local statistics cannot see the interior of a rotated patch. On synthetic lattices with both kinds of anomaly, every "
    "single score had a worst-type average precision of %s or less on an independent generator, where the site-wise maximum of a feature and a spatial score reached %s; on the "
    "development generator the maximum beat two locally adaptive rules, and choosing one score per cohort could not serve mixed lattices. The evaluation was pre-registered in two "
    "stages. The first frozen rule referenced rarity to the whole cohort and failed on its test generator (%s, against %s for a generic fused baseline), because the lattices did not "
    "share one normal distribution. A revised rule, frozen before a third generator was scored, beat every single score and three fused baselines there and tied a contextual LOF "
    "baseline. All data are synthetic." % (f2(max(G3P[s]["worst"] for s in singles)), f2(g3s[CH2]["worst"]), f3(P1s[MXRB]["worst"]), f2(P1s[ORLS]["worst"])))
assert len(ABSTRACT.split()) <= 250, len(ABSTRACT.split())

HIGHLIGHTS = [
    "Rarity and spatial-order scores detect different anomalies on embedding lattices",
    "Five propositions bound what each score, smoothing and fusion can detect",
    "Site-wise maximum fusion beat adaptive and cohort-level fusion on mixed lattices",
    "A pre-registered failure showed rarity needs a reference shared by the cohort",
    "Rotated patches of a texture remain undetected by every local statistic tested",
]
assert all(len(h) <= 85 for h in HIGHLIGHTS), [len(h) for h in HIGHLIGHTS]

DESIGN = r"""
\section{Experimental design}
\label{sec:design}
\subsection{Generators}
Table~\ref{tab:gen} summarizes the three generators. Every lattice has $32\times32$ sites and $d=16$ unless stated, and contains feature anomalies (label 1) and arrangement anomalies (label 2); a cohort is ten lattices of one condition.

\textbf{G1 (development).} A Gaussian random field with correlation length 3 (default) occupies the first $d/2$ dimensions, with independent noise (standard deviation 0.2) in all $d$; the normal embeddings therefore lie near a common subspace. A feature anomaly is a $3\times3$ block, a single site or a random-walk blob shifted by $s$ along a random unit direction of the complementary dimensions. An arrangement anomaly takes a fraction $\delta$ of the sites of a $6\times6$ block and permutes their embeddings without fixed points. Prevalences are 1.5\%% (feature) and 3.5\%% (arrangement). The strength levels $s\in\{1,1.5,2,2.5,3\}$ were set by a pilot on development seeds so that the unsmoothed rarity spans average precision (AP) %s to %s.

\textbf{G2 (stage-1 confirmation).} Sphere-native: 5--10 Voronoi domains per lattice, each with its own random mean direction, a linear drift of that direction across the domain, and its own von Mises--Fisher concentration (log-uniform between 40 and 400); the lattice therefore has domain walls and heterogeneous coherence. Feature anomalies are random-walk blobs of 1--12 sites drawn around a direction at angle $\phi$ from the local domain mean; arrangement anomalies exchange the embeddings of a fraction $\delta$ of the sites of two distant blobs of 16--32 sites and reshuffle them, so that the multiset of embeddings of the lattice is unchanged. Each lattice draws its own domain directions, so the lattices do \emph{not} share one normal distribution; this property was not recognized as decisive before the test (Section~\ref{sec:stage1}).

\textbf{G3 (stage-2 confirmation).} An oriented texture: a latent vector $[\cos\varphi,\sin\varphi,0.6\cos\psi,0.6\sin\psi]$, with a stripe phase $\varphi$ of wavelength 8 sites and slowly varying orientation and a slowly varying texture code $\psi\in[0,0.6]$, is mapped into $d$ dimensions by a random linear map \emph{shared by all lattices} (a fixed encoder), plus noise, and normalized. Feature anomalies are blobs whose texture code lies 1.2 outside the normal range; arrangement anomalies are either shuffled exchanges between two blobs, as in G2, or $6\times6$ patches rotated by $90^\circ$, in which every embedding and every local neighborhood stays ordinary.

\begin{table}[t]
\caption{The generators and their role.}
\label{tab:gen}
\centering\footnotesize
\begin{tabular}{p{1.1cm}p{3.6cm}p{3.6cm}p{3.6cm}p{2.3cm}}
\toprule
 & Background & Feature anomaly & Arrangement anomaly & Role\\
\midrule
G1 & Gaussian field in a $d/2$-dim.\ subspace; shared across lattices & shift out of the subspace (blocks, sites, blobs) & permutation inside $6\times6$ blocks & development\\
G2 & Voronoi domains, von Mises--Fisher, drift; own directions per lattice & blob drawn at angle $\phi$ from the local mean & exchange between two distant blobs & stage-1 test\\
G3 & oriented texture through a shared encoder & blob with an out-of-range texture code & exchange between blobs, or $90^\circ$ rotated patch & stage-2 test\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Fusion rules compared on the development generator}
Besides the maximum and the OR of Eq.~\eqref{eq:fusion}, five rules were compared: the cohort-level \emph{selection} of one channel by the 97th percentile of its robust $z$-score (median and MAD) \citep{Rousseeuw1993}, the fusion of the earlier version; \emph{zmax}, the site-wise maximum of the two robust $z$-scores; \emph{lcOR}, the OR rule after channel B has been standardized against the median and MAD of its own $15\times15$ neighborhood (a local calibration intended for heterogeneous backgrounds); \emph{relmax}, $\max(a_i,(1-g_i)b_i)$ with $g_i$ the cohort rank of the local median of $B$ (down-weighting channel B where the background itself is disordered); and $\max(r_i,b_i)$ with the unsmoothed rarity in place of channel A.

\subsection{Metrics}
For each lattice, the AP of a score is computed separately for feature anomalies against normal sites and for arrangement anomalies against normal sites (the other anomaly type excluded), and for all anomalies together \citep{Davis2006}. The per-type APs are averaged over the lattices of a condition, and the \emph{worst-type AP} is the smaller of the two averages; it is the primary endpoint, because a detector for mixed lattices must find both kinds. Uncertainty is a paired bootstrap over lattices (2000 resamples) of the difference between two detectors.

\subsection{Protocol}
\emph{Stage 1.} On G1 only: a pilot (strength levels), Experiment A (detectability map; 20 lattices per cell, $5\times5$ cells and a coherence axis), and Experiment B (four mixed conditions, 30 lattices each). The rule for choosing the fusion was declared in the code before Experiment B: the better of max$(A,B)$ and OR$(A,B)$ by the mean worst-type AP over the four conditions, replaced by an adaptive rule only if that rule was better and its paired 95\%% interval excluded zero. A first run of Experiment B showed the unsmoothed rarity above channel A, so max(rarity, $B$) was added as a candidate and the generic pair max(isolation forest, Shekhar) as a comparator, and Experiment B was run again; both runs are archived. The selected rule, the G2 conditions, seeds, endpoint and comparators, and the SHA-256 of the study code were then written to a freeze file and committed to version control (%s, tag \texttt{kbs-freeze}) before any G2 lattice was scored; the confirmation script refuses to run if the code differs from the frozen hash. Experiment C ran the frozen rule on three G2 conditions with 30 lattices each.

\emph{Stage 2.} After the stage-1 result, a post-hoc exploration on fresh seeds of the four G1 and the three G2 conditions (20 lattices each) compared all 24 pairs of a marginal score (cohort rarity, within-lattice rarity, LOF, isolation forest), a spatial score (channel B, Shekhar statistic, neighbor distance) and a fusion (max, OR); the selection criterion, the highest mean worst-type AP over the seven conditions, was committed before the exploration was run. The selected rule was frozen a second time (tag \texttt{kbs-freeze-2}, %s) together with generator G3, which had been written but never scored, and evaluated on two G3 conditions and on fresh seeds of one G2 and one G1 condition, 30 lattices each.

\emph{Experiment D} (descriptive, after both freezes) varies one factor at a time on G3 or G2 (dimension, lattice size, feature prevalence, window, background coherence, noise; 10 lattices per setting) and records the runtime per lattice on a desktop computer (8 cores).
""" % (f2(J("kbs_pilot.json")["s=1"]), f2(J("kbs_pilot.json")["s=3"]), FZ["frozen_at_utc"].replace("T", " ").replace("Z", " UTC"), FZ2["frozen_at_utc"].replace("T", " ").replace("Z", " UTC"))

g1max = max(grid, key=lambda k: grid[k][MX]["worst"])
RESULTS = r"""
\section{Results}
\label{sec:results}
\subsection{Checks of the analytical statements}
\emph{Blindness (Prop.~\ref{prop:perm}).} On lattices that contain only rearranged regions, the AUC of the cohort rarity for the rearranged sites was %s (G1) and %s (G2), and that of LOF %s and %s (30 lattices each): chance, as the proposition requires.

\emph{Occupancy (Prop.~\ref{prop:surv}).} Table~\ref{tab:p2} compares the predicted share of feature-anomaly sites that survive the fixed $3\times3$ mean at the cohort's top 3\%% with the measured share, for strong anomalies ($s=3$, mean anomalous rank %s--%s). The occupancy model reproduces the ordering of the shapes (single sites, then blobs, then blocks) but underestimates the measured survival, for example %s predicted against %s measured for $3\times3$ blocks, because the ranks of normal neighbors are spread rather than equal to their mean. Similarity weighting kept %s of the single sites and %s--%s of the larger shapes, close to the unsmoothed score. With weaker anomalies ($s=1.5$) the anomalous ranks are spread, the model predicts no survival at all, and 4--21\%% of the anomalous sites survived anyway; the model is a description of the strong-anomaly regime.

\begin{table}[t]
\caption{Survival of feature-anomaly sites at the cohort's top 3\%% (G1, $s=3$, 20 lattices per shape): threshold occupancy $p^{*}$ of Eq.~\eqref{eq:pstar}, predicted share of anomalous sites with occupancy $\ge p^{*}$, and measured shares above the top-3\%% threshold of each score.}
\label{tab:p2}
\centering\footnotesize
\begin{tabular}{lccccc}
\toprule
Shape & $p^{*}$ & Predicted (fixed mean) & Fixed mean & Unsmoothed & Similarity-weighted\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}

\emph{Walls (Prop.~\ref{prop:wall}).} On G2 lattices without anomalies, %s\%% of the sites lie on a domain wall. Among the top 3\%% of each statistic, the share of wall sites was %s\%% for channel B, against %s\%% for the isotropic disorder, %s\%% for the neighbor distance and %s\%% for the Shekhar statistic. The quadrant minimum removed about a third of the wall responses, as the proposition predicts for straight walls; the rest lie at corners, junctions of three domains and strongly drifting domains, where its condition fails.

\emph{Fusion cost (Prop.~\ref{prop:fuse}).} On G3 lattices with shuffled regions, at the top 3\%% of the cohort, the true-positive rate of max(isolation forest, Shekhar) for feature anomalies was %s (channels: %s and %s) and for arrangement anomalies %s (%s and %s); its false-positive rate was %s, against %s and %s for the channels, i.e.\ close to their sum.

\emph{Rotations (Prop.~\ref{prop:rot}).} On G3 lattices with rotated patches (30 lattices), the mean AUC of the Shekhar statistic was %s on the interior of the patches and %s on their border; for the neighbor distance %s and %s, and for contextual LOF %s and %s. No statistic did better than chance on the interior. Channel B did not see the border either (%s and %s): the seam of a rotated patch is a wall between two ordered regions, which the quadrant minimum ignores by construction (Prop.~\ref{prop:wall}).

\subsection{Experiment A: the detectability map}
Figure~\ref{fig:phase} shows the AP of each channel for its own anomaly type and the worst-type AP of max$(A,B)$ over feature strength and arrangement disruption on G1. Every single score has a worst-type AP of at most %s in every cell, because it is blind to the other type, whereas the maximum reaches up to %s. The feature AP of channel A is set by $s$ and hardly depends on $\delta$, and the arrangement AP of channel B grows with $\delta$ and is independent of $s$: the two channels respond to separate axes. A more coherent background helps neither: as the correlation length grows from 1 to 6 sites, the arrangement AP of channel B falls from %s to %s, because a shuffle among nearly identical vectors changes little, and the feature AP of channel A falls from %s to %s.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig_phase.png}
\caption{Experiment A (G1, mixed lattices with $3\times3$ feature blocks, 20 lattices per cell). (a) Feature-anomaly AP of channel A, (b) arrangement-anomaly AP of channel B, (c) worst-type AP of max$(A,B)$ over feature strength $s$ and arrangement disruption $\delta$. (d) Dependence on the correlation length of the background at $s=2$, $\delta=1$.}
\label{fig:phase}
\end{figure*}

\subsection{Experiment B: fusion rules on the development generator}
Table~\ref{tab:dev} lists the mean per-type APs over the four mixed G1 conditions (30 lattices each). Single scores reached at most %s in worst-type AP; the cohort-level selection, which picks one channel per cohort, did not improve on the channel it picked (%s). Among the fusion rules, the site-wise maximum was best: max(rarity, $B$) %s, max$(A,B)$ %s, OR$(A,B)$ %s. Both adaptive rules were worse than the simple maximum (paired differences %s for lcOR and %s for relmax), so neither was adopted. Two comparisons were unfavorable to the channels proposed in the earlier version: on feature anomalies the unsmoothed rarity (%s) and an isolation forest fitted per lattice (%s) were above channel A (%s), and on arrangement anomalies the Shekhar statistic (%s) was above channel B (%s). The generic pair max(isolation forest, Shekhar) reached %s, above the selected rule (%s), but it was a comparator, not a candidate, under the declared protocol.

\begin{table}[t]
\caption{Experiment B (G1, four mixed conditions, 30 lattices each): mean AP for feature anomalies, arrangement anomalies and all anomalies, and the worst-type AP.}
\label{tab:dev}
\centering\footnotesize
\begin{tabular}{lcccc}
\toprule
Score or rule & Feature & Arrangement & All & Worst type\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}

\subsection{Stage 1: the frozen rule failed on G2}
\label{sec:stage1}
Table~\ref{tab:stage1} gives the result on the 90 G2 lattices. The frozen max(rarity, $B$) reached a worst-type AP of %s, below the pre-declared generic comparators OR(LOF, Shekhar) (%s), max(isolation forest, Shekhar) (%s) and contextual LOF (%s); the paired intervals of all three differences exclude zero. The failure has a clear cause: the cohort rarity was at chance on G2's feature anomalies (AP %s), whereas LOF fitted on each lattice found them (%s). Every G2 lattice draws its own domain directions, so an embedding at $40^\circ$ from its local domain is unusual within its lattice but not in a cohort whose other lattices point in all directions. A rarity referenced to the cohort presupposes that the lattices share one normal distribution; G1 satisfied that assumption, G2 did not, and the development data could not reveal it.

\begin{table}[t]
\caption{Stage 1 (G2, three conditions pooled, 90 lattices): per-type AP, worst-type AP and the paired difference in worst-type AP of the frozen rule minus each comparator, with 95\%% bootstrap interval.}
\label{tab:stage1}
\centering\footnotesize
\begin{tabular}{lcccc}
\toprule
Score or rule & Feature & Arrangement & Worst type & Frozen minus score\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}

\subsection{Stage 2: the revised rule on a new generator}
The post-hoc exploration (G1 and fresh G2 seeds) ranked max(isolation forest, Shekhar) first among the 24 pairs (mean worst-type AP %s; with channel B instead of the Shekhar statistic %s; with OR instead of max %s). That rule was frozen and tested on G3 (Table~\ref{tab:stage2}, Figs.~\ref{fig:maps} and~\ref{fig:confirm}). With shuffled regions it reached a worst-type AP of %s (feature %s, arrangement %s), where every single score stayed at or below %s. Pooled over the two G3 conditions it was above every single score (paired differences between %s and %s, all intervals above zero), above the stage-1 rule (%s), above OR(LOF, Shekhar) (%s) and level with contextual LOF (%s). On the fresh-seed replications it reached %s on G1 and %s on G2, where OR(LOF, Shekhar) was slightly better (%s): the isolation forest finds G2's within-lattice anomalies less well than LOF.

Rotated patches were not detected by anything: the best arrangement AP of any score was %s, consistent with Proposition~\ref{prop:rot}, since only the patch borders differ from normal neighborhoods.

\begin{table*}[t]
\caption{Stage 2 (G3, 30 lattices per condition): per-type AP for shuffled regions and rotated patches, worst-type AP pooled over both conditions, and the paired difference in pooled worst-type AP of the frozen rule minus each score, with 95\%% bootstrap interval.}
\label{tab:stage2}
\centering\footnotesize
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{Shuffled regions} & \multicolumn{2}{c}{Rotated patches} & Pooled & Frozen minus score\\
Score or rule & Feature & Arrangement & Feature & Arrangement & worst type & \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_maps.png}
\caption{One G3 lattice with shuffled regions (top) and one with rotated patches (bottom): ground truth, cohort ranks of the isolation forest and of the Shekhar statistic, and the frozen stage-2 fusion; cyan contours mark the anomalies.}
\label{fig:maps}
\end{figure}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig_confirm.png}
\caption{Per-type AP in the two confirmatory stages. Left: G2 (stage 1, 90 lattices); the stage-2 rule is shown for reference only, since it was chosen after this test. Middle and right: G3 (stage 2, 30 lattices each).}
\label{fig:confirm}
\end{figure*}

\subsection{Experiment D: robustness and cost}
Table~\ref{tab:robust} varies one factor at a time. On G3 with $32\times32$ lattices and $d\ge16$, the frozen rule reached a worst-type AP of %s--%s and was the best or within 0.01 of the best of the four fused rules in %d of %d settings (dimension, feature prevalence, window and noise); contextual LOF was better at noise 0.16 (%s against %s). At $d=2$ every method failed, because two dimensions cannot hold the texture manifold and its anomalies apart. Larger G3 lattices lowered every method (worst-type AP of the frozen rule %s, %s and %s for 32, 64 and 128 sites per side). This is a property of G3 rather than of the methods: its stripe phase is computed from absolute coordinates with a slowly varying orientation, so the background varies faster far from the origin (mean Shekhar statistic of anomaly-free lattices 0.29, 0.38 and 0.54); a scaling study needs a generator without this defect. On G2, every fused rule weakened as the background became less coherent, and OR(LOF, Shekhar) was above the frozen rule at every coherence level, because LOF finds G2's within-lattice anomalies better than the isolation forest. Computing all scores of this study took %s s per $32\times32$ lattice at $d=16$, %s s at $d=384$ and %s s at $128\times128$ on a desktop computer; the isolation forest and LOF dominate the cost, and the spatial statistics are linear in the number of sites.

\begin{table}[t]
\caption{Experiment D: worst-type AP of four fused rules under one-at-a-time variation (10 lattices per setting; G3 with shuffled regions unless stated) and the time to compute all scores of the study per lattice.}
\label{tab:robust}
\centering\footnotesize
\begin{tabular}{llccccc}
\toprule
Factor & Value & max(IF, Shekhar) & max(rarity, $B$) & OR(LOF, Shekhar) & Contextual LOF & s/lattice\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (f2(PR["P1_G1"]["rarity"]["mean"]), f2(PR["P1_G2"]["rarity"]["mean"]), f2(PR["P1_G1"]["LOF"]["mean"]), f2(PR["P1_G2"]["LOF"]["mean"]),
       f2(min(v["r_a"] for v in PR2["P2b"].values())), f2(max(v["r_a"] for v in PR2["P2b"].values())), f2(PR2["P2b"]["block 3"]["predicted_survival"]),
       f2(PR2["P2b"]["block 3"]["measured_survival_mean"]), f2(PR2["P2b"]["block 1"]["measured_survival_similarity_weighted"]),
       f2(min(v["measured_survival_similarity_weighted"] for k, v in PR2["P2b"].items() if k != "block 1")), f2(max(v["measured_survival_similarity_weighted"] for k, v in PR2["P2b"].items() if k != "block 1")),
       table_p2(),
       round(100 * PR["P3"]["wall_fraction"]), round(100 * p3[B_]), round(100 * p3["isotropic disorder D"]), round(100 * p3["center-to-neighbor distance"]), round(100 * p3[SH]),
       f2(p4["max fusion"]["TPR_f"]), f2(p4["isolation forest"]["TPR_f"]), f2(p4["Shekhar"]["TPR_f"]), f2(p4["max fusion"]["TPR_a"]), f2(p4["isolation forest"]["TPR_a"]),
       f2(p4["Shekhar"]["TPR_a"]), f3(p4["max fusion"]["FPR"]), f3(p4["isolation forest"]["FPR"]), f3(p4["Shekhar"]["FPR"]),
       f2(p5[SH]["interior"]["mean"]), f2(p5[SH]["border"]["mean"]), f2(p5["center-to-neighbor distance"]["interior"]["mean"]), f2(p5["center-to-neighbor distance"]["border"]["mean"]),
       f2(p5[CLOF]["interior"]["mean"]), f2(p5[CLOF]["border"]["mean"]), f2(p5[B_]["interior"]["mean"]), f2(p5[B_]["border"]["mean"]),
       f2(singles_worst_max), f2(max(fused_worst)),
       f2(coh["corr=1"][B_]["arrangement"]), f2(coh["corr=6"][B_]["arrangement"]), f2(coh["corr=1"][A_]["feature"]), f2(coh["corr=6"][A_]["feature"]),
       f2(max(dev_single[k] for k in (A_, B_, "rarity", IF, SH, LOF, CLOF))), f2(dev_single[SEL]), f2(crit[MXRB]), f2(crit[MX]), f2(crit["fusion: OR"]),
       ci(DV["tests_vs_best_simple"]["fusion: lcOR (locally calibrated B)"]), ci(DV["tests_vs_best_simple"]["fusion: relmax (reliability-weighted B)"]),
       f2(dev_r), f2(dev_if), f2(dev_A), f2(dev_sh), f2(dev_B), f2(dev_ifs), f2(crit[MXRB]),
       table_dev(),
       f3(P1s[MXRB]["worst"]), f2(P1s[ORLS]["worst"]), f2(P1s[IFS]["worst"]), f2(P1s[CLOF]["worst"]), f2(P1s["rarity"]["feature"]), f2(P1s[LOF]["feature"]),
       table_stage1(),
       f2(cr2[CH2]), f2(cr2["max(isolation forest (embeddings) | channel B (directional disorder))"]), f2(cr2["OR(isolation forest (embeddings) | spatial outlier statistic (Shekhar))"]),
       f2(g3s[CH2]["worst"]), f2(g3s[CH2]["feature"]), f2(g3s[CH2]["arrangement"]), f2(max(g3s[s]["worst"] for s in singles)),
       sg(min(g3t[s]["worst"]["mean"] for s in singles)), sg(max(g3t[s]["worst"]["mean"] for s in singles)), ci(g3t[MXRB]["worst"]), ci(g3t[ORLS]["worst"]), ci(g3t[CLOF]["worst"]),
       f2(g1r[CH2]["worst"]), f2(g2r[CH2]["worst"]), f2(g2r[ORLS]["worst"]),
       f2(max(g3r[k]["arrangement"] for k in g3r)),
       table_stage2(),
       f2(min(rb["dimension"][v]["summary"][CH2]["worst"] for v in ("16", "64", "384"))), f2(max(rb["dimension"][v]["summary"][CH2]["worst"] for v in ("16", "64", "384"))),
       _nbest, _ntot, f2(rb["noise (G3)"]["0.16"]["summary"][CLOF]["worst"]), f2(rb["noise (G3)"]["0.16"]["summary"][CH2]["worst"]),
       f2(rb["lattice size"]["32"]["summary"][CH2]["worst"]), f2(rb["lattice size"]["64"]["summary"][CH2]["worst"]), f2(rb["lattice size"]["128"]["summary"][CH2]["worst"]),
       "%.1f" % t_dim["16"], "%.1f" % t_dim["384"], "%.1f" % t_n["128"],
       table_robust())

DISCUSSION = r"""
\section{Discussion}
\label{sec:discussion}
\subsection{What the study establishes}
Within the stated assumptions and on controlled synthetic lattices, four findings hold across generators. (i) Feature scores and spatial-order scores detect different anomalies, and each is near chance on the other kind; Proposition~\ref{prop:perm} gives the reason for one direction, and the dispersion statistics' blindness to coherent unusual regions the other. (ii) When both kinds occur in the same lattice, choosing one score per cohort cannot work, and a site-wise maximum of cohort ranks is a strong, simple fusion: it beat OR, $z$-score maxima and two locally adaptive rules on the development generator, and its frozen form beat every single score and three fused baselines on a generator it had never seen. (iii) Whether a fixed aggregation window erases an anomaly is a geometric question that Proposition~\ref{prop:surv} answers in the strong-anomaly regime. (iv) No local, rotation-invariant statistic can see the interior of a rotated patch (Proposition~\ref{prop:rot}); such anomalies require orientation-aware or non-local models of the kind developed for logical anomalies in images \citep{Bergmann2022,Batzner2024}.

\subsection{Failure modes and limits of complementarity}
\emph{The reference of a feature score.} The stage-1 failure shows that a rarity score is only as good as its reference set: referencing it to the cohort assumes that all lattices share one normal distribution. When normality is defined per lattice (per image, per specimen), a within-lattice score such as LOF or an isolation forest is needed, and when the lattices do share a normal distribution, as with a fixed encoder in G3, both kinds of reference work. Which situation applies is a property of the data that should be checked before a detector is chosen, for example by comparing within-lattice and cross-lattice nearest-neighbor distances.

\emph{The spatial score.} The directional statistic of the earlier version did what Proposition~\ref{prop:wall} says (fewer responses at walls), but it was less sensitive than the plain Shekhar statistic to the arrangement anomalies of all three generators, and the selection preferred the latter. Its wall property would matter where walls are frequent and anomalies rare; in the present generators sensitivity mattered more.

\emph{The cost of fusion.} Proposition~\ref{prop:fuse} and the measured rates show the price: the false positives of the two channels add up, so the per-type AP of the maximum is below that of the better channel for each type (Table~\ref{tab:dev}). The maximum is the right choice when both kinds of anomaly matter; when only one does, the matching single channel is better. A fused score also does not say which kind of anomaly it found; keeping the two rank maps, as in Fig.~\ref{fig:maps}, does.

\emph{Background coherence.} A shuffle among nearly identical vectors is nearly invisible, so arrangement anomalies become harder, not easier, in very smooth backgrounds (Fig.~\ref{fig:phase}d); detectability depends on the variation of the background at the scale of the anomaly.

\subsection{What is not established}
All data are synthetic and were generated by the author of the analysis; G2 and G3 were written before their tests and frozen, but they are not independent of the author's assumptions. Nothing here shows performance on natural images, medical slides or industrial data, and the generators contain neither texture statistics nor encoder artefacts of real embeddings. The stage-2 rule was selected after the stage-1 failure on data that included G2, and its G2 replication is therefore not confirmatory. The comparison is limited to classical detectors; deep models trained on normal data were not included. Statistical uncertainty is over lattices of one generator; conclusions that should hold across generators rest on three generators only. G3 has a defect that limits it to small lattices: its texture becomes irregular far from the origin, so its results for lattices larger than $32\times32$ are not informative about scaling. What would refute the main conclusions: a generator with mixed anomalies on which a single score or a cohort-level choice matches the site-wise maximum, or a real data set on which the maximum of a feature and a spatial score does not improve the worst-type detection.

\subsection{Relation to the earlier version}
The earlier version reported that a cohort-level choice between channel A and channel B raised a lower-tail AP across fifteen test scenarios, most of which had been used during development. The present study replaces that design. Its results do not support the earlier recommendation: cohort-level selection fails on mixed lattices, and the two channels proposed earlier were outperformed by generic scores (an isolation forest and the Shekhar statistic) in the role they were designed for. What survives is the decomposition into a feature channel and a spatial channel, the occupancy explanation of aggregation loss, and the wall property of the directional statistic.
"""

CONCL = r"""
\section{Conclusions}
Feature rarity and spatial order are complementary sources of evidence on lattices of embeddings, and five elementary propositions bound what each, spatial aggregation and their fusion can detect. In controlled experiments with both kinds of anomaly in the same lattice, a site-wise maximum of the calibrated ranks of a within-lattice feature score and a neighbor-difference score was the most reliable label-free combination, and it survived a pre-registered test on a generator it had never seen. A first pre-registered rule failed because its feature score was referenced to a cohort that did not share one normal distribution; that failure, and the undetected rotated patches, mark the limits of local, marginal-plus-spatial detection. Testing these conclusions on real embedding lattices is the next step.
"""

DECL = r"""
\section*{CRediT authorship contribution statement}
\textbf{Leon Sandler:} Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Visualization, Writing -- original draft, Writing -- review \& editing.

\section*{Declaration of competing interest}
The author declares that he has no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Funding}
This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

\section*{Data availability}
All data are synthetic and are regenerated by the code from the recorded seeds. Code, frozen protocols, result files and the version-control history with the freeze commits are openly available at \url{https://github.com/sandlerleon/MyUncleVision} (MIT license; modules \texttt{kbs\_study}, \texttt{kbs\_stage2}, \texttt{kbs\_robust}, \texttt{kbs\_props2}) and archived on Zenodo (\url{https://doi.org/10.5281/zenodo.24066260}).

\section*{Declaration of generative AI and AI-assisted technologies in the writing process}
During the preparation of this work the author used Claude (Anthropic) to assist with writing and testing the analysis code to the author's specification, with the figures and with language editing. The author reviewed and edited the content and takes full responsibility for the content of the published article.
"""

REFS = r"""
\begin{thebibliography}{99}
\bibitem{Aggarwal2015} C.C. Aggarwal, S. Sathe, Theoretical foundations and algorithms for outlier ensembles, ACM SIGKDD Explor. Newsl. 17 (2015) 24--47. \url{https://doi.org/10.1145/2830544.2830549}.
\bibitem{Aggarwal2017} C.C. Aggarwal, Outlier Analysis, 2nd ed., Springer, Cham, 2017. \url{https://doi.org/10.1007/978-3-319-47578-3}.
\bibitem{Banerjee2005} A. Banerjee, I.S. Dhillon, J. Ghosh, S. Sra, Clustering on the unit hypersphere using von Mises--Fisher distributions, J. Mach. Learn. Res. 6 (2005) 1345--1382.
\bibitem{Batzner2024} K. Batzner, L. Heckler, R. K{\"o}nig, EfficientAD: accurate visual anomaly detection at millisecond-level latencies, in: Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2024, pp. 127--137. \url{https://doi.org/10.1109/WACV57701.2024.00020}.
\bibitem{Bergmann2022} P. Bergmann, K. Batzner, M. Fauser, D. Sattlegger, C. Steger, Beyond dents and scratches: logical constraints in unsupervised anomaly detection and localization, Int. J. Comput. Vis. 130 (2022) 947--969. \url{https://doi.org/10.1007/s11263-022-01578-9}.
\bibitem{Breunig2000} M.M. Breunig, H.-P. Kriegel, R.T. Ng, J. Sander, LOF: identifying density-based local outliers, in: Proc. ACM SIGMOD Int. Conf. Manag. Data, 2000, pp. 93--104. \url{https://doi.org/10.1145/342009.335388}.
\bibitem{Chandola2009} V. Chandola, A. Banerjee, V. Kumar, Anomaly detection: a survey, ACM Comput. Surv. 41 (2009) 15:1--15:58. \url{https://doi.org/10.1145/1541880.1541882}.
\bibitem{Davis2006} J. Davis, M. Goadrich, The relationship between precision-recall and ROC curves, in: Proc. 23rd Int. Conf. Mach. Learn. (ICML), 2006, pp. 233--240. \url{https://doi.org/10.1145/1143844.1143874}.
\bibitem{Defard2021} T. Defard, A. Setkov, A. Loesch, R. Audigier, PaDiM: a patch distribution modeling framework for anomaly detection and localization, in: Pattern Recognition. ICPR International Workshops and Challenges, Lecture Notes in Computer Science, vol. 12664, Springer, Cham, 2021, pp. 475--489. \url{https://doi.org/10.1007/978-3-030-68799-1_35}.
\bibitem{Kriegel2011} H.-P. Kriegel, P. Kr{\"o}ger, E. Schubert, A. Zimek, Interpreting and unifying outlier scores, in: Proc. 2011 SIAM Int. Conf. Data Mining, 2011, pp. 13--24. \url{https://doi.org/10.1137/1.9781611972818.2}.
\bibitem{Kuwahara1976} M. Kuwahara, K. Hachimura, S. Eiho, M. Kinoshita, Processing of RI-angiocardiographic images, in: K. Preston, M. Onoe (Eds.), Digital Processing of Biomedical Images, Plenum Press, New York, 1976, pp. 187--202. \url{https://doi.org/10.1007/978-1-4684-0769-3_13}.
\bibitem{Liu2008} F.T. Liu, K.M. Ting, Z.-H. Zhou, Isolation forest, in: Proc. 8th IEEE Int. Conf. Data Mining (ICDM), 2008, pp. 413--422. \url{https://doi.org/10.1109/ICDM.2008.17}.
\bibitem{Mardia2000} K.V. Mardia, P.E. Jupp, Directional Statistics, Wiley, Chichester, 2000.
\bibitem{Pang2021} G. Pang, C. Shen, L. Cao, A. van den Hengel, Deep learning for anomaly detection: a review, ACM Comput. Surv. 54 (2022) 38:1--38:38. \url{https://doi.org/10.1145/3439950}.
\bibitem{Roth2022} K. Roth, L. Pemula, J. Zepeda, B. Sch{\"o}lkopf, T. Brox, P. Gehler, Towards total recall in industrial anomaly detection, in: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 14298--14308. \url{https://doi.org/10.1109/CVPR52688.2022.01392}.
\bibitem{Rousseeuw1993} P.J. Rousseeuw, C. Croux, Alternatives to the median absolute deviation, J. Am. Stat. Assoc. 88 (1993) 1273--1283. \url{https://doi.org/10.1080/01621459.1993.10476408}.
\bibitem{Ruff2021} L. Ruff, J.R. Kauffmann, R.A. Vandermeulen, G. Montavon, W. Samek, M. Kloft, T.G. Dietterich, K.-R. M{\"u}ller, A unifying review of deep and shallow anomaly detection, Proc. IEEE 109 (2021) 756--795. \url{https://doi.org/10.1109/JPROC.2021.3052449}.
\bibitem{Schubert2014} E. Schubert, A. Zimek, H.-P. Kriegel, Local outlier detection reconsidered: a generalized view on locality with applications to spatial, video, and network outlier detection, Data Min. Knowl. Discov. 28 (2014) 190--237. \url{https://doi.org/10.1007/s10618-012-0300-z}.
\bibitem{Shekhar2003} S. Shekhar, C.-T. Lu, P. Zhang, A unified approach to detecting spatial outliers, GeoInformatica 7 (2003) 139--166. \url{https://doi.org/10.1023/A:1023455925009}.
\bibitem{Song2007} X. Song, M. Wu, C. Jermaine, S. Ranka, Conditional anomaly detection, IEEE Trans. Knowl. Data Eng. 19 (2007) 631--645. \url{https://doi.org/10.1109/TKDE.2007.1009}.
\bibitem{Tomasi1998} C. Tomasi, R. Manduchi, Bilateral filtering for gray and color images, in: Proc. 6th Int. Conf. Comput. Vis. (ICCV), 1998, pp. 839--846. \url{https://doi.org/10.1109/ICCV.1998.710815}.
\end{thebibliography}
"""

intro = open(os.path.join(HERE, "intro.tex"), encoding="utf-8").read()
framework = open(os.path.join(HERE, "framework.tex"), encoding="utf-8").read()
TITLE = "Detectability and complementarity of feature and spatial-order anomalies in embedding lattices: analytical bounds and a two-stage pre-registered evaluation"
TEX = r"""\documentclass[preprint,review,12pt]{elsarticle}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[T1]{fontenc}
\usepackage{url}
\usepackage{lineno}
\linenumbers
\newtheorem{proposition}{Proposition}
\journal{Knowledge-Based Systems}
\biboptions{numbers,sort&compress}

\begin{document}
\begin{frontmatter}
\title{%s}
\author[a]{Leon Sandler\corref{cor1}}
\cortext[cor1]{Corresponding author. E-mail: sandler.leon@gmail.com; ORCID 0009-0007-4584-808X.}
\address[a]{Independent Researcher, Northbrook, IL, USA}

\begin{abstract}
%s
\end{abstract}

\begin{keyword}
anomaly detection \sep spatial outliers \sep embedding lattices \sep score fusion \sep detectability \sep pre-registration
\end{keyword}
\end{frontmatter}

%s

%s

%s

%s

%s

%s

%s

\end{document}
""" % (TITLE, ABSTRACT, intro, framework, DESIGN, RESULTS, DISCUSSION, CONCL, DECL + REFS)

TEX = TEX.replace(r"\begin{tabular}", r"\resizebox{\linewidth}{!}{\begin{tabular}").replace(r"\end{tabular}", r"\end{tabular}}")
out = os.path.join(HERE, "main.tex")
open(out, "w", encoding="utf-8").write(TEX)
open(os.path.join(HERE, "highlights.txt"), "w", encoding="utf-8").write("\n".join(HIGHLIGHTS) + "\n")
print("abstract words", len(ABSTRACT.split()), "| highlights", [len(h) for h in HIGHLIGHTS])
for _ in range(2):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=HERE, capture_output=True, text=True, errors="replace")
print("pdflatex exit", r.returncode, "| undefined refs:", r.stdout.count("undefined"))
