# Spatial aggregation of anomaly scores on embedding lattices

`myuncle_vision/gating.py` has the rules, `myuncle_vision/og_study.py` the protocol, `myuncle_vision/morphology.py` the independent generator. Everything after the rarity score is label-free; only the *selection* of the rule parameters uses development labels.

| Rule | Definition |
|---|---|
| `none` | `s = r` |
| `mean` | mean of `r` over a `w x w` window (the earlier frozen configuration for `w = 3`) |
| `simw` | `s_i = sum_j c_ij r_j / sum_j c_ij`, `c_ij = exp(-(1 - u_i.u_j)/tau)`, `tau` = median of `1 - cos` over lattice-adjacent pairs of the cohort (an established edge-preserving filter, not claimed as new) |
| `onesided` | `s_i = r_i + max(0, m_i - r_i)`, `m_i` = similarity-weighted mean over the neighbors (adapted from the asymmetric damping of the author's flowmeter paper, Zenodo 10.5281/zenodo.24036377) |
| `og_cos`, `og_disorder`, `og_vmf` | `s_i = (1 - g_i) r_i + g_i s_i^sim`, `g_i = logistic(alpha (A_i - theta))`, `A_i` = cohort rank of the mean compatibility / the resultant length `R_i` / the von Mises-Fisher information `H_chi` of the neighbors |
| `onesided_gate` | the one-sided rule inside the compatibility gate |
| persistence (`lam`) | multiplies `s` by `(1 - lam) + lam P_i`, `P_i` = share of windows 1, 3, 5, 7 at which the score exceeds its cohort top-3% threshold |

`H_chi` is a monotone function of `R_i`: **a gate on the rank of `H_chi` is a gate on the rank of `R_i`** (tested), so the von Mises-Fisher information cannot add information to a rank gate.

## Reproduce

```
python -m myuncle_vision.og_study cache dev      # rarity scores, embeddings and truth of the development scenarios
python -m myuncle_vision.og_study cache test     # the same for the independent test scenarios (new seeds)
python -m myuncle_vision.og_study grid           # 460 distinct configurations on the ten development scenarios -> results/og_grid_dev.json
python -m myuncle_vision.og_study search         # GaaR strategies vs random search on that table -> results/og_search.json
python -m myuncle_vision.og_study test           # champions on the ten test scenarios, paired bootstrap -> results/og_test.json
```
Scenarios: synthetic images (handcrafted and DINOv2 embeddings), MyUncle lattices (rotor and rotor + D), and six anomaly shapes of the independent generator (scattered sites, 3 x 3 and 6 x 6 blocks, bars of 8, irregular 12-site clusters, 3 x 3 blocks on domain boundaries; prevalence 3%, separability fixed). Fitness `F = 0.5 mean(AP) + 0.5 min(AP)` over the ten scenarios.

## Findings (independent test)

* Fixed 3 x 3 mean: `F` 0.11; no smoothing 0.23; **untuned similarity-weighted 3 x 3: 0.32**; paired 95% intervals of the differences from the fixed mean and from no smoothing exclude zero.
* The fixed mean destroyed scattered and bar-shaped anomalies (AP 0.37 -> 0.03; 0.39 -> 0.05) and helped only DINOv2 images; similarity weighting kept them (region recall 0.44 and 0.60).
* Order / von Mises-Fisher gates (`F` 0.30) did not beat ordinary similarity; the one-sided rule matched no smoothing.
* GaaR was not better than random search on the rule table (top-5% hit rate 94% for random, 70-81% for GaaR strategies): many configurations of the similarity family are nearly equivalent.
* No rule helped the handcrafted-feature images, and on rotor-only lattices similarity weighting stayed slightly below no smoothing.
