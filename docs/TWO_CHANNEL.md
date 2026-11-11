# Two label-free channels: rarity and local order

A rarity score sees **what** is at a site. It cannot see an anomaly that consists of ordinary embeddings in the **wrong arrangement**, because such an anomaly leaves the distribution of the embeddings unchanged. MyUncle's local order supplies the second channel.

| Channel | Score | Finds | Blind to |
|---|---|---|---|
| A | rarity of the embedding in the cohort, aggregated with similarity weights (`gating.py`, rule `simw`) | feature outliers of any shape | rearranged regions (average precision = prevalence) |
| B | directional disorder: the minimum over the four 2 x 2 quadrant windows that contain the site of `1 - |mean unit vector|` (`two_channel.directional_disorder`) | rearranged (disorganized) regions, thin outliers | compact outlier blocks (their interior is ordered) |

The isotropic disorder `D` of MyUncle is large at every wall between two ordered domains. A wall is ordered on at least one side, so the minimum over the quadrants stays near zero there, while a disorganized region is disordered in every direction (the idea of the Kuwahara filter, applied to the order parameter).

**Choice of the channel (label-free, per cohort).** The raw score of each channel (the nearest-neighbor distance at `k = 1%` for A, the directional disorder for B) is standardized with the cohort median and MAD; the tail index `T` is the 97th percentile of the result (1.88 for a Gaussian bulk). Channel B is used when `T_B > T_A` and `T_B > 2 x 1.88`, channel A otherwise.

## Use

```
python -m myuncle_vision run --input IMAGES --output OUT --aggregation two_channel      # or: simw (channel A only), mean (the earlier frozen default)
```
```python
from myuncle_vision import discover_images
discover_images("IMAGES", "OUT", aggregation="two_channel")
```
`cohort.Config(aggregation="two_channel")` writes `extra["channel_A"]`, `extra["channel_B"]` (cohort ranks per patch) and `extra["novelty"]` (the selected channel) and reports the tail indices and the selected channel in the cohort summary. The default stays `mean`, so that the earlier frozen results remain reproducible; `two_channel` is the recommended setting. In a cohort that contains both kinds of anomaly read both maps (`channel_A`, `channel_B`): the choice is made once per cohort.

## Reproduce

```
python -m myuncle_vision.tc_study calibrate      # separability of the feature outliers of the mixed scenarios (unsmoothed rarity, one development cohort)
python -m myuncle_vision.tc_study cache dev      # fifteen development scenarios
python -m myuncle_vision.tc_study cache test     # fifteen test scenarios (new seeds, new images and lattices)
python -m myuncle_vision.tc_study dev            # all detectors and seven fusion rules on the development scenarios -> results/two_channel_dev.json
python -m myuncle_vision.tc_study test           # the development-selected rule, once, with paired bootstrap -> results/two_channel.json
python docs/build/make_fig_channels.py           # docs/figures/fig_channels.png
```
Scenarios: the ten of `og_study` (feature outliers of six shapes, two image scenarios, two MyUncle lattice scenarios), three arrangement scenarios (the embeddings of a region are permuted among its sites; blocks of 6 and 8, irregular clusters of 24; 7% of the sites; spatially correlated background) and two mixed scenarios (rearranged 6 x 6 blocks, 3.5%, with feature outliers that leave the subspace of the normal sites, 1.5%).

## Findings (fifteen independent test scenarios; F = 0.5 mean AP + 0.5 min AP)

| Detector | ten earlier scenarios (mean AP) | arrangement (mean AP) | mixed (mean AP) | F [95% interval] |
|---|---|---|---|---|
| rarity, no smoothing | 0.35 | 0.06 | 0.23 | 0.16 [0.15, 0.18] |
| rarity, fixed 3 x 3 mean | 0.19 | 0.06 | 0.07 | 0.09 [0.08, 0.11] |
| channel A (similarity-weighted rarity) | 0.47 | 0.06 | 0.15 | 0.20 [0.19, 0.21] |
| isotropic disorder D | 0.11 | 0.57 | 0.29 | 0.12 [0.11, 0.12] |
| local manifold statistic (PCA residual, m = 2) | 0.18 | 0.48 | 0.26 | 0.13 [0.12, 0.13] |
| distance to the 8 neighbors | 0.20 | 0.86 | 0.53 | 0.19 [0.19, 0.19] |
| channel B (directional disorder) | 0.24 | 0.86 | 0.49 | 0.20 [0.20, 0.21] |
| **channel chosen by tail index** | **0.47** | **0.86** | **0.49** | **0.36 [0.29, 0.46]** |

* Rarity is at chance on rearranged regions (AP 0.06 at a prevalence of 0.06 to 0.07); channel B reaches 0.86.
* The fused detector beats channel A by +0.16 in F (paired 95% interval +0.10 to +0.25) and loses nothing on the ten earlier scenarios (paired difference of the mean AP 0.00 to +0.01).
* The tail index chose channel A in 33 of 34 test cohorts with feature anomalies only (the exception: one cohort of bars, where both channels work) and channel B in all 25 cohorts with rearranged regions.
* The directional form matters for the choice, not for the detection: the plain distance to the neighbors finds rearranged regions equally well (0.86), but it responds to every domain wall, so the selection built on it picks the wrong channel in block scenarios (F 0.34; paired difference +0.01 to +0.09); with the isotropic disorder F 0.17.
* Site-wise fusions were below the selection: maximum of the standardized scores 0.25, maximum of the ranks 0.26, tail-weighted maxima 0.34.

## Limits (measured)

* **Mixed cohorts.** The choice is per cohort. In cohorts with both kinds of anomaly it chose channel B: rearranged sites AP 0.62, feature outliers 0.02 to 0.05 (rarity alone finds them: 0.52 and 0.49). Read both maps.
* **Channel A is not uniformly better than the unsmoothed rarity.** In the smooth background of the mixed cohorts similarity weighting lowered the AP of scattered outliers from 0.49 to 0.08.
* **A first attempt failed** (`og_geometry.py`, `results/og_geometry_first_attempt.json`): fixed blends of the ranks of channel A and of the isotropic disorder, selected on thirteen development scenarios (F 0.256 against 0.238), did not beat channel A on the test scenarios (0.206 against 0.214). The directional statistic and the tail index were designed after that, so thirteen of the fifteen test scenarios had been used once before.
* When neither channel has a tail the choice rests on the default (channel A). Channel B needs a background that is locally ordered.
* All evidence is simulated; the generators were written by the author, and a permutation is an idealized arrangement anomaly.
