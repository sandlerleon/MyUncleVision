# Configuration search with GaaR

[GaaR](https://github.com/sandlerleon/GaaR) (Goal-Aware Adaptive Regulation) is an evolutionary search framework. `myuncle_vision/gaar_opt.py` uses it (vendored unmodified in `third_party/gaar`, SHA-256 recorded and tested) to select a **configuration of the MyUncleVision score** on labeled development scenarios.

**This is supervised hyperparameter selection, not label-free adaptation.** The chosen configuration is fixed and then applied to unlabeled data.

## Search space (360 configurations)

| Gene | Values |
|---|---|
| smoothing type and window | none, mean 3/5/7, Gaussian 5/7, median 3/5 |
| axis weights (appearance / state-wise disorder / contact rarity) | 1/0/0, 0.75/0.25/0, 0.5/0.5/0, 0.75/0/0.25, 0.4/0.4/0.2 |
| PCA dimension | 8, 16, 32 |
| largest rarity scale | 2%, 5%, 10% of the reference set |

The frozen MyUncleVision configuration is `mean 3 / appearance only / PCA 16 / 10%`. The smoothing options are available to every user: `Config(smooth_type="gaussian", smooth_window=5)` (the default stays `mean`, 3).

## Fitness

`F = 0.5 * mean(AP) + 0.5 * min(AP)` over four development scenarios (development images with the handcrafted and the DINOv2 embedding; development MyUncle lattices with the rotor and the rotor + D embedding). The lower-tail term keeps any scenario from being sacrificed.

## Reproduce

```
python -m myuncle_vision.gaar_opt grid        # all 360 configurations on the development scenarios (about 80 minutes on 5 workers) -> results/gaar_grid_dev.json
python -m myuncle_vision.gaar_opt search      # GaaR strategies and random search, 100 runs x 40 evaluations -> results/gaar_search_comparison.json
python -m myuncle_vision.gaar_opt test        # champions on the fresh test scenarios (new seeds; MyUncle preset superconducting_scaffold not used before) -> results/gaar_test.json
python -m myuncle_vision.gaar_opt labelfree   # Moran's I of the unsmoothed score vs the effect of smoothing -> results/gaar_labelfree.json
```
The image cohorts are expected under `MUV_BENCH_ROOT` (default `...\MyUncleVision_work\bench`); the `dev` and `held` cohorts are built by `bench.py`, the `test` cohort (seeds + 100) by the same code on first use.

## Findings (see the manuscripts, Section 5.3 / 4.6)

* The default configuration ranked 64th of 360 on the development fitness (0.265; optimum 0.351).
* With 40 evaluations, GaaR's greedy and restart strategies reached the top 1% of configurations in 59 and 60% of the runs, random search in 25%; at the top 5% level random search did as well or better (87% against 71 to 85%).
* On fresh test scenarios the restart/population champion raised the lower-tail fitness from 0.18 to 0.26 (the worst scenario, handcrafted-feature images, from AP 0.04 to 0.17; DINOv2 images fell from 0.50 to 0.41). The champions of greedy and random search reached 0.21 and the exhaustive development optimum 0.25: every development-optimized configuration beat the default, but the order did not follow the development fitness.
* **Any gain belongs to the selected configuration, not to the search algorithm** (a random-search champion also beat the default; see the intervals below). Each strategy contributed one champion; four scenarios are few; the test data come from the same generators as the development data.
* A label-free hint (exploratory): Moran's I of the unsmoothed rarity score was below 0.02 in exactly the two lattice scenarios in which smoothing lost most of the AP, and above 0.1 in the other six. It would not have helped the handcrafted-feature test images.

## Uncertainty and a controlled morphology study (v0.5.0)

* **Intervals.** `gaar_opt test` now resamples the independent units of every fresh scenario (16 images; 16 simulation runs of three snapshots) with the same draws for all configurations and stores 95% intervals of the test fitness and of the difference from the default in `results/gaar_test.json` (`bootstrap`). For the restart/population champion the difference is +0.08, 95% interval [-0.05, +0.17]: **not distinguishable from zero**.
* **Morphology study** (`python -m myuncle_vision.morphology`, `results/morphology_study.json`): an independent generator of embedding lattices (smooth background domains, 3% anomalous sites, fixed separability, anomalies as b x b blocks with their own directions; the separability was calibrated on the unsmoothed score only, then frozen). Without smoothing the average precision was between 0.43 and 0.49 at every size; the frozen 3 x 3 mean lowered it to 0.03 (b = 1) ... 0.31 (b = 6); a 5 x 5 mean was worse; the GaaR champion beat the frozen configuration for b <= 3 but stayed below no smoothing at every size. Smoothing hurt at every size, least for compact anomalies; in the development scenarios it helped DINOv2 images, which is why the search did not select "no smoothing".
* Reproducibility: the grid table, the search comparison (seeds 0-99, GaaR re-seeded with 1000 + run), the fresh-test results and the GaaR commit (4021e64, `third_party/gaar/SOURCE_COMMIT.txt`) are in the repository and in the Zenodo archive of each release.
