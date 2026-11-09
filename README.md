# MyUncleVision

An entropy-guided, label-free discovery platform for unusual tissue organization in whole-slide images (WSI), built as a computer-vision front end for the
[MyUncle](https://github.com/sandlerleon/MyUncle) maintenance-lattice core (Sandler, doi:[10.5281/zenodo.21223570](https://doi.org/10.5281/zenodo.21223570)). It was written for the
NIDDK/HeroX *Kidney AI Discovery Challenge* (Kidney Precision Medicine Project, KPMP, whole-slide images), but nothing in it is specific to the kidney.

Leon Sandler, Independent Researcher, sandler.leon@gmail.com, ORCID 0009-0007-4584-808X. MIT license (code); MyUncle is vendored unmodified (also MIT). Archived on Zenodo (all versions): software [10.5281/zenodo.24066260](https://doi.org/10.5281/zenodo.24066260), manuscript [10.5281/zenodo.24066256](https://doi.org/10.5281/zenodo.24066256). Version 0.1.1 DOIs: software 10.5281/zenodo.24066261, manuscript 10.5281/zenodo.24066257.

> **Status and honest limits (v0.2.0).** The platform runs end to end on SVS and on plain TIFF/PNG/JPEG, is covered by automated tests, was built and run in Docker (default command, offline, three slides), and is evaluated on **synthetic** slides only
> (a development cohort and a held-out cohort with settings frozen beforehand, `results/frozen_settings.json`, `results/bench_*.json`, `results/robustness_*.json`). Synthetic slides are not kidney tissue and were written by the author of the method.
> **It has not been run on KPMP slides** (data released to registered HeroX participants, 2.4 TB for Release 1), and no pathologist has reviewed any output. Everything it outputs about real tissue is a candidate for expert review, not a finding.
>
> **What the evaluation showed.** Lattice smoothing raised the spatial coherence of every scorer. On the held-out cohort the frozen score was the best method with DINOv2 and a poor one with handcrafted features; a smoothed Mahalanobis baseline scored higher on the development cohort;
> the disorder and contact axes of the lattice helped with handcrafted features and hurt with DINOv2, so they are optional diagnostics (the default weights are 1,0,0); with handcrafted features the frozen score was unstable to a coarser patch size. On real slides compare encoders (`--encoder`), scorers (`--scorer knn|mahalanobis`) and axes (`--weights`).
> The MyUncle scenario analysis is an exploratory modelling analogy, off by default (`--scenarios`).

## One command

```
docker build -t myuncle-vision .
docker run --rm -v /path/to/svs:/input:ro -v /path/to/results:/output myuncle-vision
```

The container's default command is `python -m myuncle_vision run --input /input --output /output`. Input: a directory (searched recursively) of `.svs` whole-slide images (also `.ndpi/.scn/.mrxs`, plain `.tif/.png/.jpg`).
The DINOv2 weights are downloaded once while the image is built, so the run itself needs no network. GPU: `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121` and `docker run --gpus all`.
Without Docker: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt && python -m myuncle_vision run --input DIR --output OUT`.
Useful options: `--max-patches N` (coarsens the patch grid until a slide has at most N patches), `--mpp`, `--q`, `--encoder handcrafted|dinov2`, `--hash-files`.

## What it does

1. **Tissue lattice.** Each slide is cut into non-overlapping patches (default 224 px at 0.5 µm/px) on a regular grid; the grid is the lattice. Patches that fail QC (out of focus, pen ink, folds/debris, glass) are excluded everywhere.
2. **Embeddings** of every patch (DINOv2 ViT-S/14, 384-d), centred and reduced by a cohort PCA (16 components) and normalized to the unit sphere.
3. **Order and entropy (MyUncle).** Local disorder `D = 1 - ||mean of the neighbours' unit embeddings||` is MyUncle's `MC.local_disorder` for any dimension; the von Mises-Fisher entropy of the neighbourhood and
   `H_chi = S_uniform - S_vmf` generalize MyUncle's order-entropy bridge (`Bridge.S`, `Bridge.H`) from d = 2 to d dimensions. The tests check the d = 2 equality against the vendored, unmodified MyUncle core.
4. **Tissue states and transitions.** k-means prototype states on the sphere; windowed state entropy; rarity of the contacts between states (`-log P(neighbour state | own state)` over the cohort).
5. **Novelty score** (frozen default): appearance rarity judged at five prevalence scales (0.5 to 10% of the cohort, highest rank), smoothed over the 3 x 3 lattice window. Local disorder *for the patch's own state* and the rarity of contacts are computed as diagnostics and can be added with `--weights`.
6. **Discovery.** Top-scoring patches form connected regions; regions are clustered across slides into candidate states, ranked by novelty, recurrence, reproducibility under two disjoint halves of the reference set.
7. **Explanations and maps.** Plain-language, number-only explanations per finding; heat maps of D, H_chi and novelty; masks; per-patch scores.
8. **Exploratory scenarios** (opt-in `--scenarios`; MyUncle simulation reuse, a modelling analogy, not validated and not prognosis): slide order mapped onto MyUncle's maintenance lattice, `rho_hat` against the bistable window of MyUncle's mean-field model, kinetic Monte Carlo started from the observed slide under half/equal/double repair.

## Models and files (HeroX README requirement)

| File | Function | Findings supported | Trained? |
|---|---|---|---|
| timm `vit_small_patch14_dinov2.lvd142m` (DINOv2 ViT-S/14, Oquab et al. 2023; weights loaded by `myuncle_vision/encoder.py`) | patch embeddings (384-d) | all (F1..Fk are defined on the embedding lattice) | pretrained on natural images, not fine-tuned |
| `myuncle_vision/encoder.py::HandcraftedEncoder` | deterministic colour/stain/orientation/texture features (64-d); offline fallback and tests | all, when selected | no model file |
| `third_party/myuncle/MyUncle_core.py` (MyUncle, unmodified, SHA-256 in `MyUncle_core.sha256`) | order-entropy bridge (d = 2 reference), mean-field bistability, kinetic Monte Carlo | progression scenarios; the d = 2 checks of the order/entropy definitions | no |
| `myuncle_vision/cohort.py` (PCA, k-means, k-nearest-neighbour reference fitted **at run time** on the input cohort) | cohort space, prototype states, novelty axes | all | fitted per run; nothing is stored between runs |

Every model must support at least one finding: all four are used by the ranking or by the scenario analysis; there is no unused model file.

## Outputs (`--output DIR`)

| Path | Content |
|---|---|
| `findings.json`, `findings.csv` | numbered candidate states F1..Fk: type, slides, patches, area (mm²), axes, recurrence, reproducibility, score, explanation, region coordinates (level-0 pixels) |
| `slides/<slide>/annotations.json` | region boxes per finding in level-0 pixel coordinates of the source slide |
| `slides/<slide>/patch_scores.csv` | per patch: grid position, coordinates, QC result, state, novelty and its three axes, D, H_chi, state entropy, transition rarity, candidate flag |
| `slides/<slide>/novelty_mask.png`, `novelty_map.npy`, `heatmaps.png` | binary candidate mask (thumbnail geometry), novelty lattice, four-panel heat map figure |
| `slides/<slide>/progression_scenarios.json` | only with `--scenarios`: exploratory MyUncle scenario analysis for the slide (labelled exploratory) |
| `summary_statistics.csv` | per slide: patches, artifacts, candidates, order vs spatial-shuffle null (z), D, H_chi, `rho_hat`, bistable-window flag |
| `images_used.csv` | **every input image and how it was used** (source statement: KPMP open data supplied by the user), size, optional SHA-256, resolution, patch counts, status |
| `run_manifest.json` | versions, configuration, encoder and model files, MyUncle core hash and `unmodified` flag, seeds, runtime, device |

## Evaluation (synthetic, reproducible)

`myuncle_vision/bench.py` builds two synthetic cohorts (development: 8 slides; held-out: 16 slides with three anomaly types never used in development, about 1% prevalence, stain and scanner shifts), computes every ablation and baseline
(Isolation Forest, LOF, kNN, Mahalanobis, each raw and lattice-smoothed) and the robustness suite, for the DINOv2 and the handcrafted encoder, and writes `results/bench_*.json`, `results/robustness_*.json`.
Settings were frozen in `results/frozen_settings.json` before the held-out cohort existed. `python docs/build/build_documents.py` turns the results into the manuscript and the HeroX pack; its asserts stop the build if a sentence stops being true.
`python -m myuncle_vision validate-synthetic` and `rarity-sweep` are the original v0.1 checks (`results/validation_*.json`, `rarity_sweep_*.json`, kept as development history). `python tests/test_all.py` runs the automated checks.

## Compute

DINOv2 ViT-S/14 on one CPU core processes roughly 15-30 patches per second; a slide with 10,000 patches takes minutes on a GPU and ten minutes or more on a CPU. Release 1 (6,256 slides, 2.4 TB) is a GPU/cluster workload: use `--max-patches`
to bound the work per slide, and the per-slide embedding cache (`<output>/cache`) to resume.

## Reproducibility and provenance

Seeds are fixed (`--seed`). The vendored MyUncle core is byte-identical to `github.com/sandlerleon/MyUncle` commit `48409f9` (`third_party/myuncle/SOURCE_COMMIT.txt`); the tests fail if it changes. A generative AI assistant (Claude, Anthropic) helped write the
code, tests, validation and documentation; the author reviewed and takes responsibility for them.

## Cite

See `CITATION.cff`. Pretrained model: Oquab et al., *DINOv2: Learning robust visual features without supervision*, TMLR (2024), arXiv:2304.07193.
