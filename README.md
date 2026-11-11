# MyUncleVision

A reusable framework for **unsupervised anomaly discovery in images**: it cuts images into patches, puts the patch embeddings on a lattice, judges how rare each patch's appearance is at several prevalence scales, smooths the score over the lattice so that coherent regions stand out, and returns ranked, mapped, explained candidate regions with an audit trail.
It is the computer-vision sister of [MyUncle](https://github.com/sandlerleon/MyUncle) (Sandler, doi:[10.5281/zenodo.21223570](https://doi.org/10.5281/zenodo.21223570)): the order and entropy definitions of MyUncle's maintenance lattice are generalized to d dimensions and reused here, and MyUncle is vendored **unmodified**.
Whole-slide histopathology is the first application; nothing in the code is specific to any challenge, data set or tissue, and the same pipeline runs on any collection of images (SVS/TIFF/PNG/JPEG) with your own embedding model if you want one.

Leon Sandler, Independent Researcher, sandler.leon@gmail.com, ORCID 0009-0007-4584-808X. MIT license (code); MyUncle is vendored under its own MIT license.
Archived on Zenodo (all versions): software [10.5281/zenodo.24066260](https://doi.org/10.5281/zenodo.24066260), manuscript [10.5281/zenodo.24066256](https://doi.org/10.5281/zenodo.24066256).

> **Status and honest limits (v0.2.x).** The framework runs end to end on SVS and on plain images, is covered by automated tests, was built and run in Docker, and is evaluated on **synthetic** images only (a development cohort and a held-out cohort with settings frozen beforehand:
> `results/frozen_settings.json`, `results/bench_*.json`, `results/robustness_*.json`). Those images are not tissue and were written by the author of the method. **It has not been run on real histopathology yet**; that validation, with expert review, is the planned next step (see the paper's Section 5).
> What the evaluation showed: lattice smoothing raised the spatial coherence of every scorer; the frozen score was the best method on the held-out cohort with DINOv2 and a poor one with handcrafted features; a smoothed Mahalanobis baseline scored higher on the development cohort;
> the disorder and contact axes helped with handcrafted features and hurt with DINOv2 (so the default weights are 1,0,0). Everything the tool outputs about real data is a candidate for expert review, not a finding. Compare encoders and scorers on your data before trusting a ranking.

## Quick start

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m myuncle_vision demo                                    # four small synthetic slides, about 20 seconds with the handcrafted encoder: findings, heat maps, masks
python -m myuncle_vision run --input images/ --output out/       # your images (recursive); DINOv2 by default when torch/timm are installed
```

Docker (no manual step; DINOv2 weights are baked into the image):
```
docker build -t myuncle-vision .
docker run --rm -v /path/to/images:/input:ro -v /path/to/results:/output myuncle-vision
```

From Python (any image collection):
```python
import myuncle_vision as muv
res = muv.discover_images("images/", "out/", encoder="auto", mpp=0.5, patch_px=224, q=0.02)     # a directory
res = muv.discover_arrays([img1, img2, img3], "out/")                                           # uint8 RGB numpy arrays
enc = muv.CallableEncoder(my_function, dim=768, name="my-model")                                # your own embedding model
res = muv.discover_images("images/", "out/", encoder=enc)
```
Your own encoder on the command line: `--encoder-module path/to/file.py:function --encoder-dim 768` (see `examples/custom_encoder.py`). More in [docs/USAGE.md](docs/USAGE.md).
**Invoke it like MyUncle:** "use MyUncleVision to find unusual regions in these images" (point an assistant at this repository and at `docs/USAGE.md`).

## What it does

1. **Lattice.** Each image is cut into non-overlapping patches (default 224 px at 0.5 µm/px, set `--mpp` and `--patch-px` for your resolution); the patch grid is the lattice. Patches that fail QC (out of focus, pen ink, folds/debris, glass) are excluded everywhere (heuristics: tune them for new data).
2. **Embeddings** of every patch (DINOv2 ViT-S/14, a handcrafted colour/texture encoder, or your own), centred, reduced by a cohort PCA (16 components) and normalized to the unit sphere.
3. **Novelty score** (frozen default): appearance rarity, the mean distance to the nearest patches of a reference sample of the cohort, judged at five prevalence scales (0.5 to 10% of the reference set, highest rank), then averaged over the 3 x 3 lattice window. `--scorer mahalanobis` is an alternative.
4. **Order and entropy (MyUncle).** Local disorder `D = 1 - ||mean of the neighbours' unit embeddings||` is MyUncle's `MC.local_disorder` in any dimension; the von Mises-Fisher entropy of the neighbourhood and `H_chi = S_uniform - S_vmf` generalize MyUncle's order-entropy bridge from d = 2 (tests compare with the unmodified core). State-wise disorder and the rarity of contacts between states are computed as diagnostics and can be added with `--weights`.
5. **Discovery.** Top-scoring patches form connected regions; regions are clustered across images into candidate states, ranked by novelty, recurrence and reproducibility under two disjoint halves of the reference set.
6. **Explanations and maps.** Plain-language, number-only explanations per finding; heat maps, masks, per-patch scores.
7. **Exploratory scenarios** (opt-in `--scenarios`): a modelling analogy that maps the lattice onto MyUncle's simulation; not validated, not evidence about any process.

## Adapting it to a new project

Choose the patch size and resolution (`--patch-px`, `--mpp`); decide whether the default QC heuristics (blur, pen, dark, glass) make sense for your images; pick an embedding model (`--encoder`, `--encoder-module`); set the rarity scales deliberately (`--rarity`, the multi-scale default covers 0.5 to 10%);
decide the top fraction that counts as a candidate (`--q`); and run the baselines of the paper on your data before trusting the ranking (`myuncle_vision/bench.py`: Isolation Forest, LOF, kNN and Mahalanobis, raw and lattice-smoothed). Validate on your own labelled or expert-reviewed examples; the synthetic results here do not transfer automatically.

## Bridge to MyUncle

`myuncle_vision/bridge.py` connects the two repositories in both directions ([docs/BRIDGE.md](docs/BRIDGE.md)): simulate any MyUncle preset and analyse its lattice states with MyUncleVision's order maps and label-free novelty (`python -m myuncle_vision bridge`), check numerically that the two implementations agree (`verify_bridge`), or turn an analysed lattice of embeddings into the initial state of a MyUncle Monte Carlo (`slide_to_myuncle`, exploratory). MyUncle stays unmodified.

## Configuration search with GaaR

`myuncle_vision/gaar_opt.py` uses [GaaR](https://github.com/sandlerleon/GaaR) (vendored unmodified) to select a configuration of the score (smoothing type and window, axis weights, PCA dimension, largest rarity scale) on labeled development scenarios and tests the champions on fresh scenarios; see [docs/GAAR.md](docs/GAAR.md). It is supervised selection, not label-free adaptation; in the reported fresh test the gain of the selected configuration was not distinguishable from zero (paired 95% interval) and a random-search champion also beat the default, so any gain belongs to the configuration, not to the search algorithm. A controlled study on an independent generator (`myuncle_vision/morphology.py`) found that lattice smoothing hurt at every anomaly size, least for compact anomalies. The smoothing options (`smooth_type`, `smooth_window`) are available in `Config`; the defaults are unchanged.

## Models and files

| File | Function | Trained? |
|---|---|---|
| timm `vit_small_patch14_dinov2.lvd142m` (DINOv2 ViT-S/14, Oquab et al. 2023) | patch embeddings (384-d) | pretrained on natural images, not fine-tuned |
| `myuncle_vision/encoder.py::HandcraftedEncoder` | deterministic colour/stain/orientation/texture features (64-d); offline fallback and tests | no model file |
| your own function via `CallableEncoder` / `--encoder-module` | any embedding | yours |
| `third_party/gaar/GaaR_core.py` (GaaR, unmodified, SHA-256 in `GaaR_core.sha256`) | evolutionary configuration search (`gaar_opt.py`) | no |
| `third_party/myuncle/MyUncle_core.py` (MyUncle, unmodified, SHA-256 in `MyUncle_core.sha256`) | reference order-entropy definitions, bistability, kinetic Monte Carlo (scenarios) | no |
| `myuncle_vision/cohort.py` (PCA, k-means, nearest-neighbour reference fitted **at run time** on the input) | cohort space, prototype states, novelty axes | fitted per run; nothing is stored between runs |

## Outputs (`--output DIR`)

| Path | Content |
|---|---|
| `findings.json`, `findings.csv` | numbered candidate states F1..Fk: type, images, patches, area, axes, recurrence, reproducibility, score, explanation, region coordinates (level-0 pixels) |
| `slides/<image>/annotations.json` | region boxes per finding in the coordinates of the source image |
| `slides/<image>/patch_scores.csv` | per patch: grid position, coordinates, QC result, state, novelty and its axes, D, H_chi, state entropy, transition rarity, candidate flag |
| `slides/<image>/novelty_mask.png`, `novelty_map.npy`, `heatmaps.png` | binary candidate mask, novelty lattice, four-panel heat-map figure |
| `slides/<image>/progression_scenarios.json` | only with `--scenarios`: exploratory, labelled as such |
| `summary_statistics.csv` | per image: patches, artifacts, candidates, order against a spatial-shuffle null, D, H_chi |
| `images_used.csv` | every input image and how it was used (provenance statement via `--data-source`), size, optional SHA-256, resolution, patch counts, status |
| `run_manifest.json` | versions, configuration, encoder and model files, MyUncle core hash and `unmodified` flag, seeds, runtime, device |

## Evaluation (synthetic, reproducible)

`myuncle_vision/bench.py` builds two synthetic cohorts (development: 8 slides; held-out: 16 slides with three anomaly types never used in development, about 1% prevalence, stain and scanner shifts), computes every ablation and baseline for DINOv2 and the handcrafted encoder, and a robustness suite;
settings were frozen in `results/frozen_settings.json` before the held-out cohort existed. `python docs/build/build_documents.py` turns the results into the manuscript (its asserts stop the build if a sentence stops being true). `python tests/test_all.py` runs the automated checks.
The original v0.1 checks (`validate-synthetic`, `rarity-sweep`) are kept as development history. A second testbed on lattices simulated with MyUncle (`python -m myuncle_vision.bridge_bench`, results in `results/bridge_bench_*.json`) uses the same frozen settings and baselines; see [docs/BRIDGE.md](docs/BRIDGE.md).

## Compute

DINOv2 ViT-S/14 on one CPU core processes roughly 15-30 patches per second; use a GPU for large collections (`--max-patches N` bounds the work per image; the per-image embedding cache in `<output>/cache` makes runs resumable). Docker build: `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121` and `--gpus all` for CUDA.

## Applications

* [docs/herox/](docs/herox/): the NIDDK/HeroX Kidney AI Discovery Challenge (KPMP kidney whole-slide images): submission pack draft and run guide. That application is one use of this framework, not its scope.
* Papers: `docs/prl/` (letter for Pattern Recognition Letters, Elsevier: LaTeX source, PDF, highlights, cover letter; submitted, under editorial processing), `docs/MyUncleVision_Manuscript_ChemRxiv.*` (statistical-mechanics framing, prepared for ChemRxiv) and `docs/MyUncleVision_Manuscript_PRL.*` (first, histopathology-framed version, superseded).

## Reproducibility and provenance

Seeds are fixed (`--seed`). The vendored MyUncle core is byte-identical to `github.com/sandlerleon/MyUncle` commit `48409f9`; the tests fail if it changes. A generative AI assistant (Claude, Anthropic) helped write the code, tests, evaluation and documentation; the author reviewed and takes responsibility for them.

## Cite

See `CITATION.cff`. Pretrained model: Oquab et al., *DINOv2: Learning robust visual features without supervision*, TMLR (2024), arXiv:2304.07193.
