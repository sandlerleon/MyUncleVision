# Using MyUncleVision on your own images

MyUncleVision finds **unusual regions** in a collection of images without labels. It is a framework, not a classifier: it ranks patches whose appearance is rare in your collection, makes the ranking spatially coherent on the patch lattice, and writes maps, per-patch tables, ranked candidate states and an audit trail.
It does not know what is interesting; a person has to look at the regions. Validate it on examples you trust before relying on a ranking (the evaluation in the paper is synthetic only).

## 1. Install and try

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m myuncle_vision demo
```

## 2. Run on a directory

```
python -m myuncle_vision run --input images/ --output out/ --mpp 0.5 --patch-px 224 --q 0.02
```
Images are searched recursively (SVS, NDPI, TIFF, PNG, JPEG). If an image carries no resolution, `--mpp` is assumed (recorded as `mpp_assumed` in `images_used.csv`); for ordinary photographs `--mpp` only sets the patch size in "micrometres", so choose `--patch-px` and `--mpp` such that a patch covers the scale of structure you care about.

Useful options: `--encoder dinov2|handcrafted`, `--encoder-module file.py:function --encoder-dim N` (your own model), `--scorer knn|mahalanobis`, `--weights 1,0,0` (appearance only, the frozen default; `0.4,0.4,0.2` adds the lattice axes), `--rarity 0.01`, `--q`, `--max-patches N`, `--hash-files`, `--data-source "..."`, `--seed`, `--no-figures`, `--scenarios` (exploratory).

## 3. From Python

```python
import numpy as np
import myuncle_vision as muv

res = muv.discover_images("images/", "out/", encoder="handcrafted", q=0.02)
res = muv.discover_arrays([np.asarray(img) for img in my_images], "out/")

def my_embeddings(patches):                    # uint8 (N, H, W, 3) -> float (N, D)
    return my_model(patches)

enc = muv.CallableEncoder(my_embeddings, dim=768, name="my-model", model_files=["my_model.pt"])
res = muv.discover_images("images/", "out/", encoder=enc)
print(res["findings"][0]["explanation"])
```

## 4. Reading the output

* `findings.csv`: ranked candidate states (recurrence across images, reproducibility, explanation). A finding in one image only may be an artifact of that image.
* `slides/<image>/heatmaps.png`: candidate regions, novelty, local disorder D and order information H_chi on the image.
* `slides/<image>/patch_scores.csv`: every patch with its QC result and scores; `novelty_mask.png` and `annotations.json` carry the regions in image coordinates.
* `run_manifest.json`: everything needed to reproduce the run, including the hash of the unmodified MyUncle core.

## 5. Checklist for a new domain

1. Patch size and resolution such that a patch holds one unit of structure. 2. QC heuristics (blur, pen ink, dark debris, glass) fit your artifacts? They are heuristics for stained tissue slides. 3. Embedding model: compare at least two (the paper's DINOv2 and handcrafted encoders disagreed). 4. Rarity scale: the multi-scale default covers 0.5 to 10% of the cohort; a state more common than that will not be flagged.
5. Compare against the baselines on your data (`myuncle_vision/bench.py`) and look at the unsmoothed score too: on small anomalies the smoothing can dilute the signal (it did with handcrafted features on the held-out synthetic cohort). 6. Review the top regions with someone who knows the domain, blinded to the score if you want to measure yield.

## 6. Reproduce the paper's evaluation

`python -m myuncle_vision` is the tool; the evaluation harness is `myuncle_vision/bench.py` (development and held-out synthetic cohorts, ablations, baselines, robustness) and `docs/build/build_documents.py` (manuscript from the result files). `python tests/test_all.py` runs the checks.
