# -*- coding: utf-8 -*-
"""Command line.  The single command of the HeroX container:

    python -m myuncle_vision run --input /input --output /output
"""
import argparse
import json
import os
import sys

from .pipeline import RunConfig, run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="myuncle_vision", description="MyUncleVision: unsupervised discovery of unusual tissue states in whole-slide images")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyse a directory of whole-slide images")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--encoder", default="auto", choices=["auto", "dinov2", "handcrafted"])
    r.add_argument("--device", default="auto")
    r.add_argument("--encoder-module", default=None, help="your own encoder: path/to/file.py:function, mapping a uint8 array (N, H, W, 3) to (N, dim) embeddings (needs --encoder-dim)")
    r.add_argument("--encoder-dim", type=int, default=0)
    r.add_argument("--data-source", default=None, help="provenance statement written into images_used.csv")
    r.add_argument("--patch-px", type=int, default=224)
    r.add_argument("--mpp", type=float, default=0.5, help="target micrometres per pixel of a patch")
    r.add_argument("--max-patches", type=int, default=0, help="coarsen the grid until a slide has at most this many patches (0 = no limit)")
    r.add_argument("--q", type=float, default=0.02, help="top fraction of novelty scores that become candidate patches")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--weights", default="1,0,0", help="novelty axis weights: appearance,disorder,contacts (default 1,0,0: appearance only, the frozen setting; 0.4,0.4,0.2 adds the lattice axes)")
    r.add_argument("--scorer", default="knn", choices=["knn", "mahalanobis"], help="appearance scorer: multi-scale nearest-neighbour rarity (default) or Mahalanobis distance; both are lattice-smoothed")
    r.add_argument("--rarity", type=float, default=0.01, help="rarity scale: fraction of the reference set used as k; a state is only 'rare' if it covers less than this share of the cohort")
    r.add_argument("--no-figures", action="store_true")
    r.add_argument("--scenarios", action="store_true", help="EXPLORATORY: add the MyUncle scenario analysis per slide (a modelling analogy, not validated)")
    r.add_argument("--hash-files", action="store_true")
    br = sub.add_parser("bridge", help="MyUncle <-> MyUncleVision: simulate a MyUncle preset, analyse its states with MyUncleVision (order maps + label-free novelty), check the two implementations agree")
    br.add_argument("--preset", default="chiral_scaffold_biology")
    br.add_argument("--L", type=int, default=32)
    br.add_argument("--steps", type=int, default=60)
    br.add_argument("--every", type=int, default=15)
    br.add_argument("--seed", type=int, default=0)
    br.add_argument("--output", default="muv_bridge")
    dm = sub.add_parser("demo", help="end-to-end demonstration on four small synthetic slides (about two minutes, no downloads)")
    dm.add_argument("--output", default="muv_demo")
    v = sub.add_parser("validate-synthetic", help="run the validation benchmark on synthetic slides (writes results/validation.json)")
    v.add_argument("--output", default="results")
    v.add_argument("--encoder", default="handcrafted", choices=["auto", "dinov2", "handcrafted"])
    w = sub.add_parser("rarity-sweep", help="AUROC against the rarity scale on the synthetic benchmark (writes results/rarity_sweep_<encoder>.json)")
    w.add_argument("--output", default="results")
    w.add_argument("--encoder", default="handcrafted", choices=["auto", "dinov2", "handcrafted"])
    a = ap.parse_args(argv)
    if a.cmd == "run":
        enc = None
        if a.encoder_module:
            from .encoder import load_encoder_module
            assert a.encoder_dim > 0, "--encoder-dim is required with --encoder-module"
            enc = load_encoder_module(a.encoder_module, a.encoder_dim)
        extra = {"data_source": a.data_source} if a.data_source else {}
        cfg = RunConfig(patch_px=a.patch_px, **extra, encoder=a.encoder, device=a.device, mpp=a.mpp, max_patches=a.max_patches, q=a.q, seed=a.seed, weights=tuple(float(x) for x in a.weights.split(",")), rarity_fraction=a.rarity, scorer=a.scorer, figures=not a.no_figures, progression=a.scenarios, hash_files=a.hash_files)
        run(a.input, a.output, cfg, encoder=enc)
    elif a.cmd == "rarity-sweep":
        from .validate import rarity_sweep
        rarity_sweep(a.output, a.encoder)
    elif a.cmd == "bridge":
        from . import bridge as BR
        from .bridge import json_report
        rep = json_report(a.preset, a.L, a.steps, a.every, a.seed, a.output)
        print(json.dumps(rep["check"], indent=1))
        print("report and figure in", os.path.abspath(a.output))
    elif a.cmd == "demo":
        import tempfile
        from . import synthetic as S
        with tempfile.TemporaryDirectory() as tmp:
            for seed, an in ((1, ("disorganized",)), (2, ("novel_state",)), (3, ()), (4, ())):
                img, _ = S.make_slide(seed=seed, size=2688, anomalies=an, anomaly_radius=200)
                S.write_tiff(os.path.join(tmp, "demo_%d.tif" % seed), img, 0.5)
            res = run(tmp, a.output, RunConfig(encoder="handcrafted", q=0.05, figures=True))
        print("demo output in", os.path.abspath(a.output), "- open findings.csv and slides/*/heatmaps.png (the synthetic slides contain planted anomalies; they are not real tissue)")
    elif a.cmd == "validate-synthetic":
        from .validate import validate
        res = validate(a.output, a.encoder)
        print(json.dumps(res["headline"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
