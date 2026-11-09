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
    r.add_argument("--mpp", type=float, default=0.5, help="target micrometres per pixel of a patch")
    r.add_argument("--max-patches", type=int, default=0, help="coarsen the grid until a slide has at most this many patches (0 = no limit)")
    r.add_argument("--q", type=float, default=0.02, help="top fraction of novelty scores that become candidate patches")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--weights", default="0.4,0.4,0.2", help="novelty axis weights: appearance,disorder,contacts (e.g. 1,0,0 for the appearance axis alone)")
    r.add_argument("--rarity", type=float, default=0.01, help="rarity scale: fraction of the reference set used as k; a state is only 'rare' if it covers less than this share of the cohort")
    r.add_argument("--no-figures", action="store_true")
    r.add_argument("--no-progression", action="store_true")
    r.add_argument("--hash-files", action="store_true")
    v = sub.add_parser("validate-synthetic", help="run the validation benchmark on synthetic slides (writes results/validation.json)")
    v.add_argument("--output", default="results")
    v.add_argument("--encoder", default="handcrafted", choices=["auto", "dinov2", "handcrafted"])
    w = sub.add_parser("rarity-sweep", help="AUROC against the rarity scale on the synthetic benchmark (writes results/rarity_sweep_<encoder>.json)")
    w.add_argument("--output", default="results")
    w.add_argument("--encoder", default="handcrafted", choices=["auto", "dinov2", "handcrafted"])
    a = ap.parse_args(argv)
    if a.cmd == "run":
        cfg = RunConfig(encoder=a.encoder, device=a.device, mpp=a.mpp, max_patches=a.max_patches, q=a.q, seed=a.seed, weights=tuple(float(x) for x in a.weights.split(",")), rarity_fraction=a.rarity, figures=not a.no_figures, progression=not a.no_progression, hash_files=a.hash_files)
        run(a.input, a.output, cfg)
    elif a.cmd == "rarity-sweep":
        from .validate import rarity_sweep
        rarity_sweep(a.output, a.encoder)
    elif a.cmd == "validate-synthetic":
        from .validate import validate
        res = validate(a.output, a.encoder)
        print(json.dumps(res["headline"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
