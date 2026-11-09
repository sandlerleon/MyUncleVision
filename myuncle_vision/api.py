# -*- coding: utf-8 -*-
"""Python API: use MyUncleVision from your own code, for any image collection (not tied to any challenge or tissue).

    import myuncle_vision as muv
    res = muv.discover_images("images/", "results/", encoder="handcrafted")            # a directory of images (SVS, TIFF, PNG, JPEG ...)
    res = muv.discover_arrays([img1, img2, ...], "results/")                    # numpy uint8 RGB arrays
    enc = muv.CallableEncoder(my_embedding_function, dim=768, name="mymodel")  # your own embedding model
    res = muv.discover_images("images/", "results/", encoder=enc, mpp=0.25, patch_px=256, q=0.01)

The result is the dict returned by pipeline.run (slides, findings, manifest, summary); everything is also written to the output directory.
"""
import os
import tempfile

import numpy as np
from PIL import Image

from .encoder import CallableEncoder, get_encoder       # noqa: F401
from .pipeline import RunConfig, run


def discover_images(input_dir, output_dir, encoder="auto", log=print, **options):
    """run the pipeline on a directory.  encoder: 'auto' | 'dinov2' | 'handcrafted' or an encoder object (CallableEncoder).  options: any RunConfig field
    (mpp, patch_px, q, weights, scorer, rarity_fraction, max_patches, figures, progression, data_source, ...)"""
    if not isinstance(encoder, str):
        enc, name = encoder, getattr(encoder, "name", "custom")
    else:
        enc, name = None, encoder
    cfg = RunConfig(encoder=name, **options)
    return run(input_dir, output_dir, cfg, log=log, encoder=enc)


def discover_arrays(images, output_dir, names=None, encoder="auto", log=print, **options):
    """run the pipeline on in-memory uint8 RGB arrays (written as lossless PNG into a temporary directory)"""
    with tempfile.TemporaryDirectory() as tmp:
        for i, a in enumerate(images):
            nm = names[i] if names else "image_%03d" % i
            Image.fromarray(np.asarray(a, np.uint8)).save(os.path.join(tmp, nm + ".png"))
        return discover_images(tmp, output_dir, encoder=encoder, log=log, **options)
