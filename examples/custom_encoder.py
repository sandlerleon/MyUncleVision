# -*- coding: utf-8 -*-
"""Example of your own embedding model for MyUncleVision.  Any function mapping a uint8 array (N, H, W, 3) to float embeddings (N, D) works:

    python -m myuncle_vision run --input images/ --output out/ --encoder-module examples/custom_encoder.py:encode --encoder-dim 48

Replace the body with a call to your own network (pathology-pretrained, CLIP, a domain model ...)."""
import numpy as np


def encode(patches):
    """48-d colour-histogram embedding (a stand-in for a real model)"""
    out = []
    for p in patches:
        f = [np.histogram(p[..., c], bins=16, range=(0, 255), density=True)[0] for c in range(3)]
        out.append(np.concatenate(f))
    return np.asarray(out, np.float32)
