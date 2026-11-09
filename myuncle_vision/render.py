# -*- coding: utf-8 -*-
"""Heat maps and masks.  The lattice (patch grid) is drawn on the slide thumbnail with the exact geometry of the patch grid."""
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _extent(s):
    nr, nc = s.shape
    w = nc * s.side0 / s.scale
    h = nr * s.side0 / s.scale
    return [0, w, h, 0]


def slide_figure(s, findings, path, dpi=110):
    """four panels: candidate regions, novelty, local disorder D, entropy information H_chi"""
    ext = _extent(s)
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    th = s.thumb
    for a, (key, title, cmap) in zip(ax, [(None, "candidate regions", None), ("novelty", "novelty score", "magma"), ("D", "local disorder D (MyUncle)", "viridis"), ("H_chi", "order information H_chi", "cividis")]):
        a.imshow(th, extent=[0, th.shape[1], th.shape[0], 0], interpolation="nearest")
        if key == "novelty":
            m = np.full(s.shape, np.nan)
            m[s.rows, s.cols] = s.extra["novelty"]
        elif key == "D":
            m = s.extra["maps"]["D"]
        elif key == "H_chi":
            m = s.extra["maps"]["H_chi"]
        else:
            m = None
        if m is not None:
            a.imshow(np.ma.masked_invalid(m), extent=ext, cmap=cmap, alpha=0.75, interpolation="nearest")
        a.set_title("%s: %s" % (s.name[:28], title), fontsize=8)
        a.set_xlim(0, th.shape[1]); a.set_ylim(th.shape[0], 0)
        a.axis("off")
    for f in findings:
        for r in f["regions"]:
            if r["slide_name"] != s.name:
                continue
            x0, y0, x1, y1 = [v / s.scale for v in (r["x0"], r["y0"], r["x1"], r["y1"])]
            ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1.4))
            ax[0].text(x0, y0, f["id"], color="yellow", fontsize=8, va="bottom", weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def novelty_mask_png(s, path):
    """binary PNG (thumbnail size): candidate patches painted on the thumbnail geometry"""
    th = s.thumb
    m = np.zeros(th.shape[:2], np.uint8)
    cand = s.extra["cand_mask"]
    k = s.side0 / s.scale
    for r, c in zip(*np.where(cand)):
        x0, y0 = int(round(c * k)), int(round(r * k))
        x1, y1 = int(round((c + 1) * k)), int(round((r + 1) * k))
        m[y0:y1, x0:x1] = 255
    cv2.imwrite(path, m)
