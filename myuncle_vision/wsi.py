# -*- coding: utf-8 -*-
"""Whole-slide images: reading (OpenSlide for SVS and other vendor formats, array backend for plain TIFF/PNG/JPEG), tissue mask, patch grid and patch QC.

Patch grid: non-overlapping square patches of `patch_px` pixels at a target resolution `mpp` (micrometres per pixel), placed on a regular grid so that the patch (row, col)
is the lattice site of MyUncleVision.  QC thresholds are heuristics that must be tuned on real slides; every patch keeps its QC numbers in the output."""
import os
import math

import cv2
import numpy as np
from PIL import Image

try:                                              # SVS, NDPI, ... need the OpenSlide library
    import openslide
except Exception:                                 # pragma: no cover
    openslide = None

SLIDE_EXT = (".svs", ".tif", ".tiff", ".ndpi", ".scn", ".mrxs", ".png", ".jpg", ".jpeg")
DEFAULT_MPP = 0.5                                 # assumed when a file carries no resolution (recorded as mpp_assumed)


# ------------------------------------------------------------------------------------------------ colour
_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])   # Ruifrok & Johnston stain vectors (H, E, DAB), rows
_HED_INV = np.linalg.inv(_HED / np.linalg.norm(_HED, axis=1, keepdims=True))


def hed_od(rgb):
    """optical densities of hematoxylin, eosin and the residual channel for an RGB uint8 array (..., 3)"""
    od = -np.log((np.asarray(rgb, float) + 1.0) / 256.0)
    return od @ _HED_INV


def hsv_arrays(rgb):
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    return hsv[..., 0].astype(float) * 2.0, hsv[..., 1].astype(float) / 255.0, hsv[..., 2].astype(float) / 255.0      # hue in degrees


# ------------------------------------------------------------------------------------------------ slide backends
class Slide:
    """minimal common interface: .name, .dimensions (W, H at level 0), .mpp, .mpp_assumed, thumbnail(), read(x, y, side0, out_px)"""

    def __init__(self, path):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self._os = None
        self._arr = None
        self.mpp_assumed = False
        use_os = openslide is not None and path.lower().endswith((".svs", ".ndpi", ".scn", ".mrxs", ".tif", ".tiff"))
        if use_os:
            try:
                self._os = openslide.OpenSlide(path)
            except Exception:
                self._os = None
        if self._os is not None:
            self.backend = "openslide"
            self.dimensions = tuple(self._os.dimensions)
            mp = self._os.properties.get(openslide.PROPERTY_NAME_MPP_X)
            self.mpp = float(mp) if mp else None
        else:
            self.backend = "array"
            self._arr = np.asarray(Image.open(path).convert("RGB"))
            self.dimensions = (self._arr.shape[1], self._arr.shape[0])
            self.mpp = _tiff_mpp(path)
        if not self.mpp:
            self.mpp = DEFAULT_MPP
            self.mpp_assumed = True

    def thumbnail(self, max_side=2048):
        W, H = self.dimensions
        scale = max(W, H) / float(max_side)
        scale = max(scale, 1.0)
        tw, th = int(round(W / scale)), int(round(H / scale))
        if self._os is not None:
            img = self._os.get_thumbnail((tw, th)).convert("RGB")
            arr = np.asarray(img)
        else:
            arr = cv2.resize(self._arr, (tw, th), interpolation=cv2.INTER_AREA)
        return arr, W / float(arr.shape[1])        # level-0 pixels per thumbnail pixel

    def read(self, x, y, side0, out_px):
        """RGB uint8 (out_px, out_px, 3) of the square of side `side0` level-0 pixels with top-left corner (x, y), white outside the slide"""
        if self._os is not None:
            level = self._os.get_best_level_for_downsample(max(side0 / float(out_px), 1.0))
            ds = self._os.level_downsamples[level]
            w = int(math.ceil(side0 / ds))
            reg = self._os.read_region((int(x), int(y)), level, (w, w))
            bg = Image.new("RGB", reg.size, (255, 255, 255))
            bg.paste(reg, mask=reg.split()[3])
            img = bg
            if img.size != (out_px, out_px):
                img = img.resize((out_px, out_px), Image.LANCZOS)
            return np.asarray(img)
        W, H = self.dimensions
        x0, y0 = int(x), int(y)
        x1, y1 = min(W, x0 + int(side0)), min(H, y0 + int(side0))
        crop = np.full((int(side0), int(side0), 3), 255, np.uint8)
        if x1 > x0 and y1 > y0:
            crop[:y1 - y0, :x1 - x0] = self._arr[y0:y1, x0:x1]
        if crop.shape[0] != out_px:
            crop = cv2.resize(crop, (out_px, out_px), interpolation=cv2.INTER_AREA if crop.shape[0] > out_px else cv2.INTER_LINEAR)
        return crop

    def close(self):
        if self._os is not None:
            self._os.close()


def _tiff_mpp(path):
    """micrometres per pixel from a TIFF resolution tag or an 'MPP = ' description (synthetic slides write the latter); None if absent"""
    try:
        im = Image.open(path)
        desc = str(im.tag_v2.get(270, "")) if hasattr(im, "tag_v2") else ""
        if "MPP" in desc.upper():
            for tok in desc.replace("|", ";").split(";"):
                if tok.strip().upper().startswith("MPP"):
                    return float(tok.split("=")[1])
    except Exception:
        pass
    return None


def find_slides(folder):
    """sorted list of slide files below folder"""
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(SLIDE_EXT):
                out.append(os.path.join(root, f))
    return sorted(out)


# ------------------------------------------------------------------------------------------------ tissue and grid
def tissue_mask(thumb, min_area_frac=2e-5):
    """boolean tissue mask of a thumbnail: pixels that are not bright, colourless glass (V < 0.96 and either saturation >= 0.05 or V < 0.90), morphological closing and opening, small objects removed"""
    _, S, V = hsv_arrays(thumb)
    # glass is bright (V near 1) and nearly colourless; pale H&E tissue (many lumens) can have a saturation hardly above the glass, so darkness is the main criterion and saturation the second one
    m = (V < 0.96) & ((S >= 0.05) | (V < 0.90))
    m = m.astype(np.uint8)
    k = max(3, int(round(min(m.shape) * 0.002)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_frac * m.size:
            keep |= lab == i
    return keep


def patch_grid(slide, patch_px=224, mpp=0.5, min_tissue=0.5, max_side=2048):
    """grid of candidate patches over the tissue.  Returns dict: rows, cols (int arrays of lattice indices), x0, y0 (level-0 pixels), side0, shape (nrows, ncols), tissue_frac, thumb, mask, scale"""
    thumb, scale = slide.thumbnail(max_side)
    mask = tissue_mask(thumb)
    side0 = patch_px * mpp / slide.mpp                    # patch side in level-0 pixels
    W, H = slide.dimensions
    ncols, nrows = int(math.ceil(W / side0)), int(math.ceil(H / side0))
    frac = cv2.resize(mask.astype(np.float32), (ncols, nrows), interpolation=cv2.INTER_AREA)
    # thumbnail covers the same area as the slide, so a resize to the grid shape is an area average of the mask over every patch
    r, c = np.where(frac >= min_tissue)
    return dict(rows=r, cols=c, x0=(c * side0), y0=(r * side0), side0=side0, shape=(nrows, ncols), tissue_frac=frac[r, c], thumb=thumb, mask=mask, scale=scale)


# ------------------------------------------------------------------------------------------------ patch QC and stain descriptors
def patch_qc(rgb, pen_thr=0.05, dark_thr=0.20, white_thr=0.60):
    """QC numbers and an artifact flag for one patch (uint8 RGB).  Heuristics: pen ink (saturated blue/green hues), fold or dark debris (very dark pixels),
    mostly glass (white).  Blur is judged per slide in finalize_qc.  Returns dict(blur, pen, dark, white, artifact, reason)"""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
    H, S, V = hsv_arrays(rgb)
    pen = float(np.mean((H > 90) & (H < 240) & (S > 0.60) & (V > 0.15)))
    dark = float(np.mean(V < 0.25))
    white = float(np.mean((V > 0.93) & (S < 0.08)))
    reasons = []
    if pen > pen_thr:
        reasons.append("pen")
    if dark > dark_thr:
        reasons.append("fold_or_debris")
    if white > white_thr:
        reasons.append("glass")
    return dict(blur=blur, pen=pen, dark=dark, white=white, artifact=bool(reasons), reason="+".join(reasons))


def finalize_qc(qc, blur_abs=3.0, blur_rel=0.15):
    """add the blur rule to the stored per-patch QC numbers: a patch is out of focus when its Laplacian variance is below an absolute floor or below blur_rel times the median of its own slide
    (stain and magnification change the absolute scale, the slide's own median does not).  Returns the QC dict with artifact and reason updated"""
    blur = np.asarray(qc["blur"], float)
    med = float(np.median(blur)) if len(blur) else 0.0
    isblur = (blur < blur_abs) | (blur < blur_rel * med)
    reason = np.array([("blur" + ("+" + r if r else "")) if b else r for b, r in zip(isblur, qc["reason"])], dtype=object)
    out = dict(qc)
    out["artifact"] = np.asarray(qc["artifact"], bool) | isblur
    out["reason"] = reason
    return out


def stain_descriptors(rgb):
    """hematoxylin/eosin optical-density summaries and a nuclear-density proxy (fraction of strongly hematoxylin pixels)"""
    od = hed_od(rgb[::2, ::2])
    h, e = od[..., 0], od[..., 1]
    return dict(h_mean=float(h.mean()), e_mean=float(e.mean()), nuc_frac=float(np.mean(h > 0.45)), od_sd=float(od[..., :2].std()))
