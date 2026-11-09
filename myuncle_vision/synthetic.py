# -*- coding: utf-8 -*-
"""Synthetic histology-like slides with KNOWN planted anomalies and artifacts, for validation and tests of the pipeline.

THESE ARE NOT KIDNEY TISSUE.  They imitate only what the platform measures: compartments with different local texture (tubule-like rings with nuclei, glomerulus-like discs, interstitium,
vessels), slide-to-slide stain variation, two planted anomaly types (three more for the held-out benchmark) and four artifact types.  Results on them show that the machinery detects and separates what it was built to detect;
they say nothing about the kidney.

anomalies:  disorganized  (the architecture is replaced by a patchwork of small tiles of different structure: neighbouring patches disagree, high local disorder)
            novel_state   (a coherent striped lattice in an unusual hue: ordered, but unlike anything else on the slide)
artifacts:  blur, pen (ink stroke), fold (dark band), bubble
"""
import os

import cv2
import numpy as np
from PIL import Image, TiffImagePlugin


def _noise(rng, shape, sigma, amp):
    n = rng.normal(0, 1, shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    return n / max(n.std(), 1e-6) * amp


def _pick_centres(rng, tissue, radii, margin, size):
    dist = cv2.distanceTransform(tissue, cv2.DIST_L2, 5)
    out = []
    for r in radii:
        ys, xs = np.where(dist > r + margin)
        for _ in range(400):
            i = rng.integers(len(ys))
            c = (int(xs[i]), int(ys[i]))
            if all(np.hypot(c[0] - o[0], c[1] - o[1]) > r + orr + 120 for o, orr in out):
                out.append((c, r))
                break
        else:
            raise RuntimeError("no room for a planted structure")
    return out


def make_slide(seed=0, size=4480, mpp=0.5, anomalies=(), artifacts=(), stain_jitter=0.05, anomaly_radius=360, palette_shift=0.0, soft_blur=0.8):
    """returns (uint8 RGB image, truth dict of boolean masks: tissue, <anomaly>, <artifact>, anomaly_any, artifact_any)"""
    rng = np.random.default_rng(seed)
    H = W = size
    img = np.full((H, W, 3), 245.0, np.float32)
    tissue = np.zeros((H, W), np.uint8)
    cv2.ellipse(tissue, (W // 2, H // 2), (int(W * 0.46), int(H * 0.44)), float(rng.uniform(0, 180)), 0, 360, 1, -1)
    tissue = (cv2.GaussianBlur(tissue.astype(np.float32) + _noise(rng, (H, W), 60, 0.35), (0, 0), 20) > 0.5).astype(np.uint8)
    tb = tissue.astype(bool)
    pink = np.array([228, 168, 198], np.float32) + rng.normal(0, 4, 3)
    base = pink[None, None, :] + _noise(rng, (H, W), 40, 7)[..., None] + _noise(rng, (H, W), 2.0, 5)[..., None]
    img[tb] = base[tb]
    nuc = (62, 34, 112)
    # --- structures
    occ = np.zeros((H // 16 + 1, W // 16 + 1), bool)

    def free(x, y, r):
        a = occ[max(0, (y - r) // 16):(y + r) // 16 + 1, max(0, (x - r) // 16):(x + r) // 16 + 1]
        return not a.any()

    def take(x, y, r):
        occ[max(0, (y - r) // 16):(y + r) // 16 + 1, max(0, (x - r) // 16):(x + r) // 16 + 1] = True

    def draw_glom(x, y, r):
        cv2.circle(img, (x, y), r + 10, (222, 146, 192), -1)
        cv2.circle(img, (x, y), r, (250, 238, 244), -1)
        cv2.circle(img, (x, y), int(r * 0.82), (178, 112, 172), -1)
        for _ in range(int(r * 0.7)):
            a, rr = rng.uniform(0, 2 * np.pi), r * 0.8 * np.sqrt(rng.uniform())
            cv2.circle(img, (int(x + rr * np.cos(a)), int(y + rr * np.sin(a))), 3, nuc, -1)

    def draw_tubule(x, y, r):
        cv2.circle(img, (x, y), r, (198, 128, 172), -1)
        cv2.circle(img, (x, y), r - 9, (246, 228, 238), -1)
        for k in range(int(rng.integers(9, 15))):
            a = rng.uniform(0, 2 * np.pi)
            cv2.circle(img, (int(x + (r - 4) * np.cos(a)), int(y + (r - 4) * np.sin(a))), 3, nuc, -1)

    ys, xs = np.where(tb)
    for _ in range(int(rng.integers(7, 11))):
        i = rng.integers(len(ys))
        x, y, r = int(xs[i]), int(ys[i]), int(rng.integers(80, 105))
        if tb[max(0, y - r - 12):y + r + 12, max(0, x - r - 12):x + r + 12].all() and free(x, y, r + 14):
            draw_glom(x, y, r)
            take(x, y, r + 14)
    for _ in range(int(rng.integers(2, 4))):
        i = rng.integers(len(ys))
        x, y, r = int(xs[i]), int(ys[i]), 46
        if tb[max(0, y - r - 8):y + r + 8, max(0, x - r - 8):x + r + 8].all() and free(x, y, r + 8):
            cv2.circle(img, (x, y), r, (190, 90, 120), -1)
            cv2.circle(img, (x, y), r - 12, (250, 232, 238), -1)
            take(x, y, r + 8)
    for _ in range(int(0.75 * tb.sum() / (np.pi * 40 ** 2))):
        i = rng.integers(len(ys))
        x, y, r = int(xs[i]), int(ys[i]), int(rng.integers(22, 34))
        if tb[max(0, y - r - 3):y + r + 3, max(0, x - r - 3):x + r + 3].all() and free(x, y, r + 3):
            draw_tubule(x, y, r)
            take(x, y, r + 3)
    truth = {"tissue": tissue.astype(bool)}
    centres = _pick_centres(rng, tissue, [anomaly_radius] * len(anomalies) + [220] * len(artifacts), 20, size)
    # --- anomalies
    for name, (c, r) in zip(anomalies, centres[:len(anomalies)]):
        m = np.zeros((H, W), np.uint8)
        cv2.circle(m, c, r, 1, -1)
        mb = m.astype(bool)
        if name == "disorganized":
            # architecture replaced by a patchwork: small tiles of different local structure side by side, so that neighbouring patches disagree
            patch = pink[None, None, :] + _noise(rng, (H, W), 40, 7)[..., None]
            img[mb] = patch[mb]
            canvas = img.copy()
            T = 120
            for ty in range(c[1] - r, c[1] + r, T):
                for tx in range(c[0] - r, c[0] + r, T):
                    kind = int(rng.integers(0, 5))
                    x1, y1 = min(tx + T, W), min(ty + T, H)
                    if x1 <= tx or y1 <= ty or tx < 0 or ty < 0:
                        continue
                    if kind == 0:                                                   # dense nuclei
                        for _ in range(110):
                            cv2.circle(canvas, (int(rng.integers(tx, x1)), int(rng.integers(ty, y1))), 3, nuc, -1)
                    elif kind == 1:                                                 # fibres
                        for _ in range(14):
                            p0 = (int(rng.integers(tx, x1)), int(rng.integers(ty, y1)))
                            th = rng.uniform(0, np.pi)
                            cv2.line(canvas, p0, (int(p0[0] + 40 * np.cos(th)), int(p0[1] + 40 * np.sin(th))), (180, 100, 150), 3)
                    elif kind == 2:                                                 # pale lumen
                        cv2.rectangle(canvas, (tx + 10, ty + 10), (x1 - 10, y1 - 10), (246, 232, 240), -1)
                    elif kind == 3:                                                 # tubule-like rings
                        for _ in range(2):
                            draw_tubule(int(rng.integers(tx + 28, max(tx + 29, x1 - 28))), int(rng.integers(ty + 28, max(ty + 29, y1 - 28))), 24)
                    else:                                                           # hyper-eosinophilic
                        cv2.rectangle(canvas, (tx, ty), (x1, y1), (205, 110, 150), -1)
            img[mb] = canvas[mb]
        elif name == "novel_state":
            yy, xx = np.mgrid[0:H, 0:W]
            ang = np.deg2rad(32)
            stripe = 0.5 + 0.5 * np.sin(2 * np.pi * (xx * np.cos(ang) + yy * np.sin(ang)) / 18.0)
            col = (172 + (118 - 172) * stripe[..., None] * np.array([1, 1, 1], np.float32), )[0]
            col = np.stack([172 - 52 * stripe, 190 - 45 * stripe, 214 - 30 * stripe], -1)
            img[mb] = col[mb] + _noise(rng, (H, W), 1.5, 3)[..., None][mb]
        elif name == "dense_tubules":                                                  # held-out type: tightly packed small tubules (an architecture change that stays tissue-like)
            patch = pink[None, None, :] + _noise(rng, (H, W), 40, 7)[..., None]
            img[mb] = patch[mb]
            for _ in range(int(1.0 * mb.sum() / (np.pi * 24 ** 2))):
                a_, rr = rng.uniform(0, 2 * np.pi), max(r - 30, 1) * np.sqrt(rng.uniform())
                draw_tubule(int(c[0] + rr * np.cos(a_)), int(c[1] + rr * np.sin(a_)), int(rng.integers(13, 19)))
        elif name == "nuclear_crowding":                                               # held-out type: interstitium crowded with nuclei, no other change
            patch = pink[None, None, :] + _noise(rng, (H, W), 40, 7)[..., None]
            img[mb] = patch[mb]
            for _ in range(int(0.012 * mb.sum() / 4)):
                a_, rr = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.uniform())
                cv2.circle(img, (int(c[0] + rr * np.cos(a_)), int(c[1] + rr * np.sin(a_))), 3, nuc, -1)
        elif name == "glomerular_cluster":                                             # held-out type: a cluster of glomerulus-like discs where there is normally one at a time
            patch = pink[None, None, :] + _noise(rng, (H, W), 40, 7)[..., None]
            img[mb] = patch[mb]
            for k in range(6):
                a_, rr = 2 * np.pi * k / 6 + rng.uniform(-0.2, 0.2), 0.55 * r
                draw_glom(int(c[0] + rr * np.cos(a_)), int(c[1] + rr * np.sin(a_)), int(rng.integers(52, 66)))
        truth[name] = mb
    # --- slide appearance first (noise, mild blur), then the artifacts that sit on top of it
    img += _noise(rng, (H, W), 1.0, 2.5)[..., None]
    img = cv2.GaussianBlur(img, (0, 0), soft_blur)
    img = img * (1.0 + palette_shift * np.array([0.07, -0.05, 0.06], np.float32))[None, None, :]      # stain / scanner shift of the whole slide
    for name, (c, r) in zip(artifacts, centres[len(anomalies):]):
        m = np.zeros((H, W), np.uint8)
        if name == "blur":
            cv2.circle(m, c, r, 1, -1)
            bl = cv2.GaussianBlur(img, (0, 0), 14)
            img[m.astype(bool)] = bl[m.astype(bool)]
        elif name == "pen":
            cv2.line(m, (c[0] - r, c[1] - r // 2), (c[0] + r, c[1] + r // 2), 1, 34)
            img[m.astype(bool)] = (28, 58, 190)
        elif name == "fold":
            cv2.ellipse(m, c, (r, 70), float(rng.uniform(0, 180)), 0, 360, 1, -1)
            img[m.astype(bool)] *= 0.22
        elif name == "bubble":
            cv2.circle(m, c, r, 1, 12)
            img[m.astype(bool)] = (232, 232, 238)
        truth[name] = m.astype(bool)
    img *= (1.0 + rng.normal(0, stain_jitter, 3))[None, None, :]
    out = np.clip(img, 0, 255).astype(np.uint8)
    fold = truth.get("fold", np.zeros((H, W), bool))
    glass = ~tb & ~fold
    out[glass] = np.maximum(out[glass], 238)
    truth["anomaly_any"] = np.any([truth[a] for a in anomalies], axis=0) if anomalies else np.zeros((H, W), bool)
    truth["artifact_any"] = np.any([truth[a] for a in artifacts], axis=0) if artifacts else np.zeros((H, W), bool)
    return out, truth


def write_tiff(path, img, mpp=0.5):
    """plain RGB TIFF with the resolution in the image description ('MPP=0.5'), read by the array backend"""
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[270] = "MPP=%g" % mpp
    Image.fromarray(img).save(path, tiffinfo=ifd, compression="tiff_deflate")


def patch_truth(truth, grid, names, thr=0.5):
    """per-patch fraction of overlap with every named truth mask, for the patches of a patch grid (rows, cols, side0 as returned by wsi.patch_grid)"""
    out = {}
    side = grid["side0"]
    for n in names:
        m = truth[n].astype(np.float32)
        fr = cv2.resize(m, (grid["shape"][1], grid["shape"][0]), interpolation=cv2.INTER_AREA)
        out[n] = fr[grid["rows"], grid["cols"]]
    return out
