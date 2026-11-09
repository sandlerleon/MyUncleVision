# -*- coding: utf-8 -*-
"""Patch encoders.

    dinov2       DINOv2 ViT-S/14 (Oquab et al., 2023; timm weights 'vit_small_patch14_dinov2.lvd142m', natural-image pretraining, 384-d CLS embedding).  The default for real slides.
    handcrafted  deterministic colour / stain / gradient-orientation / texture features (no model file, 64-d).  Used by the tests and as the offline fallback; it is reported in the run manifest.

Every encoder maps uint8 patches (N, H, W, 3) to float32 embeddings (N, D).  Neither encoder is trained on kidney tissue; the platform is unsupervised and the choice is a documented limitation."""
import numpy as np

from .wsi import hed_od, hsv_arrays

DINO_NAME = "vit_small_patch14_dinov2.lvd142m"


class HandcraftedEncoder:
    name = "handcrafted"
    dim = 64
    model_files = []

    def encode(self, patches):
        feats = []
        for p in patches:
            feats.append(self._one(p))
        return np.asarray(feats, np.float32)

    @staticmethod
    def _one(p):
        import cv2
        small = p[::2, ::2]
        od = hed_od(small)
        H, S, V = hsv_arrays(small)
        f = []
        for ch in (od[..., 0], od[..., 1], S, V):                       # stain and colour: mean, sd, 3 quantiles
            f += [ch.mean(), ch.std(), *np.quantile(ch, [0.1, 0.5, 0.9])]
        hh, _ = np.histogram(H, bins=12, range=(0, 360), weights=S)
        f += list(hh / max(hh.sum(), 1e-9))
        g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        mag, ang = np.hypot(gx, gy), np.arctan2(gy, gx) % np.pi
        ho, _ = np.histogram(ang, bins=8, range=(0, np.pi), weights=mag)
        ho = ho / max(ho.sum(), 1e-9)
        f += list(ho)                                                   # orientation histogram (structure direction)
        f += [mag.mean() / 255.0, mag.std() / 255.0, float(np.mean(np.abs(ho - 1 / 8)))]
        lap = cv2.Laplacian(g, cv2.CV_32F)
        f += [lap.var() / 1e4, float(np.mean(od[..., 0] > 0.45)), float(np.mean(od[..., 1] > 0.5))]
        for sh in (1, 2, 4):                                            # coarse spatial layout: 2x2 block means of the hematoxylin density at three scales
            b = od[..., 0][:: sh, :: sh]
            h2, w2 = b.shape[0] // 2, b.shape[1] // 2
            f += [b[:h2, :w2].mean(), b[:h2, w2:].mean(), b[h2:, :w2].mean(), b[h2:, w2:].mean()]
        f = np.asarray(f, np.float32)
        out = np.zeros(HandcraftedEncoder.dim, np.float32)
        out[:min(len(f), 64)] = f[:64]
        return out


class DinoV2Encoder:
    name = "dinov2_vits14"
    model_files = [DINO_NAME]

    def __init__(self, device="auto", batch=32):
        import torch
        import timm
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch = batch
        self.model = timm.create_model(DINO_NAME, pretrained=True, num_classes=0, img_size=224).eval().to(device)   # 224 px input (16 x 16 tokens); the position embeddings are interpolated
        self.dim = int(self.model.num_features)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def encode(self, patches):
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(patches), self.batch):
                x = torch.from_numpy(np.ascontiguousarray(patches[i:i + self.batch])).to(self.device).permute(0, 3, 1, 2).float() / 255.0
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False) if x.shape[-1] != 224 else x
                z = self.model((x - self.mean) / self.std)
                out.append(z.float().cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.dim), np.float32)


def get_encoder(name="auto", device="auto"):
    """'dinov2', 'handcrafted' or 'auto' (dinov2 when torch+timm+weights are available, otherwise handcrafted)"""
    if name == "handcrafted":
        return HandcraftedEncoder()
    try:
        return DinoV2Encoder(device)
    except Exception as e:
        if name == "dinov2":
            raise
        print("[encoder] DINOv2 unavailable (%s); using the handcrafted encoder" % type(e).__name__)
        return HandcraftedEncoder()


if __name__ == "__main__":                       # python -m myuncle_vision.encoder --prefetch  (downloads the weights, e.g. while building the Docker image)
    import sys
    if "--prefetch" in sys.argv:
        e = DinoV2Encoder("cpu")
        print("DINOv2 ready, dim", e.dim)
