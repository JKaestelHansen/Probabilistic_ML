"""Frozen image backbones: image -> fixed-length embedding.

    "handcrafted"   no download; focus / noise / exposure / object-morphology features.
                    A strong QC baseline and what the tests run on.
    "dinov2"        facebook/dinov2-base (self-supervised ViT; best general-purpose features)
    "dinov2-small"  facebook/dinov2-small
    "siglip2"       google/siglip2-base-patch16-224 image tower
    "hf:<model_id>" any Hugging Face vision model loadable with AutoModel (mean-pooled last hidden state),
                    e.g. "hf:facebook/dinov3-vitb16-pretrain-lvd1689m" or a microscopy model.
"""
import numpy as np
from scipy import ndimage


ALIASES = {
    "dinov2": "facebook/dinov2-base",
    "dinov2-small": "facebook/dinov2-small",
    "siglip2": "google/siglip2-base-patch16-224",
}


def normalize_image(img, low=1.0, high=99.8):
    """Any 2D (grayscale / microscopy, any dtype) or HxWxC image -> float32 HxWx3 in [0, 1].

    Percentile normalisation is the usual choice for fluorescence data with arbitrary intensity scales.
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] > 3:
        img = img[..., :3]
    lo, hi = np.percentile(img, [low, high])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def _morphology_features(gray):
    """Threshold + connected components -> aggregated per-object shape statistics."""
    smooth = ndimage.gaussian_filter(gray, 1.0)
    bg = np.median(smooth)
    mad = np.median(np.abs(smooth - bg)) * 1.4826 + 1e-6
    mask = smooth > bg + 3 * mad
    mask = ndimage.binary_opening(mask, iterations=1)
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros(9, dtype=np.float32)
    feats = []
    for sl, i in zip(ndimage.find_objects(lab), range(1, n + 1)):
        obj = lab[sl] == i
        area = obj.sum()
        if area < 4:
            continue
        yy, xx = np.nonzero(obj)
        cov = np.cov(np.stack([yy, xx])) + 1e-6 * np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))
        ecc = np.sqrt(1 - ev[0] / ev[1])
        perim = area - ndimage.binary_erosion(obj).sum()
        circularity = 4 * np.pi * area / max(perim, 1) ** 2
        extent = area / obj.size
        feats.append([np.log(area), ecc, circularity, extent])
    if not feats:
        return np.zeros(9, dtype=np.float32)
    f = np.array(feats)
    return np.concatenate([[np.log1p(len(f))], f.mean(0), f.std(0)]).astype(np.float32)


def handcrafted_features(img):
    """Image-quality + morphology descriptor (~30 dims) computed on the raw intensities."""
    raw = np.asarray(img, dtype=np.float32)
    gray = raw.mean(-1) if raw.ndim == 3 else raw
    g = (gray - gray.mean()) / (gray.std() + 1e-6)

    lap = ndimage.laplace(g)
    gx, gy = ndimage.sobel(g, 0), ndimage.sobel(g, 1)
    # high-frequency noise estimate (robust, Immerkaer-style) relative to signal range
    noise = np.median(np.abs(lap)) / 0.6745
    # radial power spectrum: blur removes high-frequency power
    f = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = g.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
    bands = [np.log1p(f[(r >= a) & (r < b)].mean()) for a, b in zip(np.linspace(0, 1, 9)[:-1], np.linspace(0, 1, 9)[1:])]

    hi = raw.max()
    exposure = [
        np.mean(gray >= hi) if hi > 0 else 0.0,  # fraction at max value (saturation / clipping)
        np.log1p(np.percentile(gray, 99.9) - np.percentile(gray, 50)),  # dynamic range above background
        np.log1p(gray.std()),
    ]
    sharp = [np.log(lap.var() + 1e-8), np.log(np.mean(gx**2 + gy**2) + 1e-8), np.log(noise + 1e-8),
             float(np.mean(g**3)), float(np.mean(g**4))]
    return np.concatenate([bands, exposure, sharp, _morphology_features(gray)]).astype(np.float32)


class Backbone:
    def __init__(self, name="handcrafted", device=None):
        self.name = name
        self.device = device
        self.model = None
        if name != "handcrafted":
            self._load_hf(ALIASES.get(name, name.removeprefix("hf:")))

    def _load_hf(self, model_id):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.device = self.device or ("cuda" if torch.cuda.is_available()
                                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        # CLIP/SigLIP-style dual encoders: keep only the image tower
        self.model = (getattr(model, "vision_model", None) or model).to(self.device).eval()

    def __call__(self, images, batch_size=16):
        if self.model is None:
            return np.stack([handcrafted_features(im) for im in images])
        import torch

        out = []
        for i in range(0, len(images), batch_size):
            batch = [(normalize_image(im) * 255).astype(np.uint8) for im in images[i:i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                res = self.model(**inputs)
            hidden = res.last_hidden_state
            # pooled token (DINO CLS / SigLIP attention pool) + mean of patch tokens
            pooled = getattr(res, "pooler_output", None)
            if pooled is None:
                pooled = hidden[:, 0]
            out.append(torch.cat([pooled, hidden.mean(1)], -1).float().cpu().numpy())
        return np.concatenate(out)
