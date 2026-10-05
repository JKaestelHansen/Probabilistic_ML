"""Synthetic microscopy-like benchmark built on ../generate_blob_images.py (Gaussian background, bright
blobs, Poisson noise), extended with a morphology class and acquisition defects so every decision
primitive has ground truth:

    morphology  Choice  round / elongated / irregular   (shape of the blobs in the image)
    focus       Score   very blurry < blurry < soft < sharp   (defocus level)
    usable      Noul    sharp-or-soft focus AND enough signal AND no saturation
"""
import numpy as np
from scipy import ndimage

from .schema import Choice, Noul, Score

MORPHOLOGIES = ["round", "elongated", "irregular"]
FOCUS_LEVELS = ["very blurry", "blurry", "soft", "sharp"]
FOCUS_SIGMA = {0: 4.0, 1: 2.5, 2: 1.2, 3: 0.0}  # Gaussian defocus sigma per level

QUESTIONS = [
    Choice("morphology", "What is the dominant shape of the objects?", MORPHOLOGIES),
    Score("focus", "How well focused is the image?", FOCUS_LEVELS),
    Noul("usable", "Is this image usable for analysis?",
         yes_text="a sharp, well exposed fluorescence microscopy image",
         no_text="a blurry, dark or saturated fluorescence microscopy image"),
]


def _blob_mask(rng, H, W, kind, size):
    yy, xx = np.mgrid[:H, :W]
    cy, cx = rng.uniform(0.1, 0.9) * H, rng.uniform(0.1, 0.9) * W
    if kind == "round":
        return np.hypot(yy - cy, xx - cx) <= size / 2
    if kind == "elongated":
        a, b = size * rng.uniform(0.9, 1.3), size / rng.uniform(3.0, 4.5)
        th = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        return (u / a) ** 2 + (v / b) ** 2 <= 1
    # irregular: union of a few offset discs
    m = np.zeros((H, W), bool)
    for _ in range(rng.integers(3, 6)):
        oy, ox = rng.normal(0, size * 0.45, 2)
        m |= np.hypot(yy - cy - oy, xx - cx - ox) <= size * rng.uniform(0.2, 0.4)
    return m


def make_blob_dataset(n=600, size=128, seed=0):
    """Returns images (n, size, size) float64 and labels {key: list}."""
    rng = np.random.default_rng(seed)
    images, labels = [], {"morphology": [], "focus": [], "usable": []}
    for _ in range(n):
        kind = rng.integers(3)
        focus = rng.choice(4, p=[0.2, 0.2, 0.2, 0.4])
        signal = rng.uniform(20, 40) if rng.random() > 0.15 else rng.uniform(3, 6)  # 15% low-signal
        saturated = rng.random() < 0.1

        bg = rng.normal(100, 10)
        img = np.full((size, size), bg)
        for _ in range(rng.integers(3, 8)):
            img[_blob_mask(rng, size, size, MORPHOLOGIES[kind], rng.uniform(10, 22))] += signal
        if FOCUS_SIGMA[focus] > 0:
            img = ndimage.gaussian_filter(img, FOCUS_SIGMA[focus])
        img = rng.poisson(np.clip(img, 0, None)).astype(np.float64)
        if saturated:
            img = np.minimum(img, np.percentile(img, rng.uniform(40, 70)))  # detector clipping

        images.append(img)
        labels["morphology"].append(MORPHOLOGIES[kind])
        labels["focus"].append(FOCUS_LEVELS[focus])
        labels["usable"].append(bool(focus >= 2 and signal >= 10 and not saturated))
    return np.stack(images), labels
