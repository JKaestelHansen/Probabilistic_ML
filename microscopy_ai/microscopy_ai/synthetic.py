"""Synthetic fluorescence fields with per-object ground truth, extending ../../generate_blob_images.py
(Gaussian background ~N(100, 10), bright objects, Poisson noise).

Object types:
    round_cell, elongated_cell, budding_cell   the "known" taxonomy
    fiber                                      contamination (long, thin, dimmer, often crosses others)
    ring                                       NOVEL: deliberately not in the default taxonomy, to test the
                                               unknown -> discovery path
Images get a random defocus level, so quality questions have ground truth too.
"""
import numpy as np
from scipy import ndimage

KNOWN = ["round_cell", "elongated_cell", "budding_cell"]
CONTAMINATION = ["fiber"]
NOVEL = ["ring"]
ALL_TYPES = KNOWN + CONTAMINATION + NOVEL

TAXONOMY = {
    "categories": [
        {"name": "round_cell", "description": "compact, near-circular cell with a smooth boundary"},
        {"name": "elongated_cell", "description": "rod or oval shaped cell, about 2-4x longer than wide"},
        {"name": "budding_cell", "description": "irregular cell made of a main body with one or more lobes or buds"},
        {"name": "fiber", "description": "long thin filament, often dimmer than cells; dust or fabric contamination",
         "contamination": True},
    ]
}


def _shape(kind, rng, H, W, cy, cx, size):
    yy, xx = np.mgrid[:H, :W]
    dy, dx = yy - cy, xx - cx
    if kind == "round_cell":
        return np.hypot(dy, dx) <= size / 2
    if kind == "elongated_cell":
        th = rng.uniform(0, np.pi)
        u, v = dx * np.cos(th) + dy * np.sin(th), -dx * np.sin(th) + dy * np.cos(th)
        return (u / (size * 0.75)) ** 2 + (v / (size / rng.uniform(4.5, 6))) ** 2 <= 1
    if kind == "budding_cell":
        m = np.hypot(dy, dx) <= size * 0.42
        for _ in range(rng.integers(1, 4)):
            a = rng.uniform(0, 2 * np.pi)
            r = size * 0.42
            m |= np.hypot(dy - r * np.sin(a), dx - r * np.cos(a)) <= size * rng.uniform(0.2, 0.3)
        return m
    if kind == "fiber":
        th = rng.uniform(0, np.pi)
        u, v = dx * np.cos(th) + dy * np.sin(th), -dx * np.sin(th) + dy * np.cos(th)
        v = v - 0.004 * size * (u / size) ** 2 * size  # slight curvature
        return (np.abs(u) <= size * rng.uniform(1.5, 2.2)) & (np.abs(v) <= 1.6)
    if kind == "ring":
        r = np.hypot(dy, dx)
        return (r <= size / 2) & (r >= size / 2 - 2.5)
    raise ValueError(kind)


def make_field(rng, size=256, n_objects=(6, 12), type_probs=None, focus_sigma=None):
    """One image -> (image, instance label map, {instance id: type}, focus sigma)."""
    type_probs = type_probs or {"round_cell": 0.27, "elongated_cell": 0.27, "budding_cell": 0.26, "fiber": 0.1, "ring": 0.1}
    types, p = list(type_probs), np.array(list(type_probs.values()), dtype=float)
    sigma = focus_sigma if focus_sigma is not None else rng.choice([0.0, 0.0, 0.0, 0.8, 2.0, 3.5])
    bg = rng.normal(100, 10)
    signal = np.zeros((size, size))
    inst = np.zeros((size, size), np.int32)
    gt = {}
    for _ in range(rng.integers(*n_objects)):
        kind = types[rng.choice(len(types), p=p / p.sum())]
        for _attempt in range(30):
            s = rng.uniform(16, 26)
            cy, cx = rng.uniform(s, size - s, 2)
            m = _shape(kind, rng, size, size, cy, cx, s)
            # objects mostly avoid each other; some fibers cross cells (realistic contamination / overlap)
            if not (inst[m] > 0).any() or (kind == "fiber" and rng.random() < 0.3):
                break
        else:
            continue
        i = len(gt) + 1
        gt[i] = kind
        inst[m & (inst == 0)] = i
        signal[m] += rng.uniform(20, 28) if kind == "fiber" else rng.uniform(25, 45)
    img = bg + signal
    if sigma > 0:
        img = ndimage.gaussian_filter(img, sigma)
    img = rng.poisson(np.clip(img, 0, None)).astype(np.float64)
    return img, inst, gt, float(sigma)


def make_dataset(n_images=20, size=256, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [make_field(rng, size, **kw) for _ in range(n_images)]
