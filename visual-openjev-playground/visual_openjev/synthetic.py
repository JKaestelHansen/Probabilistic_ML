"""Synthetic object images with per-object ground truth, for tests, demos and sanity-checking a VLM backend.

Shapes are 2D projections of common particle morphologies:
    sphere      disc with radial shading
    cube        rotated square with facet shading
    rod         capsule (rounded rectangle), aspect 3-6
    hexagonal   regular hexagon plate
    aggregate   cluster of small fused primary particles
    triangle    NOVEL: deliberately not in the default taxonomy, to exercise the unknown -> discovery path

Imaging: random polarity (bright objects on dark, as in fluorescence / SEM; or dark on bright, as in
brightfield / TEM), uneven illumination, defocus and Poisson noise.
"""
import numpy as np
from scipy import ndimage

KNOWN = ["sphere", "cube", "rod", "aggregate", "hexagonal"]
NOVEL = ["triangle"]
ALL_TYPES = KNOWN + NOVEL


def _rot(dy, dx, th):
    return dx * np.cos(th) + dy * np.sin(th), -dx * np.sin(th) + dy * np.cos(th)


def _polygon(dy, dx, n, r, th):
    """Regular n-gon with circumradius r, rotated by th."""
    a = np.arctan2(dy, dx) - th
    rho = np.hypot(dy, dx)
    sector = 2 * np.pi / n
    return rho * np.cos((a % sector) - sector / 2) <= r * np.cos(sector / 2)


def _object(kind, rng, yy, xx, cy, cx, size):
    """-> (mask, intensity profile in [0, 1] over the mask)."""
    dy, dx = yy - cy, xx - cx
    th = rng.uniform(0, 2 * np.pi)
    if kind == "sphere":
        r = size / 2
        rho = np.hypot(dy, dx)
        return rho <= r, np.sqrt(np.clip(1 - (rho / r) ** 2, 0, 1)) * 0.6 + 0.4
    if kind == "cube":
        u, v = _rot(dy, dx, th)
        h = size * 0.42
        m = (np.abs(u) <= h) & (np.abs(v) <= h * rng.uniform(0.85, 1.0))
        return m, 0.75 + 0.25 * (u > 0)  # two facet brightnesses
    if kind == "rod":
        u, v = _rot(dy, dx, th)
        w = size * rng.uniform(0.17, 0.25)
        L = w * rng.uniform(2.5, 5)  # half-length; length/width = L/w = 2.5-5
        m = (np.abs(v) <= w) & ((np.abs(u) <= L - w) | (np.hypot(np.abs(u) - (L - w), v) <= w))
        return m, np.sqrt(np.clip(1 - (v / w) ** 2, 0, 1)) * 0.5 + 0.5
    if kind == "hexagonal":
        return _polygon(dy, dx, 6, size * 0.5, th), np.full(dy.shape, 0.9)
    if kind == "triangle":
        return _polygon(dy, dx, 3, size * 0.6, th), np.full(dy.shape, 0.9)
    if kind == "aggregate":
        m = np.zeros(dy.shape, bool)
        prof = np.zeros(dy.shape)
        py, px = 0.0, 0.0
        for _ in range(rng.integers(5, 10)):
            r = size * rng.uniform(0.12, 0.2)
            rho = np.hypot(dy - py, dx - px)
            mi = rho <= r
            m |= mi
            prof = np.maximum(prof, np.sqrt(np.clip(1 - (rho / r) ** 2, 0, 1)) * mi)
            a = rng.uniform(0, 2 * np.pi)
            py, px = np.clip([py + 1.4 * r * np.sin(a), px + 1.4 * r * np.cos(a)], -size * 0.45, size * 0.45)
        return m, prof * 0.6 + 0.4
    raise ValueError(kind)


def make_field(rng, size=256, n_objects=(6, 12), type_probs=None, focus_sigma=None, polarity=None):
    """One image -> (image, instance label map, {instance id: type}, info dict)."""
    type_probs = type_probs or {"sphere": 0.18, "cube": 0.18, "rod": 0.18, "aggregate": 0.18, "hexagonal": 0.18,
                                "triangle": 0.10}
    types, p = list(type_probs), np.array(list(type_probs.values()), dtype=float)
    sigma = focus_sigma if focus_sigma is not None else float(rng.choice([0.0, 0.0, 0.0, 0.8, 2.0]))
    polarity = polarity or str(rng.choice(["bright", "dark"]))
    yy, xx = np.mgrid[:size, :size]
    signal = np.zeros((size, size))
    inst = np.zeros((size, size), np.int32)
    gt = {}
    for _ in range(rng.integers(*n_objects)):
        kind = types[rng.choice(len(types), p=p / p.sum())]
        for _attempt in range(40):
            s = rng.uniform(20, 34)
            cy, cx = rng.uniform(s, size - s, 2)
            m, prof = _object(kind, rng, yy, xx, cy, cx, s)
            if not ndimage.binary_dilation(inst > 0, iterations=3)[m].any():
                break
        else:
            continue
        i = len(gt) + 1
        gt[i] = kind
        inst[m] = i
        signal[m] += prof[m] * rng.uniform(40, 70)
    bg = rng.uniform(80, 120) * (1 + 0.15 * (xx / size - 0.5) * rng.uniform(-1, 1))  # uneven illumination
    img = bg + signal if polarity == "bright" else bg + 80 - signal * 0.9
    if sigma > 0:
        img = ndimage.gaussian_filter(img, sigma)
    img = rng.poisson(np.clip(img, 0, None)).astype(np.float64)
    return img, inst, gt, {"focus_sigma": sigma, "polarity": polarity}


def make_dataset(n_images=20, size=256, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [make_field(rng, size, **kw) for _ in range(n_images)]
