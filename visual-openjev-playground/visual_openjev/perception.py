"""Perception: "what is visually there?"  image -> QC metrics -> object discovery -> object records.

segment(method="threshold")  classical: flatten uneven illumination, robust threshold, connected components.
                              Works for bright-on-dark (fluorescence, SEM) and dark-on-bright (brightfield, TEM)
                              images via `polarity`. No download; fine for well separated objects.
segment(method="cellpose")   Cellpose-SAM (`pip install cellpose`): better for touching objects and cells.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from .objects import ObjectRecord


def image_quality(img):
    """Whole-image QC metrics (raw numbers; the decision layer / rules interpret them)."""
    img = np.asarray(img, dtype=np.float64)
    g = (img - img.mean()) / (img.std() + 1e-9)
    bg = np.median(img)
    mad = np.median(np.abs(img - bg)) * 1.4826 + 1e-9
    return {
        "focus_laplacian_logvar": float(np.log(ndimage.laplace(ndimage.gaussian_filter(g, 1)).var() + 1e-12)),
        "saturated_fraction": float(np.mean(img >= img.max())) if img.max() > 0 else 0.0,
        "background": float(bg),
        "noise_mad": float(mad),
        "signal_to_noise": float((np.percentile(img, 99.5) - bg) / mad),
    }


def detect_polarity(img):
    """'bright' if objects are brighter than the background, else 'dark' (objects are the minority tail)."""
    lo, med, hi = np.percentile(img, [0.5, 50, 99.5])
    return "bright" if hi - med >= med - lo else "dark"


def object_signal(img, polarity="auto", flatten=31):
    """Image -> background-flattened signal where objects are positive, whatever the imaging polarity."""
    img = np.asarray(img, dtype=np.float64)
    if polarity == "auto":
        polarity = detect_polarity(img)
    s = img if polarity == "bright" else img.max() - img
    if flatten:
        # morphological opening removes objects smaller than `flatten` px, leaving the illumination profile
        bg = ndimage.gaussian_filter(ndimage.grey_opening(s, size=(flatten, flatten)), flatten / 3)
        s = s - bg
    return s


def segment(img, method="threshold", min_area=30, k_sigma=3.0, smooth=1.5, polarity="auto", flatten=61,
            cellpose_model=None):
    """-> instance label map (0 = background)."""
    img = np.asarray(img, dtype=np.float64)
    if method == "cellpose":
        if cellpose_model is None:
            from cellpose import models

            import torch

            cellpose_model = models.CellposeModel(gpu=torch.cuda.is_available())
        masks = cellpose_model.eval(img)[0]
        return masks.astype(np.int32)
    s = ndimage.gaussian_filter(object_signal(img, polarity, flatten), smooth)
    bg = np.median(s)
    mad = np.median(np.abs(s - bg)) * 1.4826 + 1e-9
    fg = ndimage.binary_opening(s > bg + k_sigma * mad, iterations=1)
    fg = ndimage.binary_fill_holes(fg)
    lab, _ = ndimage.label(fg)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_area
    keep[0] = False
    lab = np.where(keep[lab], lab, 0)
    # relabel consecutively
    _, inv = np.unique(lab, return_inverse=True)
    return inv.reshape(lab.shape).astype(np.int32)


def _solidity(mask):
    yy, xx = np.nonzero(mask)
    pts = np.stack([yy, xx], 1).astype(float)
    if len(pts) < 5:
        return 1.0
    # pixel corners so the hull encloses whole pixels
    corners = np.concatenate([pts + d for d in ([-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5])])
    try:
        return float(mask.sum() / ConvexHull(corners).volume)
    except QhullError:
        return 1.0


def outline_symmetry(mask, n_bins=72):
    """Fourier analysis of the boundary's radial profile r(theta) around the centroid.

    Returns (order, strength): the dominant rotational symmetry among 3..8 (3 triangle, 4 square/cube,
    6 hexagon) and its amplitude relative to the mean radius. Round objects have strength ~0; elongation
    shows up in harmonic 2, reported separately as `elongation_harmonic`.
    """
    edge = mask & ~ndimage.binary_erosion(mask)
    yy, xx = np.nonzero(edge)
    if len(yy) < 12:
        return 0, 0.0, 0.0
    cy, cx = np.nonzero(mask)
    th = np.arctan2(yy - cy.mean(), xx - cx.mean())
    r = np.hypot(yy - cy.mean(), xx - cx.mean())
    b = ((th + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    prof = np.array([r[b == k].max() if (b == k).any() else np.nan for k in range(n_bins)])
    ok = ~np.isnan(prof)
    prof = np.interp(np.arange(n_bins), np.arange(n_bins)[ok], prof[ok], period=n_bins)
    amp = np.abs(np.fft.rfft(prof)) / n_bins
    rel = 2 * amp / amp[0]
    k = 3 + int(np.argmax(rel[3:9]))
    return k, float(rel[k]), float(rel[2])


def object_features(img, lab, i, sl, bg):
    m = lab[sl] == i
    area = int(m.sum())
    yy, xx = np.nonzero(m)
    cov = np.cov(np.stack([yy, xx])) + 1e-6 * np.eye(2) if area > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov))
    # 4-neighbour boundary edge count, scaled by pi/4 (city-block -> Euclidean length for smooth shapes)
    mp = np.pad(m, 1)
    perim = float((np.abs(np.diff(mp, axis=0)).sum() + np.abs(np.diff(mp, axis=1)).sum()) * np.pi / 4)
    vals = img[sl][m]
    ring = ndimage.binary_dilation(m, iterations=2) & ~m
    local_bg = float(np.median(img[sl][ring])) if ring.any() else bg
    contrast = float(vals.mean() - local_bg)
    gy, gx = np.gradient(ndimage.gaussian_filter(img[sl].astype(float), 0.7))
    edge = m & ~ndimage.binary_erosion(m)
    edge_grad = float(np.hypot(gy, gx)[edge].mean()) if edge.any() else 0.0
    # touching another object (possible overlap / clump)
    neigh = ndimage.binary_dilation(m, iterations=2) & (lab[sl] > 0) & (lab[sl] != i)
    return {
        "area": area,
        "perimeter": perim,
        "circularity": float(4 * np.pi * area / max(perim, 1) ** 2),
        "aspect_ratio": float(np.sqrt(ev[1] / ev[0])),
        "eccentricity": float(np.sqrt(1 - ev[0] / ev[1])),
        "solidity": _solidity(m),
        # area / area of the circle through the farthest boundary point: circle 1, hexagon .83, square .64, triangle .41
        "circumcircle_fill": float(area / (np.pi * (np.hypot(yy - yy.mean(), xx - xx.mean()).max() + 0.5) ** 2)),
        **dict(zip(("symmetry_order", "symmetry_strength", "elongation_harmonic"), outline_symmetry(m))),
        "mean_intensity": float(vals.mean()),
        "contrast": contrast,
        # edge gradient relative to contrast: ~scale-free sharpness of the boundary, drops with defocus
        "edge_sharpness": float(edge_grad / max(abs(contrast), 1e-6)),
        "touches_other": bool(neigh.any()),
    }


def extract_objects(img, image_id, lab=None, pad=6, **seg_kw):
    """Segment (unless a label map is given) and build one ObjectRecord per object."""
    img = np.asarray(img, dtype=np.float64)
    lab = segment(img, **seg_kw) if lab is None else lab
    H, W = img.shape[:2]
    qc = image_quality(img)
    qc["polarity"] = seg_kw.get("polarity", "auto")
    if qc["polarity"] == "auto":
        qc["polarity"] = detect_polarity(img)
    sig = object_signal(img, qc["polarity"], flatten=0)  # features on the "objects are bright" signal
    records = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        f = object_features(sig, lab, i, sl, float(np.median(sig)))
        y0, x0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
        y1, x1 = min(sl[0].stop + pad, H), min(sl[1].stop + pad, W)
        f["touches_border"] = bool(sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W)
        records.append(ObjectRecord(
            object_id=f"{image_id}_obj_{i:04d}", image_id=image_id, bbox=[y0, x0, y1, x1], features=f,
            crop=img[y0:y1, x0:x1].astype(np.float32), mask=(lab[y0:y1, x0:x1] == i),
            meta={"image_qc": qc, "instance": i},
        ))
    return records, lab


def match_ground_truth(records, lab, gt_inst, gt_types, min_iou=0.3):
    """Synthetic data: attach the ground-truth type of the best-overlapping true object (IoU >= min_iou)."""
    for r in records:
        m = lab == r.meta["instance"]
        ids, counts = np.unique(gt_inst[m], return_counts=True)
        best, best_iou = None, 0.0
        for t, c in zip(ids, counts):
            if t == 0:
                continue
            iou = c / (m.sum() + (gt_inst == t).sum() - c)
            if iou > best_iou:
                best, best_iou = t, iou
        r.meta["gt"] = gt_types[best] if best is not None and best_iou >= min_iou else "no_match"
        r.meta["gt_iou"] = float(best_iou)
    return records
