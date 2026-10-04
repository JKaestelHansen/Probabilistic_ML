"""Discovery: "what have we not understood yet?"  Zero-label bootstrap and taxonomy growth.

    clusters = kmeans(E, k)                        group similar objects
    neighbors(E, i, k)                             "show me objects like this one"
    pick = select_for_review(records, n)           diverse + uncertain objects for a human to label
    export_review_html(records, "review.html")     contact sheet, one section per cluster + labels.csv template
    apply_labels(records, read_labels("labels.csv"))
"""
import base64
import csv
import html
import io

import numpy as np
from scipy import ndimage


def _standardise(E):
    return (E - E.mean(0)) / (E.std(0) + 1e-9)


def kmeans(E, k, n_iter=100, seed=0):
    """k-means++ on standardised embeddings. Returns (labels, centres)."""
    rng = np.random.default_rng(seed)
    X = _standardise(np.asarray(E, dtype=np.float64))
    k = min(k, len(X))
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    C = np.array(C)
    for _ in range(n_iter):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab, C


def neighbors(E, i, k=8):
    """Indices of the k nearest objects to object i (cosine similarity on standardised embeddings)."""
    X = _standardise(np.asarray(E, dtype=np.float64))
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    sim = X @ X[i]
    sim[i] = -np.inf
    return np.argsort(-sim)[:k]


def uncertainty_of(r):
    """Highest available uncertainty signal for an object (review score > decision entropy > none)."""
    if r.review:
        return r.review["score"]
    for d in r.decisions.values():
        if "category" in d:
            return 1 - d["category"]["confidence"]
    return 0.0


def select_for_review(records, n, n_clusters=None, seed=0):
    """Active learning: cluster, then take the most uncertain unlabelled object from each cluster in turn
    (round-robin), so a small labelling budget covers the whole visual space."""
    pool = [i for i, r in enumerate(records) if r.label is None]
    if not pool:
        return []
    E = np.stack([records[i].embedding for i in pool])
    lab, _ = kmeans(E, n_clusters or max(2, min(n, len(pool) // 3)), seed=seed)
    queues = {}
    for j, i in zip(lab, pool):
        queues.setdefault(j, []).append(i)
    for j in queues:
        queues[j].sort(key=lambda i: -uncertainty_of(records[i]))
    picked = []
    while len(picked) < min(n, len(pool)):
        for j in list(queues):
            if queues[j]:
                picked.append(queues[j].pop(0))
            if len(picked) >= n:
                break
    return [records[i] for i in picked]


def _png_b64(crop, mask=None, size=96):
    from PIL import Image

    from openjev.deciders import to_uint8_rgb

    rgb = to_uint8_rgb(crop).copy()
    if mask is not None:
        rgb[mask & ~ndimage.binary_erosion(mask)] = [255, 80, 80]
    im = Image.fromarray(rgb)
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def export_review_html(records, path, groups=None, title="Object review", csv_path=None):
    """Self-contained HTML contact sheet (crops with mask outline, id, current decision) grouped by cluster,
    plus a labels CSV template (object_id,label) for the reviewer to fill in."""
    groups = groups if groups is not None else [0] * len(records)
    by = {}
    for r, g in zip(records, groups):
        by.setdefault(g, []).append(r)
    parts = [f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>",
             "<style>body{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}"
             ".g{display:flex;flex-wrap:wrap;gap:8px}.c{background:#fff;border:1px solid #ddd;border-radius:6px;"
             "padding:4px;width:110px;font-size:11px}.c img{width:100px;height:100px;object-fit:contain;"
             "image-rendering:pixelated;background:#000}.r{color:#b00}"
             "@media (prefers-color-scheme:dark){body{background:#111;color:#ddd}.c{background:#1c1c1c;border-color:#333}}"
             "</style>", f"<h1>{html.escape(title)}</h1>"]
    for g in sorted(by, key=str):
        parts.append(f"<h2>Group {html.escape(str(g))} ({len(by[g])} objects)</h2><div class='g'>")
        for r in by[g]:
            cur = r.label or (r.review.get("category") if r.review else None)
            flag = ""
            if r.review.get("needs_review"):
                d = r.decisions.get(r.review["source"], {}).get("category", {})
                cur = f"{d.get('value')}? ({d.get('confidence', 0):.2f})"
                flag = f" <span class='r' title='{html.escape(', '.join(r.review['reasons']))}'>review</span>"
            parts.append(f"<div class='c'><img src='data:image/png;base64,{_png_b64(r.crop, r.mask)}'>"
                         f"<div>{html.escape(r.object_id)}</div><div>{html.escape(str(cur))}{flag}</div></div>")
        parts.append("</div>")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    csv_path = csv_path or path.rsplit(".", 1)[0] + "_labels.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object_id", "group", "label"])
        for r, g in zip(records, groups):
            w.writerow([r.object_id, g, r.label or ""])
    return path, csv_path


def read_labels(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    # short rows (missing trailing columns) give None
    return {r["object_id"]: (r.get("label") or "").strip() for r in rows if (r.get("label") or "").strip()}


def apply_labels(records, labels):
    n = 0
    for r in records:
        if r.object_id in labels:
            r.label = labels[r.object_id]
            n += 1
    return n
