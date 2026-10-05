"""Measurement: "how many / how much?"  Only objects whose final decision says they count are counted."""
import csv
from collections import defaultdict

import numpy as np


def final_category(r, source=None):
    """Reviewed human label > accepted review category > raw decision of `source`."""
    if r.label:
        return r.label
    if r.review and not r.review["needs_review"]:
        return r.review["category"]
    if source and source in r.decisions:
        return r.decisions[source]["category"]["value"]
    return None


def is_counted(r, source, exclude=()):
    if r.label:
        return r.label not in exclude
    d = r.decisions.get(source, {})
    return bool(d.get("countable", {}).get("decision", False)) and not (r.review and r.review["needs_review"])


def summarize(records, source, pixel_size_um=None, exclude=("unknown",)):
    """Per image and category: count, mean/total area; plus pending-review counts. Returns list of row dicts.

    exclude: human-label categories never counted (pass taxonomy contamination types + "unknown")."""
    scale = (pixel_size_um or 1.0) ** 2
    unit = "um2" if pixel_size_um else "px"
    agg = defaultdict(list)
    pending = defaultdict(int)
    for r in records:
        if r.review and r.review["needs_review"] and not r.label:
            pending[r.image_id] += 1
            continue
        if is_counted(r, source, exclude):
            agg[(r.image_id, final_category(r, source))].append(r.features["area"] * scale)
    rows = []
    for (img, cat), areas in sorted(agg.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        rows.append({"image_id": img, "category": cat, "count": len(areas),
                     f"mean_area_{unit}": float(np.mean(areas)), f"total_area_{unit}": float(np.sum(areas)),
                     "pending_review_in_image": pending[img]})
    return rows


def write_csv(rows, path):
    if not rows:
        open(path, "w").close()
        return path
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path
