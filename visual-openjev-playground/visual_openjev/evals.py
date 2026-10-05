"""Compare architectures (A: VLM-only, B: VLM -> decision layer, C: classifier -> decision) on objects with
ground truth (record.meta["gt"] for synthetic data, or record.label for a held-out human-labelled set).

Objects whose true type is not in the taxonomy count as correct only if predicted "unknown".
"""
import numpy as np

from openjev import calibration as cal

from .decisions import UNKNOWN


def truth(r, tax, gt_key="gt"):
    t = r.label or r.meta.get(gt_key)
    if t is None or t == "no_match":
        return None
    return t if t in tax.names else UNKNOWN


def evaluate(records, tax, keys, gt_key="gt"):
    names = tax.names
    res = {}
    for key in keys:
        rs = [r for r in records if key in r.decisions and truth(r, tax, gt_key) is not None]
        if not rs:
            continue
        P = np.array([[r.decisions[key]["category"]["probs"].get(n, 0.0) for n in names] for r in rs])
        P = P / P.sum(1, keepdims=True)
        y = np.array([names.index(truth(r, tax, gt_key)) for r in rs])
        m = cal.summarize(P, y)
        unk = y == names.index(UNKNOWN)
        pred = P.argmax(1)
        m["unknown_recall"] = float(np.mean(pred[unk] == y[unk])) if unk.any() else float("nan")
        m["known_accuracy"] = float(np.mean(pred[~unk] == y[~unk])) if (~unk).any() else float("nan")
        has = np.array([r.review.get("source") == key for r in rs])
        if has.any():
            ok = has & np.array([not r.review.get("needs_review", True) for r in rs])
            m["auto_accept_rate"] = float(ok[has].mean())
            m["accuracy_on_auto_accepted"] = float(np.mean(pred[ok] == y[ok])) if ok.any() else float("nan")
        res[key] = m
    return res
