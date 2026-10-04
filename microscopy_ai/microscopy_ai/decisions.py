"""Decision layer: "what does it mean according to our taxonomy?"

The taxonomy is data, not code: it starts small, always includes "unknown", and grows as discovery turns
recurring unknowns into named categories. Decisions are typed openjev questions asked of any backend:

    A  VLM-only            crop image + taxonomy  -> VLM decider
    B  VLM -> decision      typed VLM observations + features (JSON state) -> any text/decision backend
    C  classifier -> layer  classifier probabilities + OOD score (+ features/observations) -> decision backend
                            (or classification.classifier_decisions for the no-LLM fast path)

`review()` then combines independent signals into a review score; model probability alone is never treated
as scientific confidence.
"""
import json

import numpy as np

from openjev import Choice, Noul

from .vlm import DEFAULT_CONTEXT, compact

UNKNOWN = "unknown"


class Taxonomy:
    def __init__(self, categories):
        self.categories = [dict(c) for c in categories if c["name"] != UNKNOWN]

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path))["categories"])

    def save(self, path):
        json.dump({"categories": self.categories}, open(path, "w"), indent=1)

    @property
    def names(self):
        return [c["name"] for c in self.categories] + [UNKNOWN]

    @property
    def contamination(self):
        return [c["name"] for c in self.categories if c.get("contamination")]

    def add(self, name, description, contamination=False):
        """Grow the taxonomy, e.g. after reviewing a cluster of unknowns."""
        self.categories.append({"name": name, "description": description, "contamination": contamination})

    def describe(self):
        lines = [f"- {c['name']}: {c.get('description', '')}" for c in self.categories]
        return "\n".join(lines + [f"- {UNKNOWN}: does not clearly match any category above, or never seen before"])


def decision_questions(tax):
    return [
        Choice("category", "Which category best describes this object?\nCategories:\n" + tax.describe(), tax.names),
        Noul("is_contamination", "Is this object contamination or debris rather than a sample object? "
             f"Contamination types: {', '.join(tax.contamination) or 'none defined'}."),
        Noul("countable", "Should this object be included in quantitative counts, i.e. a single, whole, "
             "in-focus object of a known non-contamination category?"),
    ]


def object_state(r, features=True, vlm=True, classifier=True):
    """Everything known about one object, as a JSON-able dict for the decision backend."""
    s = {}
    if features:
        f = r.features
        s["measurements"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()}
        qc = r.meta.get("image_qc", {})
        s["image_quality"] = {k: round(qc[k], 3) for k in ("focus_laplacian_logvar", "signal_to_noise") if k in qc}
    if vlm and "vlm" in r.observations:
        s["visual_observations"] = compact(r.observations["vlm"])
    if classifier and r.classifier:
        s["classifier"] = {"probs": {k: round(v, 3) for k, v in r.classifier["probs"].items()},
                           "out_of_distribution_score": round(r.classifier.get("ood", 0.0), 3)}
    return s


def run_vlm_only(records, vlm_decider, tax, key="A_vlm_only", context=DEFAULT_CONTEXT):
    qs = decision_questions(tax)
    for r in records:
        r.decisions[key] = vlm_decider.decide({"text": context, "images": [r.crop]}, qs)
    return records


def run_decision_layer(records, decider, tax, key="B_vlm_decision", **state_kw):
    qs = decision_questions(tax)
    for r in records:
        r.decisions[key] = decider.decide(object_state(r, **state_kw), qs)
    return records


def _category_probs(r, key, names):
    p = r.decisions.get(key, {}).get("category", {}).get("probs")
    return None if p is None else np.array([p.get(n, 0.0) for n in names])


def review(records, tax, primary, compare=(), threshold=0.5):
    """Combine independent signals into one review score (max of components, each in [0, 1]).

    uncertainty    1 - confidence of the primary category decision
    unknown        P(unknown) from the primary decision
    disagreement   total-variation distance between primary and each comparison source (VLM vs classifier...)
    low_quality    1 - P(in focus) from VLM observations, if available
    ood            classifier out-of-distribution score, if available
    """
    names = tax.names
    for r in records:
        d = r.decisions[primary]["category"]
        comp = {"uncertainty": 1 - d["confidence"], "unknown": d["probs"].get(UNKNOWN, 0.0)}
        p = _category_probs(r, primary, names)
        for k in compare:
            q = _category_probs(r, k, names)
            if q is not None:
                comp[f"disagreement:{k}"] = float(0.5 * np.abs(p - q).sum())
        if "vlm" in r.observations and "in_focus" in r.observations["vlm"]:
            comp["low_quality"] = 1 - r.observations["vlm"]["in_focus"]["value"]
        if "ood" in r.classifier:
            comp["ood"] = float(r.classifier["ood"])
        score = max(comp.values())
        needs = score >= threshold
        r.review = {
            "score": float(score), "components": {k: float(v) for k, v in comp.items()},
            "reasons": sorted(k for k, v in comp.items() if v >= threshold), "needs_review": bool(needs),
            "category": None if needs else d["value"], "source": primary,
        }
    return records
