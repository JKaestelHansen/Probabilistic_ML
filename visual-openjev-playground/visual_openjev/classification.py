"""Specialised classifier: the cheap, high-throughput path once some labels exist (architecture C).

    vectors = object_vectors(records, backbone)       crop embedding (+ morphology features)
    clf = ObjectClassifier(known_categories).fit(records_with_labels)
    clf.predict(records)                               -> record.classifier = {probs incl. "unknown", ood}
    accepted, deferred = cascade(records, tax)         confident + in-distribution -> accept, rest -> VLM

"unknown" is not a trained class: it comes from an out-of-distribution score (k-NN distance to the labelled
embeddings, relative to the labelled set's own leave-one-out distances), so the classifier is never forced to
pick a known class for something it has never seen.
"""
import numpy as np

from openjev import Choice, format_answer
from openjev.vision import Backbone, VisionJev

from .decisions import UNKNOWN

MORPH_FEATURES = ["area", "circularity", "aspect_ratio", "solidity", "circumcircle_fill", "symmetry_strength",
                  "elongation_harmonic", "contrast", "edge_sharpness"]


def object_vectors(records, backbone="handcrafted", morphology=True, batch_size=32):
    """Embed crops with a frozen backbone and store on record.embedding; returns (N, D)."""
    bb = backbone if isinstance(backbone, Backbone) else Backbone(backbone)
    # embed every crop as "bright object on dark background" so clusters group by shape, not imaging polarity
    crops = [r.crop if r.meta.get("image_qc", {}).get("polarity", "bright") == "bright" else r.crop.max() - r.crop
             for r in records]
    E = bb(crops, batch_size=batch_size).astype(np.float32)
    if morphology:
        M = np.array([[np.log1p(abs(r.features[k])) * np.sign(r.features[k]) for k in MORPH_FEATURES]
                      for r in records], dtype=np.float32)
        E = np.concatenate([E, M], 1)
    for r, e in zip(records, E):
        r.embedding = e
    return E


def _knn_dist(Q, B, k, exclude_self=False):
    d2 = (Q ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * Q @ B.T
    d = np.sqrt(np.maximum(d2, 0))
    if exclude_self:
        np.fill_diagonal(d, np.inf)
    return np.sort(d, 1)[:, :k].mean(1)


class ObjectClassifier:
    def __init__(self, categories, n_members=5, k=5, version="clf-v1"):
        self.categories = list(categories)
        self.k, self.version = k, version
        self.model = VisionJev([Choice("category", "category", self.categories)], n_members=n_members)

    def fit(self, records, epochs=200, **kw):
        lab = [r for r in records if r.label in self.categories]
        X = np.stack([r.embedding for r in lab])
        self.model.fit(X, {"category": [r.label for r in lab]}, epochs=epochs, **kw)
        # OOD reference: standardised embeddings + leave-one-out k-NN distances of the labelled set
        self.ref = (X - self.model.mu) / self.model.sd
        d = _knn_dist(self.ref, self.ref, self.k, exclude_self=True)
        self.q95, self.q995 = np.quantile(d, 0.95), np.quantile(d, 0.995)
        return self

    def ood_scores(self, X):
        """0 for typical embeddings, rising to 1 beyond the 99.5th percentile of labelled-set distances."""
        d = _knn_dist((X - self.model.mu) / self.model.sd, self.ref, self.k)
        return np.clip((d - self.q95) / max(self.q995 - self.q95, 1e-9), 0, 1)

    def predict(self, records):
        X = np.stack([r.embedding for r in records])
        probs, epi = self.model.predict_proba(X)["category"]
        ood = self.ood_scores(X)
        for r, p, o, e in zip(records, probs, ood, epi):
            full = {c: float(pi * (1 - o)) for c, pi in zip(self.categories, p)}
            full[UNKNOWN] = float(o)
            r.classifier = {"probs": full, "ood": float(o), "epistemic": float(e), "model": self.version}
        return records


def classifier_decisions(records, tax, key="C_classifier"):
    """No-LLM fast path: map classifier output straight onto the typed decision questions."""
    names = tax.names
    q_cat = Choice("category", "category", names)
    contam = set(tax.contamination)
    for r in records:
        p = np.array([r.classifier["probs"].get(n, 0.0) for n in names])
        p = p / p.sum()
        p_contam = float(sum(pi for n, pi in zip(names, p) if n in contam))
        p_known_sample = float(sum(pi for n, pi in zip(names, p) if n not in contam and n != UNKNOWN))
        single = 0.0 if r.features.get("touches_other") else 1.0
        r.decisions[key] = {
            "category": format_answer(q_cat, p),
            "is_contamination": {"type": "noul", "value": p_contam, "decision": p_contam >= 0.5,
                                 "probs": {"no": 1 - p_contam, "yes": p_contam}, "confidence": max(p_contam, 1 - p_contam)},
            "countable": {"type": "noul", "value": p_known_sample * single, "decision": p_known_sample * single >= 0.5,
                          "probs": {"no": 1 - p_known_sample * single, "yes": p_known_sample * single},
                          "confidence": max(p_known_sample * single, 1 - p_known_sample * single)},
        }
    return records


def cascade(records, accept=0.9, max_ood=0.5):
    """Split into (accepted by the classifier, deferred to the VLM / review)."""
    accepted, deferred = [], []
    for r in records:
        p = r.classifier["probs"]
        best = max((v, k) for k, v in p.items() if k != UNKNOWN)
        (accepted if best[0] >= accept and r.classifier["ood"] < max_ood else deferred).append(r)
    return accepted, deferred
