"""Stand-in backends so the whole stack runs offline (tests, demos, CI). NOT real perception.

MockVLM looks at the crop with classical image analysis and answers the same typed questions a real VLM
would get, with deliberate noise. Swap in openjev.VLMDecider / OpenAICompatDecider for real runs.
mock_decision_backend reasons over the JSON state (observations + measurements) with rules, standing in
for a text LLM / hosted decision API in architecture B.
"""
import numpy as np

from openjev import RuleDecider

from .perception import object_features, segment

SHAPE_TO_CATEGORY = {"round": "round_cell", "oval or rod-shaped": "elongated_cell",
                     "irregular or lobed": "budding_cell", "long thin filament": "fiber"}


def _central_features(crop):
    lab = segment(crop, min_area=10)
    if lab.max() == 0:
        return None
    H, W = lab.shape
    c = lab[H // 2, W // 2]
    if c == 0:  # pick the largest component
        c = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    sl = tuple(slice(0, s) for s in lab.shape)
    f = object_features(crop, lab, c, sl, float(np.median(crop)))
    f["n_components"] = int(lab.max())
    return f


def _shape_probs(f):
    if f is None:
        return {"fragment or unclear": 1.0}
    ar, circ, sol, sharp = f["aspect_ratio"], f["circularity"], f["solidity"], f["edge_sharpness"]
    s = {"round": 2.5 * (ar < 1.35) * (sol > 0.85), "oval or rod-shaped": 2.5 * (1.8 < ar < 5),
         "irregular or lobed": 2.5 * (sol < 0.9) * (ar < 2.2) * (f["area"] > 250),
         "long thin filament": 3.0 * (ar >= 5),
         "ring or hollow": 3.0 * (sharp > 0.75) * (ar < 1.3) * (f["contrast"] < 16),
         "fragment or unclear": 1.0 * (f["area"] < 120)}
    return {k: v + 0.3 for k, v in s.items()}


def _choice(q, scores, rng, noise):
    z = np.array([scores.get(o, 0.0) for o in q.labels]) + rng.normal(0, noise, len(q.labels))
    e = np.exp(z - z.max())
    return e / e.sum()


def _yes(p):
    return [1 - p, p]


class MockVLM(RuleDecider):
    def __init__(self, noise=0.4, seed=0):
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        super().__init__({})

    def option_logits(self, state, questions):
        f = _central_features(np.asarray(state["images"][0], dtype=float))
        shape = _shape_probs(f)
        out = []
        for q in questions:
            if q.key == "shape":
                p = _choice(q, shape, self.rng, self.noise)
            elif q.key == "category":
                cat = {SHAPE_TO_CATEGORY.get(k, "unknown"): v for k, v in shape.items()}
                p = _choice(q, cat, self.rng, self.noise)
            elif q.key == "boundary":
                sharp = 0 if f is None else f["edge_sharpness"]
                p = _choice(q, {"sharp": 2 * (sharp > 0.4), "slightly soft": 1.5 * (0.25 < sharp <= 0.4),
                                "indistinct": 2 * (sharp <= 0.25)}, self.rng, self.noise)
            elif q.key == "texture":
                hollow = shape.get("ring or hollow", 0) > 1
                p = _choice(q, {"hollow centre": 3 * hollow, "uniform": 1.5, "granular": 0.5}, self.rng, self.noise)
            elif q.key == "multiple_objects":
                p = _yes(0.8 if f is not None and (f["n_components"] > 1 or f["touches_other"]) else 0.1)
            elif q.key == "in_focus":
                p = _yes(float(np.clip(0 if f is None else (f["edge_sharpness"] - 0.15) * 3, 0.02, 0.98)))
            elif q.key == "is_contamination":
                p = _yes(0.85 if shape.get("long thin filament", 0) > 1 else 0.08)
            elif q.key == "countable":
                p = _yes(0.1 if f is None or shape.get("fragment or unclear", 0) > 1 else 0.8)
            else:
                p = np.ones(len(q.labels)) / len(q.labels)
            out.append(np.log(np.clip(np.asarray(p, dtype=float), 1e-9, None)))
        return out


def mock_decision_backend(noise=0.3, seed=0):
    """Architecture-B stand-in: decides from the JSON state's visual observations + measurements."""
    rng = np.random.default_rng(seed)

    def category(state, q):
        obs = state.get("visual_observations", {})
        shape = obs.get("shape", {}).get("value")
        conf = obs.get("shape", {}).get("confidence", 0.5)
        m = state.get("measurements", {})
        scores = {SHAPE_TO_CATEGORY.get(shape, "unknown"): 3 * conf}
        if m.get("aspect_ratio", 1) >= 5:
            scores["fiber"] = scores.get("fiber", 0) + 1.5  # measurement evidence
        if obs.get("texture", {}).get("value") == "hollow centre":
            scores["unknown"] = scores.get("unknown", 0) + 2
        return _choice(q, scores, rng, noise)

    def contamination(state, q):
        p = 0.85 if state.get("visual_observations", {}).get("shape", {}).get("value") == "long thin filament" else 0.08
        return _yes(p)

    def countable(state, q):
        obs = state.get("visual_observations", {})
        p = 0.85
        p *= 1 - 0.8 * obs.get("multiple_objects", 0)
        p *= 0.3 + 0.7 * obs.get("in_focus", 1)
        return _yes(float(np.clip(p, 0.01, 0.99)))

    return RuleDecider({"category": category, "is_contamination": contamination, "countable": countable})
