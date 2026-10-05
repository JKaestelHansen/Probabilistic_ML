"""Stand-in backends so the whole stack runs offline (tests, demos, CI). NOT real perception.

MockVLM looks at the crop with classical image analysis and answers the same typed questions a real VLM
would get, with deliberate noise. Swap in openjev.VLMDecider / OpenAICompatDecider for real runs.
mock_decision_backend reasons over the JSON state (observations + measurements) with rules, standing in
for a text LLM / hosted decision API in architecture B.
"""
import numpy as np

from openjev import RuleDecider

from .perception import object_features, object_signal, segment

OUTLINE_TO_CATEGORY = {"circular": "sphere", "square or rectangular": "cube", "hexagonal": "hexagonal",
                       "elongated (rod, wire or needle)": "rod", "lumpy cluster of smaller particles": "aggregate"}


def _central_features(crop):
    crop = np.asarray(crop, dtype=float)
    lab = segment(crop, min_area=10, flatten=0)
    if lab.max() == 0:
        return None
    H, W = lab.shape
    c = lab[H // 2, W // 2]
    if c == 0:  # pick the largest component
        c = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    sig = object_signal(crop, flatten=0)
    sl = tuple(slice(0, s) for s in lab.shape)
    f = object_features(sig, lab, c, sl, float(np.median(sig)))
    f["n_components"] = int(lab.max())
    return f


def _outline_scores(f):
    if f is None:
        return {"fragment or unclear": 3.0}
    ar, sol, fill = f["aspect_ratio"], f["solidity"], f["circumcircle_fill"]
    order, strength = f["symmetry_order"], f["symmetry_strength"]
    s = {
        "elongated (rod, wire or needle)": 3.0 * (ar >= 2.0) * (sol > 0.85),
        "lumpy cluster of smaller particles": 3.0 * (sol < 0.9) * (ar < 2.0) * (f["elongation_harmonic"] > 0.12),
        "circular": 3.0 * (ar < 1.3) * (strength < 0.035),
        "hexagonal": 3.0 * (ar < 1.3) * (order == 6) * (strength >= 0.035),
        "square or rectangular": 3.0 * (ar < 1.6) * (order == 4) * (strength >= 0.06),
        "triangular": 3.0 * (order == 3) * (strength >= 0.15) * (sol > 0.85),
        "fragment or unclear": 2.0 * (f["area"] < 80),
    }
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
        f = _central_features(state["images"][0])
        outline = _outline_scores(f)
        out = []
        for q in questions:
            if q.key == "outline":
                p = _choice(q, outline, self.rng, self.noise)
            elif q.key == "category":
                cat = {}
                for k, v in outline.items():
                    c = OUTLINE_TO_CATEGORY.get(k, "unknown")
                    cat[c] = max(cat.get(c, 0), v)
                p = _choice(q, cat, self.rng, self.noise)
            elif q.key == "edges":
                faceted = f is not None and f["symmetry_strength"] >= 0.035 and f["solidity"] > 0.9
                p = _choice(q, {"straight edges with sharp corners (faceted)": 2.5 * faceted,
                                "smoothly curved, no corners": 2.5 * (not faceted),
                                "rough or bumpy": 2.0 * (f is not None and f["solidity"] < 0.88)}, self.rng, self.noise)
            elif q.key == "surface":
                multi = outline.get("lumpy cluster of smaller particles", 0) > 1
                p = _choice(q, {"made of several sub-particles": 3 * multi, "uniform": 1.5,
                                "shaded like a 3D sphere or rod": 1.0}, self.rng, self.noise)
            elif q.key == "multiple_objects":
                p = _yes(0.8 if f is not None and (f["n_components"] > 1 or f["touches_other"]) else 0.1)
            elif q.key == "in_focus":
                p = _yes(float(np.clip(0 if f is None else (f["edge_sharpness"] - 0.12) * 4, 0.02, 0.98)))
            elif q.key == "countable":
                p = _yes(0.1 if f is None or outline.get("fragment or unclear", 0) > 1 else 0.85)
            else:
                p = np.ones(len(q.labels)) / len(q.labels)
            out.append(np.log(np.clip(np.asarray(p, dtype=float), 1e-9, None)))
        return out


def mock_decision_backend(noise=0.3, seed=0):
    """Architecture-B stand-in: decides from the JSON state's visual observations + measurements."""
    rng = np.random.default_rng(seed)

    def category(state, q):
        obs = state.get("visual_observations", {})
        outline = obs.get("outline", {}).get("value")
        conf = obs.get("outline", {}).get("confidence", 0.5)
        m = state.get("measurements", {})
        scores = {OUTLINE_TO_CATEGORY.get(outline, "unknown"): 3 * conf}
        if m.get("aspect_ratio", 1) >= 2.2:
            scores["rod"] = scores.get("rod", 0) + 1.5  # measurement evidence
        if obs.get("surface", {}).get("value") == "made of several sub-particles":
            scores["aggregate"] = scores.get("aggregate", 0) + 1.0
        return _choice(q, scores, rng, noise)

    def countable(state, q):
        obs = state.get("visual_observations", {})
        p = 0.85
        p *= 1 - 0.8 * obs.get("multiple_objects", 0)
        p *= 0.3 + 0.7 * obs.get("in_focus", 1)
        return _yes(float(np.clip(p, 0.01, 0.99)))

    def contamination(state, q):
        return _yes(0.05)

    return RuleDecider({"category": category, "countable": countable, "is_contamination": contamination})
