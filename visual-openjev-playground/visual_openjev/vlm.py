"""VLM perception: extract visual evidence from object crops (particles, crystals, grains, cells) as TYPED
observations, not prose.

Each observation is itself a bounded decision asked of a vision backend (openjev VLMDecider /
OpenAICompatDecider). The answers (with probabilities) are stored in record.observations["vlm"] and become
part of the state that the decision layer reasons over.
"""
from openjev import Choice, Noul

DEFAULT_CONTEXT = ("The image is a crop around a single detected object from a microscopy image (optical, "
                   "fluorescence or electron microscopy; the object may be brighter or darker than the background). "
                   "Judge only the object in the centre.")

OBSERVATION_QUESTIONS = [
    Choice("outline", "What is the outline of the central object?",
           ["circular", "square or rectangular", "hexagonal", "triangular", "elongated (rod, wire or needle)",
            "lumpy cluster of smaller particles", "irregular", "fragment or unclear"]),
    Choice("edges", "What do the object's edges and corners look like?",
           ["straight edges with sharp corners (faceted)", "smoothly curved, no corners", "rough or bumpy", "indistinct"]),
    Choice("surface", "What does the object's interior look like?", ["uniform", "shaded like a 3D sphere or rod",
                                                                      "made of several sub-particles", "hollow or ring-like", "unclear"]),
    Noul("multiple_objects", "Does the crop's central region contain more than one separate object touching or overlapping?"),
    Noul("in_focus", "Is the object in focus?"),
]


def observe(records, decider, questions=OBSERVATION_QUESTIONS, context=DEFAULT_CONTEXT, key="vlm"):
    """Ask every observation question about every crop; one decider call per object (all questions batched)."""
    for r in records:
        ans = decider.decide({"text": context, "images": [r.crop]}, questions)
        r.observations[key] = ans
    return records


def compact(answers):
    """Typed answers -> short {key: value (+ confidence)} dict for downstream prompts / reports."""
    out = {}
    for k, a in answers.items():
        if a["type"] == "noul":
            out[k] = round(a["value"], 3)  # P(yes)
        else:
            out[k] = {"value": a.get("level", a["value"]), "confidence": round(a["confidence"], 3)}
    return out
