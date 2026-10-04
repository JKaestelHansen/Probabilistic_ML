"""VLM perception: extract visual evidence from object crops as TYPED observations, not prose.

Each observation is itself a bounded decision asked of a vision backend (openjev VLMDecider /
OpenAICompatDecider). The answers (with probabilities) are stored in record.observations["vlm"] and become
part of the state that the decision layer reasons over.
"""
from openjev import Choice, Noul

DEFAULT_CONTEXT = ("The image is a crop around a single detected object from a fluorescence microscopy image "
                   "(bright = fluorescent signal). Judge only the object in the centre.")

OBSERVATION_QUESTIONS = [
    Choice("shape", "What is the overall shape of the central object?",
           ["round", "oval or rod-shaped", "irregular or lobed", "long thin filament", "ring or hollow", "fragment or unclear"]),
    Choice("boundary", "How does the boundary of the object look?", ["sharp", "slightly soft", "indistinct"]),
    Choice("texture", "What is the internal texture of the object?", ["uniform", "granular", "hollow centre", "unclear"]),
    Noul("multiple_objects", "Does the crop's central region contain more than one overlapping or touching object?"),
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
