"""Typed decision questions and answers.

Three primitives, mirroring the decision-API pattern (Jev / OpenAI Decisions):

- Choice: pick one of N labelled options.
- Score:  place the input on an ordered ladder of levels (low -> high).
- Noul:   calibrated yes/no, returned as P(yes).

Every question is internally a categorical decision over a list of option labels
(Score levels are ordered, Noul is ["no", "yes"]), which lets all three share heads,
losses and calibration code.
"""
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class Choice:
    key: str
    question: str
    options: list
    type: str = field(default="choice", init=False)

    @property
    def labels(self):
        return list(self.options)


@dataclass
class Score:
    key: str
    question: str
    levels: list  # ordered from lowest to highest
    type: str = field(default="score", init=False)

    @property
    def labels(self):
        return list(self.levels)


@dataclass
class Noul:
    key: str
    question: str
    # Optional natural-language descriptions used by zero-shot (CLIP/SigLIP) backends.
    yes_text: str = None
    no_text: str = None
    type: str = field(default="noul", init=False)

    @property
    def labels(self):
        return ["no", "yes"]


QUESTION_TYPES = {"choice": Choice, "score": Score, "noul": Noul}


def question_to_dict(q):
    return asdict(q)


def question_from_dict(d):
    d = dict(d)
    cls = QUESTION_TYPES[d.pop("type")]
    return cls(**d)


def encode_label(q, value):
    """Map a raw label (option string, level string/index, bool) to a class index. None -> -1."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -1
    if isinstance(q, Noul):
        if isinstance(value, str):
            return int(value.lower() in ("yes", "true", "1"))
        return int(bool(value))
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    return q.labels.index(value)


def format_answer(q, probs, conformal_q=None, epistemic=None):
    """Turn a probability vector over q.labels into a typed, JSON-serialisable answer."""
    probs = np.asarray(probs, dtype=float)
    labels = q.labels
    out = {"type": q.type, "probs": {l: float(p) for l, p in zip(labels, probs)}}
    if isinstance(q, Choice):
        k = int(np.argmax(probs))
        out["value"] = labels[k]
        out["confidence"] = float(probs[k])
        if conformal_q is not None:
            # Split-conformal prediction set: options whose nonconformity 1-p is within the threshold.
            out["set"] = [l for l, p in zip(labels, probs) if 1 - p <= conformal_q]
    elif isinstance(q, Score):
        expected = float(np.sum(np.arange(len(labels)) * probs))
        out["value"] = expected  # fractional position on the ladder, 0 .. L-1
        out["level"] = labels[int(round(expected))]
        out["confidence"] = float(np.max(probs))
    else:
        p_yes = float(probs[1])
        out["value"] = p_yes
        out["decision"] = p_yes >= 0.5
        out["confidence"] = max(p_yes, 1 - p_yes)
    if epistemic is not None:
        out["epistemic"] = float(epistemic)
    return out
