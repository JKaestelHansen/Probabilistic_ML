"""Every detected object is an entity that accumulates evidence over time; nothing is overwritten.

    record.features        classical CV measurements (area, circularity, solidity, contrast, ...)
    record.embedding       visual embedding of the crop (DINOv2 / SigLIP / handcrafted)
    record.observations    perception outputs, e.g. {"vlm": {question key: typed answer}}
    record.classifier      small-classifier output {"probs": {...}, "ood": float, "model": version}
    record.decisions       decision-layer outputs per architecture / backend
    record.review          final review score, reasons, needs_review
    record.label           human label (None until reviewed)
    record.meta            free-form: ground truth for synthetic data, model versions, provenance

ObjectStore persists to a directory: objects.jsonl (everything JSON-able), arrays.npz (crops, masks,
embeddings keyed by object id). Re-analysing later never requires re-segmenting the raw images.
"""
import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class ObjectRecord:
    object_id: str
    image_id: str
    bbox: list  # [y0, x0, y1, x1] in image pixels
    features: dict = field(default_factory=dict)
    crop: np.ndarray = None
    mask: np.ndarray = None
    embedding: np.ndarray = None
    observations: dict = field(default_factory=dict)
    classifier: dict = field(default_factory=dict)
    decisions: dict = field(default_factory=dict)
    review: dict = field(default_factory=dict)
    label: str = None
    meta: dict = field(default_factory=dict)

    ARRAYS = ("crop", "mask", "embedding")

    def to_json(self):
        d = asdict(self)
        for k in self.ARRAYS:
            d.pop(k)
        return d


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class ObjectStore:
    def __init__(self, records=None, info=None):
        self.records = list(records or [])
        self.info = dict(info or {})  # model versions, pipeline config, ...

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def by_id(self):
        return {r.object_id: r for r in self.records}

    def embeddings(self):
        return np.stack([r.embedding for r in self.records])

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "objects.jsonl"), "w") as f:
            for r in self.records:
                f.write(json.dumps(r.to_json(), default=_jsonable) + "\n")
        arrays = {}
        for r in self.records:
            for k in ObjectRecord.ARRAYS:
                v = getattr(r, k)
                if v is not None:
                    arrays[f"{r.object_id}::{k}"] = v
        np.savez_compressed(os.path.join(path, "arrays.npz"), **arrays)
        with open(os.path.join(path, "info.json"), "w") as f:
            json.dump(self.info, f, indent=1, default=_jsonable)

    @classmethod
    def load(cls, path):
        arrays = np.load(os.path.join(path, "arrays.npz"))
        records = []
        with open(os.path.join(path, "objects.jsonl")) as f:
            for line in f:
                r = ObjectRecord(**json.loads(line))
                for k in ObjectRecord.ARRAYS:
                    key = f"{r.object_id}::{k}"
                    if key in arrays:
                        setattr(r, k, arrays[key])
                records.append(r)
        info_path = os.path.join(path, "info.json")
        info = json.load(open(info_path)) if os.path.exists(info_path) else {}
        return cls(records, info)
