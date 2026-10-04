"""Train and evaluate a VisionJev model.

Real data:
    python scripts/train_vision.py --images-dir imgs/ --labels labels.csv --questions questions.json \
        --backbone dinov2 --out qc_model.pt
  labels.csv:     filename,<question key>,<question key>,...   (empty cell = unlabelled)
  questions.json: [{"type": "noul", "key": "usable", "question": "Is this image usable?"},
                   {"type": "choice", "key": "morphology", "question": "...", "options": ["round", "elongated"]}]

Synthetic demo (no data needed):
    python scripts/train_vision.py --synthetic 800 --out demo.pt
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from openjev import question_from_dict  # noqa: E402
from openjev.vision import VisionJev  # noqa: E402


def load_image(path):
    if path.endswith(".npy"):
        return np.load(path)
    from PIL import Image

    return np.asarray(Image.open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir")
    ap.add_argument("--labels")
    ap.add_argument("--questions")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--backbone", default="handcrafted")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out", default="vision_jev.pt")
    a = ap.parse_args()

    if a.synthetic:
        from openjev.data import QUESTIONS, make_blob_dataset

        questions = QUESTIONS
        images, labels = make_blob_dataset(a.synthetic)
    else:
        questions = [question_from_dict(d) for d in json.load(open(a.questions))]
        rows = list(csv.DictReader(open(a.labels)))
        images = [load_image(os.path.join(a.images_dir, r["filename"])) for r in rows]
        labels = {q.key: [r.get(q.key) or None for r in rows] for q in questions}

    jev = VisionJev(questions, backbone=a.backbone, n_members=a.members)
    print(f"embedding {len(images)} images with {a.backbone} ...")
    X = jev.embed(images)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * a.test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    sub = lambda s: {k: [v[i] for i in s] for k, v in labels.items()}
    jev.fit(X[tr], sub(tr), epochs=a.epochs)
    print(json.dumps({"temperatures": jev.temperatures, "test": jev.evaluate(X[te], sub(te))}, indent=2))
    jev.save(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
