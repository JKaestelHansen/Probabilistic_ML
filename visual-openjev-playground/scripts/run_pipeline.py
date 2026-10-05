"""Run the perception + decision stack on object images (particles, MOFs, grains, cells, ...).

Offline demo (synthetic shapes, stand-in backends):
    python scripts/run_pipeline.py --synthetic 40 --out runs/demo

Your images, zero labels (architectures A + B), local VLM on CPU (presets: cpu-tiny, cpu-small, cpu, gpu):
    python scripts/run_pipeline.py --images data/raw --vlm cpu --out runs/r1

VLM behind an OpenAI-compatible server that returns logprobs (llama.cpp, vLLM, SGLang), optional separate
text LLM for the decision step:
    python scripts/run_pipeline.py --images data/raw --vlm-url http://localhost:8080/v1 --vlm-model local \
        --decider-llm Qwen/Qwen3-1.7B --out runs/r1

After filling in runs/r1/label_batch_labels.csv, add the trained classifier (architecture C):
    python scripts/run_pipeline.py --resume runs/r1 --labels runs/r1/label_batch_labels.csv --out runs/r1
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from visual_openjev import measurements  # noqa: E402
from visual_openjev.classification import object_vectors  # noqa: E402
from visual_openjev.decisions import Taxonomy  # noqa: E402
from visual_openjev.discovery import apply_labels, export_review_html, kmeans, read_labels, select_for_review  # noqa: E402
from visual_openjev.evals import evaluate  # noqa: E402
from visual_openjev.objects import ObjectStore  # noqa: E402
from visual_openjev.pipeline import perceive, run_classifier, run_zero_label  # noqa: E402


def load_images(folder):
    from PIL import Image

    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".npy")))
    imgs = [np.load(os.path.join(folder, f)) if f.endswith(".npy") else np.asarray(Image.open(os.path.join(folder, f)))
            for f in files]
    # RGB / multi-channel images: segment on the mean channel (customise for your stains / detectors)
    imgs = [im.mean(-1) if im.ndim == 3 else im for im in imgs]
    return imgs, [os.path.splitext(f)[0] for f in files]


def build_backends(a):
    if a.vlm:
        from openjev import VLMDecider

        vlm = VLMDecider(a.vlm, mode=a.mode, image_size=a.image_size)
    elif a.vlm_url:
        from openjev import OpenAICompatDecider

        vlm = OpenAICompatDecider(a.vlm_url, a.vlm_model, api_key=os.environ.get("OPENAI_API_KEY"), mode=a.mode,
                                  image_size=a.image_size)
    else:
        from visual_openjev.mock import MockVLM, mock_decision_backend

        print("no --vlm given: using offline stand-in backends (demo only, not real perception)")
        return MockVLM(), mock_decision_backend()
    decider = None
    if a.decider_llm:
        from openjev import LLMDecider

        decider = LLMDecider(a.decider_llm, mode=a.mode)
    return vlm, decider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--resume", help="existing run directory (ObjectStore) to continue from")
    ap.add_argument("--taxonomy")
    ap.add_argument("--vlm", help="preset (cpu-tiny, cpu-small, cpu, gpu) or Hugging Face VLM id")
    ap.add_argument("--image-size", type=int, default=224, help="longer side of crops shown to the VLM")
    ap.add_argument("--polarity", default="auto", choices=["auto", "bright", "dark"],
                    help="objects brighter (fluorescence, SEM) or darker (brightfield, TEM) than background")
    ap.add_argument("--segmentation", default="threshold", choices=["threshold", "cellpose"])
    ap.add_argument("--vlm-url", help="OpenAI-compatible base URL")
    ap.add_argument("--vlm-model")
    ap.add_argument("--decider-llm", help="Hugging Face text LLM id for the decision layer (default: the VLM)")
    ap.add_argument("--mode", default="letters", choices=["letters", "isolated"])
    ap.add_argument("--backbone", default="handcrafted")
    ap.add_argument("--labels", help="filled-in labels CSV -> train classifier (architecture C)")
    ap.add_argument("--n-review", type=int, default=100)
    ap.add_argument("--pixel-size-um", type=float)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tax = Taxonomy.load(a.taxonomy) if a.taxonomy else Taxonomy.default()

    if a.resume:
        store = ObjectStore.load(a.resume)
    else:
        if a.synthetic:
            from visual_openjev.synthetic import make_dataset

            data = make_dataset(a.synthetic)
            store = perceive([d[0] for d in data], ground_truth=[(d[1], d[2]) for d in data])
        else:
            imgs, ids = load_images(a.images)
            store = perceive(imgs, ids, method=a.segmentation, polarity=a.polarity)
        print(f"{len(store)} objects")
        object_vectors(store.records, a.backbone)
        store.info["backbone"] = a.backbone
        vlm, decider = build_backends(a)
        run_zero_label(store, tax, vlm, decider,
                       versions={"vlm": a.vlm or a.vlm_model or "mock", "decider": a.decider_llm or "vlm"})

    primary = "B_vlm_decision"
    if a.labels:
        print(f"applied {apply_labels(store.records, read_labels(a.labels))} labels")
        run_classifier(store, tax)
        primary = "C_classifier"

    # review package: cluster the objects that need review + an active-learning pick
    pending = [r for r in store.records if r.review.get("needs_review") and not r.label]
    if pending:
        groups, _ = kmeans(np.stack([r.embedding for r in pending]), min(12, max(1, len(pending) // 5)))
        export_review_html(pending, os.path.join(a.out, "review.html"), groups, title="Objects needing review")
    pick = select_for_review(store.records, a.n_review)
    if pick:
        export_review_html(pick, os.path.join(a.out, "label_batch.html"), title="Next labelling batch")

    rows = measurements.summarize(store.records, primary, a.pixel_size_um, exclude=tax.contamination + ["unknown"])
    measurements.write_csv(rows, os.path.join(a.out, "measurements.csv"))
    keys = [k for k in ("A_vlm_only", "B_vlm_decision", "C_classifier") if any(k in r.decisions for r in store.records)]
    if any("gt" in r.meta for r in store.records):
        test = [r for r in store.records if not r.label]
        res = evaluate(test, tax, keys)
        json.dump(res, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
        print(json.dumps(res, indent=1))
    store.save(a.out)
    print(f"{len(pending)} objects need review; outputs in {a.out}")


if __name__ == "__main__":
    main()
