"""Sanity-check a VLM backend on synthetic objects with known shapes BEFORE trusting it on your data.

Prints per-object answers, accuracy, calibration and seconds per object, so you can compare models on your
machine (e.g. preset "cpu-small" vs "cpu"). Synthetic objects are easy; a model that fails here will fail on
real images.

    python scripts/check_backend.py --vlm cpu-small --n 30              # local Hugging Face model on CPU
    python scripts/check_backend.py --vlm-url http://localhost:8080/v1 --vlm-model any   # llama.cpp / vLLM server
    python scripts/check_backend.py --mock                              # offline stand-in (no model)
"""
import argparse
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from visual_openjev.decisions import Taxonomy, run_vlm_only  # noqa: E402
from visual_openjev.evals import evaluate  # noqa: E402
from visual_openjev.pipeline import perceive  # noqa: E402
from visual_openjev.synthetic import make_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", help="preset (cpu-tiny, cpu-small, cpu, gpu) or Hugging Face model id")
    ap.add_argument("--vlm-url")
    ap.add_argument("--vlm-model", default="local")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mode", default="letters", choices=["letters", "isolated"])
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--n", type=int, default=30, help="number of objects to test")
    a = ap.parse_args()

    if a.mock:
        from visual_openjev.mock import MockVLM

        vlm = MockVLM()
    elif a.vlm_url:
        from openjev import OpenAICompatDecider

        vlm = OpenAICompatDecider(a.vlm_url, a.vlm_model, mode=a.mode, image_size=a.image_size,
                                  api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        from openjev import VLMDecider

        print(f"loading {a.vlm or 'cpu'} ...")
        vlm = VLMDecider(a.vlm or "cpu", mode=a.mode, image_size=a.image_size)

    data = make_dataset(12, seed=123)
    store = perceive([d[0] for d in data], ground_truth=[(d[1], d[2]) for d in data])
    # interleave object types so even a small --n covers every shape (including the novel one)
    by = {}
    for r in store.records:
        if r.meta["gt"] != "no_match":
            by.setdefault(r.meta["gt"], []).append(r)
    recs = [r for group in itertools.zip_longest(*by.values()) for r in group if r is not None][: a.n]
    tax = Taxonomy.default()
    t0 = time.time()
    run_vlm_only(recs, vlm, tax, key="check")
    dt = (time.time() - t0) / len(recs)
    for r in recs:
        c = r.decisions["check"]["category"]
        print(f"{r.object_id}  true={r.meta['gt']:<10} pred={c['value']:<10} p={c['confidence']:.2f}")
    res = evaluate(recs, tax, ["check"])["check"]
    res["seconds_per_object"] = dt
    print(json.dumps(res, indent=1))
    print("note: 'triangle' objects are not in the taxonomy; the correct answer for them is 'unknown'.")


if __name__ == "__main__":
    main()
