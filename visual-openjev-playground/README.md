# visual-openjev-playground

Jev-style **typed decisions on top of vision-language models**, for images of objects: particles, MOFs,
grains, crystals, cells. You define the answer space (a taxonomy that always includes `unknown`). The
VLM's answers are read as **probabilities over those options**, with no free text to parse. Objects that are
uncertain, unknown or disputed go to a review queue, so the taxonomy can grow from your data.

Default taxonomy: **sphere · cube · rod · aggregate · hexagonal · unknown** (`configs/taxonomy.json`).

```
image ─► PERCEPTION   flatten illumination · bright- or dark-object segmentation (or Cellpose-SAM)
                      per object: crop, mask, area, aspect ratio, solidity, circumcircle fill,
                      outline symmetry (3/4/6-fold), contrast, edge sharpness
      ─► EVIDENCE     crop embedding (DINOv2 / SigLIP / handcrafted) · typed VLM observations
                      (outline, edges, surface, overlap, focus)
      ─► DECISION     category (taxonomy + unknown) · countable          ← openjev decision layer
                      A: VLM-only   B: VLM observations → decision model   C: classifier → decision
      ─► REVIEW       score = max(uncertainty, P(unknown), A-vs-B disagreement, low focus, OOD) + reasons
      ─► MEASUREMENT  counts / areas per image and category (only accepted, countable objects)
      ─► DISCOVERY    clusters of unknowns · active-learning batches · HTML review sheets → labels
                      → train classifier (C) → cascade: confident → accept, rest → VLM / human
```

## No GPU? Which VLM to use

The decision layer only needs **one forward pass per object** over a small crop (default 224 px), with no
text generation. That's what makes CPU use practical.

| Preset | Model | Size | Where |
|---|---|---|---|
| `cpu-tiny` | HuggingFaceTB/SmolVLM-256M-Instruct | ~0.5 GB | any laptop; fastest, weakest |
| `cpu-small` | HuggingFaceTB/SmolVLM-500M-Instruct | ~1 GB | any laptop |
| `cpu` (default) | Qwen/Qwen3-VL-2B-Instruct | ~4 GB | 8–16 GB RAM laptop; best quality on CPU |
| `gpu` | Qwen/Qwen2.5-VL-7B-Instruct | ~15 GB | **free** Google Colab / Kaggle T4: `notebooks/colab_free_gpu.ipynb` |

Other routes:
- **Quantised models on CPU via llama.cpp:** run `llama-server -hf ggml-org/Qwen2.5-VL-3B-Instruct-GGUF` (or
  `ggml-org/gemma-3-4b-it-GGUF`, `ggml-org/SmolVLM-500M-Instruct-GGUF`), then pass
  `--vlm-url http://localhost:8080/v1`. This needs a server build that returns `top_logprobs`; the backend
  raises a clear error if yours doesn't.
- **Any vLLM / SGLang server or hosted endpoint** that returns logprobs, via `--vlm-url`.

Always run `scripts/check_backend.py` first. It scores the model on synthetic shapes with known answers,
including triangles, which aren't in the taxonomy and should come back as `unknown`. It also reports
seconds per object, so you can pick a model for your machine.

## Quick start

```bash
pip install -e ".[hf,dev]"                  # CPU torch is fine
pytest -q                                   # offline, ~30 s (tiny random-weight models, no downloads)

python scripts/check_backend.py --mock      # pipeline check, no model
python scripts/check_backend.py --vlm cpu-small --n 30
python scripts/check_backend.py --vlm cpu --n 30

# your images (tif / png / jpg / npy), zero labels: architectures A + B
python scripts/run_pipeline.py --images data/raw --vlm cpu --out runs/r1
#   -> runs/r1/review.html          flagged objects, clustered
#   -> runs/r1/label_batch.html     next objects worth labelling (+ label_batch_labels.csv to fill in)
#   -> runs/r1/measurements.csv     counts / areas per image and category
#   -> runs/r1/objects.jsonl + arrays.npz   all evidence per object (re-analyse without re-segmenting)

# after labelling: train the fast classifier (architecture C) and re-review
python scripts/run_pipeline.py --resume runs/r1 --labels runs/r1/label_batch_labels.csv --out runs/r1
```

Useful options:
- `--taxonomy my.json`: your own categories (add `"contamination": true` to flag debris types).
- `--mode isolated`: scores each option on its own, so option order can't bias the answer.
- `--backbone dinov2`: better embeddings for clustering and the classifier.
- `--pixel-size-um 0.05`: report areas in µm².

## Using the decision layer directly

```python
from openjev import Choice, Noul, VLMDecider
vlm = VLMDecider("cpu")   # or "cpu-small", "gpu", or any Hugging Face VLM id
vlm.decide({"text": "Crop of one particle from an SEM image.", "images": [crop]},
           [Choice("shape", "Which shape?", ["sphere", "cube", "rod", "aggregate", "hexagonal", "unknown"]),
            Noul("in_focus", "Is the particle in focus?")])
# {'shape': {'value': 'cube', 'probs': {...}, 'confidence': 0.81}, 'in_focus': {'value': 0.93, ...}}
```

Backends: `VLMDecider` (local Hugging Face VLM), `OpenAICompatDecider` (server with logprobs), `LLMDecider`
(text, for architecture B), and `RuleDecider` (guardrails). Calibrate with
`decider.fit_temperature(states, question, labels)` once you have a few labels.

## Layout

```
openjev/            decision layer: schema (Choice/Score/Noul), deciders, calibration (ECE, temperature,
                    conformal, epistemic/aleatoric), fast trained heads (vision/), text benchmark (text/), API
visual_openjev/     objects, perception, vlm, decisions, classification, discovery, measurements, evals,
                    synthetic data, offline mock backends, pipeline
scripts/            check_backend.py, run_pipeline.py, train_vision.py, benchmark_text.py
notebooks/          colab_free_gpu.ipynb
configs/            taxonomy.json
tests/              31 offline tests (`pytest -q` from this folder)
```

## Status

- **Tested offline:**
  - Every module and the full loop (zero-label → review → labels → classifier → cascade → measurements).
  - The real `VLMDecider` path, on a tiny random-weight Qwen2-VL.
  - `OpenAICompatDecider`, against a fake server.
- **Not yet measured:** real VLM accuracy on real images. Numbers from `--mock` / `--synthetic` come from
  classical-CV stand-ins and say nothing about VLM quality. Use `check_backend.py`, then a small
  hand-labelled set of your own images with `visual_openjev.evals.evaluate`.
- **Hosted Jev:** currently accepts only text/objects as state, so it can only plug in at architecture B
  (over the JSON evidence), as a `Decider` subclass once you have API access.
