# microscopy_ai

A microscopy **perception + decision** stack for labelling images when you start with zero labels. Every
detected object becomes a record that collects evidence over time. Typed decisions (Choice / Score / yes-no,
always with an `unknown` option) come from a provider-agnostic decision layer
([`../openjev`](../openjev)). Unknowns are clustered for human review, so the taxonomy grows as the data
reveals new categories.

```
image ─► PERCEPTION      QC metrics · segmentation (threshold | Cellpose-SAM) · per-object crop, mask, features
      ─► EVIDENCE        crop embedding (DINOv2 / SigLIP / handcrafted) · typed VLM observations
                         (shape, boundary, texture, overlap, focus)
      ─► DECISION        category (taxonomy + unknown) · is_contamination · countable
                         A: VLM-only   B: VLM observations → decision backend   C: classifier → decision
      ─► REVIEW          score = max(uncertainty, P(unknown), source disagreement, low quality, OOD) + reasons
      ─► MEASUREMENT     counts / areas per image and category, counting only accepted, countable objects
      ─► DISCOVERY       clustering · nearest neighbours · active-learning batches · HTML contact sheets
                         → human labels → classifier (C) → cascade: confident → accept, rest → VLM / human
```

## Modules (`microscopy_ai/`)

| Module | Role |
|---|---|
| `objects.py` | `ObjectRecord`, plus `ObjectStore`, which saves to `objects.jsonl` + `arrays.npz`. Nothing is overwritten; each architecture writes its own key. |
| `perception.py` | Image QC and segmentation. Per object: area, perimeter, circularity, aspect ratio, solidity, contrast, edge sharpness, touches-other/border. |
| `vlm.py` | Typed observation questions asked of a VLM on each crop. |
| `decisions.py` | `Taxonomy` (JSON, always includes `unknown`, grows with `.add`), decision questions, architectures A/B, and `review()` |
| `classification.py` | Crop embeddings. Small ensemble classifier with a k-NN OOD score that becomes P(unknown). No-LLM fast path and cascade. |
| `discovery.py` | k-means, neighbours, active-learning selection, review HTML + labels CSV |
| `measurements.py` | Per-image / per-category counts and areas (px² or µm²) |
| `evals.py` | Compares A/B/C: accuracy, NLL, Brier, ECE, unknown recall, auto-accept rate, accuracy on auto-accepted objects |
| `synthetic.py` | Synthetic fields with ground truth: 3 known cell types, a contamination type (fiber), and a **novel** type (ring) left out of the taxonomy |
| `mock.py` | Offline stand-in backends (classical analysis behind the typed interface), for tests and demos only |

## Decision backends (from `openjev`)

| Backend | Use |
|---|---|
| `VLMDecider("Qwen/Qwen2.5-VL-7B-Instruct")` | local VLM; any Hugging Face image-text-to-text model (Qwen-VL, InternVL-hf, Gemma 3) |
| `OpenAICompatDecider(url, model)` | a VLM served by vLLM / SGLang, or a hosted model that returns logprobs |
| `LLMDecider("Qwen/Qwen3-4B")` | text-only decision backend for architecture B (reasons over the JSON evidence) |
| `RuleDecider({...})` | deterministic guardrails, e.g. "objects under 30 px are never countable" |

No backend generates free text. Option probabilities are read from the next-token distribution in one of
two modes:
- `mode="letters"`: one batched pass per object.
- `mode="isolated"`: each candidate is judged on its own, so the result doesn't depend on option order.

Fit per-question temperatures with `decider.fit_temperature(...)` once a few labels exist.

## Run

```bash
pip install -e ../openjev[hf] && pip install pillow   # + `cellpose` for Cellpose-SAM
pytest -q tests                                        # offline, ~10 s

# offline demo (stand-in backends, synthetic data)
python scripts/run_pipeline.py --synthetic 40 --out runs/demo

# real data, zero labels: local VLM for perception + decisions
python scripts/run_pipeline.py --images data/raw --taxonomy taxonomy.json \
    --vlm Qwen/Qwen2.5-VL-7B-Instruct --backbone dinov2 --out runs/r1
# open runs/r1/label_batch.html, fill runs/r1/label_batch_labels.csv, then add the classifier (C):
python scripts/run_pipeline.py --resume runs/r1 --labels runs/r1/label_batch_labels.csv --out runs/r1
```

Outputs:
- `objects.jsonl` + `arrays.npz`: the full evidence store.
- `review.html`: flagged objects, clustered.
- `label_batch.html` + CSV: the next active-learning batch.
- `measurements.csv`, and `eval.json` when ground truth exists.

`taxonomy.json`:
```json
{"categories": [{"name": "type_a", "description": "what it looks like"},
                {"name": "dust", "description": "...", "contamination": true}]}
```

## What is and isn't validated

- **Tested offline:**
  - Every module and the full loop (zero-label → review → labels → classifier → cascade → measurements).
  - The real `VLMDecider` code path, through a tiny random-weight Qwen2-VL.
  - `OpenAICompatDecider`, against a fake server.
- **Not yet measured:** accuracy with real VLMs on real microscopy. The numbers from `--synthetic` use the
  stand-in backends and say nothing about VLM quality. Benchmark A/B/C with `evals.evaluate` on a small
  human-labelled set of your own images first.
- **Hosted decision APIs:** Jev currently takes only text/objects as state, so it fits architecture B (on the
  JSON evidence), not raw images. Add it, or OpenAI Decisions, as a `Decider` subclass once you have API access.
