# openjev

A provider-agnostic **typed decision layer** that sits alongside an LLM or VLM. You give a state (text, a
JSON object, and/or images) and typed questions with bounded answers, and get back structured answers with
**calibrated probabilities**. No free text is generated or parsed. This follows the decision-API pattern of
TypeSafe's Jev and OpenAI's Decisions API, built on open models. The microscopy stack that uses it lives in
[`../microscopy_ai`](../microscopy_ai).

| Primitive | Returns |
|---|---|
| `Choice(key, question, options)` | `value` (an option), `probs`, `confidence` |
| `Score(key, question, levels)`   | `value` (fractional position on the ordered ladder), `level`, `probs` |
| `Noul(key, question)`            | `value` = P(yes), `decision`, `confidence` |

## Deciders (`openjev/deciders.py`)

| Backend | State | Notes |
|---|---|---|
| `LLMDecider(hf_id)` | text / JSON | local causal LM (e.g. Qwen3) |
| `VLMDecider(hf_id)` | text / JSON + images | local image-text-to-text model (Qwen-VL, InternVL-hf, Gemma 3) |
| `OpenAICompatDecider(url, model)` | text / JSON + images | vLLM / SGLang / hosted endpoint returning `top_logprobs` |
| `RuleDecider({key: fn})` | anything | deterministic guardrails and baselines |

Two scoring modes, neither of which decodes an answer:
- `"letters"`: options shown as A, B, C…; probabilities come from the next-token distribution. All of a
  state's questions go in one batched forward pass.
- `"isolated"`: each candidate is judged yes/no on its own, and P(yes) is normalised across candidates.
  Order-invariant by construction.

Every decider has `decide(state, questions)` and `fit_temperature(states, question, labels)`.

```python
from openjev import Choice, Noul, VLMDecider
dec = VLMDecider("Qwen/Qwen2.5-VL-7B-Instruct")
dec.decide({"text": "Crop of one object from a fluorescence image.", "images": [crop]},
           [Choice("category", "Which category?", ["type_a", "type_b", "unknown"]),
            Noul("in_focus", "Is the object in focus?")])
```

## Also included

- `calibration.py`: ECE/MCE, Brier, NLL, AUROC, temperature scaling, split conformal sets, and
  epistemic/aleatoric entropy decomposition.
- `vision/`: `VisionJev`, a fast path trained on frozen embeddings (deep-ensemble heads, per-question
  temperatures, conformal sets). Used as the specialised classifier in `microscopy_ai`. `ZeroShotVision`
  (SigLIP2).
- `text/`: `TextJevLite`, a small trainable open-vocabulary decision head that accepts distilled soft
  labels, plus a synthetic ticket benchmark.
- `api.py`: a FastAPI `/v1/decide` endpoint.

## Recommended image backbones (for embeddings / the fast path)

| Use | Backbone |
|---|---|
| General image features (start here) | `dinov2` (facebook/dinov2-base), or DINOv3 via `hf:<id>` |
| Zero-shot before you have labels | `ZeroShotVision("siglip2")` with one prompt per option |
| Fluorescence / Cell Painting | a microscopy ViT via `hf:<id>` (e.g. Recursion's OpenPhenom) |
| Per-cell morphology | segment with Cellpose-SAM, crop each cell, run VisionJev on the crops |
| No GPU, no downloads, QC baseline | `handcrafted` (focus spectrum, noise, saturation, object shape stats) |

Text: `LLMDecider("Qwen/Qwen3-1.7B")` (or any chat LLM); `TextJevLite("answerdotai/ModernBERT-base")`.

## Run

```bash
pip install -e ".[hf,api,dev]"
pytest -q tests                                     # ~30 s on CPU, no downloads

# vision: synthetic demo, then your data
python scripts/train_vision.py --synthetic 800 --out demo.pt
python scripts/train_vision.py --images-dir imgs/ --labels labels.csv \
       --questions questions.json --backbone dinov2 --out qc.pt

# text benchmark (add --llm for the zero-shot LLM baseline, --data for your JSONL)
python scripts/benchmark_text.py --llm Qwen/Qwen3-1.7B

# serve both behind one decision endpoint
OPENJEV_VISION=qc.pt OPENJEV_TEXT_LLM=Qwen/Qwen3-1.7B uvicorn openjev.api:app
curl -X POST localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "text": "My card was charged twice for the same order.",
  "questions": [{"type": "choice", "key": "queue", "question": "Which team?",
                 "options": ["billing", "technical", "shipping", "fraud"]}]}'
```

`examples/quickstart_vision.py` walks through the same steps cell by cell, including a reliability diagram.

## Synthetic benchmarks

- `openjev.data.make_blob_dataset` extends `../generate_blob_images.py`. It covers three blob morphologies,
  four defocus levels, low signal and detector saturation, so every primitive has ground truth.
- `openjev.text.benchmark_data.make_ticket_benchmark` covers ticket routing, urgency and sentiment.
- Both are deliberately easy. They check the plumbing and calibration, not real-world accuracy.

## Roadmap

1. **Port shared-prefix candidate scoring with LoRA** from
   [IamBusy/OpenJev](https://github.com/IamBusy/OpenJev) (Apache-2.0): fine-tune a small LM/VLM to score
   candidates once labels exist.
2. **Distillation:** pre-label with `VLMDecider`, then train the fast path (`VisionJev` / `TextJevLite`) on
   soft labels.
3. **Hosted adapters:** TypeSafe Jev (text/object state only) and OpenAI Decisions, as `Decider`
   subclasses once their request schemas are available, so all backends can be benchmarked side by side.
4. **Open-vocabulary image head:** SigLIP2 patch tokens plus text-tower question/option embeddings fed to
   `QuestionConditionedHead`.
