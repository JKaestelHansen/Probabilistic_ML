# openjev

Open-source **typed decision models**: you give a state (an image or text) and typed questions, and get back
structured answers with **calibrated probabilities**. There is no free text to parse. This follows the
decision-API pattern of TypeSafe's Jev and OpenAI's Decisions API, built from open models. The focus is on
images: QC of good vs bad images, and morphology. A text version is included for benchmarking.

| Primitive | Returns |
|---|---|
| `Choice(key, question, options)` | `value` (an option), `probs`, `confidence`, `set` (split-conformal prediction set) |
| `Score(key, question, levels)`   | `value` (fractional position on the ordered ladder), `level`, `probs` |
| `Noul(key, question)`            | `value` = P(yes), `decision`, `confidence` |

Image answers also carry `epistemic`: the ensemble mutual information, i.e. "the model doesn't know".
This is separate from aleatoric ambiguity.

## Architecture

```
image ─► frozen backbone ─► embedding ─► deep ensemble of TypedHeads ─► per-question temperature ─► typed answers
         (DINOv2/v3, SigLIP2,            (shared MLP trunk, then         (fit on held-out half A)     + conformal sets
          microscopy ViT, or              softmax / CORAL ordinal /                                     (held-out half B)
          handcrafted QC features)        sigmoid head per question)

text ─► LLMDecider:  open LLM, all questions in one batched forward pass, option-letter probabilities (no decoding)
     └► TextJevLite: encoder tokens + question + option embeddings ─► cross-attention head ─► scores any option set
```

- **Training:** proper scoring rules only (cross-entropy, CORAL BCE, BCE). Missing labels are masked, so
  partially labelled datasets work.
- **Calibration:** temperature scaling and split conformal prediction. Metrics: ECE/MCE, Brier, NLL,
  AUROC, reliability bins, and total/aleatoric/epistemic entropy decomposition (`openjev/calibration.py`).
  These are the classification counterparts of `../uncertainty_quantification`.
- `QuestionConditionedHead` is modality-agnostic. The roadmap item below reuses it for open-vocabulary
  image questions.

## Recommended backbones

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

1. **Open-vocabulary image questions:** feed SigLIP2 patch tokens to `QuestionConditionedHead` as the
   state, with the question and options encoded by the SigLIP2 text tower. Train on many (image, question,
   answer) triples.
2. **VLM decider:** the `LLMDecider` letter-probability trick with Qwen-VL / InternVL / Gemma 3, for
   arbitrary image questions with no training.
3. **Distillation:** pre-label images with the VLM decider or a hosted API, then train VisionJev on the
   soft labels (TextJevLite already accepts probability-vector targets).
4. **Hosted-API adapters:** TypeSafe Jev and OpenAI Decisions, for benchmarking. Their request schemas
   were not publicly verifiable when this was written, so no adapter is included. Any object with
   `.decide(state, questions)` plugs into `scripts/benchmark_text.py`.
