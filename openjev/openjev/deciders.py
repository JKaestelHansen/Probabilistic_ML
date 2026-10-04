"""Provider-agnostic decision layer: typed questions over a state -> calibrated typed answers.

A state is any of
    "free text"
    {"any": "json-serialisable", "observations": ...}            (rendered as JSON for the model)
    {"text": "...", "images": [array, ...]}                       (images only for vision backends)
A dict may carry "images" next to other fields; everything except "images" is rendered as text.

Backends (all share prompting, scoring modes, calibration and answer formatting):
    LLMDecider           local Hugging Face causal LM (text)
    VLMDecider           local Hugging Face vision-language model (text + images), e.g. Qwen-VL
    OpenAICompatDecider  any OpenAI-compatible /chat/completions server that returns logprobs
                         (vLLM / SGLang serving an open model, or a hosted provider)
    RuleDecider          deterministic python rules (guardrails, tests, hand-written baselines)

Scoring modes (no answer text is generated in either):
    "letters"   options listed as A, B, C...; probabilities read from the next-token distribution.
                One forward pass for all questions of a state. Fast, but can be sensitive to option order.
    "isolated"  each candidate judged on its own ("is this candidate correct? yes/no"), P(yes) normalised
                across candidates. Order-invariant by construction; costs one sequence per option.
"""
import base64
import io
import json
import string

import numpy as np

from .schema import Noul, Score, format_answer
from . import calibration as cal

SYSTEM = ("You are a decision model. Read the state and answer the question by replying with the letter of "
          "exactly one option. Reply with the letter only.")
LETTERS = string.ascii_uppercase


# ------------------------------------------------------------------------------------------- state & prompts
def split_state(state):
    """-> (state text, list of images)."""
    if isinstance(state, str):
        return state, []
    state = dict(state)
    images = list(state.pop("images", []) or [])
    if set(state) == {"text"}:
        return state["text"], images
    return json.dumps(state, indent=1, default=_json_default), images


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def options_of(q):
    return ["No", "Yes"] if isinstance(q, Noul) else [str(o) for o in q.labels]


def letter_prompt(state_text, q, n_images=0):
    options = options_of(q)
    if len(options) > len(LETTERS):
        raise ValueError(f"{q.key}: letter mode supports at most {len(LETTERS)} options; use mode='isolated'")
    lines = []
    if n_images:
        lines.append(f"The image{'s' if n_images > 1 else ''} above {'are' if n_images > 1 else 'is'} part of the state.")
    if state_text:
        lines += [f"State:\n{state_text}", ""]
    lines.append(f"Question: {q.question}")
    lines.append("Options (ordered from lowest to highest):" if isinstance(q, Score) else "Options:")
    lines += [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    lines += ["", "Answer with a single letter."]
    return "\n".join(lines), len(options)


def candidate_prompt(state_text, q, option, n_images=0):
    yn = Noul(q.key, f"Question: {q.question}\nCandidate answer: {option}\nIs the candidate answer correct?")
    return letter_prompt(state_text, yn, n_images)


def softmax(z, T=1.0):
    z = np.asarray(z, dtype=float) / T
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ------------------------------------------------------------------------------------------- base
class Decider:
    """Subclasses implement `_letter_logprobs(state_text, images, prompts)`.

    prompts: list of (user prompt, n_options). Returns a list of (n_options,) arrays of log-probabilities of
    the option letters A.. as the next token.
    """

    def __init__(self, mode="letters"):
        assert mode in ("letters", "isolated")
        self.mode = mode
        self.temperatures = {}

    def _letter_logprobs(self, state_text, images, prompts):
        raise NotImplementedError

    def option_logits(self, state, questions):
        """Unnormalised log-scores over each question's options, list of (K,) arrays."""
        text, images = split_state(state)
        n_img = len(images)
        if self.mode == "letters":
            return self._letter_logprobs(text, images, [letter_prompt(text, q, n_img) for q in questions])
        prompts, owners = [], []
        for qi, q in enumerate(questions):
            if isinstance(q, Noul):
                prompts.append(letter_prompt(text, q, n_img))
                owners.append((qi, None))
            else:
                for o in options_of(q):
                    prompts.append(candidate_prompt(text, q, o, n_img))
                    owners.append((qi, o))
        lp = self._letter_logprobs(text, images, prompts)
        out = [[] for _ in questions]
        for (qi, o), l in zip(owners, lp):
            if o is None:
                out[qi] = l
            else:
                out[qi].append(l[1] - np.logaddexp(l[0], l[1]))  # log P(yes | candidate)
        return [np.asarray(o, dtype=float) for o in out]

    def decide(self, state, questions):
        """All questions about one state -> {key: typed answer}."""
        logits = self.option_logits(state, questions)
        return {q.key: format_answer(q, softmax(z, self.temperatures.get(q.key, 1.0)))
                for q, z in zip(questions, logits)}

    def fit_temperature(self, states, q, y):
        """Fit the temperature of one question on labelled states (y = class indices)."""
        z = np.stack([self.option_logits(s, [q])[0] for s in states])
        self.temperatures[q.key] = cal.fit_temperature(lambda T: softmax(z, T), np.asarray(y))
        return self.temperatures[q.key]


def _device(device):
    import torch

    return device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def _letter_ids(tok):
    # each letter can be tokenised with or without a leading space; pool both
    return [sorted({tok.encode(v, add_special_tokens=False)[0] for v in (L, " " + L)}) for L in LETTERS]


def _gather_letters(logp, prompts, letter_ids):
    import torch

    return [np.array([torch.logsumexp(logp[i, letter_ids[j]], 0).item() for j in range(k)])
            for i, (_, k) in enumerate(prompts)]


# ------------------------------------------------------------------------------------------- local text LLM
class LLMDecider(Decider):
    def __init__(self, model_id="Qwen/Qwen3-1.7B", device=None, dtype="auto", model=None, tokenizer=None,
                 mode="letters", batch_size=16):
        """Pass a Hugging Face model id, or an already-loaded (model, tokenizer) pair."""
        super().__init__(mode)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = _device(device)
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = model or AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self.model = model.to(self.device).eval()
        self.letter_ids = _letter_ids(self.tok)
        self.batch_size = batch_size

    def _chat(self, user):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        if self.tok.chat_template:
            # enable_thinking=False switches off reasoning traces for hybrid models (Qwen3); ignored elsewhere
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        return f"{SYSTEM}\n\n{user}\n\nAnswer:"

    def _letter_logprobs(self, state_text, images, prompts):
        import torch

        out = []
        for b in range(0, len(prompts), self.batch_size):
            chunk = prompts[b:b + self.batch_size]
            batch = self.tok([self._chat(p) for p, _ in chunk], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                logp = self.model(**batch).logits[:, -1].float().log_softmax(-1).cpu()
            out += _gather_letters(logp, chunk, self.letter_ids)
        return out


# ------------------------------------------------------------------------------------------- local VLM
def to_uint8_rgb(img):
    """Microscopy-friendly conversion: percentile-normalise any 2D/3D array to uint8 RGB."""
    img = np.asarray(img)
    if img.dtype == np.uint8 and img.ndim == 3 and img.shape[-1] == 3:
        return img
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, -1)
    img = img[..., :3]
    lo, hi = np.percentile(img, [1, 99.8])
    return (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


class VLMDecider(Decider):
    """Hugging Face image-text-to-text model, e.g. "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen3-VL-8B-Instruct",
    "OpenGVLab/InternVL3-8B-hf", "google/gemma-3-4b-it". Images in the state are shown before the question."""

    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", device=None, dtype="auto", model=None,
                 processor=None, mode="letters", batch_size=8):
        super().__init__(mode)
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = _device(device)
        self.proc = processor or AutoProcessor.from_pretrained(model_id)
        self.tok = self.proc.tokenizer
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = model or AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
        self.model = model.to(self.device).eval()
        self.letter_ids = _letter_ids(self.tok)
        self.batch_size = batch_size

    def _chat(self, user, n_images):
        content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": user}]
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": content}]
        return self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _letter_logprobs(self, state_text, images, prompts):
        import torch

        imgs = [to_uint8_rgb(im) for im in images]
        out = []
        for b in range(0, len(prompts), self.batch_size):
            chunk = prompts[b:b + self.batch_size]
            texts = [self._chat(p, len(imgs)) for p, _ in chunk]
            kw = {"images": [im for _ in chunk for im in imgs]} if imgs else {}
            batch = self.proc(text=texts, padding=True, return_tensors="pt", **kw).to(self.device)
            with torch.no_grad():
                logp = self.model(**batch).logits[:, -1].float().log_softmax(-1).cpu()
            out += _gather_letters(logp, chunk, self.letter_ids)
        return out


# ------------------------------------------------------------------------------------------- remote server
class OpenAICompatDecider(Decider):
    """Any OpenAI-compatible chat endpoint that returns `top_logprobs` (vLLM, SGLang, llama.cpp server, or a
    hosted provider that exposes logprobs for the model you use). One request per prompt, max_tokens=1.

        vllm serve Qwen/Qwen2.5-VL-7B-Instruct --enable-prefix-caching
        OpenAICompatDecider("http://localhost:8000/v1", "Qwen/Qwen2.5-VL-7B-Instruct")
    """

    def __init__(self, base_url, model, api_key=None, mode="letters", top_logprobs=20, timeout=60):
        super().__init__(mode)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.top_logprobs, self.timeout = model, api_key, top_logprobs, timeout

    @staticmethod
    def _data_url(img):
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(to_uint8_rgb(img)).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _post(self, payload):
        import urllib.request

        req = urllib.request.Request(self.url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def _letter_logprobs(self, state_text, images, prompts):
        image_parts = [{"type": "image_url", "image_url": {"url": self._data_url(im)}} for im in images]
        out = []
        for user, k in prompts:
            content = image_parts + [{"type": "text", "text": user}] if image_parts else user
            resp = self._post({
                "model": self.model, "max_tokens": 1, "temperature": 0, "logprobs": True,
                "top_logprobs": self.top_logprobs,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            })
            top = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            seen = {}
            for t in top:
                L = t["token"].strip()
                if L in LETTERS[:k]:
                    seen[L] = np.logaddexp(seen.get(L, -np.inf), t["logprob"])
            # letters outside the returned top-k get a floor below the smallest observed logprob
            floor = min(t["logprob"] for t in top) - np.log(10)
            out.append(np.array([seen.get(L, floor) for L in LETTERS[:k]]))
        return out


# ------------------------------------------------------------------------------------------- rules
class RuleDecider(Decider):
    """rules: {question key: fn(state, question) -> probabilities over the question's options}.

    For guardrails ("objects under 20 px are never counted"), hand-written baselines and tests.
    """

    def __init__(self, rules):
        super().__init__("letters")
        self.rules = rules

    def option_logits(self, state, questions):
        return [np.log(np.clip(np.asarray(self.rules[q.key](state, q), dtype=float), 1e-9, None))
                for q in questions]
