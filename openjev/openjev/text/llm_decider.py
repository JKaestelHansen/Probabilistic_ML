"""LLMDecider: typed decisions from an open causal LLM in a single parallel forward pass.

No generation: every question for a state is put in one left-padded batch, and the probability of each
option is read from the next-token distribution over option letters (A, B, C, ...). Score = expected
level, Noul = P(Yes). Then one temperature per question key fixes calibration.

This is the closest open approximation of the decision-API interface and the text benchmark baseline.
For throughput in production, serve the same prompts with vLLM / SGLang (prefix caching shares the state
across questions) and request top logprobs for one token.
"""
import string

import numpy as np

from ..schema import Choice, Noul, Score, format_answer
from .. import calibration as cal

SYSTEM = ("You are a decision model. Read the state and answer the question by replying with the letter of "
          "exactly one option. Reply with the letter only.")
LETTERS = string.ascii_uppercase


def build_prompt(state, q):
    if isinstance(q, Noul):
        options = ["No", "Yes"]
    else:
        options = q.labels
    if len(options) > len(LETTERS):
        raise ValueError(f"{q.key}: LLMDecider supports at most {len(LETTERS)} options")
    lines = [f"State:\n{state}", "", f"Question: {q.question}"]
    if isinstance(q, Score):
        lines.append("Options (ordered from lowest to highest):")
    else:
        lines.append("Options:")
    lines += [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    lines += ["", "Answer with a single letter."]
    return "\n".join(lines), len(options)


class LLMDecider:
    def __init__(self, model_id="Qwen/Qwen3-1.7B", device=None, dtype="auto", model=None, tokenizer=None):
        """Pass a Hugging Face model id, or an already-loaded (model, tokenizer) pair."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = model or AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self.model = model.to(self.device).eval()
        # each letter can be tokenised with or without a leading space; pool both
        self.letter_ids = [sorted({self.tok.encode(v, add_special_tokens=False)[0] for v in (L, " " + L)})
                           for L in LETTERS]
        self.temperatures = {}

    def _chat(self, user):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        if self.tok.chat_template:
            # enable_thinking=False switches off reasoning traces for hybrid models (Qwen3); ignored elsewhere
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        return f"{SYSTEM}\n\n{user}\n\nAnswer:"

    def option_logits(self, state, questions):
        """One forward pass for all questions about `state`. Returns a list of (K,) log-score arrays."""
        import torch

        prompts, ks = zip(*[build_prompt(state, q) for q in questions])
        batch = self.tok([self._chat(p) for p in prompts], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            logp = self.model(**batch).logits[:, -1].float().log_softmax(-1).cpu()
        out = []
        for i, k in enumerate(ks):
            out.append(np.array([torch.logsumexp(logp[i, self.letter_ids[j]], 0).item() for j in range(k)]))
        return out

    @staticmethod
    def _softmax(z, T=1.0):
        z = np.asarray(z) / T
        e = np.exp(z - z.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    def decide(self, state, questions):
        logits = self.option_logits(state, questions)
        return {q.key: format_answer(q, self._softmax(z, self.temperatures.get(q.key, 1.0)))
                for q, z in zip(questions, logits)}

    def fit_temperature(self, states, q, y):
        """Fit the temperature of one question on labelled states (y = class indices)."""
        z = np.stack([self.option_logits(s, [q])[0] for s in states])
        self.temperatures[q.key] = cal.fit_temperature(lambda T: self._softmax(z, T), np.asarray(y))
        return self.temperatures[q.key]
