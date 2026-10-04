"""TextJevLite: a small trainable open-vocabulary decision model.

state text  -> encoder tokens ─┐
question    -> encoder, pooled ─┼─> QuestionConditionedHead -> logits over the given options (one pass)
options     -> encoder, pooled ─┘

Because options are inputs, it can answer questions/options it was not trained on. Train it on labelled
(state, question, answer) triples, or distil soft labels from LLMDecider / a frontier API
(target = probability vector), then fit one global temperature on held-out data.

encoder="hash" is a tiny trainable bag-of-words encoder (no download, used in tests);
otherwise any Hugging Face encoder id, e.g. "answerdotai/ModernBERT-base" or "Qwen/Qwen3-Embedding-0.6B"
(kept frozen; a linear projection is trained on top).
"""
import re
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..heads import QuestionConditionedHead
from ..schema import Choice, Noul, encode_label, format_answer
from .. import calibration as cal


class HashEncoder(nn.Module):
    def __init__(self, d, n_buckets=2**15, max_len=128):
        super().__init__()
        self.emb = nn.Embedding(n_buckets, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        self.n_buckets, self.max_len = n_buckets, max_len

    def forward(self, texts):
        ids = [[1 + zlib.crc32(w.encode()) % (self.n_buckets - 1) for w in re.findall(r"\w+", t.lower())][: self.max_len] or [0]
               for t in texts]
        T = max(len(i) for i in ids)
        dev = self.emb.weight.device
        x = torch.zeros(len(ids), T, dtype=torch.long, device=dev)
        for r, i in enumerate(ids):
            x[r, : len(i)] = torch.tensor(i)
        mask = x == 0
        h = self.emb(x) + self.pos(torch.arange(T, device=dev))[None]
        return h, mask


class HFEncoder(nn.Module):
    def __init__(self, model_id, d, max_len=512):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.proj = nn.Linear(self.model.config.hidden_size, d)
        self.max_len = max_len

    def forward(self, texts):
        dev = self.proj.weight.device
        b = self.tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=self.max_len).to(dev)
        with torch.no_grad():
            h = self.model(**b).last_hidden_state
        return self.proj(h.float()), b["attention_mask"] == 0


def _pool(h, mask):
    w = (~mask).float()[..., None]
    return (h * w).sum(1) / w.sum(1).clamp_min(1)


def _labels(q):
    # Natural-language option text for the encoder
    return ["no", "yes"] if isinstance(q, Noul) else [str(o) for o in q.labels]


class TextJevLite(nn.Module):
    def __init__(self, encoder="hash", d=256, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = HashEncoder(d) if encoder == "hash" else HFEncoder(encoder, d)
        self.head = QuestionConditionedHead(d, n_heads, n_layers, dropout)
        self.temperature = 1.0

    def forward(self, states, questions):
        """states: list[str]; questions: list of typed questions (one per state). Returns (B, Kmax) logits."""
        s_tok, s_mask = self.encoder(states)
        q_tok, q_mask = self.encoder([q.question for q in questions])
        q_emb = _pool(q_tok, q_mask)
        opts = [_labels(q) for q in questions]
        K = max(len(o) for o in opts)
        flat = [o for os in opts for o in os]
        o_tok, o_mask = self.encoder(flat)
        o_flat = _pool(o_tok, o_mask)
        o_emb = torch.zeros(len(states), K, o_flat.shape[-1], device=o_flat.device)
        o_pad = torch.ones(len(states), K, dtype=torch.bool, device=o_flat.device)
        i = 0
        for b, os in enumerate(opts):
            o_emb[b, : len(os)] = o_flat[i: i + len(os)]
            o_pad[b, : len(os)] = False
            i += len(os)
        return self.head(s_tok, q_emb, o_emb, s_mask, o_pad)

    @staticmethod
    def _targets(examples, K, device):
        """Hard labels (index / raw label) or soft labels (probability vectors) -> (B, K) target distribution."""
        t = torch.zeros(len(examples), K, device=device)
        for b, (_, q, y) in enumerate(examples):
            if isinstance(y, (list, tuple, np.ndarray)):
                t[b, : len(y)] = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
            else:
                t[b, encode_label(q, y)] = 1.0
        return t

    @staticmethod
    def _shuffle_options(ex, rng):
        """Permute Choice options (remapping the target) so the model matches option meaning, not position."""
        state, q, y = ex
        if not isinstance(q, Choice):
            return ex
        perm = rng.permutation(len(q.options))
        q2 = Choice(q.key, q.question, [q.options[i] for i in perm])
        if isinstance(y, (list, tuple, np.ndarray)):
            return state, q2, np.asarray(y)[perm]
        return state, q2, int(np.flatnonzero(perm == encode_label(q, y))[0])

    def fit(self, examples, epochs=20, lr=2e-3, batch_size=32, shuffle_options=True, seed=0, verbose=False):
        """examples: list of (state, question, target); target = label / class index / probability vector."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        dev = next(self.parameters()).device
        for ep in range(epochs):
            self.train()
            perm = rng.permutation(len(examples))
            total = 0.0
            for b in range(0, len(perm), batch_size):
                ex = [examples[i] for i in perm[b:b + batch_size]]
                if shuffle_options:
                    ex = [self._shuffle_options(e, rng) for e in ex]
                logits = self([e[0] for e in ex], [e[1] for e in ex])
                t = self._targets(ex, logits.shape[1], dev)
                logp = logits.log_softmax(-1).masked_fill(t == 0, 0.0)  # avoid -inf * 0 on padded options
                loss = -(t * logp).sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(ex)
            if verbose:
                print(f"epoch {ep}: loss {total / len(examples):.4f}")
        return self

    def option_probs(self, states, questions, temperature=None):
        self.eval()
        T = self.temperature if temperature is None else temperature
        with torch.no_grad():
            logits = self(states, questions)
        p = F.softmax(logits / T, -1).cpu().numpy()
        return [p[i, : len(_labels(q))] for i, q in enumerate(questions)]

    def fit_temperature(self, examples):
        """One global temperature on held-out examples with hard labels."""
        self.eval()
        with torch.no_grad():
            logits = self([e[0] for e in examples], [e[1] for e in examples]).cpu().numpy()
        y = np.array([encode_label(q, t) for _, q, t in examples])

        def probs(T):
            z = logits / T
            z = z - np.nanmax(np.where(np.isfinite(z), z, np.nan), 1, keepdims=True)
            e = np.where(np.isfinite(z), np.exp(z), 0.0)
            return e / e.sum(1, keepdims=True)

        self.temperature = cal.fit_temperature(probs, y)
        return self.temperature

    def decide(self, state, questions):
        """All questions about one state in one batched forward pass."""
        P = self.option_probs([state] * len(questions), questions)
        return {q.key: format_answer(q, p) for q, p in zip(questions, P)}
