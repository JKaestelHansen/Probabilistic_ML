"""Decision heads.

TypedHeads              fixed question set, one small head per question on top of frozen embeddings
                        (the workhorse for image QC / morphology).
QuestionConditionedHead open-vocabulary: the question and its options are inputs, so new questions can
                        be asked at inference time (the Jev-style interface). Modality-agnostic: the
                        state is any sequence of token embeddings (text tokens, image patches).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import Choice, Score, Noul


class TypedHeads(nn.Module):
    def __init__(self, d_in, questions, hidden=256, dropout=0.2):
        super().__init__()
        self.questions = questions
        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict()
        self.coral_bias = nn.ParameterDict()
        for q in questions:
            if isinstance(q, Choice):
                self.heads[q.key] = nn.Linear(hidden, len(q.labels))
            elif isinstance(q, Score):
                # CORAL: one shared weight vector, L-1 ordered thresholds -> rank-consistent ordinal output
                self.heads[q.key] = nn.Linear(hidden, 1, bias=False)
                self.coral_bias[q.key] = nn.Parameter(torch.linspace(1, -1, len(q.labels) - 1))
            else:
                self.heads[q.key] = nn.Linear(hidden, 1)

    def forward(self, z):
        h = self.trunk(z)
        out = {}
        for q in self.questions:
            logit = self.heads[q.key](h)
            if isinstance(q, Score):
                logit = logit + self.coral_bias[q.key]
            out[q.key] = logit
        return out


def logits_to_probs(q, logits, temperature=1.0):
    """Raw head output -> probabilities over q.labels, shape (N, L)."""
    logits = logits / temperature
    if isinstance(q, Choice):
        return logits.softmax(-1)
    if isinstance(q, Noul):
        p = torch.sigmoid(logits[:, 0])
        return torch.stack([1 - p, p], -1)
    # Score / CORAL: P(y > k); enforce monotonicity, then difference into level probabilities
    gt = torch.sigmoid(logits)
    gt = torch.cummin(gt, dim=-1).values
    ones = torch.ones_like(gt[:, :1])
    zeros = torch.zeros_like(gt[:, :1])
    cum = torch.cat([ones, gt, zeros], -1)
    return (cum[:, :-1] - cum[:, 1:]).clamp_min(0)


def typed_loss(q, logits, y):
    """Proper-scoring-rule loss for one question. y is a class index tensor; -1 = unlabelled (masked)."""
    mask = y >= 0
    if mask.sum() == 0:
        return logits.sum() * 0.0
    logits, y = logits[mask], y[mask]
    if isinstance(q, Choice):
        return F.cross_entropy(logits, y)
    if isinstance(q, Noul):
        return F.binary_cross_entropy_with_logits(logits[:, 0], y.float())
    k = torch.arange(logits.shape[1], device=y.device)
    target = (y[:, None] > k[None, :]).float()
    return F.binary_cross_entropy_with_logits(logits, target)


class QuestionConditionedHead(nn.Module):
    """Answer arbitrary typed questions about a state.

    state_tokens:  (B, T, d)  embeddings of the state (text tokens or image patches)
    state_mask:    (B, T)     True for padding
    question_emb:  (B, d)     embedding of the question text
    option_emb:    (B, K, d)  embeddings of the K option labels (K may vary per question; pad + option_mask)

    The question attends over the state (cross-attention), the result is fused with the question and scored
    against every option embedding. All options are scored in one parallel pass - no decoding.
    """

    def __init__(self, d, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(d, n_heads, 4 * d, dropout, batch_first=True, norm_first=True)
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d)
        self.opt_proj = nn.Linear(d, d)
        self.log_scale = nn.Parameter(torch.tensor(2.3))  # ~10, learnable logit scale

    def forward(self, state_tokens, question_emb, option_emb, state_mask=None, option_mask=None):
        h = question_emb[:, None, :]
        for layer in self.layers:
            h = layer(h, state_tokens, memory_key_padding_mask=state_mask)
        h = F.normalize(self.norm(h[:, 0]), dim=-1)
        o = F.normalize(self.opt_proj(option_emb), dim=-1)
        logits = self.log_scale.exp() * torch.einsum("bd,bkd->bk", h, o)
        if option_mask is not None:
            logits = logits.masked_fill(option_mask, float("-inf"))
        return logits
