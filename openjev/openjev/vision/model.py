"""VisionJev: frozen image backbone + deep ensemble of typed decision heads, temperature-scaled per question,
with split-conformal prediction sets for Choice questions and an epistemic (mutual information) estimate.

    jev = VisionJev([Noul("usable", "Is this image usable?"),
                     Choice("morphology", "Dominant object shape?", ["round", "elongated", "irregular"])],
                    backbone="dinov2")
    X = jev.embed(images)                 # cache once, the backbone is frozen
    jev.fit(X, {"usable": y_usable, "morphology": y_morph})
    jev.predict(features=X_new)           # -> list of {key: typed answer}
"""
import numpy as np
import torch

from ..schema import encode_label, format_answer, question_to_dict, question_from_dict, Choice, Score
from ..heads import TypedHeads, logits_to_probs, typed_loss
from .. import calibration as cal
from .backbones import Backbone


class VisionJev:
    def __init__(self, questions, backbone="handcrafted", hidden=256, dropout=0.2, n_members=5, device="cpu"):
        self.questions = list(questions)
        self.backbone_name = backbone
        self.hidden, self.dropout, self.n_members = hidden, dropout, n_members
        self.device = device
        self._backbone = None
        self.members = []
        self.temperatures = {q.key: 1.0 for q in self.questions}
        self.conformal_q = {}

    # ---------------------------------------------------------------- features
    @property
    def backbone(self):
        if self._backbone is None:
            self._backbone = Backbone(self.backbone_name)
        return self._backbone

    def embed(self, images, batch_size=16):
        return self.backbone(images, batch_size=batch_size).astype(np.float32)

    def _encode_labels(self, labels, n):
        Y = {}
        for q in self.questions:
            raw = labels.get(q.key, [None] * n)
            Y[q.key] = np.array([encode_label(q, v) for v in raw], dtype=np.int64)
        return Y

    def _member_logits(self, X):
        Xt = torch.as_tensor((X - self.mu) / self.sd, dtype=torch.float32, device=self.device)
        outs = []
        with torch.no_grad():
            for m in self.members:
                m.eval()
                outs.append({k: v for k, v in m(Xt).items()})
        return outs

    def _probs(self, member_logits, q, T):
        """Ensemble-averaged probabilities (N, L) and per-member probabilities (M, N, L)."""
        per = torch.stack([logits_to_probs(q, lg[q.key], T) for lg in member_logits]).cpu().numpy()
        return per.mean(0), per

    # ---------------------------------------------------------------- training
    def fit(self, X, labels, val_frac=0.25, epochs=300, lr=1e-3, weight_decay=1e-4, batch_size=64,
            alpha=0.1, seed=0, verbose=False):
        """X: (N, D) embeddings from .embed(). labels: {question key: list of raw labels (None = missing)}.

        The held-out split is halved: one half fits temperatures, the other the conformal thresholds,
        so the coverage guarantee is not spoiled by reusing calibration data.
        """
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.float32)
        Y = self._encode_labels(labels, len(X))
        idx = rng.permutation(len(X))
        n_val = int(len(X) * val_frac)
        val, tr = idx[:n_val], idx[n_val:]
        val_t, val_c = val[: n_val // 2], val[n_val // 2:]

        self.mu, self.sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr = torch.as_tensor((X[tr] - self.mu) / self.sd, device=self.device)
        Ytr = {k: torch.as_tensor(v[tr], device=self.device) for k, v in Y.items()}

        self.members = []
        for m_i in range(self.n_members):
            torch.manual_seed(seed + m_i)
            model = TypedHeads(X.shape[1], self.questions, self.hidden, self.dropout).to(self.device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
            # bootstrap resample per member -> more diverse ensemble
            boot = torch.as_tensor(rng.integers(0, len(tr), len(tr)), device=self.device)
            for ep in range(epochs):
                model.train()
                perm = boot[torch.randperm(len(boot), device=self.device)]
                for b in range(0, len(perm), batch_size):
                    bi = perm[b:b + batch_size]
                    out = model(Xtr[bi])
                    loss = sum(typed_loss(q, out[q.key], Ytr[q.key][bi]) for q in self.questions)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                sched.step()
            if verbose:
                print(f"member {m_i}: final train loss {loss.item():.4f}")
            self.members.append(model)

        if len(val_t):
            lg = self._member_logits(X[val_t])
            for q in self.questions:
                y = Y[q.key][val_t]
                if (y >= 0).sum() > 1:
                    m = y >= 0
                    self.temperatures[q.key] = cal.fit_temperature(
                        lambda T: self._probs(lg, q, T)[0][m], y[m])
        if len(val_c):
            lg = self._member_logits(X[val_c])
            for q in self.questions:
                y = Y[q.key][val_c]
                if isinstance(q, Choice) and (y >= 0).sum() > 1:
                    m = y >= 0
                    p = self._probs(lg, q, self.temperatures[q.key])[0][m]
                    self.conformal_q[q.key] = cal.conformal_threshold(p, y[m], alpha)
        return self

    # ---------------------------------------------------------------- inference
    def predict_proba(self, X):
        """{key: (mean probs (N, L), epistemic MI (N,))} for all questions."""
        lg = self._member_logits(np.asarray(X, dtype=np.float32))
        out = {}
        for q in self.questions:
            mean, per = self._probs(lg, q, self.temperatures[q.key])
            out[q.key] = (mean, cal.decompose_uncertainty(per)[2])
        return out

    def predict(self, images=None, features=None):
        """Typed answers, one dict per image: {question key: answer}."""
        X = features if features is not None else self.embed(images)
        P = self.predict_proba(X)
        return [
            {q.key: format_answer(q, P[q.key][0][i], self.conformal_q.get(q.key), P[q.key][1][i])
             for q in self.questions}
            for i in range(len(X))
        ]

    def evaluate(self, X, labels):
        Y = self._encode_labels(labels, len(X))
        P = self.predict_proba(X)
        res = {}
        for q in self.questions:
            res[q.key] = cal.summarize(P[q.key][0], Y[q.key], ordinal=isinstance(q, Score))
            if q.key in self.conformal_q:
                m = Y[q.key] >= 0
                p, y = P[q.key][0][m], Y[q.key][m]
                sets = (1 - p) <= self.conformal_q[q.key]
                res[q.key]["set_coverage"] = float(sets[np.arange(len(y)), y].mean())
                res[q.key]["set_size"] = float(sets.sum(1).mean())
        return res

    # ---------------------------------------------------------------- persistence
    def save(self, path):
        torch.save({
            "questions": [question_to_dict(q) for q in self.questions],
            "backbone": self.backbone_name, "hidden": self.hidden, "dropout": self.dropout,
            "mu": self.mu, "sd": self.sd, "temperatures": self.temperatures, "conformal_q": self.conformal_q,
            "members": [m.state_dict() for m in self.members],
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        s = torch.load(path, map_location=device, weights_only=False)
        qs = [question_from_dict(d) for d in s["questions"]]
        jev = cls(qs, s["backbone"], s["hidden"], s["dropout"], len(s["members"]), device)
        jev.mu, jev.sd = s["mu"], s["sd"]
        jev.temperatures, jev.conformal_q = s["temperatures"], s["conformal_q"]
        for sd in s["members"]:
            m = TypedHeads(len(jev.mu), qs, jev.hidden, jev.dropout).to(device)
            m.load_state_dict(sd)
            jev.members.append(m)
        return jev
