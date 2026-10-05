"""Calibration metrics and post-hoc calibration for categorical decisions (numpy only).

The regression-style metrics in ../uncertainty_quantification assume Gaussian predictive distributions;
decisions are categorical, so these are the classification counterparts.
"""
import numpy as np


def _onehot(y, k):
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1
    return out


def nll(probs, y, eps=1e-12):
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], eps, None))))


def brier(probs, y):
    return float(np.mean(np.sum((probs - _onehot(y, probs.shape[1])) ** 2, axis=1)))


def reliability_bins(probs, y, n_bins=10):
    """Top-label reliability diagram: per-bin (mean confidence, accuracy, count)."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    mean_conf, acc, count = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        count[b] = m.sum()
        if count[b]:
            mean_conf[b], acc[b] = conf[m].mean(), correct[m].mean()
    return mean_conf, acc, count


def expected_calibration_error(probs, y, n_bins=10):
    mean_conf, acc, count = reliability_bins(probs, y, n_bins)
    return float(np.sum(count / count.sum() * np.abs(acc - mean_conf)))


def max_calibration_error(probs, y, n_bins=10):
    mean_conf, acc, count = reliability_bins(probs, y, n_bins)
    return float(np.max(np.abs(acc - mean_conf)[count > 0]))


def entropy(probs, eps=1e-12):
    return -np.sum(probs * np.log(probs + eps), axis=-1)


def decompose_uncertainty(member_probs):
    """member_probs: (M, N, K) from an ensemble / MC dropout.

    total = H[E p], aleatoric = E H[p], epistemic = total - aleatoric (mutual information).
    """
    mean = member_probs.mean(0)
    total = entropy(mean)
    aleatoric = entropy(member_probs).mean(0)
    return total, aleatoric, np.maximum(total - aleatoric, 0.0)


def fit_temperature(probs_fn, y, grid=np.exp(np.linspace(-2.5, 2.5, 101))):
    """Grid search for the temperature minimising NLL. probs_fn(T) -> (N, K) probabilities."""
    losses = [nll(probs_fn(T), y) for T in grid]
    return float(grid[int(np.argmin(losses))])


def conformal_threshold(probs, y, alpha=0.1):
    """Split conformal (LAC score s = 1 - p_true). Sets {k: 1 - p_k <= q} cover the truth w.p. >= 1 - alpha."""
    n = len(y)
    scores = 1 - probs[np.arange(n), y]
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def summarize(probs, y, ordinal=False, n_bins=10):
    """Standard metric bundle for one question. Labels < 0 are ignored."""
    m = y >= 0
    probs, y = probs[m], y[m]
    out = {
        "n": int(len(y)),
        "accuracy": float(np.mean(probs.argmax(1) == y)),
        "nll": nll(probs, y),
        "brier": brier(probs, y),
        "ece": expected_calibration_error(probs, y, n_bins),
    }
    if ordinal:
        out["mae_levels"] = float(np.mean(np.abs(probs @ np.arange(probs.shape[1]) - y)))
    if probs.shape[1] == 2:
        out["auroc"] = auroc(probs[:, 1], y)
    return out


def auroc(score, y):
    """Mann-Whitney AUROC, ties counted half."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0) + 0.5 * (diff == 0)))
