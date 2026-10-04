import numpy as np

from openjev import calibration as cal


def test_perfectly_calibrated_has_low_ece():
    rng = np.random.default_rng(0)
    p1 = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p1).astype(int)
    probs = np.stack([1 - p1, p1], 1)
    assert cal.expected_calibration_error(probs, y) < 0.02


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, (5000, 3))
    p_true = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in p_true])
    T = cal.fit_temperature(lambda T: np.exp(3 * z / T) / np.exp(3 * z / T).sum(1, keepdims=True), y)
    assert 2.4 < T < 3.7  # logits were inflated 3x


def test_conformal_coverage():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1.5, (4000, 5))
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = np.array([rng.choice(5, p=pp) for pp in p])
    q = cal.conformal_threshold(p[:2000], y[:2000], alpha=0.1)
    sets = (1 - p[2000:]) <= q
    assert sets[np.arange(2000), y[2000:]].mean() >= 0.88


def test_uncertainty_decomposition():
    agree = np.tile([[0.5, 0.5]], (4, 1, 1))
    disagree = np.array([[[0.99, 0.01]], [[0.01, 0.99]]])
    _, alea, epi = cal.decompose_uncertainty(agree)
    assert epi[0] < 1e-9 and alea[0] > 0.69
    _, alea, epi = cal.decompose_uncertainty(disagree)
    assert epi[0] > 0.6
