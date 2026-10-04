import numpy as np
import pytest

from openjev import Choice, Noul, Score
from openjev import calibration as cal
from openjev.schema import encode_label
from openjev.text import TextJevLite
from openjev.text.benchmark_data import QUEUE, make_ticket_benchmark


def _eval(model, examples):
    probs = model.option_probs([e[0] for e in examples], [e[1] for e in examples])
    y = np.array([encode_label(q, t) for _, q, t in examples])
    return np.mean([p.argmax() == t for p, t in zip(probs, y)])


@pytest.fixture(scope="module")
def lite():
    data = make_ticket_benchmark(300, seed=0)
    model = TextJevLite("hash", d=64, n_heads=4, n_layers=1).fit(data[:600], epochs=15)
    model.fit_temperature(data[600:750])
    return model, data[750:]


def test_jev_lite_learns(lite):
    model, test = lite
    for key, floor in [("queue", 0.9), ("urgent", 0.9), ("sentiment", 0.8)]:
        assert _eval(model, [e for e in test if e[1].key == key]) > floor, key


def test_jev_lite_uses_option_meaning_not_position(lite):
    model, test = lite
    rev = Choice("queue", QUEUE.question, QUEUE.options[::-1])
    swapped = [(s, rev, t) for s, q, t in test if q.key == "queue"]
    assert _eval(model, swapped) > 0.9


def test_jev_lite_decide_returns_typed_answers(lite):
    model, _ = lite
    out = model.decide("I was charged twice for order 1234.", [
        QUEUE, Noul("urgent", "Does this need a response within the hour?"),
        Score("sentiment", "How satisfied is the customer?", ["very unhappy", "unhappy", "neutral", "happy"])])
    assert out["queue"]["value"] == "billing"
    assert out["urgent"]["type"] == "noul" and 0 <= out["urgent"]["value"] <= 1
    assert 0 <= out["sentiment"]["value"] <= 3


def test_jev_lite_soft_label_distillation():
    data = make_ticket_benchmark(60, seed=1)
    soft = [(s, q, np.eye(len(q.labels))[encode_label(q, t)] * 0.9 + 0.1 / len(q.labels)) for s, q, t in data]
    TextJevLite("hash", d=32, n_heads=4, n_layers=1).fit(soft, epochs=2)
