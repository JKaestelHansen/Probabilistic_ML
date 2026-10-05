import numpy as np
import pytest

from openjev.data import QUESTIONS, make_blob_dataset
from openjev.vision import VisionJev


@pytest.fixture(scope="module")
def trained():
    imgs, labels = make_blob_dataset(360, size=96, seed=0)
    jev = VisionJev(QUESTIONS, backbone="handcrafted", n_members=2)
    X = jev.embed(imgs)
    sub = lambda s: {k: v[s] for k, v in labels.items()}
    jev.fit(X[:280], sub(slice(0, 280)), epochs=60)
    return jev, X[280:], sub(slice(280, None)), imgs[280:]


def test_learns_all_primitives(trained):
    jev, X, labels, _ = trained
    res = jev.evaluate(X, labels)
    assert res["usable"]["accuracy"] > 0.85
    assert res["morphology"]["accuracy"] > 0.5  # chance 0.33
    assert res["focus"]["mae_levels"] < 0.8
    for r in res.values():
        assert r["ece"] < 0.2


def test_typed_answers(trained):
    jev, X, _, imgs = trained
    ans = jev.predict(images=imgs[:2])
    assert len(ans) == 2
    a = ans[0]
    assert a["morphology"]["value"] in QUESTIONS[0].options
    assert abs(sum(a["morphology"]["probs"].values()) - 1) < 1e-5
    assert 0 <= a["focus"]["value"] <= 3
    assert 0 <= a["usable"]["value"] <= 1 and isinstance(a["usable"]["decision"], bool)
    assert "set" in a["morphology"] and a["morphology"]["epistemic"] >= 0


def test_missing_labels_are_masked():
    imgs, labels = make_blob_dataset(80, size=64, seed=3)
    labels["morphology"] = [None if i % 2 else v for i, v in enumerate(labels["morphology"])]
    jev = VisionJev(QUESTIONS, n_members=1)
    jev.fit(jev.embed(imgs), labels, epochs=5)


def test_save_load_roundtrip(trained, tmp_path):
    jev, X, _, _ = trained
    jev.save(tmp_path / "m.pt")
    jev2 = VisionJev.load(tmp_path / "m.pt")
    a, b = jev.predict_proba(X[:5]), jev2.predict_proba(X[:5])
    for k in a:
        np.testing.assert_allclose(a[k][0], b[k][0], atol=1e-6)
