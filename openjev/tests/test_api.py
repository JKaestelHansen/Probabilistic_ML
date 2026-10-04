import base64
import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from openjev.api import create_app
from openjev.data import QUESTIONS, make_blob_dataset
from openjev.text import TextJevLite
from openjev.text.benchmark_data import make_ticket_benchmark
from openjev.vision import VisionJev


@pytest.fixture(scope="module")
def client():
    imgs, labels = make_blob_dataset(80, size=64, seed=0)
    vision = VisionJev(QUESTIONS, n_members=1)
    vision.fit(vision.embed(imgs), labels, epochs=5)
    text = TextJevLite("hash", d=32, n_heads=4, n_layers=1).fit(make_ticket_benchmark(40), epochs=1)
    return TestClient(create_app(vision, text)), imgs[0]


def _b64_npy(a):
    buf = io.BytesIO()
    np.save(buf, a)
    return base64.b64encode(buf.getvalue()).decode()


def test_image_decision(client):
    c, img = client
    r = c.post("/v1/decide", json={"image": _b64_npy(img), "questions": [{"key": "usable"}]})
    assert r.status_code == 200
    assert set(r.json()["answers"]) == {"usable"}


def test_text_decision(client):
    c, _ = client
    r = c.post("/v1/decide", json={"text": "My card was charged twice.", "questions": [
        {"type": "choice", "key": "queue", "question": "Which team?", "options": ["billing", "fraud"]}]})
    assert r.status_code == 200
    assert r.json()["answers"]["queue"]["value"] in ("billing", "fraud")


def test_bad_requests(client):
    c, img = client
    assert c.post("/v1/decide", json={}).status_code == 400
    assert c.post("/v1/decide", json={"image": _b64_npy(img), "questions": [{"key": "nope"}]}).status_code == 400
    assert c.post("/v1/decide", json={"text": "x", "questions": [{"type": "bogus"}]}).status_code == 400
