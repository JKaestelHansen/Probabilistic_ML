import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from openjev import Choice, Noul, Score, LLMDecider, VLMDecider, OpenAICompatDecider, RuleDecider
from openjev.deciders import split_state

from tiny_models import tiny_llm, tiny_vlm

QS = [Choice("category", "Which category?", ["round", "elongated", "unknown"]),
      Noul("contamination", "Is this contamination?"),
      Score("severity", "How severe?", ["none", "low", "high"])]


def _check(out):
    assert set(out) == {"category", "contamination", "severity"}
    for a in out.values():
        assert abs(sum(a["probs"].values()) - 1) < 1e-6
    assert set(out["contamination"]["probs"]) == {"no", "yes"}


def test_split_state():
    assert split_state("hi") == ("hi", [])
    assert split_state({"text": "hi", "images": [1]}) == ("hi", [1])
    text, imgs = split_state({"area": np.float32(3.0), "images": [1, 2]})
    assert json.loads(text) == {"area": 3.0} and imgs == [1, 2]


@pytest.mark.parametrize("mode", ["letters", "isolated"])
def test_llm_decider(mode):
    model, tok = tiny_llm()
    _check(LLMDecider(model=model, tokenizer=tok, device="cpu", mode=mode).decide({"area": 120}, QS))


@pytest.mark.parametrize("mode", ["letters", "isolated"])
def test_vlm_decider_with_images(mode):
    model, proc = tiny_vlm()
    dec = VLMDecider(model=model, processor=proc, device="cpu", mode=mode, batch_size=2)
    crop = np.random.default_rng(0).poisson(100, (40, 52)).astype(float)
    _check(dec.decide({"text": "A microscopy object.", "images": [crop]}, QS))
    _check(dec.decide("text only state", QS))


def test_isolated_mode_is_order_invariant():
    model, tok = tiny_llm()
    dec = LLMDecider(model=model, tokenizer=tok, device="cpu", mode="isolated")
    q1 = Choice("c", "Which?", ["round", "elongated", "unknown"])
    q2 = Choice("c", "Which?", ["unknown", "round", "elongated"])
    p1, p2 = dec.decide("state", [q1])["c"]["probs"], dec.decide("state", [q2])["c"]["probs"]
    for k in p1:
        assert abs(p1[k] - p2[k]) < 1e-5


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.last = body
        top = [{"token": "B", "logprob": -0.2}, {"token": " A", "logprob": -1.9}, {"token": "x", "logprob": -4.0}]
        resp = {"choices": [{"logprobs": {"content": [{"token": "B", "logprob": -0.2, "top_logprobs": top}]}}]}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_openai_compat_decider():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        dec = OpenAICompatDecider(f"http://127.0.0.1:{srv.server_port}/v1", "some-vlm")
        out = dec.decide({"text": "obj", "images": [np.ones((8, 8))]}, QS)
        _check(out)
        assert out["category"]["value"] == "elongated"  # B had the highest logprob
        assert out["category"]["probs"]["unknown"] < out["category"]["probs"]["round"]  # C unseen -> floor
        msg = _Handler.last["messages"][1]["content"]
        assert msg[0]["type"] == "image_url" and msg[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert _Handler.last["max_tokens"] == 1 and _Handler.last["logprobs"] is True
    finally:
        srv.shutdown()


def test_rule_decider_and_temperature():
    dec = RuleDecider({"contamination": lambda s, q: [0.2, 0.8] if s["bright"] else [0.9, 0.1]})
    q = QS[1]
    assert dec.decide({"bright": True}, [q])["contamination"]["decision"] is True
    T = dec.fit_temperature([{"bright": True}, {"bright": False}] * 10, q, [1, 0] * 10)
    assert T < 1  # rules were under-confident on perfectly separable data -> sharpen
