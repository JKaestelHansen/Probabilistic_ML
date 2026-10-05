"""Benchmark text deciders on accuracy AND calibration.

    python scripts/benchmark_text.py                         # TextJevLite (hash encoder) on synthetic tickets
    python scripts/benchmark_text.py --llm Qwen/Qwen3-1.7B   # + zero-shot LLMDecider
    python scripts/benchmark_text.py --data my.jsonl --encoder answerdotai/ModernBERT-base --llm Qwen/Qwen3-4B

my.jsonl lines: {"state": "...", "question": {"type": "choice", "key": ..., "question": ..., "options": [...]},
                 "label": "billing"}

To benchmark a hosted decision API (TypeSafe Jev, OpenAI Decisions), wrap its client in an object with
`.decide(state, questions) -> {key: {"probs": {...}}}` and add it to `deciders` below.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from openjev import Score, question_from_dict  # noqa: E402
from openjev import calibration as cal  # noqa: E402
from openjev.schema import encode_label  # noqa: E402
from openjev.text import LLMDecider, TextJevLite  # noqa: E402
from openjev.text.benchmark_data import make_ticket_benchmark  # noqa: E402


def evaluate(decider, examples):
    by_key = defaultdict(lambda: ([], [], None))
    t0 = time.time()
    for state, q, label in examples:
        ans = decider.decide(state, [q])[q.key]
        probs, ys, _ = by_key[q.key]
        probs.append(list(ans["probs"].values()))
        ys.append(encode_label(q, label))
        by_key[q.key] = (probs, ys, q)
    out = {k: cal.summarize(np.array(p), np.array(y), ordinal=isinstance(q, Score)) for k, (p, y, q) in by_key.items()}
    out["ms_per_decision"] = 1000 * (time.time() - t0) / len(examples)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--encoder", default="hash")
    ap.add_argument("--llm")
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()

    if a.data:
        rows = [json.loads(l) for l in open(a.data)]
        data = [(r["state"], question_from_dict(r["question"]), r["label"]) for r in rows]
    else:
        data = make_ticket_benchmark(400)
    rng = np.random.default_rng(0)
    data = [data[i] for i in rng.permutation(len(data))]
    n = len(data)
    train, calib, test = data[: int(0.6 * n)], data[int(0.6 * n): int(0.75 * n)], data[int(0.75 * n):]

    deciders = {}
    lite = TextJevLite(a.encoder).fit(train, epochs=a.epochs)
    lite.fit_temperature(calib)
    deciders["text_jev_lite"] = lite
    if a.llm:
        llm = LLMDecider(a.llm)
        for q in {q.key: q for _, q, _ in calib}.values():
            ex = [(s, t) for s, qq, t in calib if qq.key == q.key]
            llm.fit_temperature([s for s, _ in ex], q, [encode_label(q, t) for _, t in ex])
        deciders[f"llm:{a.llm}"] = llm

    print(json.dumps({name: evaluate(d, test) for name, d in deciders.items()}, indent=2))


if __name__ == "__main__":
    main()
