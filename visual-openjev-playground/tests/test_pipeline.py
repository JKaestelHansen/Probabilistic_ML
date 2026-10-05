import numpy as np
import pytest

from visual_openjev import measurements
from visual_openjev.classification import cascade, object_vectors
from visual_openjev.decisions import Taxonomy, decision_questions, object_state
from visual_openjev.discovery import apply_labels, export_review_html, kmeans, neighbors, read_labels, select_for_review
from visual_openjev.evals import evaluate
from visual_openjev.mock import MockVLM, mock_decision_backend
from visual_openjev.objects import ObjectStore
from visual_openjev.pipeline import perceive, run_classifier, run_zero_label
from visual_openjev.synthetic import make_dataset


@pytest.fixture(scope="module")
def run():
    data = make_dataset(24, seed=1)
    store = perceive([d[0] for d in data], ground_truth=[(d[1], d[2]) for d in data])
    tax = Taxonomy.default()
    object_vectors(store.records)
    run_zero_label(store, tax, MockVLM(), mock_decision_backend())
    return store, tax


def test_taxonomy_always_has_unknown_and_grows():
    tax = Taxonomy.default()
    assert tax.names == ["sphere", "cube", "rod", "aggregate", "hexagonal", "unknown"]
    assert [q.key for q in decision_questions(tax)] == ["category", "countable"]
    tax.add("triangular_plate", "flat triangle")
    tax.add("dust", "debris", contamination=True)
    assert tax.names[-1] == "unknown" and tax.contamination == ["dust"]
    qs = decision_questions(tax)
    assert "triangular_plate" in qs[0].question and qs[0].options == tax.names
    assert [q.key for q in qs] == ["category", "is_contamination", "countable"]


def test_taxonomy_file_matches_default():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "taxonomy.json")
    assert Taxonomy.load(path).categories == Taxonomy.default().categories


def test_perception_records(run):
    store, _ = run
    assert len(store) > 50
    r = store.records[0]
    assert r.crop.shape == r.mask.shape and r.mask.any()
    for k in ("area", "circularity", "solidity", "aspect_ratio", "edge_sharpness", "touches_other"):
        assert k in r.features
    assert r.embedding is not None


def test_zero_label_decisions_and_review(run):
    store, tax = run
    r = store.records[0]
    assert set(r.decisions) == {"A_vlm_only", "B_vlm_decision"}
    assert set(r.observations["vlm"]) == {"outline", "edges", "surface", "multiple_objects", "in_focus"}
    assert {"score", "components", "reasons", "needs_review", "category"} <= set(r.review)
    assert "disagreement:A_vlm_only" in r.review["components"]
    s = object_state(r)
    assert "visual_observations" in s and "measurements" in s
    res = evaluate(store.records, tax, ["A_vlm_only", "B_vlm_decision"])
    assert res["B_vlm_decision"]["accuracy"] > 0.4  # stand-in backends; checks plumbing, not real accuracy


def test_unknown_objects_are_flagged(run):
    store, _ = run
    novel = [r for r in store.records if r.meta["gt"] == "triangle"]
    assert novel
    flagged = [r for r in novel if r.review["needs_review"] or r.review["category"] == "unknown"]
    assert len(flagged) / len(novel) > 0.5


def test_discovery_and_label_loop(run, tmp_path):
    store, tax = run
    E = store.embeddings()
    lab, C = kmeans(E, 5)
    assert lab.shape == (len(store),) and C.shape[0] == 5
    assert len(neighbors(E, 0, 4)) == 4
    pick = select_for_review(store.records, 30)
    assert len(pick) == 30 and len({r.object_id for r in pick}) == 30
    html_path, csv_path = export_review_html(pick, str(tmp_path / "rev.html"))
    assert "data:image/png;base64" in open(html_path).read()
    lines = open(csv_path).read().splitlines()
    filled = [lines[0]] + [l + r.meta["gt"] for l, r in zip(lines[1:], pick)]
    open(csv_path, "w").write("\n".join(filled))
    assert len(read_labels(csv_path)) == 30


def test_classifier_cascade_and_measurements(run, tmp_path):
    store, tax = run
    for r in store.records[::2]:
        if r.meta["gt"] in tax.names:
            r.label = r.meta["gt"]
    run_classifier(store, tax, n_members=2, epochs=60)
    r = store.records[1]
    assert "unknown" in r.classifier["probs"] and 0 <= r.classifier["ood"] <= 1
    assert abs(sum(r.classifier["probs"].values()) - 1) < 1e-4
    acc, dfr = cascade(store.records, accept=0.9)
    assert len(acc) + len(dfr) == len(store)
    rows = measurements.summarize(store.records, "C_classifier", pixel_size_um=0.5)
    assert rows and "mean_area_um2" in rows[0]
    assert sum(x["count"] for x in rows) <= len(store)
    rows = measurements.summarize(store.records, "C_classifier", exclude=tax.contamination + ["unknown"])
    assert all(x["category"] != "unknown" for x in rows)
    # persistence keeps everything
    store.save(tmp_path / "s")
    s2 = ObjectStore.load(tmp_path / "s")
    assert len(s2) == len(store)
    a, b = store.records[3], s2.records[3]
    assert a.decisions == b.decisions and a.review == b.review and a.label == b.label
    np.testing.assert_array_equal(a.crop, b.crop)
    np.testing.assert_allclose(a.embedding, b.embedding)


def test_real_vlm_decider_plumbing_on_object_crops(run):
    """Tiny random Qwen2-VL: the real VLMDecider path runs on object crops (answers are random)."""
    from openjev import VLMDecider
    from visual_openjev.decisions import run_vlm_only
    from visual_openjev.vlm import observe
    from tiny_models import tiny_vlm

    store, tax = run
    model, proc = tiny_vlm()
    vlm = VLMDecider(model=model, processor=proc, device="cpu")
    recs = [r.__class__(**{**r.__dict__, "decisions": {}, "observations": {}}) for r in store.records[:2]]
    observe(recs, vlm)
    run_vlm_only(recs, vlm, tax)
    assert recs[0].observations["vlm"]["outline"]["value"] in recs[0].observations["vlm"]["outline"]["probs"]
    assert recs[1].decisions["A_vlm_only"]["category"]["value"] in tax.names
