"""End-to-end run: images -> objects -> embeddings -> (VLM observations, decisions A/B/C) -> review ->
measurements, all recorded in an ObjectStore.

    store = perceive(images, image_ids)                       # segmentation + features + crops
    object_vectors(store.records, "dinov2")                   # embeddings (classification.py)
    run_zero_label(store, tax, vlm=VLMDecider(...), decider=LLMDecider(...))   # architectures A and B
    ... human labels a few hundred objects (discovery.export_review_html / apply_labels) ...
    run_classifier(store, tax)                                # architecture C + cascade
"""
from .classification import ObjectClassifier, classifier_decisions, object_vectors
from .decisions import review, run_decision_layer, run_vlm_only
from .objects import ObjectStore
from .perception import extract_objects, match_ground_truth
from .vlm import observe


def perceive(images, image_ids=None, ground_truth=None, **seg_kw):
    """ground_truth: optional list of (instance map, {id: type}) per image (synthetic data)."""
    records = []
    for k, img in enumerate(images):
        iid = image_ids[k] if image_ids else f"img_{k:04d}"
        recs, lab = extract_objects(img, iid, **seg_kw)
        if ground_truth is not None:
            match_ground_truth(recs, lab, *ground_truth[k])
        records += recs
    return ObjectStore(records, {"segmentation": seg_kw or {"method": "threshold"}})


def run_zero_label(store, tax, vlm, decider=None, versions=None):
    """A: VLM-only decisions. B: VLM observations -> decision backend (defaults to the VLM's own text side
    if no separate decider is given). Review uses B as primary, A as the comparison source."""
    recs = store.records
    run_vlm_only(recs, vlm, tax, key="A_vlm_only")
    observe(recs, vlm)
    run_decision_layer(recs, decider or vlm, tax, key="B_vlm_decision", classifier=False)
    review(recs, tax, primary="B_vlm_decision", compare=["A_vlm_only"])
    store.info.setdefault("versions", {}).update(versions or {})
    return store


def run_classifier(store, tax, decider=None, n_members=5, epochs=200):
    """C: train on human-labelled objects, predict all, decide (fast path or via decider), review."""
    recs = store.records
    if any(r.embedding is None for r in recs):
        object_vectors(recs)
    clf = ObjectClassifier([n for n in tax.names if n != "unknown"], n_members=n_members).fit(recs, epochs=epochs)
    clf.predict(recs)
    classifier_decisions(recs, tax, key="C_classifier")
    if decider is not None:
        run_decision_layer(recs, decider, tax, key="C_classifier_decision")
    compare = [k for k in ("B_vlm_decision", "A_vlm_only") if any(k in r.decisions for r in recs)]
    review(recs, tax, primary="C_classifier", compare=compare)
    store.info.setdefault("versions", {})["classifier"] = clf.version
    return clf
