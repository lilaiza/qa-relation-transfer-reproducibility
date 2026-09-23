from __future__ import annotations

import sys

import pytest

from qa_relation_transfer import run_interventions
from qa_relation_transfer.dataset import RELATIONS, RelationFact, build_examples
from qa_relation_transfer.schemas import Split
from qa_relation_transfer.smoke import SMOKE_LAYERS, assess_smoke, stratified_smoke_examples


def _facts(entity_id: str) -> list[RelationFact]:
    facts = []
    for relation_id in ("P19", "P569", "P106"):
        _, _, phrase = RELATIONS[relation_id]
        answer = f"{entity_id}-{relation_id}"
        facts.append(RelationFact(
            entity_id=entity_id,
            entity_label=entity_id,
            relation_id=relation_id,
            sentence=f"{entity_id} has {phrase} {answer}.",
            answer=answer,
            source_id=f"{entity_id}-{relation_id}",
        ))
    return facts


def _examples():
    examples = build_examples([fact for entity in range(4) for fact in _facts(f"entity_{entity}")])
    return [example.model_copy(update={"split": Split.CALIBRATION}) for example in examples]


def _row(example, condition: str, target_score: float, donor_score: float, layer: int) -> dict:
    passages = []
    for passage in example.passages:
        score = target_score if passage.id == example.target_passage_id else donor_score if passage.id == example.donor_passage_id else 0.1
        passages.append({**passage.model_dump(mode="json"), "score": score, "rank": 0})
    passages.sort(key=lambda passage: -passage["score"])
    for rank, passage in enumerate(passages, start=1):
        passage["rank"] = rank
    return {
        "example_id": example.id,
        "condition": condition,
        "layer": layer,
        "target_passage_id": example.target_passage_id,
        "donor_passage_id": example.donor_passage_id,
        "retrieval": {"passages": passages},
    }


def _layer_payloads(*, promising: bool = True) -> list[dict]:
    first, second = _examples()[:2]
    payloads = []
    for layer in SMOKE_LAYERS:
        donor_scores = (0.2, 0.9) if promising else (0.8, 0.2)
        rows = []
        for example in (first, second):
            rows.extend([
                _row(example, "baseline", 0.8, 0.2, layer),
                _row(example, "same_entity_donor", *donor_scores, layer),
                _row(example, "same_entity_control", 0.6, 0.3, layer),
                _row(example, "different_entity_donor", 0.6, 0.35, layer),
                _row(example, "self_patch", 0.8, 0.2, layer),
            ])
        payloads.append({"rows": rows})
    return payloads


def _metadata(*, complete: bool = True) -> dict:
    return {
        "profile": "smoke",
        "full_split_examples": 10,
        "smoke": {
            "selected_example_ids": ["a", "b"],
            "coverage": {"complete": complete, "missing_directions": [], "undersampled_directions": []},
        },
        "timing": {"model_initialization_seconds": 2.0, "layer_seconds": {"1": 2.0, "3": 2.0, "6": 2.0}, "total_wall_seconds": 8.0},
    }


def test_stratified_smoke_sampling_is_reproducible_and_balanced_for_available_directions():
    selected_first, coverage_first = stratified_smoke_examples(_examples(), per_direction=2)
    selected_second, coverage_second = stratified_smoke_examples(reversed(_examples()), per_direction=2)
    assert [example.id for example in selected_first] == [example.id for example in selected_second]
    assert coverage_first == coverage_second
    assert all(count == 2 for count in coverage_first["selected_counts"].values() if count)


def test_assessment_recommends_scaling_for_promising_affordable_smoke():
    report = assess_smoke(_metadata(), _layer_payloads())
    assert report["recommendation"] == "escalar"
    assert report["promising_layers"] == list(SMOKE_LAYERS)
    assert report["timing"]["projected_within_target"] is True
    assert report["advisory_only"] is True


def test_assessment_recommends_diagnostic_scaling_for_ambiguous_affordable_smoke():
    report = assess_smoke(_metadata(), _layer_payloads(promising=False))
    assert report["recommendation"] == "escalar por diagnóstico"


def test_assessment_flags_incomplete_coverage_and_excessive_projection():
    report = assess_smoke(_metadata(complete=False), _layer_payloads(), target_seconds=10)
    assert report["recommendation"] == "iterar"
    assert "smoke coverage is incomplete" in report["errors"]
    assert report["timing"]["projected_within_target"] is False


def test_assessment_uses_artifact_layer_when_baseline_rows_are_unpatched():
    payloads = _layer_payloads()
    for payload, layer in zip(payloads, SMOKE_LAYERS, strict=True):
        payload["configuration"] = {"layer": layer}
        for row in payload["rows"]:
            if row["condition"] == "baseline":
                row["layer"] = None
    report = assess_smoke(_metadata(), payloads)
    assert report["errors"] == []
    assert [summary["layer"] for summary in report["layer_summaries"]] == list(SMOKE_LAYERS)


def test_test_split_still_requires_a_frozen_layer(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "run_interventions",
        "--data", str(tmp_path / "data.jsonl"),
        "--split", "test",
        "--output-dir", str(tmp_path / "results"),
    ])
    with pytest.raises(ValueError, match="frozen-layer-file"):
        run_interventions.main()
