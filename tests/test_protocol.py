from __future__ import annotations

from qa_relation_transfer.agents import donor_for_condition, patch_relation_span, relation_token_indices
from qa_relation_transfer.dataset import RELATIONS, RelationFact, assert_no_entity_leakage, build_examples, load_zsre_records, question_and_span
from qa_relation_transfer.evaluation import paired_condition_effect, retrieval_metrics, summarize_layer
from qa_relation_transfer.schemas import PatchCondition, Split


def facts_for(entity_id: str, label: str):
    facts = []
    for relation_id in ("P19", "P569", "P106"):
        _, _, phrase = RELATIONS[relation_id]
        answer = f"{entity_id}-{relation_id}-answer"
        facts.append(RelationFact(
            entity_id=entity_id,
            entity_label=label,
            relation_id=relation_id,
            sentence=f"{label} has {phrase} value {answer}.",
            answer=answer,
            source_id=f"{entity_id}-{relation_id}",
        ))
    return facts


def examples():
    return build_examples([
        *facts_for("entity_1", "Entity One"),
        *facts_for("entity_2", "Entity Two"),
        *facts_for("entity_3", "Entity Three"),
        *facts_for("entity_4", "Entity Four"),
    ])


def row(example, condition: str, target_score: float, donor_score: float):
    passages = []
    for index, passage in enumerate(example.passages, start=1):
        score = 0.1
        if passage.id == example.target_passage_id:
            score = target_score
        if passage.id == example.donor_passage_id:
            score = donor_score
        passages.append({**passage.model_dump(mode="json"), "rank": index, "score": score})
    passages.sort(key=lambda passage: -passage["score"])
    for index, passage in enumerate(passages, start=1):
        passage["rank"] = index
    return {
        "example_id": example.id,
        "condition": condition,
        "layer": 2,
        "target_passage_id": example.target_passage_id,
        "donor_passage_id": example.donor_passage_id,
        "retrieval": {"passages": passages},
    }


def test_question_span_is_exact_and_token_offsets_select_only_relation_tokens():
    question, span = question_and_span("Ada Lovelace", "P569")
    assert question[span[0]:span[1]] == "birth date"
    offsets = __import__("torch").tensor([[0, 4], [5, 7], [8, 13], [14, 18], [0, 0]])
    assert relation_token_indices(offsets, span) == [2, 3]


def test_masked_patch_changes_only_relation_positions():
    torch = __import__("torch")
    original = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    donor = torch.full((2, 4), 99.0)
    patched, mode = patch_relation_span(original, [1, 3], donor, [2, 4], self_patch=False)
    assert mode == "tokenwise-donor"
    assert torch.equal(patched[0, 0], original[0, 0])
    assert torch.equal(patched[0, 2], original[0, 2])
    assert torch.equal(patched[0, 4], original[0, 4])
    assert torch.equal(patched[0, [1, 3]], donor)


def test_contrastive_examples_are_entity_disjoint_and_have_bidirectional_pairs():
    built = examples()
    assert len(built) == 24
    assert_no_entity_leakage(built)
    first = built[0]
    assert len(first.passages) == 6
    assert {passage.role.value for passage in first.passages} == {"target", "donor", "control"}
    donor_relation = next(passage.evidence_relation_id for passage in first.passages if passage.id == first.donor_passage_id)
    inverse_id = first.paired_example_ids[donor_relation]
    inverse = next(example for example in built if example.id == inverse_id)
    assert inverse.relation_id == donor_relation
    assert next(passage.evidence_relation_id for passage in inverse.passages if passage.id == inverse.donor_passage_id) == first.relation_id


def test_natural_donor_controls_keep_entity_and_relation_contracts():
    built = examples()
    target = built[0]
    by_id = {example.id: example for example in built}
    same_entity = donor_for_condition(target, by_id, built, PatchCondition.SAME_ENTITY_DONOR)
    same_control = donor_for_condition(target, by_id, built, PatchCondition.SAME_ENTITY_CONTROL)
    different_entity = donor_for_condition(target, by_id, built, PatchCondition.DIFFERENT_ENTITY_DONOR)
    assert same_entity.entity_id == target.entity_id
    assert same_control.entity_id == target.entity_id
    assert different_entity.entity_id != target.entity_id
    assert different_entity.relation_id == same_entity.relation_id


def test_primary_metric_is_paired_donor_advantage_change():
    first, second = examples()[:2]
    baseline = [row(first, "baseline", 0.8, 0.2), row(second, "baseline", 0.7, 0.1)]
    donor = [row(first, "same_entity_donor", 0.3, 0.9), row(second, "same_entity_donor", 0.2, 0.8)]
    effect = paired_condition_effect(baseline, donor)
    assert effect["mean_donor_advantage_change"] > 1.0
    assert effect["positive_rate"] == 1.0
    metrics = retrieval_metrics(donor[0])
    assert metrics["donor_top_1"] is True


def test_layer_summary_compares_treatment_with_natural_controls():
    first, second = examples()[:2]
    rows = [
        row(first, "baseline", 0.8, 0.2), row(second, "baseline", 0.8, 0.2),
        row(first, "same_entity_donor", 0.2, 0.9), row(second, "same_entity_donor", 0.2, 0.9),
        row(first, "same_entity_control", 0.5, 0.4), row(second, "same_entity_control", 0.5, 0.4),
        row(first, "different_entity_donor", 0.5, 0.45), row(second, "different_entity_donor", 0.5, 0.45),
        row(first, "self_patch", 0.8, 0.2), row(second, "self_patch", 0.8, 0.2),
    ]
    summary = summarize_layer(rows)
    assert summary["layer"] == 2
    assert summary["specificity_margin"] > 0


def test_official_raw_tsv_layout_reads_template_before_entity(tmp_path):
    source = tmp_path / "positive_examples"
    source.write_text(
        "place of birth\tWhere was XXX born?\tAda Lovelace\tAda Lovelace was born in London.\tLondon\n"
        "occupation\tWhat is XXX's job?\tAda Lovelace\tAda Lovelace was a mathematician.\tmathematician\n",
        encoding="utf-8",
    )
    records = load_zsre_records(source)
    assert [(record.relation_id, record.entity_label, record.answer) for record in records] == [
        ("P19", "Ada Lovelace", "London"),
        ("P106", "Ada Lovelace", "mathematician"),
    ]


def test_bounded_raw_import_selects_complete_entities_deterministically(tmp_path):
    source = tmp_path / "positive_examples"
    rows = []
    for entity in ("Ada", "Grace", "Katherine"):
        for relation, template, answer in (
            ("place of birth", "Where was XXX born?", "London"),
            ("date of birth", "When was XXX born?", "1900"),
            ("occupation", "What is XXX's job?", "mathematician"),
        ):
            rows.append(f"{relation}\t{template}\t{entity}\t{entity} has value {answer}.\t{answer}\n")
    source.write_text("".join(rows), encoding="utf-8")
    first = load_zsre_records(source, max_entities=2, candidate_multiplier=2)
    second = load_zsre_records(source, max_entities=2, candidate_multiplier=2)
    assert [record.source_id for record in first] == [record.source_id for record in second]
    assert len({record.entity_id for record in first}) == 2
    assert {record.relation_id for record in first} == {"P19", "P569", "P106"}


def test_bounded_raw_import_excludes_existing_entities(tmp_path):
    source = tmp_path / "positive_examples"
    rows = []
    for entity in ("Ada", "Grace", "Katherine"):
        for relation, template, answer in (
            ("place of birth", "Where was XXX born?", "London"),
            ("date of birth", "When was XXX born?", "1900"),
            ("occupation", "What is XXX's job?", "mathematician"),
        ):
            rows.append(f"{relation}\t{template}\t{entity}\t{entity} has value {answer}.\t{answer}\n")
    source.write_text("".join(rows), encoding="utf-8")
    records = load_zsre_records(
        source, max_entities=2, candidate_multiplier=2, excluded_entity_ids={"Ada"}
    )
    assert {record.entity_id for record in records} == {"Grace", "Katherine"}


def test_controlled_evidence_removes_cross_relation_answers_and_keeps_provenance():
    facts = [
        RelationFact(
            entity_id="ada", entity_label="Ada", relation_id=relation_id,
            sentence="Ada was born in London in 1900 and was a mathematician.",
            answer=answer, source_id=relation_id,
            source_sentence="Ada was born in London in 1900 and was a mathematician.",
        )
        for relation_id, answer in (("P19", "London"), ("P569", "1900"), ("P106", "mathematician"))
    ]
    facts.extend(facts_for("grace", "Grace"))
    built = build_examples(facts, evidence_mode="controlled")
    ada = next(example for example in built if example.entity_id == "ada")
    target = next(passage for passage in ada.passages if passage.id == ada.target_passage_id)
    assert target.text == "Ada's birth date is 1900." or target.text == "Ada's birth place is London." or target.text == "Ada's occupation is mathematician."
    assert target.source_text == "Ada was born in London in 1900 and was a mathematician."
    assert target.evidence_mode == "controlled"
