from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from heapq import heapreplace, heappush
from itertools import combinations, product
from pathlib import Path
from typing import Iterable

from .schemas import PassageRole, RelationExample, RelationPassage, Split


SEED = 42
RELATIONS = {
    "P19": ("birth_place", "What is the birth place of {entity}?", "birth place"),
    "P569": ("birth_date", "What is the birth date of {entity}?", "birth date"),
    "P106": ("occupation", "What is the occupation of {entity}?", "occupation"),
    "P27": ("citizenship", "What is the citizenship of {entity}?", "citizenship"),
}
RELATION_ALIASES = {
    "place of birth": "P19",
    "birth place": "P19",
    "date of birth": "P569",
    "birth date": "P569",
    "occupation": "P106",
    "citizenship": "P27",
    "country of citizenship": "P27",
}


@dataclass(frozen=True)
class RelationFact:
    entity_id: str
    entity_label: str
    relation_id: str
    sentence: str
    answer: str
    source_id: str
    source_sentence: str | None = None

    @property
    def passage_id(self) -> str:
        digest = hashlib.sha256(self.sentence.encode("utf-8")).hexdigest()[:16]
        return f"{self.entity_id}:{self.relation_id}:{digest}"


def controlled_sentence(fact: RelationFact) -> str:
    """Render one relation-exclusive evidence statement from a sourced fact."""
    _, _, relation_phrase = RELATIONS[fact.relation_id]
    return f"{fact.entity_label}'s {relation_phrase} is {fact.answer}."


def controlled_fact(fact: RelationFact) -> RelationFact:
    return replace(
        fact,
        sentence=controlled_sentence(fact),
        source_sentence=fact.source_sentence or fact.sentence,
    )


def stable_seed(*parts: str, seed: int = SEED) -> int:
    payload = "\x1f".join((str(seed), *map(str, parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def canonical_relation(value: str) -> str | None:
    normalized = " ".join(value.casefold().replace("_", " ").split())
    if value.upper() in RELATIONS:
        return value.upper()
    return RELATION_ALIASES.get(normalized)


def question_and_span(entity_label: str, relation_id: str) -> tuple[str, tuple[int, int]]:
    _, template, relation_phrase = RELATIONS[relation_id]
    question = template.format(entity=entity_label)
    start = question.index(relation_phrase)
    return question, (start, start + len(relation_phrase))


def split_for_entity(entity_id: str, seed: int = SEED) -> Split:
    fraction = stable_seed(entity_id, seed=seed) / (2**64 - 1)
    if fraction < 0.70:
        return Split.TRAIN
    if fraction < 0.85:
        return Split.CALIBRATION
    return Split.TEST


def _first(mapping: dict, *names: str) -> str | None:
    for name in names:
        value = mapping.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _answer_from_values(sentence: str, values: Iterable[object]) -> str | None:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    for value in cleaned:
        if value in sentence:
            return value
        match = re.fullmatch(r"(\d+)\s*[,;:\-]\s*(\d+)", value)
        if match:
            start, end = map(int, match.groups())
            if 0 <= start < end <= len(sentence):
                return sentence[start:end]
    numeric = [int(value) for value in cleaned if value.isdigit()]
    if len(numeric) >= 2:
        start, end = numeric[0], numeric[1]
        if 0 <= start < end <= len(sentence):
            return sentence[start:end]
    return None


def _official_raw_layout(columns: list[str]) -> bool:
    """The released ``positive_examples`` layout is relation, template, entity, sentence, answer."""
    return len(columns) >= 5 and ("XXX" in columns[1] or "{entity}" in columns[1])


def _tsv_fact(columns: list[str], index: int) -> RelationFact | None:
    if len(columns) < 5:
        return None
    relation = canonical_relation(columns[0])
    if _official_raw_layout(columns):
        entity, sentence, answer_values = columns[2].strip(), columns[3].strip(), columns[4:]
    else:
        entity, sentence, answer_values = columns[1].strip(), columns[3].strip(), columns[4:]
    answer = _answer_from_values(sentence, answer_values)
    if not (relation and entity and sentence and answer):
        return None
    return RelationFact(
        entity_id="_".join(entity.split()),
        entity_label=entity,
        relation_id=relation,
        sentence=sentence,
        answer=answer,
        source_id=str(index),
        source_sentence=sentence,
    )


def _sample_official_raw_records(
    path: Path, *, max_entities: int, candidate_multiplier: int, evidence_mode: str,
    excluded_entity_ids: set[str],
) -> list[RelationFact]:
    """Read a multi-GB official export without retaining all of its facts in memory.

    A first pass selects a deterministic, hash-ranked reservoir of P19 entities.
    A second pass identifies entities with at least three target relations. A
    final pass keeps several distinct evidence sentences per relation for a
    bounded subset, so compatibility is not decided by an arbitrary question
    template duplicate.
    """
    if max_entities < 1 or candidate_multiplier < 1:
        raise ValueError("max_entities and candidate_multiplier must be positive")
    candidate_limit = max_entities * candidate_multiplier
    selected_scores: dict[str, int] = {}
    worst_first: list[tuple[int, str]] = []
    with path.open(encoding="utf-8", newline="") as source:
        for index, columns in enumerate(csv.reader(source, delimiter="\t")):
            fact = _tsv_fact(columns, index)
            if (
                fact is None or fact.relation_id != "P19" or fact.entity_id in selected_scores
                or fact.entity_id in excluded_entity_ids
            ):
                continue
            score = stable_seed("raw-zsre-candidate", fact.entity_id)
            if len(selected_scores) < candidate_limit:
                selected_scores[fact.entity_id] = score
                heappush(worst_first, (-score, fact.entity_id))
            elif score < -worst_first[0][0]:
                _, displaced = heapreplace(worst_first, (-score, fact.entity_id))
                del selected_scores[displaced]
                selected_scores[fact.entity_id] = score

    facts_by_key: dict[tuple[str, str], RelationFact] = {}
    with path.open(encoding="utf-8", newline="") as source:
        for index, columns in enumerate(csv.reader(source, delimiter="\t")):
            fact = _tsv_fact(columns, index)
            if fact is None or fact.entity_id not in selected_scores:
                continue
            key = (fact.entity_id, fact.relation_id)
            previous = facts_by_key.get(key)
            if previous is None or stable_seed("raw-zsre-fact", fact.source_id) < stable_seed("raw-zsre-fact", previous.source_id):
                facts_by_key[key] = fact

    relations_by_entity: dict[str, set[str]] = defaultdict(set)
    for entity_id, relation_id in facts_by_key:
        relations_by_entity[entity_id].add(relation_id)
    eligible = [entity_id for entity_id, relation_ids in relations_by_entity.items() if len(relation_ids) >= 3]
    triad_pool_size = candidate_limit
    triad_candidates = set(sorted(
        eligible,
        key=lambda entity_id: (stable_seed("raw-zsre-triad-candidate", entity_id), entity_id),
    )[:triad_pool_size])

    # The export repeats question templates but can contain many independent
    # evidence sentences for the same entity/relation. Retaining 24 distinct
    # sentences preserves the answer-exclusivity rule without an arbitrary
    # early rejection of otherwise valid entity triads.
    facts_per_relation = 24
    rich_facts: dict[tuple[str, str], list[RelationFact]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as source:
        for index, columns in enumerate(csv.reader(source, delimiter="\t")):
            fact = _tsv_fact(columns, index)
            if fact is None or fact.entity_id not in triad_candidates:
                continue
            key = (fact.entity_id, fact.relation_id)
            candidates = rich_facts[key]
            signature = (fact.sentence, fact.answer)
            if any((candidate.sentence, candidate.answer) == signature for candidate in candidates):
                continue
            if len(candidates) < facts_per_relation:
                candidates.append(fact)
                continue
            current_worst = max(candidates, key=lambda candidate: stable_seed("raw-zsre-evidence", candidate.source_id))
            if stable_seed("raw-zsre-evidence", fact.source_id) < stable_seed("raw-zsre-evidence", current_worst.source_id):
                candidates.remove(current_worst)
                candidates.append(fact)

    facts_by_entity: dict[str, list[RelationFact]] = defaultdict(list)
    for (entity_id, _), facts in rich_facts.items():
        facts_by_entity[entity_id].extend(facts)
    compatible = _has_controlled_triad if evidence_mode == "controlled" else _has_compatible_triad
    valid_entities = sorted(
        (entity_id for entity_id, facts in facts_by_entity.items() if compatible(facts)),
        key=lambda entity_id: (stable_seed("raw-zsre-final", entity_id), entity_id),
    )[:max_entities]
    chosen_entities = set(valid_entities)
    return [fact for entity_id, facts in facts_by_entity.items() if entity_id in chosen_entities for fact in facts]


def load_zsre_records(
    path: Path,
    *,
    max_entities: int | None = None,
    candidate_multiplier: int = 100,
    evidence_mode: str = "raw",
    excluded_entity_ids: set[str] | None = None,
) -> list[RelationFact]:
    """Load positive zsRE examples from JSONL or the official tab-separated export.

    Supported TSV layouts are either relation/entity/template/sentence/offsets
    or the official raw export's relation/template/entity/sentence/answer.
    ``max_entities`` activates bounded two-pass streaming for the multi-GB raw
    export and should be used for practical benchmark materialisation.
    """
    if evidence_mode not in {"raw", "controlled"}:
        raise ValueError("evidence_mode must be 'raw' or 'controlled'")
    if not path.exists():
        raise FileNotFoundError(path)
    excluded = excluded_entity_ids or set()
    suffix = path.suffix.casefold()
    records: list[RelationFact] = []
    if suffix in {".json", ".jsonl"}:
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            row = json.loads(line)
            relation = canonical_relation(_first(row, "relation", "relation_id", "predicate") or "")
            entity = _first(row, "entity", "subject", "x", "entity_label")
            sentence = _first(row, "sentence", "context", "evidence")
            answer_value = row.get("answer") or row.get("answers") or row.get("object")
            if isinstance(answer_value, list):
                answer = _answer_from_values(sentence or "", answer_value)
            else:
                answer = _answer_from_values(sentence or "", [answer_value])
            if relation and entity and sentence and answer:
                normalized_entity = "_".join(entity.split())
                if normalized_entity in excluded:
                    continue
                records.append(RelationFact(
                    entity_id=normalized_entity,
                    entity_label=entity,
                    relation_id=relation,
                    sentence=sentence,
                    answer=answer,
                    source_id=str(row.get("id", index)),
                    source_sentence=sentence,
                ))
        return records

    with path.open(encoding="utf-8", newline="") as source:
        first_row = next(csv.reader(source, delimiter="\t"), [])
    if max_entities is not None and _official_raw_layout(first_row):
        return _sample_official_raw_records(
            path,
            max_entities=max_entities,
            candidate_multiplier=candidate_multiplier,
            evidence_mode=evidence_mode,
            excluded_entity_ids=excluded,
        )

    with path.open(encoding="utf-8", newline="") as source:
        for index, columns in enumerate(csv.reader(source, delimiter="\t")):
            fact = _tsv_fact(columns, index)
            if fact is not None and fact.entity_id not in excluded:
                records.append(fact)
    return records


def _compatible_triad(facts: tuple[RelationFact, RelationFact, RelationFact]) -> bool:
    if len({fact.sentence for fact in facts}) != 3:
        return False
    if len({fact.answer.casefold() for fact in facts}) != 3:
        return False
    for fact in facts:
        other_answers = [other.answer.casefold() for other in facts if other != fact]
        sentence = fact.sentence.casefold()
        if any(answer in sentence for answer in other_answers):
            return False
    return True


def _has_compatible_triad(facts: Iterable[RelationFact]) -> bool:
    by_relation: dict[str, list[RelationFact]] = defaultdict(list)
    for fact in facts:
        by_relation[fact.relation_id].append(fact)
    for relation_ids in combinations(sorted(by_relation), 3):
        for triad in product(*(by_relation[relation_id] for relation_id in relation_ids)):
            if _compatible_triad(triad):
                return True
    return False


def _has_controlled_triad(facts: Iterable[RelationFact]) -> bool:
    """Check whether sourced facts can form distinct controlled statements."""
    return _has_compatible_triad(controlled_fact(fact) for fact in facts)


def _pick(candidates: list[RelationFact], *key: str) -> RelationFact:
    if not candidates:
        raise ValueError("candidate selection received an empty list")
    ordered = sorted(candidates, key=lambda fact: (fact.entity_id, fact.relation_id, fact.passage_id))
    return ordered[stable_seed(*key) % len(ordered)]


def _pick_triad(candidates: list[tuple[RelationFact, RelationFact, RelationFact]], entity_id: str) -> tuple[RelationFact, RelationFact, RelationFact]:
    ordered = sorted(candidates, key=lambda triad: tuple(fact.passage_id for fact in triad))
    return ordered[stable_seed(entity_id, "triad") % len(ordered)]


def _make_passage(fact: RelationFact, role: PassageRole, *, evidence_mode: str) -> RelationPassage:
    return RelationPassage(
        id=fact.passage_id,
        text=fact.sentence,
        entity_id=fact.entity_id,
        evidence_relation_id=fact.relation_id,
        answer=fact.answer,
        role=role,
        source_id=fact.source_id,
        source_text=fact.source_sentence,
        evidence_mode=evidence_mode,
    )


def build_examples(records: Iterable[RelationFact], *, evidence_mode: str = "raw") -> list[RelationExample]:
    """Create one deterministic three-relation contrast set per eligible entity."""
    if evidence_mode not in {"raw", "controlled"}:
        raise ValueError("evidence_mode must be 'raw' or 'controlled'")
    if evidence_mode == "controlled":
        records = (controlled_fact(fact) for fact in records)
    deduplicated: dict[tuple[str, str, str], RelationFact] = {}
    for fact in records:
        key = (fact.entity_id, fact.relation_id, fact.sentence)
        deduplicated.setdefault(key, fact)
    by_entity: dict[str, dict[str, list[RelationFact]]] = defaultdict(lambda: defaultdict(list))
    for fact in deduplicated.values():
        by_entity[fact.entity_id][fact.relation_id].append(fact)

    selected: dict[str, tuple[RelationFact, RelationFact, RelationFact]] = {}
    for entity_id, relation_facts in by_entity.items():
        available = sorted(relation for relation in relation_facts if relation in RELATIONS)
        triads: list[tuple[RelationFact, RelationFact, RelationFact]] = []
        for relation_ids in combinations(available, 3):
            options = [sorted(relation_facts[relation], key=lambda fact: fact.passage_id) for relation in relation_ids]
            for first in options[0]:
                for second in options[1]:
                    for third in options[2]:
                        triad = (first, second, third)
                        if _compatible_triad(triad):
                            triads.append(triad)
        if triads:
            selected[entity_id] = _pick_triad(triads, entity_id)

    facts_by_relation: dict[str, list[RelationFact]] = defaultdict(list)
    all_facts: list[RelationFact] = []
    for triad in selected.values():
        all_facts.extend(triad)
        for fact in triad:
            facts_by_relation[fact.relation_id].append(fact)

    examples: list[RelationExample] = []
    for entity_id, triad in sorted(selected.items()):
        ordered = sorted(triad, key=lambda fact: fact.relation_id)
        for target_index, target in enumerate(ordered):
            for donor_index, donor in enumerate(ordered):
                if donor_index == target_index:
                    continue
                control = next(fact for index, fact in enumerate(ordered) if index not in {target_index, donor_index})
                target_negatives = [fact for fact in facts_by_relation[target.relation_id] if fact.entity_id != entity_id]
                donor_negatives = [fact for fact in facts_by_relation[donor.relation_id] if fact.entity_id != entity_id]
                unrelated_candidates = [
                    fact
                    for fact in all_facts
                    if fact.entity_id != entity_id and fact.relation_id not in {target.relation_id, donor.relation_id}
                ]
                if not target_negatives or not donor_negatives or not unrelated_candidates:
                    continue
                target_negative = _pick(target_negatives, entity_id, target.relation_id, donor.relation_id, "target-negative")
                donor_negative = _pick(donor_negatives, entity_id, target.relation_id, donor.relation_id, "donor-negative")
                unrelated = _pick(unrelated_candidates, entity_id, target.relation_id, donor.relation_id, "unrelated")
                question, span = question_and_span(target.entity_label, target.relation_id)
                passages = [
                    _make_passage(target, PassageRole.TARGET, evidence_mode=evidence_mode),
                    _make_passage(donor, PassageRole.DONOR, evidence_mode=evidence_mode),
                    _make_passage(control, PassageRole.CONTROL, evidence_mode=evidence_mode),
                    _make_passage(target_negative, PassageRole.CONTROL, evidence_mode=evidence_mode),
                    _make_passage(donor_negative, PassageRole.CONTROL, evidence_mode=evidence_mode),
                    _make_passage(unrelated, PassageRole.CONTROL, evidence_mode=evidence_mode),
                ]
                if len({passage.id for passage in passages}) != 6:
                    continue
                examples.append(RelationExample(
                    id=f"{entity_id}:{target.relation_id}->{donor.relation_id}",
                    entity_id=entity_id,
                    entity_label=target.entity_label,
                    relation_id=target.relation_id,
                    question=question,
                    relation_span=span,
                    answer=target.answer,
                    target_passage_id=target.passage_id,
                    donor_passage_id=donor.passage_id,
                    control_passage_id=control.passage_id,
                    paired_example_ids={
                        donor.relation_id: f"{entity_id}:{donor.relation_id}->{target.relation_id}",
                        control.relation_id: f"{entity_id}:{control.relation_id}->{target.relation_id}",
                    },
                    passages=passages,
                    split=split_for_entity(entity_id),
                ))
    return examples


def assert_no_entity_leakage(examples: Iterable[RelationExample]) -> None:
    seen: dict[str, set[Split]] = defaultdict(set)
    for example in examples:
        seen[example.entity_id].add(example.split)
    leaking = {entity: splits for entity, splits in seen.items() if len(splits) > 1}
    if leaking:
        raise ValueError(f"entity leakage across splits: {leaking}")


def benchmark_summary(examples: Iterable[RelationExample]) -> dict:
    materialized = list(examples)
    assert_no_entity_leakage(materialized)
    entity_counts: dict[str, int] = Counter(example.split.value for example in {example.entity_id: example for example in materialized}.values())
    direction_counts: dict[str, int] = Counter()
    for example in materialized:
        donor_relation = next(
            passage.evidence_relation_id for passage in example.passages if passage.id == example.donor_passage_id
        )
        direction_counts[f"{example.split.value}:{example.relation_id}->{donor_relation}"] += 1
    return {
        "examples": len(materialized),
        "entities_by_split": dict(sorted(entity_counts.items())),
        "directions": dict(sorted(direction_counts.items())),
        "relations": dict(sorted(Counter(example.relation_id for example in materialized).items())),
    }


def validate_benchmark_size(examples: Iterable[RelationExample], *, allow_reduced: bool = False) -> dict:
    materialized = list(examples)
    summary = benchmark_summary(materialized)
    total_entities = sum(summary["entities_by_split"].values())
    directional_counts = summary["directions"]
    expected_directions = {
        f"{split}:{source}->{donor}"
        for split in (Split.CALIBRATION.value, Split.TEST.value)
        for source in RELATIONS
        for donor in RELATIONS
        if source != donor
    }
    missing_directions = sorted(expected_directions - directional_counts.keys())
    calibration_test = [directional_counts.get(key, 0) for key in expected_directions]
    full_ready = total_entities >= 200 and not missing_directions and min(calibration_test) >= 30
    reduced_ready = total_entities >= 120 and not missing_directions and min(calibration_test) >= 20
    if not full_ready and not (allow_reduced and reduced_ready):
        raise ValueError(
            "benchmark does not meet required coverage: need >=200 entities and >=30 observations per "
            "calibration/test direction; use --allow-reduced only for >=120 entities and >=20 per direction"
        )
    summary["coverage_tier"] = "full" if full_ready else "reduced"
    summary["missing_required_directions"] = missing_directions
    return summary


def save_jsonl(path: Path, examples: Iterable[RelationExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for example in examples:
            output.write(example.model_dump_json() + "\n")


def load_jsonl(path: Path) -> list[RelationExample]:
    with path.open(encoding="utf-8") as source:
        return [RelationExample.model_validate_json(line) for line in source if line.strip()]
