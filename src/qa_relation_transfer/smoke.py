"""Deterministic smoke-run sampling and advisory assessment utilities."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .dataset import RELATIONS, stable_seed
from .evaluation import summarize_layer
from .schemas import RelationExample


SMOKE_LAYERS = (1, 3, 6)
SMOKE_PER_DIRECTION = 2
SMOKE_TARGET_SECONDS = 2 * 60 * 60


def donor_relation_id(example: RelationExample) -> str:
    return next(passage.evidence_relation_id for passage in example.passages if passage.id == example.donor_passage_id)


def direction_id(example: RelationExample) -> str:
    return f"{example.relation_id}->{donor_relation_id(example)}"


def expected_directions() -> list[str]:
    return [f"{source}->{donor}" for source in RELATIONS for donor in RELATIONS if source != donor]


def stratified_smoke_examples(
    examples: Iterable[RelationExample],
    *,
    per_direction: int = SMOKE_PER_DIRECTION,
    seed: int = 42,
) -> tuple[list[RelationExample], dict[str, Any]]:
    """Sample up to ``per_direction`` examples for every directed relation pair.

    The order is derived only from the example ID and seed, so a smoke can be
    reproduced even when input JSONL ordering changes.
    """
    if per_direction < 1:
        raise ValueError("smoke examples per direction must be at least one")

    groups: dict[str, list[RelationExample]] = defaultdict(list)
    for example in examples:
        groups[direction_id(example)].append(example)

    selected: list[RelationExample] = []
    selected_counts: dict[str, int] = {}
    available_counts: dict[str, int] = {}
    for direction in expected_directions():
        candidates = groups.get(direction, [])
        available_counts[direction] = len(candidates)
        ordered = sorted(candidates, key=lambda example: (stable_seed("smoke", direction, example.id, seed=seed), example.id))
        chosen = ordered[:per_direction]
        selected.extend(chosen)
        selected_counts[direction] = len(chosen)

    selected.sort(key=lambda example: (direction_id(example), example.id))
    missing = [direction for direction in expected_directions() if available_counts[direction] == 0]
    undersampled = [
        direction
        for direction in expected_directions()
        if 0 < available_counts[direction] < per_direction
    ]
    coverage = {
        "per_direction_requested": per_direction,
        "expected_directions": expected_directions(),
        "available_counts": available_counts,
        "selected_counts": selected_counts,
        "missing_directions": missing,
        "undersampled_directions": undersampled,
        "complete": not missing and not undersampled,
    }
    return selected, coverage


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _promising_layers(summaries: Sequence[dict]) -> list[int]:
    promising = []
    for summary in summaries:
        effects = summary.get("effects", {})
        treatment = effects.get("same_entity_donor")
        controls = [effects.get(name) for name in ("same_entity_control", "different_entity_donor")]
        if treatment is None or any(control is None for control in controls):
            continue
        treatment_mean = treatment["mean_donor_advantage_change"]
        if treatment_mean > 0 and all(treatment_mean > control["mean_donor_advantage_change"] for control in controls):
            promising.append(summary["layer"])
    return promising


def assess_smoke(
    metadata: dict[str, Any],
    layer_payloads: Sequence[dict[str, Any]],
    *,
    target_seconds: float = SMOKE_TARGET_SECONDS,
) -> dict[str, Any]:
    """Create an advisory, never-blocking assessment of a retrieval smoke run."""
    errors: list[str] = []
    if metadata.get("profile") != "smoke":
        errors.append("run metadata does not identify a smoke profile")
    configuration = metadata.get("configuration", {})
    if configuration.get("reader_enabled") or configuration.get("verifier_model"):
        errors.append("smoke profile must be retrieval-only")

    coverage = metadata.get("smoke", {}).get("coverage", {})
    if not coverage.get("complete", False):
        errors.append("smoke coverage is incomplete")

    summaries = []
    expected_layers = set(SMOKE_LAYERS)
    found_layers = set()
    for payload in layer_payloads:
        try:
            configured_layer = payload.get("configuration", {}).get("layer")
            summary = summarize_layer(payload["rows"], layer=configured_layer)
            summaries.append(summary)
            found_layers.add(summary["layer"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"invalid layer output: {error}")
    if found_layers != expected_layers:
        errors.append(f"smoke must contain layers {sorted(expected_layers)}, found {sorted(found_layers)}")

    timing = metadata.get("timing", {})
    layer_seconds = timing.get("layer_seconds", {})
    sample_size = len(metadata.get("smoke", {}).get("selected_example_ids", []))
    full_size = metadata.get("full_split_examples", 0)
    per_example = []
    if sample_size:
        per_example = [float(seconds) / sample_size for seconds in layer_seconds.values()]
    if not per_example or not full_size:
        errors.append("timing or full calibration size is missing")
        projected_seconds = None
    else:
        projected_seconds = (
            float(timing.get("model_initialization_seconds", 0.0))
            + max(per_example) * full_size * 6 * 1.2
        )

    promising_layers = _promising_layers(summaries)
    cost_within_target = projected_seconds is not None and projected_seconds <= target_seconds
    if errors:
        recommendation = "iterate"
        rationale = "Coverage, artifact, or timing problems must be resolved before interpreting the signal."
    elif promising_layers and cost_within_target:
        recommendation = "scale"
        rationale = "At least one layer shows a selective donor advantage and the projection fits the time budget."
    elif not promising_layers and cost_within_target:
        recommendation = "scale for diagnosis"
        rationale = "The smoke run is technically valid and inexpensive, but the signal is ambiguous; a full sweep may resolve the uncertainty."
    else:
        recommendation = "iterate"
        rationale = "The signal or cost favors optimizing the design before a full run; this advisory result does not prevent a documented decision to scale."

    return {
        "kind": "smoke_assessment",
        "advisory_only": True,
        "recommendation": recommendation,
        "rationale": rationale,
        "target_seconds": target_seconds,
        "coverage": coverage,
        "errors": errors,
        "timing": {
            "observed_wall_seconds": timing.get("total_wall_seconds"),
            "model_initialization_seconds": timing.get("model_initialization_seconds"),
            "worst_seconds_per_example_layer": max(per_example) if per_example else None,
            "projected_full_calibration_seconds": projected_seconds,
            "projected_within_target": cost_within_target,
        },
        "promising_layers": promising_layers,
        "layer_summaries": summaries,
    }
