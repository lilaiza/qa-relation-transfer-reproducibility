"""Audit the strict-entity Verifier replica without loading any model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time


def stable_seed(*parts: str, seed: int = 42) -> int:
    payload = "\x1f".join((str(seed), *map(str, parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def split_for_entity(entity_id: str) -> str:
    value = stable_seed(entity_id) / (2**64 - 1)
    return "train" if value < 0.70 else "calibration" if value < 0.85 else "test"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--old-trace", type=Path, required=True)
    parser.add_argument("--new-trace", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = datetime.now(timezone.utc)
    timer = time.perf_counter()

    train_pair_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    train_passage_ids: set[str] = set()
    train_passage_entities: set[str] = set()
    unique_pairs: set[tuple[str, str]] = set()
    target_entities: defaultdict[str, set[str]] = defaultdict(set)
    examples = 0
    with args.data.open() as source:
        for line in source:
            row = json.loads(line)
            target_entities[row["split"]].add(row["entity_id"])
            if row["split"] != "train":
                continue
            examples += 1
            kept = 0
            for passage in row["passages"]:
                if split_for_entity(passage["entity_id"]) != "train":
                    continue
                kept += 1
                label = passage["id"] == row["target_passage_id"]
                label_counts["positive" if label else "negative"] += 1
                train_passage_ids.add(passage["id"])
                train_passage_entities.add(passage["entity_id"])
                unique_pairs.add((row["question"], passage["id"]))
            train_pair_counts[str(kept)] += 1

    old = json.loads(args.old_trace.read_text())["rows"]
    new_payload = json.loads(args.new_trace.read_text())
    new = new_payload["rows"]
    if len(old) != len(new):
        raise AssertionError("trace lengths differ")
    stable_fields = (
        "example_id", "condition", "top_passage_id", "top_passage_relation_id",
        "reader_answer", "reader_confidence", "retrieval",
    )
    changed_fields = {field: 0 for field in stable_fields}
    verifier_threshold_changes = 0
    verifier_differences: defaultdict[str, list[float]] = defaultdict(list)
    top_entity_splits: defaultdict[str, Counter[str]] = defaultdict(Counter)
    seen_top_passages: Counter[str] = Counter()
    for previous, current in zip(old, new, strict=True):
        for field in stable_fields:
            changed_fields[field] += previous[field] != current[field]
        verifier_threshold_changes += (
            previous["verifier_support_probability"] >= 0.5
        ) != (current["verifier_support_probability"] >= 0.5)
        condition = current["condition"]
        verifier_differences[condition].append(
            current["verifier_support_probability"]
            - previous["verifier_support_probability"]
        )
        top = next(
            passage for passage in current["retrieval"]["passages"]
            if passage["id"] == current["top_passage_id"]
        )
        top_entity_splits[condition][split_for_entity(top["entity_id"])] += 1
        if top["id"] in train_passage_ids:
            seen_top_passages[condition] += 1

    metadata = json.loads(args.training_metadata.read_text())
    report = {
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - timer,
        "python": platform.python_version(),
        "hashes": {
            "data": sha256(args.data),
            "old_trace": sha256(args.old_trace),
            "new_trace": sha256(args.new_trace),
            "training_metadata": sha256(args.training_metadata),
            "checkpoint": sha256(args.checkpoint),
            "script": sha256(Path(__file__)),
        },
        "strict_training": {
            "examples": examples,
            "pairs": sum(label_counts.values()),
            "unique_question_passage_pairs": len(unique_pairs),
            "labels": dict(label_counts),
            "pairs_per_example": dict(sorted(train_pair_counts.items())),
            "passage_entities": len(train_passage_entities),
            "all_passage_entities_are_train": all(
                split_for_entity(entity_id) == "train" for entity_id in train_passage_entities
            ),
            "overlap_with_calibration_target_entities": len(
                train_passage_entities & target_entities["calibration"]
            ),
            "overlap_with_test_target_entities": len(
                train_passage_entities & target_entities["test"]
            ),
        },
        "metadata_consistency": {
            "pairs_match": metadata["pairs"] == sum(label_counts.values()),
            "examples_match": metadata["examples"] == examples,
            "strict_flag": metadata["require_train_passage_entities"],
            "seed": metadata.get("seed", 42),
            "checkpoint_saved": metadata["checkpoint_saved"],
        },
        "trace_comparison": {
            "rows": len(new),
            "changed_non_verifier_fields": changed_fields,
            "verifier_threshold_0_5_changes": verifier_threshold_changes,
            "verifier_probability_difference": {
                condition: {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
                for condition, values in verifier_differences.items()
            },
            "top_passage_entity_splits": {
                condition: dict(counts) for condition, counts in top_entity_splits.items()
            },
            "top_passages_seen_in_strict_training": dict(seen_top_passages),
            "frozen_layer_file": new_payload["configuration"]["frozen_layer_file"],
            "layer": new_payload["configuration"]["layer"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
