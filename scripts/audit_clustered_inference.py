"""Recompute aggregate test inference with the target entity as the unit."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def donor_advantage(row: dict) -> float:
    passages = {item["id"]: item for item in row["retrieval"]["passages"]}
    return float(
        passages[row["donor_passage_id"]]["score"]
        - passages[row["target_passage_id"]]["score"]
    )


def cluster_summary(values: dict[str, list[float]], *, seed: int = 42) -> dict:
    entity_means = np.asarray(
        [np.mean(entity_values) for _, entity_values in sorted(values.items())],
        dtype=float,
    )
    generator = np.random.default_rng(seed)
    bootstrap = generator.choice(
        entity_means, size=(10_000, len(entity_means)), replace=True
    ).mean(axis=1)
    observed = abs(float(entity_means.mean()))
    signs = generator.choice(
        np.asarray([-1.0, 1.0]), size=(50_000, len(entity_means))
    )
    permutations = np.abs((signs * entity_means).mean(axis=1))
    return {
        "entities": len(entity_means),
        "examples": sum(map(len, values.values())),
        "examples_per_entity": sorted({len(items) for items in values.values()}),
        "mean": float(entity_means.mean()),
        "bootstrap_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "sign_permutation_pvalue": float(
            (np.count_nonzero(permutations >= observed) + 1)
            / (len(permutations) + 1)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.trace.read_text())
    indexed = {
        (row["example_id"], row["condition"]): row for row in payload["rows"]
    }
    by_entity: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    specifications = {
        "same_entity_donor_minus_baseline": ("same_entity_donor", "baseline"),
        "relation_C_minus_baseline": ("same_entity_control", "baseline"),
        "same_entity_donor_minus_relation_C": (
            "same_entity_donor",
            "same_entity_control",
        ),
        "different_entity_donor_minus_baseline": (
            "different_entity_donor",
            "baseline",
        ),
    }
    for (example_id, condition), row in indexed.items():
        if condition != "baseline":
            continue
        entity = row["entity_id"]
        for name, (first, second) in specifications.items():
            difference = donor_advantage(indexed[(example_id, first)]) - donor_advantage(
                indexed[(example_id, second)]
            )
            by_entity[name][entity].append(difference)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace": str(args.trace),
        "unit": "target entity; six directed examples retained together",
        "seed": 42,
        "bootstrap_samples": 10_000,
        "permutation_samples": 50_000,
        "results": {
            name: cluster_summary(entity_values)
            for name, entity_values in by_entity.items()
        },
        "interpretation": (
            "Post-hoc robustness analysis. It does not replace the historical "
            "pre-specified example-level intervals and p-values."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
