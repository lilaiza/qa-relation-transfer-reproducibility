"""Recompute downstream metrics from the preserved held-out trace."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path


TRACE = Path("artifacts/test/test_layer1.json.gz")
EXPECTED = Path("docs/downstream_metrics.json")


def normalized(value: object) -> str:
    return str(value).strip().casefold()


def main() -> None:
    with gzip.open(TRACE, "rt", encoding="utf-8") as source:
        rows = json.load(source)["rows"]

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    computed = {}
    for condition, condition_rows in sorted(grouped.items()):
        count = len(condition_rows)
        reader_target = 0
        reader_donor = 0
        for row in condition_rows:
            answers = {passage["id"]: passage["answer"] for passage in row["retrieval"]["passages"]}
            answer = normalized(row["reader_answer"])
            reader_target += answer == normalized(answers[row["target_passage_id"]])
            reader_donor += answer == normalized(answers[row["donor_passage_id"]])
        computed[condition] = {
            "n": count,
            "top_target_rate": sum(row["top_passage_id"] == row["target_passage_id"] for row in condition_rows) / count,
            "top_donor_rate": sum(row["top_passage_id"] == row["donor_passage_id"] for row in condition_rows) / count,
            "reader_target_exact_match": reader_target / count,
            "reader_donor_exact_match": reader_donor / count,
            "mean_verifier_passage_support": sum(row["verifier_support_probability"] for row in condition_rows) / count,
        }

    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["conditions"]
    for condition, metrics in computed.items():
        for metric, actual in metrics.items():
            reference = expected[condition][metric]
            if isinstance(actual, float):
                if abs(actual - reference) > 1e-9:
                    raise AssertionError(f"{condition}.{metric}: {actual} != {reference}")
            elif actual != reference:
                raise AssertionError(f"{condition}.{metric}: {actual} != {reference}")
    print(json.dumps(computed, indent=2, sort_keys=True))
    print(f"\nVerified against {EXPECTED}: all metrics match within 1e-9.")


if __name__ == "__main__":
    main()
