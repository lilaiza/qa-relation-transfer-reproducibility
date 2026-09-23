from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .evaluation import choose_layer, choose_relation_semantics_layer, relation_semantics_summary, relation_semantics_verdict, selective_transfer_verdict, summarize_layer


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _direction(row: dict) -> str:
    donor_id = row["donor_passage_id"]
    donor = next(passage for passage in row["retrieval"]["passages"] if passage["id"] == donor_id)
    return f"{row['relation_id']}->{donor['evidence_relation_id']}"


def _directional_summaries(rows: list[dict], *, layer: int, summary_function=summarize_layer) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_direction(row)].append(row)
    return {
        direction: summary_function(direction_rows, layer=layer)
        for direction, direction_rows in sorted(grouped.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise paired relational donor-transfer experiments")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--select-layer", type=Path)
    parser.add_argument("--test", action="store_true", help="apply the pre-registered held-out verdict")
    parser.add_argument(
        "--selection-objective",
        choices=("selective", "relation-semantics"),
        default="relation-semantics",
        help="layer-selection and verdict objective; the thesis-facing protocol uses relation-semantics",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*_layer*.json"))
    if not files:
        raise FileNotFoundError(f"no intervention results in {args.input_dir}")
    payloads = [_load(path) for path in files]
    summary_function = relation_semantics_summary if args.selection_objective == "relation-semantics" else summarize_layer
    summaries = [
        summary_function(payload["rows"], layer=payload.get("configuration", {}).get("layer"))
        for payload in payloads
    ]
    report = {
        "layers": summaries,
        "directions": {
            str(payload.get("configuration", {}).get("layer")): _directional_summaries(
                payload["rows"], layer=payload["configuration"]["layer"], summary_function=summary_function
            )
            for payload in payloads
        },
    }
    if args.select_layer:
        selection = choose_relation_semantics_layer(summaries) if args.selection_objective == "relation-semantics" else choose_layer(summaries)
        args.select_layer.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        report["frozen_selection"] = selection
    if args.test:
        if len(summaries) != 1:
            raise ValueError("held-out verdict requires exactly one frozen layer result")
        report["held_out_verdict"] = relation_semantics_verdict(summaries[0]) if args.selection_objective == "relation-semantics" else selective_transfer_verdict(summaries[0])
        selected_layer = summaries[0]["layer"]
        verdict_function = relation_semantics_verdict if args.selection_objective == "relation-semantics" else selective_transfer_verdict
        report["held_out_verdict_by_direction"] = {
            direction: verdict_function(summary)
            for direction, summary in report["directions"][str(selected_layer)].items()
        }
    output = args.output or args.input_dir / "transfer_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
