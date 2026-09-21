from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import benchmark_summary, build_examples, load_zsre_records, save_jsonl, validate_benchmark_size


def _excluded_entities(paths: list[Path]) -> set[str]:
    entities = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    entities.add(json.loads(line)["entity_id"])
    return entities


def main() -> None:
    parser = argparse.ArgumentParser(description="Build entity-disjoint contrastive relation-transfer data from zsRE")
    parser.add_argument("--input", type=Path, required=True, help="official zsRE TSV or normalized JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allow-reduced", action="store_true")
    parser.add_argument("--inspect-only", action="store_true", help="write coverage diagnostics without materialising an invalid benchmark")
    parser.add_argument("--max-entities", type=int, default=600, help="bounded deterministic sample from the official multi-GB raw export")
    parser.add_argument("--candidate-multiplier", type=int, default=250, help="P19 candidate reservoir multiplier for bounded raw exports")
    parser.add_argument("--evidence-mode", choices=("raw", "controlled"), default="raw",
                        help="raw source sentences, or relation-exclusive factual statements with source provenance")
    parser.add_argument("--exclude-data", type=Path, action="append", default=[],
                        help="JSONL whose entity_id values must not appear in this benchmark")
    args = parser.parse_args()

    excluded_entities = _excluded_entities(args.exclude_data)
    records = load_zsre_records(
        args.input,
        max_entities=args.max_entities,
        candidate_multiplier=args.candidate_multiplier,
        evidence_mode=args.evidence_mode,
        excluded_entity_ids=excluded_entities,
    )
    if not records:
        raise ValueError("no positive zsRE facts could be parsed from input")
    examples = build_examples(records, evidence_mode=args.evidence_mode)
    details = {
        "source": str(args.input),
        "output": str(args.output),
        "records_parsed": len(records),
        "max_entities": args.max_entities,
        "candidate_multiplier": args.candidate_multiplier,
        "evidence_mode": args.evidence_mode,
        "excluded_entity_count": len(excluded_entities),
        "exclude_data": [str(path) for path in args.exclude_data],
    }
    manifest = args.manifest or args.output.with_suffix(".manifest.json")
    if args.inspect_only:
        summary = benchmark_summary(examples) | details | {"inspection_only": True}
        manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    summary = validate_benchmark_size(examples, allow_reduced=args.allow_reduced) | details
    save_jsonl(args.output, examples)
    manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
