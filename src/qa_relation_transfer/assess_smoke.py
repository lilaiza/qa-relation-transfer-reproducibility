from __future__ import annotations

import argparse
import json
from pathlib import Path

from .smoke import SMOKE_TARGET_SECONDS, assess_smoke


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an advisory assessment for a retrieval smoke run")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-seconds", type=float, default=SMOKE_TARGET_SECONDS)
    args = parser.parse_args()

    metadata_path = args.input_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"smoke metadata not found: {metadata_path}")
    layer_paths = sorted(args.input_dir.glob("calibration_layer*.json"))
    if not layer_paths:
        raise FileNotFoundError(f"no calibration layer outputs found in {args.input_dir}")
    report = assess_smoke(_load(metadata_path), [_load(path) for path in layer_paths], target_seconds=args.target_seconds)
    output = args.output or args.input_dir / "smoke_assessment.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
