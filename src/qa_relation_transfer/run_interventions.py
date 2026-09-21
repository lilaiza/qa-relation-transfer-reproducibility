from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from .agents import DenseRelationRetriever, EvidenceVerifier, ExtractiveReader, donor_for_condition
from .dataset import load_jsonl
from .pipeline import RelationQAPipeline
from .schemas import PatchCondition, Split
from .smoke import SMOKE_LAYERS, SMOKE_PER_DIRECTION, file_sha256, stratified_smoke_examples


PATCH_CONDITIONS = (
    PatchCondition.BASELINE,
    PatchCondition.SELF_PATCH,
    PatchCondition.SAME_ENTITY_DONOR,
    PatchCondition.SAME_ENTITY_CONTROL,
    PatchCondition.DIFFERENT_ENTITY_DONOR,
)


def _frozen_layer(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    layer = payload.get("layer")
    if not isinstance(layer, int):
        raise ValueError(f"frozen layer file {path} contains no integer layer")
    return layer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run masked relation-span donor patches")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=[split.value for split in Split], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--frozen-layer-file", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--verifier-model")
    parser.add_argument("--skip-reader", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--profile", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--smoke-per-direction", type=int, default=SMOKE_PER_DIRECTION)
    args = parser.parse_args()

    split = Split(args.split)
    if split == Split.TEST and args.frozen_layer_file is None:
        raise ValueError("test evaluation requires --frozen-layer-file selected on calibration")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but PyTorch cannot access CUDA/ROCm")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory {args.output_dir}")

    all_examples = load_jsonl(args.data)
    split_examples = [example for example in all_examples if example.split == split]
    if args.profile == "smoke":
        if split != Split.CALIBRATION:
            raise ValueError("the smoke profile is restricted to the calibration split")
        if args.frozen_layer_file is not None:
            raise ValueError("the smoke profile cannot use a frozen layer")
        if args.layers is not None and tuple(args.layers) != SMOKE_LAYERS:
            raise ValueError(f"the smoke profile uses fixed layers {list(SMOKE_LAYERS)}")
        if args.limit is not None:
            raise ValueError("use --smoke-per-direction instead of --limit with the smoke profile")
        if args.verifier_model is not None:
            raise ValueError("the smoke profile is retrieval-only and cannot use a verifier")
        examples, coverage = stratified_smoke_examples(split_examples, per_direction=args.smoke_per_direction)
        layers = list(SMOKE_LAYERS)
        reader_enabled = False
        verifier_model = None
    else:
        examples = split_examples[: args.limit] if args.limit else split_examples
        coverage = None
        layers = [_frozen_layer(args.frozen_layer_file)] if args.frozen_layer_file else (args.layers or list(range(1, 7)))
        reader_enabled = not args.skip_reader
        verifier_model = args.verifier_model
    if not examples:
        raise ValueError(f"no examples in split {split.value}")
    examples_by_id = {example.id: example for example in all_examples}

    started_at = datetime.now(UTC)
    initialization_started = time.perf_counter()
    retriever = DenseRelationRetriever(device=args.device)
    if any(not 1 <= layer <= retriever.available_layers() for layer in layers):
        raise ValueError(f"layers must be in 1..{retriever.available_layers()}")
    reader = ExtractiveReader(device=args.device) if reader_enabled else None
    verifier = EvidenceVerifier(verifier_model, device=args.device) if verifier_model else None
    pipeline = RelationQAPipeline(retriever, reader, verifier)
    initialization_seconds = time.perf_counter() - initialization_started
    args.output_dir.mkdir(parents=True)

    layer_seconds: dict[str, float] = {}
    for layer in layers:
        layer_started = time.perf_counter()
        rows = []
        for example in examples:
            for condition in PATCH_CONDITIONS:
                donor = donor_for_condition(example, examples_by_id, split_examples, condition)
                rows.append(pipeline.run(example, condition=condition, layer=None if condition == PatchCondition.BASELINE else layer, donor=donor))
        elapsed = time.perf_counter() - layer_started
        layer_seconds[str(layer)] = elapsed
        output = args.output_dir / f"{split.value}_layer{layer}.json"
        output.write_text(json.dumps({
            "configuration": {
                "data": str(args.data),
                "split": split.value,
                "layer": layer,
                "device": args.device,
                "profile": args.profile,
                "reader_enabled": reader is not None,
                "verifier_model": verifier_model,
                "frozen_layer_file": str(args.frozen_layer_file) if args.frozen_layer_file else None,
            },
            "timing": {"layer_wall_seconds": elapsed},
            "rows": rows,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"saved {output} ({len(rows)} rows)")

    device_metadata = {"requested": args.device, "torch": torch.__version__, "hip": torch.version.hip}
    if args.device == "cuda":
        device_metadata.update({"device_count": torch.cuda.device_count(), "device_name": torch.cuda.get_device_name(0)})
    metadata = {
        "kind": "intervention_run",
        "profile": args.profile,
        "started_at_utc": started_at.isoformat(),
        "configuration": {
            "data": str(args.data),
            "data_sha256": file_sha256(args.data),
            "split": split.value,
            "layers": layers,
            "device": device_metadata,
            "reader_enabled": reader is not None,
            "verifier_model": verifier_model,
            "frozen_layer_file": str(args.frozen_layer_file) if args.frozen_layer_file else None,
        },
        "full_split_examples": len(split_examples),
        "timing": {
            "model_initialization_seconds": initialization_seconds,
            "layer_seconds": layer_seconds,
            "total_wall_seconds": time.perf_counter() - initialization_started,
        },
    }
    if coverage is not None:
        metadata["smoke"] = {"coverage": coverage, "selected_example_ids": [example.id for example in examples]}
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
