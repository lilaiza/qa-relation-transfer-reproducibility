from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .agents import VERIFIER_MODEL
from .dataset import load_jsonl, stable_seed
from .hf_config import hf_token
from .schemas import Split


def verifier_rows(examples):
    rows = []
    for example in examples:
        for passage in example.passages:
            rows.append((example.question, passage.text, int(passage.id == example.target_passage_id)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the Evidence Verifier on train entities only")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=VERIFIER_MODEL)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--profile", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--limit", type=int, help="maximum train examples; smoke default is 32")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but PyTorch cannot access CUDA/ROCm")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite verifier output {args.output}")

    all_train_examples = [example for example in load_jsonl(args.data) if example.split == Split.TRAIN]
    if args.profile == "smoke":
        limit = args.limit or 32
        epochs = args.epochs or 1
        batch_size = args.batch_size or 8
    else:
        limit = args.limit
        epochs = args.epochs or 2
        batch_size = args.batch_size or 16
    examples = sorted(all_train_examples, key=lambda example: (stable_seed("verifier-smoke", example.id, seed=args.seed), example.id))
    if limit:
        examples = examples[:limit]
    rows = verifier_rows(examples)
    if not rows:
        raise ValueError("no train verifier rows")
    started_at = datetime.now(UTC)
    initialization_started = time.perf_counter()
    token = hf_token()
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        token=token,
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(args.device)
    initialization_seconds = time.perf_counter() - initialization_started

    def collate(batch):
        questions, passages, labels = zip(*batch, strict=True)
        encoded = tokenizer(list(questions), list(passages), padding=True, truncation=True, max_length=384, return_tensors="pt")
        encoded["labels"] = torch.tensor(labels)
        return {key: value.to(args.device) for key, value in encoded.items()}

    loader = DataLoader(rows, batch_size=batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    training_started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for batch in loader:
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    args.output.mkdir(parents=True)
    if args.profile == "formal":
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
    metadata = {
        "profile": args.profile,
        "started_at_utc": started_at.isoformat(),
        "base_model": args.model,
        "split": Split.TRAIN.value,
        "full_train_examples": len(all_train_examples),
        "examples": len(examples),
        "pairs": len(rows),
        "selected_example_ids": [example.id for example in examples],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "model_initialization_seconds": initialization_seconds,
        "training_seconds": time.perf_counter() - training_started,
        "duration_seconds": time.perf_counter() - initialization_started,
        "checkpoint_saved": args.profile == "formal",
    }
    (args.output / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
