# QA relation-transfer reproducibility snapshot

This repository preserves the code, frozen configuration, aggregate results,
and held-out trace used for the final end-to-end experiment reported in the
thesis *Constraint-Weitergabe in Multi-Agenten-Systemen: Eine mechanistische
Fallstudie zur dokumentbasierten Fragebeantwortung*.

## Status and provenance

This is a post-experiment archival snapshot assembled on 2026-09-21 from the
project files that remained after the runs. It is not a Git commit captured at
the original execution time. The historical record did not retain the exact
shell command, Python version, ROCm package version, or Transformers version;
these values are therefore not reconstructed retrospectively.

The preserved metadata records PyTorch
`2.9.1+rocm7.2.1.gitff65f5bc`, HIP `7.2.53211-e1a6bc5663`, device `cuda`, and
an AMD Radeon RX 9070 XT. The experiment log documents all known limitations.

## Contents

- `src/`: experiment implementation.
- `tests/`: protocol and smoke tests.
- `data/`: manifests and checksums, but not the redistributable/raw datasets.
- `artifacts/calibration/`: layer-selection result and frozen layer.
- `artifacts/test/`: aggregate held-out results and the compressed row-level
  test trace.
- `artifacts/verifier/`: training metadata and lightweight model
  configuration. The 738 MB fine-tuned checkpoint is represented by checksum
  and is not stored in this repository.
- `docs/experiment_runs.md`: final experiment record reconstructed from the
  preserved run artifacts, with explicit provenance limitations.
- `docs/downstream_metrics.json`: independently recomputed downstream values.
- `CHECKSUMS.sha256`: integrity hashes for every preserved file.

## Main experimental facts

The holdout contains 4,000 entities and 24,000 directed examples. Every
example has six candidate passages. The entity-disjoint split contains 16,488
train, 3,834 calibration, and 3,678 test examples. Layer 1 was selected only
on calibration and then frozen before the held-out test.

The Verifier was trained from `cross-encoder/nli-deberta-v3-base` for two
epochs over 98,928 question-passage pairs. The Retriever was
`sentence-transformers/msmarco-MiniLM-L6-cos-v5`; the Reader was
`deepset/roberta-base-squad2`.

## Integrity and result verification

Verify the snapshot itself:

```bash
sha256sum --check CHECKSUMS.sha256
```

Recompute the downstream metrics from the preserved compressed trace:

```bash
python scripts/recompute_downstream.py
```

Run the unit tests after installing the project in a compatible environment:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Do not install the generic CPU PyTorch extra into an existing ROCm
environment. Install a PyTorch wheel compatible with the local ROCm, Python,
and GPU combination, then run `python scripts/check_rocm.py` before any GPU
campaign.

## Reproducing the GPU experiment

The raw zsRE source and the generated JSONL are not included. Obtain the zsRE
positive-example source, place it at `data/raw_data/positive_examples`, and
use the preparation command documented in `PROTOCOL.md` and
`docs/experiment_runs.md`. The generated final holdout must match:

```text
e57e8675f100eb8b9918082f6fd2069d07f3b9e937ae3349569e0ab1d8c12d85
```

The full sequence is: prepare the deterministic dataset, train the Verifier on
the train split, evaluate layers 1–6 on calibration, freeze the selected layer,
and evaluate that configuration once on test. The historical commands in the
experiment log that are marked as reconstructed must not be treated as exact
recorded shell history.

## Large omitted artifacts

The following retained local artifacts are omitted from GitHub because they
are large and are not required to audit the published aggregate metrics:

- final holdout JSONL — SHA-256
  `e57e8675f100eb8b9918082f6fd2069d07f3b9e937ae3349569e0ab1d8c12d85`;
- uncompressed held-out trace — SHA-256
  `a1c8df8cf0e5833065ac31ab94471685bc8b492739568fd52e7beb77459418e4`;
- fine-tuned Verifier checkpoint — SHA-256
  `9dd748facf04ddd37143098e2e4a8159081d441b72ea639f0219b73f8c82faa6`.

The compressed held-out trace is included and expands to the exact
uncompressed trace hash listed above.
