# QA relation-transfer reproducibility snapshot

This repository preserves the code, frozen configuration, aggregate results,
and held-out trace used for the final end-to-end experiment reported in the
thesis *Constraint-Weitergabe in Multi-Agenten-Systemen: Eine mechanistische
Fallstudie zur dokumentbasierten Fragebeantwortung*.

## Status and provenance

Version 1.0.0 is the post-experiment archival snapshot assembled on 2026-09-21.
Version 1.1.0 adds the corrected Verifier replication run from
2026-09-23. The historical release remains unchanged. Neither version is a Git
commit captured at the original 2026-09-08 execution time.

The corrected run records Python 3.12.14, PyTorch
`2.9.1+rocm7.2.1.gitff65f5bc`, HIP `7.2.53211-e1a6bc5663`, Transformers
5.14.1, local Arch ROCm 7.2.4 libraries, and an AMD Radeon RX 9070 XT.

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
- `artifacts/audit/`: strict split and entity-clustered robustness audits.
- `CHECKSUMS.sha256`: integrity hashes for every preserved file.

## Main experimental facts

The holdout contains 4,000 entities and 24,000 directed examples. Every
example has six candidate passages. The entity-disjoint split contains 16,488
train, 3,834 calibration, and 3,678 test examples. Layer 1 was selected only
on calibration and then frozen before the held-out test.

The corrected Verifier was trained from `cross-encoder/nli-deberta-v3-base`
for two epochs over 83,437 question-passage pairs. Every retained passage
belongs to a train entity; 15,491 candidate pairs from calibration or test
entities were excluded. The Retriever was
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

The historical sequence was: prepare the deterministic dataset, train the
Verifier, evaluate layers 1–6 on calibration, freeze layer 1, and evaluate it
on test. A later audit found that globally selected external negative passages
had exposed the historical Verifier to reserved entities. On 2026-09-23 the
Verifier was retrained with strict train-entity passage filtering and its
outputs were recomputed on the already observed test split. The frozen layer,
Retriever outputs, and Reader outputs were unchanged. This corrected run is a
post-hoc robustness check, not a second blind held-out test.

## Large omitted artifacts

The following retained local artifacts are omitted from GitHub because they
are large and are not required to audit the published aggregate metrics:

- final holdout JSONL — SHA-256
  `e57e8675f100eb8b9918082f6fd2069d07f3b9e937ae3349569e0ab1d8c12d85`;
- corrected uncompressed held-out trace — SHA-256
  `63fc19253187a6544918b056800d85edf7352c2d59909cacbcdb6587db7b0c72`;
- fine-tuned Verifier checkpoint — SHA-256
  `b5ea6c7b069dac901d6cb91ea916d7de3ac9f00e971f574dfb0b87144f974060`.

The compressed held-out trace is included and expands to the exact
uncompressed trace hash listed above.
