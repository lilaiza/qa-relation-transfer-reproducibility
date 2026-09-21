# Final experiment record

## `2026-09-08__end_to_end_complete_layer1__v1`

- **Status:** Completed formal run family: Verifier training, calibration, and
  one frozen held-out test.
- **Date and operator:** 2026-09-08; executed by the thesis author.
- **Objective:** Determine whether a localized, relation-specific causal
  redirection in the Retriever propagates without further intervention to the
  Reader and Verifier on a third entity-disjoint holdout.
- **Historical Git state:** No readable Git repository or execution-time
  commit survived in the archived workspace. The current public repository is
  a post-experiment preservation snapshot, not the original run commit.
- **Data:** `data/zsre_end_to_end_holdout.jsonl`; SHA-256
  `e57e8675f100eb8b9918082f6fd2069d07f3b9e937ae3349569e0ab1d8c12d85`.
  The holdout contains 4,000 entities disjoint from the 8,000 entities used in
  the two earlier datasets: 16,488 train, 3,834 calibration, and 3,678 test
  examples. Each directed example contains six candidate passages.
- **Seed:** 42 for dataset preparation and entity-level splits. Statistical
  procedures follow the frozen protocol.
- **Models:** Retriever
  `sentence-transformers/msmarco-MiniLM-L6-cos-v5`; Reader
  `deepset/roberta-base-squad2`; Verifier initialized from
  `cross-encoder/nli-deberta-v3-base` and fine-tuned for two epochs with batch
  size 16 and learning rate 2e-5.
- **Intervention conditions:** Baseline, self patch, same-entity donor B,
  same-entity relation-C control, and different-entity donor B.
- **Recorded environment:** PyTorch
  `2.9.1+rocm7.2.1.gitff65f5bc`; HIP `7.2.53211-e1a6bc5663`; AMD Radeon RX
  9070 XT through PyTorch's `cuda` interface. The historical Python,
  Transformers, and ROCm package versions were not recorded completely and
  are not reconstructed after the fact.
- **Historical command:** The literal shell command was not retained. The
  stored configuration shows the following sequence: train the Verifier on
  the train split; evaluate Retriever layers 1–6 on calibration with Reader
  and Verifier enabled; select the layer by the same-entity donor-B minus
  relation-C contrast; serialize the frozen selection; run the test split once
  using only that frozen configuration; aggregate the held-out results. This
  description is a reconstruction from artifacts, not recorded shell history.
- **Timing:** Verifier training started at `2026-09-07T23:41:58Z` and lasted
  1,217.89 s. Calibration started at `2026-09-08T00:02:52Z` and lasted
  1,928.24 s. Test started at `2026-09-08T07:54:57Z` and lasted 322.81 s.
- **Frozen selection:** Layer 1, chosen exclusively from 3,834 calibration
  examples. The test split was not used for layer selection.
- **Held-out retrieval results:** n=3,678; donor B minus relation C = 0.19062,
  95% CI [0.18637, 0.19463], permutation p-value 0.00005; cross-entity donor B
  effect = 0.36269, 95% CI [0.35867, 0.36666], permutation p-value 0.00005.
  All twelve directed relation pairs have positive B-minus-C contrasts and
  positive cross-entity donor effects.
- **Held-out downstream results:** Baseline target top-1 = 0.95976,
  Reader-A exact match = 0.95922, mean Verifier support = 0.95976.
  Same-entity donor-B top-1 = 0.95160, Reader-A exact match = 0.02664, mean
  Verifier support = 0.02664.
- **Interpretation:** The localized intervention selectively redirects
  retrieval, and the resulting context change reaches the Reader and Verifier.
  This downstream propagation does not establish an independently localized
  causal relation representation inside either downstream model.
- **Known limitation:** The preserved artifacts support independent metric
  recomputation but cannot guarantee bit-for-bit reconstruction of the
  historical execution because the original commit, literal command, and
  complete environment versions were not retained.

## Artifact map

- Frozen calibration decision:
  `artifacts/calibration/frozen_layer.json`.
- Calibration aggregate:
  `artifacts/calibration/transfer_summary.json`.
- Held-out aggregate:
  `artifacts/test/transfer_summary.json`.
- Compressed row-level held-out trace:
  `artifacts/test/test_layer1.json.gz`.
- Verifier training metadata:
  `artifacts/verifier/training_metadata.json`.
- Independently recomputed downstream metrics:
  `docs/downstream_metrics.json`.

