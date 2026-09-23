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

## `2026-09-23__verifier_strict_train_entities__v1`

- **Status:** Completed corrective Verifier training and post-hoc test
  evaluation; executed by the thesis author.
- **Reason:** The historical training questions were train-only, but globally
  chosen external negative passages exposed the model to all 639 calibration
  and 613 test entities. This contradicted the claimed entity-level isolation.
- **Correction:** Retain a question-passage pair only when the passage entity
  also belongs to train. The run used seed 42, two epochs, batch size 16 and
  learning rate 2e-5.
- **Training data:** 16,488 questions; 83,437 retained pairs from 2,748 train
  passage entities: 16,488 positive and 66,949 negative. It excluded 15,491
  of the 98,928 original candidate pairs. Examples retain between three and
  six pairs because reserved negatives are filtered rather than replaced.
- **Environment:** Python 3.12.14; PyTorch
  `2.9.1+rocm7.2.1.gitff65f5bc`; HIP `7.2.53211-e1a6bc5663`;
  Transformers 5.14.1; local Arch ROCm 7.2.4 libraries; AMD Radeon RX 9070 XT
  as `cuda:0`.
- **Timing:** Training began `2026-09-23T13:37:50Z`; training took 1,022.93 s
  and total training-process time was 1,024.85 s. Corrected evaluation began
  `2026-09-23T14:04:24Z` and took 332.13 s total.
- **Frozen configuration:** Historical layer 1 was reused without reselection.
  Test was not used to select a layer.
- **Audit:** All 18,390 evaluated Top-1 passages belong to test entities and
  none appeared in corrected training. Retriever rankings/scores and Reader
  outputs equal the historical trace row for row. No Verifier decision at
  threshold 0.5 changed.
- **Corrected means:** Baseline and self patch 0.959749518; same-entity donor
  and relation-C control 0.026649742; different-entity donor 0.026649741.
- **Interpretation:** The exposure flaw does not explain the downstream H4
  pattern. This is a post-hoc correction using an already observed test split,
  not a second blind test. The variable number of retained negatives also
  limits causal attribution of tiny probability changes solely to removing
  reserved entities.

## `2026-09-23__entity_clustered_inference__v1`

- **Status:** Completed post-hoc CPU robustness analysis on the corrected
  stored trace; no model inference.
- **Reason:** The 3,678 directed examples comprise six observations for each
  of 613 target entities. The historical inference resampled examples rather
  than keeping each entity cluster together.
- **Method:** Average the six paired effects within each entity, bootstrap the
  613 entity means with 10,000 samples, and perform 50,000 entity-level sign
  permutations; seed 42.
- **Results:** Same-entity donor minus baseline = 0.36361, 95% interval
  [0.35891, 0.36835]; relation C minus baseline = 0.17299
  [0.17066, 0.17534]; donor B minus relation C = 0.19062
  [0.18816, 0.19311]; different-entity donor minus baseline = 0.36269
  [0.35809, 0.36735]. All four sign-permutation p-values are 0.00002.
- **Interpretation:** The aggregate conclusions remain unchanged when target
  entity is the resampling unit. This is a post-hoc robustness analysis and
  does not retrospectively replace the pre-specified example-level inference.
- **Artifacts:** `artifacts/audit/clustered_inference.json` and
  `scripts/audit_clustered_inference.py`.
