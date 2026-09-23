# Frozen experimental protocol

## Research question

Can a relation-specific state in the Retrieval Agent causally redirect
document retrieval from target relation A to donor relation B while the entity
and all document embeddings remain fixed?

## Experimental unit

One unit is a directed entity-matched pair `A -> B`.  The target question asks
about relation A for entity E.  The donor question asks about relation B for
the same entity E.  Relation C for E is a same-entity semantic control.

## Data rules

- Source: positive zsRE facts only, restricted to `P19`, `P569`, `P106`, and
  `P27`.
- A valid entity has three relation-specific, distinct, answer-exclusive
  evidence passages. The `raw` mode uses the original zsRE sentence. The
  documented `controlled` mode renders a relation-exclusive factual statement
  from the source entity/relation/answer and retains the original row and
  sentence as provenance in the JSONL passage.
- Split by entity with deterministic seed 42: train 70%, calibration 15%,
  test 15%.
- Candidate documents are always the same six passages for a target run.
- A full benchmark needs at least 200 valid entities and 30 examples for every
  calibration/test relation direction.  The documented reduced tier requires
  120 entities and 20 examples per direction.

## Intervention rules

- Documents are encoded without interventions.
- A patch changes only contextual hidden states whose tokenizer offsets overlap
  the canonical relation phrase.
- A tokenwise donor copy is used when source and target relation spans have
  equal token counts; otherwise the donor relation-span mean is copied only to
  target relation tokens.
- Conditions are baseline, self patch, same-entity donor, same-entity relation
  C control, and same-relation/different-entity donor control.
- Zero ablation, global means, coordinate rotations and all-question patches
  are excluded from the thesis-facing experiment.

## Selection and held-out test

1. Evaluate layers 1–6 on calibration only.
2. Select exactly one layer by the largest mean same-entity donor advantage
   over the same-entity relation-C control (`B - C`).
3. Write the selected layer to `frozen_layer.json`.
4. Run the identical five conditions on test with that file.  The test runner
   rejects an unfrozen layer.

## Primary outcome and claim

For target evidence `E_A` and donor evidence `E_B`:

`delta = [score(E_B) - score(E_A)]patch - [score(E_B) - score(E_A)]baseline`

Selective transfer is supported only when held-out same-entity donor patches
have a positive bootstrap lower confidence bound, paired permutation p < .05,
and a larger mean delta than both natural controls.  The report must include
all directions separately.  Reader and Verifier changes are secondary evidence
of downstream propagation.

## Corrected Verifier training

The Verifier uses questions from the train split. A question-passage pair is
retained only if the passage entity also maps to the train split under the
same deterministic entity split. This produces 83,437 pairs: 16,488 positives
and 66,949 negatives. Filtering rather than replacing reserved external
negatives leaves between three and six pairs per training example.

The 2026-09-23 corrected evaluation reuses historical layer 1 without new
selection. Because the test results were already observed, it is reported as
a post-hoc correction and robustness analysis. It does not restore the status
of an untouched blind test for downstream Verifier probabilities.
