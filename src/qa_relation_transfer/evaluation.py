from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def retrieval_metrics(row: dict) -> dict[str, float | bool]:
    ranked = row["retrieval"]["passages"]
    target_id = row["target_passage_id"]
    donor_id = row["donor_passage_id"]
    target = next(passage for passage in ranked if passage["id"] == target_id)
    donor = next(passage for passage in ranked if passage["id"] == donor_id)
    target_rank, donor_rank = target["rank"], donor["rank"]
    return {
        "target_score": float(target["score"]),
        "donor_score": float(donor["score"]),
        "target_rank": target_rank,
        "donor_rank": donor_rank,
        "target_recall_at_1": target_rank == 1,
        "target_recall_at_3": target_rank <= 3,
        "target_mrr": 1.0 / target_rank,
        "donor_top_1": donor_rank == 1,
        "donor_advantage": float(donor["score"] - target["score"]),
    }


def bootstrap_interval(values: Iterable[float], *, samples: int = 2_000, seed: int = 42) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    means = generator.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_permutation_pvalue(values: Iterable[float], *, samples: int = 20_000, seed: int = 42) -> float:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return float("nan")
    observed = abs(float(array.mean()))
    generator = np.random.default_rng(seed)
    signs = generator.choice(np.array([-1.0, 1.0]), size=(samples, len(array)))
    simulated = np.abs((signs * array).mean(axis=1))
    return float((np.count_nonzero(simulated >= observed) + 1) / (samples + 1))


def paired_condition_effect(baseline: list[dict], intervention: list[dict]) -> dict:
    baseline_by_id = {row["example_id"]: row for row in baseline}
    intervention_by_id = {row["example_id"]: row for row in intervention}
    ids = sorted(set(baseline_by_id) & set(intervention_by_id))
    if not ids:
        raise ValueError("no overlapping example IDs for paired comparison")
    differences = []
    for example_id in ids:
        before = retrieval_metrics(baseline_by_id[example_id])
        after = retrieval_metrics(intervention_by_id[example_id])
        differences.append(after["donor_advantage"] - before["donor_advantage"])
    low, high = bootstrap_interval(differences)
    return {
        "n": len(differences),
        "mean_donor_advantage_change": float(np.mean(differences)),
        "bootstrap_low": low,
        "bootstrap_high": high,
        "permutation_pvalue": paired_permutation_pvalue(differences),
        "positive_rate": float(np.mean(np.asarray(differences) > 0)),
    }


def paired_condition_contrast(first: list[dict], second: list[dict]) -> dict:
    """Paired difference in donor advantage between two intervention conditions."""
    first_by_id = {row["example_id"]: row for row in first}
    second_by_id = {row["example_id"]: row for row in second}
    ids = sorted(set(first_by_id) & set(second_by_id))
    if not ids:
        raise ValueError("no overlapping example IDs for paired contrast")
    differences = [
        retrieval_metrics(first_by_id[example_id])["donor_advantage"]
        - retrieval_metrics(second_by_id[example_id])["donor_advantage"]
        for example_id in ids
    ]
    low, high = bootstrap_interval(differences)
    return {
        "n": len(differences),
        "mean_donor_advantage_difference": float(np.mean(differences)),
        "bootstrap_low": low,
        "bootstrap_high": high,
        "permutation_pvalue": paired_permutation_pvalue(differences),
    }


def summarize_layer(rows: list[dict], *, layer: int | None = None) -> dict:
    """Summarise one layer artifact.

    Baseline rows intentionally have ``layer=None`` because they are
    unpatched.  The enclosing artifact configuration is therefore the
    authoritative layer label when it is available.
    """
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    baseline = by_condition.get("baseline", [])
    if not baseline:
        raise ValueError("layer result has no baseline rows")
    row_layers = {row.get("layer") for row in rows if row.get("layer") is not None}
    if layer is None:
        if len(row_layers) != 1:
            raise ValueError(f"layer result must contain exactly one intervention layer, found {sorted(row_layers)}")
        layer = next(iter(row_layers))
    if not isinstance(layer, int):
        raise ValueError("layer result has no integer layer label")
    if row_layers and row_layers != {layer}:
        raise ValueError(f"layer label {layer} does not match intervention rows {sorted(row_layers)}")
    effects = {
        condition: paired_condition_effect(baseline, values)
        for condition, values in by_condition.items()
        if condition != "baseline"
    }
    treatment = effects.get("same_entity_donor")
    if treatment is None:
        raise ValueError("layer result has no same_entity_donor rows")
    control_means = [
        effects[name]["mean_donor_advantage_change"]
        for name in ("same_entity_control", "different_entity_donor")
        if name in effects
    ]
    specificity_margin = treatment["mean_donor_advantage_change"] - (sum(control_means) / len(control_means))
    return {
        "layer": layer,
        "n": len(baseline),
        "baseline": {
            "target_recall_at_1": float(np.mean([retrieval_metrics(row)["target_recall_at_1"] for row in baseline])),
            "target_recall_at_3": float(np.mean([retrieval_metrics(row)["target_recall_at_3"] for row in baseline])),
            "target_mrr": float(np.mean([retrieval_metrics(row)["target_mrr"] for row in baseline])),
        },
        "effects": effects,
        "specificity_margin": specificity_margin,
    }


def choose_layer(summaries: Iterable[dict]) -> dict:
    candidates = list(summaries)
    if not candidates:
        raise ValueError("no calibration summaries supplied")
    selected = max(candidates, key=lambda summary: (summary["specificity_margin"], -summary["layer"]))
    return {
        "layer": selected["layer"],
        "selection_metric": "same_entity_donor advantage minus natural-control mean",
        "calibration_specificity_margin": selected["specificity_margin"],
        "calibration_summary": selected,
    }


def selective_transfer_verdict(test_summary: dict) -> dict:
    effects = test_summary["effects"]
    treatment = effects["same_entity_donor"]
    controls = [effects[name] for name in ("same_entity_control", "different_entity_donor") if name in effects]
    controls_below_treatment = all(
        treatment["mean_donor_advantage_change"] > control["mean_donor_advantage_change"] for control in controls
    )
    positive = treatment["bootstrap_low"] > 0 and treatment["permutation_pvalue"] < 0.05
    return {
        "selective_transfer_supported": bool(positive and controls_below_treatment),
        "retrieval_propagation_supported": bool(positive),
        "limitations": (
            "A positive retrieval verdict establishes only relation-directed movement in the Retrieval Agent; "
            "Reader and Verifier outcomes remain secondary propagation readouts."
        ),
    }


def relation_semantics_summary(rows: list[dict], *, layer: int | None = None) -> dict:
    summary = summarize_layer(rows, layer=layer)
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    contrast = paired_condition_contrast(
        by_condition["same_entity_donor"], by_condition["same_entity_control"]
    )
    summary["relation_semantics"] = {
        "donor_vs_relation_control": contrast,
        "portable_donor_effect": summary["effects"]["different_entity_donor"],
    }
    return summary


def choose_relation_semantics_layer(summaries: Iterable[dict]) -> dict:
    selected = max(summaries, key=lambda item: item["relation_semantics"]["donor_vs_relation_control"]["mean_donor_advantage_difference"])
    return {"layer": selected["layer"], "selection_metric": "same-entity donor advantage minus same-entity relation-C control", "calibration_relation_semantics": selected}


def relation_semantics_verdict(summary: dict) -> dict:
    contrast = summary["relation_semantics"]["donor_vs_relation_control"]
    portable = summary["relation_semantics"]["portable_donor_effect"]
    supported = contrast["bootstrap_low"] > 0 and contrast["permutation_pvalue"] < 0.05 and portable["bootstrap_low"] > 0 and portable["permutation_pvalue"] < 0.05
    return {"relation_semantics_supported": supported, "portable_relation_code_supported": portable["bootstrap_low"] > 0 and portable["permutation_pvalue"] < 0.05}
