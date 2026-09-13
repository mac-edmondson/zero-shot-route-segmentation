"""Evaluate route grouping independently from hold detection."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from ..interfaces.data_models import Hold, ImageRecord, Route
from ..route_discriminator.route_discriminator import RouteDiscriminator
from .results import (
    ConditionEvaluation,
    EvaluationCaseResult,
    EvaluationReport,
    EvaluationStatus,
    EvaluationTarget,
)


def _ground_truth_routes(holds: Sequence[Hold]) -> tuple[Route, ...]:
    groups: dict[int, set[Hold]] = {}
    for hold in holds:
        route_id = hold.attributes.get("route_id")
        if isinstance(route_id, int) and not isinstance(route_id, bool):
            groups.setdefault(route_id, set()).add(hold)
    return tuple(Route(group, route_id) for route_id, group in sorted(groups.items()))


def _labels(routes: Sequence[Route], holds: Sequence[Hold]) -> list[int | None]:
    labels: dict[Hold, int] = {}
    for route_index, route in enumerate(routes):
        for hold in route.holds:
            labels[hold] = route_index
    return [labels.get(hold) for hold in holds]


def _combination(value: int) -> int:
    return value * (value - 1) // 2


def _clustering_metrics(
    predicted: Sequence[Route], truth: Sequence[Route], holds: Sequence[Hold]
) -> dict[str, float | int]:
    predicted_labels = _labels(predicted, holds)
    truth_labels = _labels(truth, holds)
    valid = [
        (prediction, actual)
        for prediction, actual in zip(predicted_labels, truth_labels, strict=True)
        if prediction is not None and actual is not None
    ]
    predicted_labels = [prediction for prediction, _ in valid]
    truth_labels = [actual for _, actual in valid]
    total_pairs = _combination(len(valid))
    predicted_pairs = sum(
        _combination(count) for count in Counter(predicted_labels).values()
    )
    truth_pairs = sum(_combination(count) for count in Counter(truth_labels).values())
    same_pairs = sum(
        _combination(count)
        for count in (
            sum(
                1
                for prediction, actual in valid
                if prediction == label and actual == truth_label
            )
            for label in set(predicted_labels)
            for truth_label in set(truth_labels)
        )
    )
    pair_precision = (
        same_pairs / predicted_pairs
        if predicted_pairs
        else (1.0 if not truth_pairs else 0.0)
    )
    pair_recall = (
        same_pairs / truth_pairs
        if truth_pairs
        else (1.0 if not predicted_pairs else 0.0)
    )
    pair_f1 = (
        2 * pair_precision * pair_recall / (pair_precision + pair_recall)
        if pair_precision + pair_recall
        else 0.0
    )
    contingency: dict[tuple[int, int], int] = Counter(
        zip(predicted_labels, truth_labels, strict=True)
    )
    predicted_sizes = Counter(predicted_labels)
    truth_sizes = Counter(truth_labels)
    purity = (
        sum(
            max(
                (
                    count
                    for (prediction, _), count in contingency.items()
                    if prediction == label
                ),
                default=0,
            )
            for label in predicted_sizes
        )
        / len(valid)
        if valid
        else 0.0
    )
    sum_comb = sum(_combination(count) for count in contingency.values())
    expected = (
        (
            sum(_combination(count) for count in predicted_sizes.values())
            * sum(_combination(count) for count in truth_sizes.values())
        )
        / total_pairs
        if total_pairs
        else 0.0
    )
    maximum = (
        sum(_combination(count) for count in predicted_sizes.values())
        + sum(_combination(count) for count in truth_sizes.values())
    ) / 2
    ari = (sum_comb - expected) / (maximum - expected) if maximum != expected else 1.0
    n = len(valid)
    mutual_information = (
        sum(
            (count / n)
            * math.log(
                (count * n) / (predicted_sizes[prediction] * truth_sizes[actual])
            )
            for (prediction, actual), count in contingency.items()
        )
        if n
        else 0.0
    )
    entropy_predicted = (
        -sum((count / n) * math.log(count / n) for count in predicted_sizes.values())
        if n
        else 0.0
    )
    entropy_truth = (
        -sum((count / n) * math.log(count / n) for count in truth_sizes.values())
        if n
        else 0.0
    )
    nmi = (
        mutual_information / math.sqrt(entropy_predicted * entropy_truth)
        if entropy_predicted and entropy_truth
        else (1.0 if entropy_predicted == entropy_truth else 0.0)
    )
    return {
        "hold_count": len(valid),
        "predicted_route_count": len(predicted),
        "ground_truth_route_count": len(truth),
        "pairwise_precision": pair_precision,
        "pairwise_recall": pair_recall,
        "pairwise_f1": pair_f1,
        "purity": purity,
        "adjusted_rand_index": ari,
        "normalized_mutual_information": nmi,
    }


def _descriptor(discriminator: RouteDiscriminator) -> dict[str, Any]:
    implementation_id = getattr(
        discriminator, "implementation_id", type(discriminator).__name__
    )
    configuration = getattr(discriminator, "configuration", None)
    if callable(configuration):
        configuration = configuration()
    return {
        "implementation_id": implementation_id,
        "configuration": configuration or {},
    }


def evaluate_route_discriminator(
    discriminator: RouteDiscriminator,
    records: Sequence[ImageRecord],
    *,
    run_id: str,
) -> EvaluationReport:
    """Evaluate a discriminator using annotated holds, excluding detection quality."""
    records = tuple(records)
    if any(not isinstance(record, ImageRecord) for record in records):
        raise TypeError("records must contain ImageRecord values.")
    cases: dict[str, list[EvaluationCaseResult]] = {"clean": [], "distorted": []}
    successful: dict[str, list[dict[str, Any]]] = {"clean": [], "distorted": []}
    for record in records:
        condition = "clean" if record.condition == "clean" else "distorted"
        holds = tuple(
            hold
            for hold in record.annotations
            if hold.attributes.get("hold_type") != "volume"
            and isinstance(hold.attributes.get("route_id"), int)
        )
        truth = _ground_truth_routes(holds)
        started = perf_counter()
        if not holds or not truth:
            cases[condition].append(
                EvaluationCaseResult(
                    record.image_id,
                    record.image_id,
                    record.source_id or record.image_id,
                    condition,
                    EvaluationStatus.UNEVALUABLE,
                    record.split,
                    record.parent_image_id,
                    record.augmentation_metadata,
                    perf_counter() - started,
                    annotations=truth,
                    error="ground truth holds with route_id are required",
                )
            )
            continue
        try:
            outputs = discriminator.get_routes([record.image], [holds])
            if len(outputs) != 1:
                raise ValueError(
                    "discriminator must return one route sequence per image"
                )
            predicted = tuple(outputs[0])
            elapsed = perf_counter() - started
            metrics = _clustering_metrics(predicted, truth, holds)
            cases[condition].append(
                EvaluationCaseResult(
                    record.image_id,
                    record.image_id,
                    record.source_id or record.image_id,
                    condition,
                    elapsed_seconds=elapsed,
                    predictions=predicted,
                    annotations=truth,
                    metrics=metrics,
                )
            )
            successful[condition].append({"metrics": metrics, "elapsed": elapsed})
        except Exception as error:  # noqa: BLE001 - preserve per-image failures
            cases[condition].append(
                EvaluationCaseResult(
                    record.image_id,
                    record.image_id,
                    record.source_id or record.image_id,
                    condition,
                    EvaluationStatus.ERROR,
                    record.split,
                    record.parent_image_id,
                    record.augmentation_metadata,
                    perf_counter() - started,
                    annotations=truth,
                    error=f"{type(error).__name__}: {error}",
                )
            )

    def block(condition: str) -> ConditionEvaluation:
        condition_cases = cases[condition]
        valid = successful[condition]
        aggregate: dict[str, float | int] = {
            "case_count": len(condition_cases),
            "evaluated_case_count": len(valid),
            "unevaluable_case_count": sum(
                case.status is EvaluationStatus.UNEVALUABLE for case in condition_cases
            ),
            "error_case_count": sum(
                case.status is EvaluationStatus.ERROR for case in condition_cases
            ),
        }
        for name in (
            "hold_count",
            "predicted_route_count",
            "ground_truth_route_count",
            "pairwise_precision",
            "pairwise_recall",
            "pairwise_f1",
            "purity",
            "adjusted_rand_index",
            "normalized_mutual_information",
        ):
            values = [float(item["metrics"][name]) for item in valid]
            if values:
                aggregate[name] = sum(values) / len(values)
        if valid:
            aggregate["mean_inference_seconds"] = sum(
                float(item["elapsed"]) for item in valid
            ) / len(valid)
        return ConditionEvaluation(condition, tuple(condition_cases), aggregate)

    return EvaluationReport(
        run_id,
        EvaluationTarget.ROUTE_DISCRIMINATION,
        {"record_count": len(records)},
        _descriptor(discriminator),
        block("clean"),
        block("distorted"),
        {
            "ground_truth": "ImageRecord.annotations",
            "ground_truth_filter": "hold_type != volume and route_id is int",
            "protocol": "ground_truth_holds_only",
        },
    )


def report_json(report: EvaluationReport, *, indent: int | None = None) -> str:
    return json.dumps(report.to_dict(), indent=indent, allow_nan=False, sort_keys=True)


def write_report(
    report: EvaluationReport, path: str | Path, *, indent: int = 2
) -> None:
    Path(path).write_text(report_json(report, indent=indent) + "\n", encoding="utf-8")
