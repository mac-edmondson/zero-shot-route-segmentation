"""Dataset-level AP evaluation for hold detectors."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from ..hold_detector.hold_detector import HoldDetector
from ..interfaces.data_models import Hold, ImageRecord, Polygon
from .results import (
    ConditionEvaluation,
    EvaluationCaseResult,
    EvaluationReport,
    EvaluationStatus,
    EvaluationTarget,
)

DEFAULT_IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


def _polygon_mask(polygon: Polygon, size: tuple[int, int]) -> np.ndarray:
    mask = PILImage.new("1", size)
    ImageDraw.Draw(mask).polygon(
        [(point.x, point.y) for point in polygon.points], fill=1
    )
    return np.asarray(mask, dtype=bool)


def _geometry_iou(
    prediction: Hold, annotation: Hold, size: tuple[int, int]
) -> tuple[float, str]:
    candidate = prediction.attributes.get("mask")
    predicted_mask = None
    if candidate is not None:
        try:
            array = np.asarray(candidate, dtype=bool)
            if array.shape == (size[1], size[0]):
                predicted_mask = array
        except (TypeError, ValueError):
            predicted_mask = None
    if predicted_mask is None:
        predicted_mask = _polygon_mask(prediction.polygon, size)
        source = "polygon"
    else:
        source = "mask"
    annotation_mask = _polygon_mask(annotation.polygon, size)
    union = np.logical_or(predicted_mask, annotation_mask).sum()
    return (
        float(np.logical_and(predicted_mask, annotation_mask).sum() / union)
        if union
        else 0.0
    ), source


def _confidence(hold: Hold, fallback: float) -> tuple[float, str]:
    value = hold.attributes.get("confidence")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
    ):
        return float(value), "attribute"
    return fallback, "fallback"


def _average_precision(
    detections: Sequence[tuple[float, int, int]],
    predictions: Sequence[Sequence[Hold]],
    truths: Sequence[Sequence[Hold]],
    sizes: Sequence[tuple[int, int]],
    threshold: float,
) -> float:
    total_truths = sum(len(items) for items in truths)
    if not total_truths:
        return 0.0
    ordered = sorted(detections, key=lambda item: (-item[0], item[1], item[2]))
    used: set[tuple[int, int]] = set()
    true_positives: list[int] = []
    false_positives: list[int] = []
    for _, case_index, prediction_index in ordered:
        prediction = predictions[case_index][prediction_index]
        candidates = [
            (
                _geometry_iou(prediction, annotation, sizes[case_index])[0],
                annotation_index,
            )
            for annotation_index, annotation in enumerate(truths[case_index])
            if (case_index, annotation_index) not in used
        ]
        best = max(candidates, default=(0.0, -1))
        if best[0] >= threshold:
            used.add((case_index, best[1]))
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)
    tp = np.cumsum(true_positives)
    fp = np.cumsum(false_positives)
    recall = tp / total_truths
    precision = tp / np.maximum(tp + fp, 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall_points = np.concatenate(([0.0], recall, [1.0]))
    precision_points = np.concatenate(
        ([precision[0] if len(precision) else 0.0], precision, [0.0])
    )
    return float(
        np.sum((recall_points[1:] - recall_points[:-1]) * precision_points[1:])
    )


def _json_prediction(hold: Hold) -> Hold:
    attributes = dict(hold.attributes)
    mask = attributes.pop("mask", None)
    if mask is not None:
        try:
            attributes["mask_available"] = True
            attributes["mask_shape"] = tuple(np.asarray(mask).shape)
        except (TypeError, ValueError):
            attributes["mask_available"] = False
    attributes["geometry_source"] = (
        "mask" if attributes.get("mask_available") else "polygon"
    )
    return Hold(hold.polygon, attributes)


def _descriptor(detector: HoldDetector) -> dict[str, Any]:
    implementation_id = getattr(detector, "implementation_id", type(detector).__name__)
    configuration = getattr(detector, "configuration", None)
    if callable(configuration):
        configuration = configuration()
    return {
        "implementation_id": implementation_id,
        "configuration": configuration or {},
    }


def evaluate_hold_detector(
    detector: HoldDetector,
    records: Sequence[ImageRecord],
    *,
    run_id: str,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    confidence_fallback: float = 1.0,
    predictions_by_image: Mapping[str, Sequence[Hold]] | None = None,
    inference_seconds_by_image: Mapping[str, float] | None = None,
) -> EvaluationReport:
    """Evaluate one detector and return a clean/distorted report."""
    thresholds = tuple(float(value) for value in iou_thresholds)
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("iou_thresholds must contain values in [0, 1].")
    if not 0 <= confidence_fallback <= 1:
        raise ValueError("confidence_fallback must be in [0, 1].")
    records = tuple(records)
    if any(not isinstance(record, ImageRecord) for record in records):
        raise TypeError("records must contain ImageRecord values.")

    case_data: list[tuple[ImageRecord, Sequence[Hold], Sequence[Hold], float]] = []
    cases: dict[str, list[EvaluationCaseResult]] = {"clean": [], "distorted": []}
    for record in records:
        condition = "clean" if record.condition == "clean" else "distorted"
        truths = tuple(
            annotation
            for annotation in record.annotations
            if annotation.attributes.get("hold_type") != "volume"
        )
        started = perf_counter()
        try:
            if predictions_by_image is None:
                outputs = detector.get_holds([record.image])
                if len(outputs) != 1:
                    raise ValueError("detector must return one hold sequence per image")
                predictions = tuple(outputs[0])
                elapsed = perf_counter() - started
            else:
                predictions = tuple(predictions_by_image.get(record.image_id, ()))
                elapsed = float((inference_seconds_by_image or {}).get(
                    record.image_id, perf_counter() - started
                ))
            status = (
                EvaluationStatus.EVALUATED if truths else EvaluationStatus.UNEVALUABLE
            )
            case_data.append((record, predictions, truths, elapsed))
            confidence_sources = {"attribute": 0, "fallback": 0}
            for prediction in predictions:
                confidence_sources[_confidence(prediction, confidence_fallback)[1]] += 1
            cases[condition].append(
                EvaluationCaseResult(
                    case_id=record.image_id,
                    image_id=record.image_id,
                    source_id=record.source_id or record.image_id,
                    condition=condition,
                    status=status,
                    split=record.split,
                    parent_image_id=record.parent_image_id,
                    augmentation_metadata=record.augmentation_metadata,
                    elapsed_seconds=elapsed,
                    predictions=tuple(
                        _json_prediction(prediction) for prediction in predictions
                    ),
                    annotations=tuple(
                        _json_prediction(annotation) for annotation in truths
                    ),
                    metrics={
                        "prediction_count": len(predictions),
                        "ground_truth_count": len(truths),
                        "confidence_attribute_count": confidence_sources["attribute"],
                        "confidence_fallback_count": confidence_sources["fallback"],
                    },
                )
            )
        except Exception as error:  # noqa: BLE001 - preserve per-image failures in the report
            elapsed = perf_counter() - started
            cases[condition].append(
                EvaluationCaseResult(
                    case_id=record.image_id,
                    image_id=record.image_id,
                    source_id=record.source_id or record.image_id,
                    condition=condition,
                    split=record.split,
                    parent_image_id=record.parent_image_id,
                    augmentation_metadata=record.augmentation_metadata,
                    elapsed_seconds=elapsed,
                    annotations=truths,
                    status=EvaluationStatus.ERROR,
                    error=f"{type(error).__name__}: {error}",
                )
            )

    aggregates: dict[str, dict[str, float | int]] = {}
    for condition in ("clean", "distorted"):
        selected = [
            (record, predictions, truths, elapsed)
            for record, predictions, truths, elapsed in case_data
            if ("clean" if record.condition == "clean" else "distorted") == condition
            and truths
        ]
        if not selected:
            aggregates[condition] = {
                "case_count": len(cases[condition]),
                "evaluated_case_count": 0,
                "unevaluable_case_count": sum(
                    case.status is EvaluationStatus.UNEVALUABLE
                    for case in cases[condition]
                ),
                "error_case_count": sum(
                    case.status is EvaluationStatus.ERROR for case in cases[condition]
                ),
                "mAP": 0.0,
            }
            continue
        detections = [
            (confidence, index, prediction_index)
            for index, (_, predictions, _, _) in enumerate(selected)
            for prediction_index, prediction in enumerate(predictions)
            for confidence, _ in [_confidence(prediction, confidence_fallback)]
        ]
        aps = {
            threshold: _average_precision(
                detections,
                [item[1] for item in selected],
                [item[2] for item in selected],
                [item[0].image.size for item in selected],
                threshold,
            )
            for threshold in thresholds
        }
        aggregates[condition] = {
            "case_count": len(cases[condition]),
            "evaluated_case_count": len(selected),
            "unevaluable_case_count": sum(
                case.status is EvaluationStatus.UNEVALUABLE for case in cases[condition]
            ),
            "error_case_count": sum(
                case.status is EvaluationStatus.ERROR for case in cases[condition]
            ),
            "ground_truth_count": sum(len(item[2]) for item in selected),
            "prediction_count": sum(len(item[1]) for item in selected),
            "mean_inference_seconds": float(np.mean([item[3] for item in selected])),
        }
        aggregates[condition].update(
            {f"AP{threshold:.2f}": value for threshold, value in aps.items()}
        )
        aggregates[condition]["AP50"] = aps.get(0.5, 0.0)
        aggregates[condition]["AP75"] = aps.get(0.75, 0.0)
        aggregates[condition]["mAP"] = float(np.mean(tuple(aps.values())))

    def block(condition: str) -> ConditionEvaluation:
        return ConditionEvaluation(
            condition, tuple(cases[condition]), aggregates[condition]
        )

    return EvaluationReport(
        run_id,
        EvaluationTarget.HOLD_DETECTION,
        {"record_count": len(records)},
        _descriptor(detector),
        block("clean"),
        block("distorted"),
        {
            "iou_thresholds": thresholds,
            "confidence_fallback": confidence_fallback,
            "ground_truth_filter": "hold_type != volume",
            "ap_protocol": "dataset_global_interpolated",
        },
    )


def report_json(report: EvaluationReport, *, indent: int | None = None) -> str:
    return json.dumps(report.to_dict(), indent=indent, allow_nan=False, sort_keys=True)


def write_report(
    report: EvaluationReport, path: str | Path, *, indent: int = 2
) -> None:
    Path(path).write_text(report_json(report, indent=indent) + "\n", encoding="utf-8")
