"""Pure metrics for shared hold and route evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean

from PIL import Image as PILImage
from PIL import ImageDraw

from ..interfaces.data_models import Hold, Polygon, Route


@dataclass(frozen=True)
class HoldMatch:
    prediction_index: int
    annotation_index: int
    iou: float


def polygon_iou(
    first: Polygon, second: Polygon, image_size: tuple[int, int] | None = None
) -> float:
    points = (*first.points, *second.points)
    size = image_size or (
        max(point.x for point in points) + 1,
        max(point.y for point in points) + 1,
    )
    masks = [PILImage.new("1", size), PILImage.new("1", size)]
    for mask, polygon in zip(masks, (first, second), strict=True):
        ImageDraw.Draw(mask).polygon(
            [(point.x, point.y) for point in polygon.points], fill=1
        )
    intersection = sum(
        a and b for a, b in zip(masks[0].getdata(), masks[1].getdata(), strict=True)
    )
    union = sum(
        a or b for a, b in zip(masks[0].getdata(), masks[1].getdata(), strict=True)
    )
    return intersection / union if union else 0.0


def match_holds(
    predictions: Sequence[Hold],
    annotations: Sequence[Hold],
    *,
    threshold: float = 0.5,
    image_size: tuple[int, int] | None = None,
) -> tuple[HoldMatch, ...]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")
    candidates = sorted(
        (
            (polygon_iou(p.polygon, a.polygon, image_size), pi, ai)
            for pi, p in enumerate(predictions)
            for ai, a in enumerate(annotations)
        ),
        reverse=True,
    )
    used_predictions, used_annotations, matches = set(), set(), []
    for iou, pi, ai in candidates:
        if iou < threshold or pi in used_predictions or ai in used_annotations:
            continue
        used_predictions.add(pi)
        used_annotations.add(ai)
        matches.append(HoldMatch(pi, ai, iou))
    return tuple(sorted(matches, key=lambda match: match.prediction_index))


def hold_metrics(
    predictions: Sequence[Hold],
    annotations: Sequence[Hold],
    *,
    threshold: float = 0.5,
    image_size: tuple[int, int] | None = None,
) -> dict[str, float | int]:
    matches = match_holds(
        predictions, annotations, threshold=threshold, image_size=image_size
    )
    tp, fp, fn = (
        len(matches),
        len(predictions) - len(matches),
        len(annotations) - len(matches),
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean([match.iou for match in matches]) if matches else 0.0,
    }


def routes_from_annotations(annotations: Sequence[Hold]) -> tuple[Route, ...]:
    groups: dict[int, set[Hold]] = {}
    for hold in annotations:
        route_id = hold.attributes.get("route_id")
        if isinstance(route_id, int) and not isinstance(route_id, bool):
            groups.setdefault(route_id, set()).add(hold)
    return tuple(Route(holds, route_id) for route_id, holds in sorted(groups.items()))


def route_metrics(
    predictions: Sequence[Route],
    annotations: Sequence[Route],
    matches: Sequence[HoldMatch],
) -> dict[str, float | int]:
    predicted_to_annotation = {
        match.prediction_index: match.annotation_index for match in matches
    }
    prediction_sets = [
        {
            predicted_to_annotation[index]
            for index, _ in enumerate(route.holds)
            if index in predicted_to_annotation
        }
        for route in predictions
    ]
    annotation_sets = [
        {hold_index for hold_index, _ in enumerate(route.holds)}
        for route in annotations
    ]
    candidates = sorted(
        (
            (
                2 * len(predicted & annotation) / (len(predicted) + len(annotation)),
                pi,
                ai,
            )
            for pi, predicted in enumerate(prediction_sets)
            for ai, annotation in enumerate(annotation_sets)
            if predicted and annotation and predicted & annotation
        ),
        reverse=True,
    )
    used_p, used_a, scores = set(), set(), []
    for score, pi, ai in candidates:
        if pi not in used_p and ai not in used_a:
            used_p.add(pi)
            used_a.add(ai)
            scores.append(score)
    precision = len(scores) / len(predictions) if predictions else 0.0
    recall = len(scores) / len(annotations) if annotations else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matched_routes": len(scores),
        "predicted_routes": len(predictions),
        "annotated_routes": len(annotations),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_route_f1": mean(scores) if scores else 0.0,
    }
