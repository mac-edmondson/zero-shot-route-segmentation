"""Pure metrics for shared hold and route evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean

import numpy as np
import torch
from PIL import Image as PILImage
from PIL import ImageDraw

from ..interfaces.data_models import Hold, Polygon, Route


@dataclass(frozen=True)
class HoldMatch:
    prediction_index: int
    annotation_index: int
    iou: float


def _rasterize_hold(
    hold: Hold, size: tuple[int, int], *, use_attribute_mask: bool
) -> np.ndarray:
    if use_attribute_mask:
        candidate = hold.attributes.get("mask")
        if candidate is not None:
            try:
                mask = np.asarray(candidate, dtype=bool)
                if mask.shape == (size[1], size[0]):
                    return np.ascontiguousarray(mask)
            except (TypeError, ValueError):
                pass
    mask = PILImage.new("1", size)
    ImageDraw.Draw(mask).polygon(
        [(point.x, point.y) for point in hold.polygon.points], fill=1
    )
    return np.ascontiguousarray(np.asarray(mask, dtype=bool))


def pairwise_hold_iou(
    predictions: Sequence[Hold],
    annotations: Sequence[Hold],
    image_size: tuple[int, int],
    *,
    device: str | torch.device | None = None,
    chunk_size: int = 16,
) -> np.ndarray:
    """Compute the exact full-resolution prediction/annotation IoU matrix once."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not predictions or not annotations:
        return np.zeros((len(predictions), len(annotations)), dtype=np.float64)
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    prediction_masks = [
        _rasterize_hold(hold, image_size, use_attribute_mask=True)
        for hold in predictions
    ]
    annotation_masks = np.stack(
        [
            _rasterize_hold(hold, image_size, use_attribute_mask=False)
            for hold in annotations
        ]
    ).reshape(len(annotations), -1)
    truth = torch.from_numpy(annotation_masks).to(target, dtype=torch.float32)
    truth_areas = truth.sum(1)
    rows = []
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    if target.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.inference_mode():
            for start in range(0, len(prediction_masks), chunk_size):
                batch = np.stack(prediction_masks[start : start + chunk_size]).reshape(
                    -1, annotation_masks.shape[1]
                )
                predicted = torch.from_numpy(batch).to(target, dtype=torch.float32)
                intersections = predicted @ truth.T
                unions = (
                    predicted.sum(1)[:, None] + truth_areas[None, :] - intersections
                )
                rows.append(
                    torch.where(unions > 0, intersections / unions, 0).cpu().numpy()
                )
    finally:
        if target.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return np.concatenate(rows).astype(np.float64, copy=False)


def polygon_iou(
    first: Polygon, second: Polygon, image_size: tuple[int, int] | None = None
) -> float:
    points = (*first.points, *second.points)
    size = image_size or (
        max(point.x for point in points) + 1,
        max(point.y for point in points) + 1,
    )
    masks = []
    for polygon in (first, second):
        mask = PILImage.new("1", size)
        ImageDraw.Draw(mask).polygon(
            [(point.x, point.y) for point in polygon.points], fill=1
        )
        masks.append(np.asarray(mask, dtype=bool))
    intersection = np.count_nonzero(masks[0] & masks[1])
    union = np.count_nonzero(masks[0] | masks[1])
    return intersection / union if union else 0.0


def match_holds(
    predictions: Sequence[Hold],
    annotations: Sequence[Hold],
    *,
    threshold: float = 0.5,
    image_size: tuple[int, int] | None = None,
    iou_matrix: np.ndarray | None = None,
) -> tuple[HoldMatch, ...]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")
    matrix = iou_matrix
    if matrix is None:
        if image_size is None:
            points = tuple(
                point
                for hold in (*predictions, *annotations)
                for point in hold.polygon.points
            )
            image_size = (
                max((point.x for point in points), default=0) + 1,
                max((point.y for point in points), default=0) + 1,
            )
        matrix = pairwise_hold_iou(predictions, annotations, image_size)
    if matrix.shape != (len(predictions), len(annotations)):
        raise ValueError("iou_matrix shape must match predictions and annotations.")
    candidates = sorted(
        (
            (float(matrix[pi, ai]), pi, ai)
            for pi in range(len(predictions))
            for ai in range(len(annotations))
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
