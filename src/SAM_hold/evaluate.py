"""Run the small local text-versus-exemplar SAM 3 hold evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import largest_mask, load_hold_masks, load_image
from .sam3_hold_wrapper import SAM3HoldWrapper

TEST_FILENAMES = ("0003.jpg", "0006.jpg")
EXEMPLAR_FILENAME = "0000.jpg"


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    return 0.0 if union == 0 else float(np.count_nonzero(first & second) / union)


def score_instances(predictions: list[np.ndarray], ground_truth: list[np.ndarray], threshold: float = 0.5) -> dict[str, float | int]:
    pairs = sorted(((mask_iou(prediction, target), prediction_index, target_index) for prediction_index, prediction in enumerate(predictions) for target_index, target in enumerate(ground_truth)), reverse=True)
    matched_predictions: set[int] = set()
    matched_targets: set[int] = set()
    matched_ious: list[float] = []
    for iou, prediction_index, target_index in pairs:
        if iou < threshold:
            break
        if prediction_index not in matched_predictions and target_index not in matched_targets:
            matched_predictions.add(prediction_index)
            matched_targets.add(target_index)
            matched_ious.append(iou)
    true_positive = len(matched_ious)
    false_positive = len(predictions) - true_positive
    false_negative = len(ground_truth) - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {"gt_holds": len(ground_truth), "predicted_holds": len(predictions), "tp": true_positive, "fp": false_positive, "fn": false_negative, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0}


def overlay(image: Image.Image, masks: list[np.ndarray], color: tuple[int, int, int]) -> Image.Image:
    result = np.asarray(image.convert("RGB")).copy()
    for mask in masks:
        result[mask] = (0.55 * result[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
    return Image.fromarray(result)


def run_evaluation(data_dir: Path, annotation_path: Path, output_dir: Path, model_dir: Path, device: str | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    exemplar_image = load_image(data_dir / EXEMPLAR_FILENAME)
    exemplar_masks = load_hold_masks(annotation_path, EXEMPLAR_FILENAME, exemplar_image.size)
    exemplar_mask = largest_mask(exemplar_masks)
    model = SAM3HoldWrapper(model_dir=model_dir, device=device)
    model.load_model()
    rows: list[dict[str, object]] = []
    for filename in TEST_FILENAMES:
        image = load_image(data_dir / filename)
        ground_truth = load_hold_masks(annotation_path, filename, image.size)
        text_masks = model.generate_masks(image)
        exemplar_masks_predicted = model.generate_masks(image, exemplar_image=exemplar_image, exemplar_mask=exemplar_mask)
        image_dir = output_dir / Path(filename).stem
        image_dir.mkdir(exist_ok=True)
        image.save(image_dir / "original.png")
        overlay(image, ground_truth, (0, 255, 0)).save(image_dir / "ground_truth.png")
        overlay(image, text_masks, (255, 80, 0)).save(image_dir / "text_only.png")
        overlay(image, exemplar_masks_predicted, (0, 120, 255)).save(image_dir / "text_exemplar.png")
        panels = [image, overlay(image, ground_truth, (0, 255, 0)), overlay(image, text_masks, (255, 80, 0)), overlay(image, exemplar_masks_predicted, (0, 120, 255))]
        combined = Image.new("RGB", (image.width * 2, image.height * 2))
        for index, panel in enumerate(panels):
            combined.paste(panel, ((index % 2) * image.width, (index // 2) * image.height))
        combined.save(image_dir / "comparison.png")
        for method, masks in (("text_only", text_masks), ("text_exemplar", exemplar_masks_predicted)):
            rows.append({"filename": filename, "method": method, **score_instances(masks, ground_truth)})
    with (output_dir / "per_image_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate: dict[str, dict[str, float]] = {}
    for method in ("text_only", "text_exemplar"):
        method_rows = [row for row in rows if row["method"] == method]
        tp, fp, fn = (sum(int(row[key]) for row in method_rows) for key in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        aggregate[method] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "mean_matched_iou": float(np.mean([float(row["mean_matched_iou"]) for row in method_rows]))}
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--annotations", type=Path, default=Path(__file__).parent / "bh-annotation.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--model-dir", type=Path, default=Path("/home/vault/v123be/v123be56/LIT/models/sam3"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run_evaluation(args.data_dir, args.annotations, args.output_dir, args.model_dir, args.device)
