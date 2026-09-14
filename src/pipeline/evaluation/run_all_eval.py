"""Run the configured hold-detector and route-discriminator benchmark suite."""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from ..hold_detector.hold_detector_factory import hold_detector_factory
from ..interfaces.data_models import Hold, ImageRecord, Route
from ..route_discriminator.route_discriminator_factory import (
    route_discriminator_factory,
)
from ..utility.ground_truth_loader import load_coco_restructured
from .hold_detector_evaluator import evaluate_hold_detector
from .metrics import match_holds, pairwise_hold_iou
from .route_discriminator_evaluator import evaluate_route_discriminator

DEFAULT_DETECTORS = ("Mask-RCNN", "YOLO", "SAM 3")
DEFAULT_ROUTE_DISCRIMINATORS = ("Triplet MLP", "DINO Clustering", "DINO Learning")


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    config: dict[str, Any]


def _annotation_jsons(root: Path, split: str) -> list[Path]:
    candidates = sorted(
        set(root.rglob("*_annotations.coco.json"))
        | set(root.rglob("*_restructured.json"))
    )
    selected = [
        path
        for path in candidates
        if split.lower() in path.parts or split.lower() in path.stem.lower()
    ]
    return selected or candidates


def _image_path(root: Path, annotation_file: Path, file_name: str) -> Path:
    candidates = (
        annotation_file.parent / file_name,
        root / file_name,
        root / "images" / Path(file_name).name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(root.rglob(Path(file_name).name))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find image '{file_name}' for {annotation_file}."
    )


def load_benchmark_records(
    dataset_root: str | Path, *, split: str = "test"
) -> list[ImageRecord]:
    """Load Roboflow COCO images and annotations into shared ImageRecords."""
    root = Path(dataset_root)
    annotation_files = _annotation_jsons(root, split)
    if not annotation_files:
        raise FileNotFoundError(
            f"No COCO annotation JSON found under {root}. Expected '*_annotations.coco.json' or '*_restructured.json'."
        )
    records: list[ImageRecord] = []
    for annotation_file in annotation_files:
        with annotation_file.open(encoding="utf-8") as file:
            data = json.load(file)
        loaded_holds = load_coco_restructured(annotation_file)
        annotations_by_image: dict[int, list[dict[str, Any]]] = {
            item["id"]: [] for item in data["images"]
        }
        for annotation in data.get("annotations", []):
            annotations_by_image.setdefault(annotation["image_id"], []).append(
                annotation
            )
        for image, holds in zip(data["images"], loaded_holds, strict=True):
            normalized: list[Hold] = []
            for hold, annotation in zip(
                holds, annotations_by_image[image["id"]], strict=True
            ):
                attributes = dict(hold.attributes)
                route_label = annotation.get("route_label")
                if isinstance(route_label, str) and route_label.lower().startswith(
                    "route_"
                ):
                    try:
                        attributes["route_id"] = int(route_label.removeprefix("route_"))
                    except ValueError:
                        pass
                normalized.append(Hold(hold.polygon, attributes))
            image_file = _image_path(root, annotation_file, image["file_name"])
            condition = (
                "distorted"
                if any(
                    part.lower() in {"distorted", "augmented"}
                    for part in annotation_file.parts
                )
                else "clean"
            )
            records.append(
                ImageRecord(
                    image["file_name"],
                    Image.open(image_file).convert("RGB"),
                    tuple(normalized),
                    split=split,
                    condition="augmented" if condition == "distorted" else "clean",
                )
            )
    return records


def run_all_evaluations(
    records: list[ImageRecord],
    *,
    detectors: tuple[ComponentSpec, ...] = tuple(
        ComponentSpec(name, {}) for name in DEFAULT_DETECTORS
    ),
    route_discriminators: tuple[ComponentSpec, ...] = tuple(
        ComponentSpec(name, {}) for name in DEFAULT_ROUTE_DISCRIMINATORS
    ),
    output_path: str | Path = "evaluation-results.json",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run every configured component and write one consolidated JSON result."""
    run_id = run_id or datetime.now(UTC).strftime("eval-%Y%m%dT%H%M%SZ")
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "record_count": len(records),
        "detectors": [],
        "route_discriminators": [],
        "errors": [],
    }
    for spec in detectors:
        try:
            detector = hold_detector_factory(spec.name, spec.config)
            report = evaluate_hold_detector(
                detector, records, run_id=f"{run_id}:{spec.name}"
            )
            result["detectors"].append(report.to_dict())
        except Exception as error:  # noqa: BLE001 - consolidate component failures
            result["errors"].append(
                {
                    "component_type": "detector",
                    "name": spec.name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    for spec in route_discriminators:
        try:
            discriminator = route_discriminator_factory(spec.name, spec.config)
            report = evaluate_route_discriminator(
                discriminator, records, run_id=f"{run_id}:{spec.name}"
            )
            result["route_discriminators"].append(report.to_dict())
        except Exception as error:  # noqa: BLE001 - consolidate component failures
            result["errors"].append(
                {
                    "component_type": "route_discriminator",
                    "name": spec.name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    Path(output_path).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


MATRIX_ROUTE_SPECS = (
    ComponentSpec("Triplet MLP", {}),
    ComponentSpec("Color Only", {"n_clusters": 6}),
    *(
        ComponentSpec(
            "DINO Clustering", {"clustering_method": method, "color_weight": weight}
        )
        for method in ("hdbscan", "dbscan", "agglomerative")
        for weight in (0.0, 1.0)
    ),
    ComponentSpec("DINO Learning", {"pooling": "weighted"}),
    ComponentSpec("DINO Learning", {"pooling": "mean"}),
    ComponentSpec("Ground Truth", {}),
    ComponentSpec("Mock", {"seed": 12345}),
)


def _relabel_predictions(
    record: ImageRecord,
    predictions: Sequence[Hold],
    threshold: float = 0.5,
    iou_matrix=None,
) -> tuple[Hold, ...]:
    annotations = tuple(
        a for a in record.annotations if a.attributes.get("hold_type") != "volume"
    )
    matches = match_holds(
        predictions,
        annotations,
        threshold=threshold,
        image_size=record.image.size,
        iou_matrix=iou_matrix,
    )
    route_ids = {
        m.prediction_index: annotations[m.annotation_index].attributes.get("route_id")
        for m in matches
    }
    return tuple(
        Hold(
            p.polygon,
            {
                **p.attributes,
                **(
                    {"route_id": route_ids[i]}
                    if isinstance(route_ids.get(i), int)
                    else {}
                ),
            },
        )
        for i, p in enumerate(predictions)
    )


class _MappedRouteDiscriminator:
    def __init__(self, discriminator: Any, threshold: float = 0.5):
        self.discriminator, self.threshold = discriminator, threshold
        self.implementation_id = getattr(
            discriminator, "implementation_id", type(discriminator).__name__
        )
        self.configuration = getattr(discriminator, "configuration", {})
        if callable(self.configuration):
            self.configuration = self.configuration()

    def get_routes(self, images, holds):
        outputs = self.discriminator.get_routes(images, holds)
        result = []
        for image, source, routes in zip(images, holds, outputs, strict=True):
            used = set()
            mapped = []
            source_by_polygon = {
                hold.polygon: (index, hold) for index, hold in enumerate(source)
            }
            for route_index, route in enumerate(routes):
                members = set()
                for output in route.holds:
                    index, hold = source_by_polygon.get(output.polygon, (-1, None))
                    if hold is not None and index not in used:
                        used.add(index)
                        members.add(hold)
                if members:
                    mapped.append(Route(members, route_index))
            result.append(mapped)
        return result


def _quiet_call(function, *args, **kwargs):
    # Long benchmark jobs may outlive their CLI output pipe; suppress model-loader progress bars.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return function(*args, **kwargs)


def _slug(spec: ComponentSpec) -> str:
    config = "-".join(f"{key}-{value}" for key, value in sorted(spec.config.items()))
    return (
        "-".join((spec.name.lower().replace(" ", "_"), config))
        .strip("-")
        .replace(".", "_")
    )


def predict_records(
    detector: Any, records: Sequence[ImageRecord]
) -> tuple[dict[str, tuple[Hold, ...]], dict[str, float]]:
    """Run detector batches once and return per-image predictions and timings."""
    batch_sizes = {
        "mask_rcnn_hold_detector": 2,
        "yolov8_hold_detector": 8,
        "sam_hold_detector": 1,
    }
    batch_size = batch_sizes.get(
        getattr(detector, "implementation_id", type(detector).__name__), 1
    )
    predictions: dict[str, tuple[Hold, ...]] = {}
    timings: dict[str, float] = {}
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        started = perf_counter()
        outputs = detector.get_holds([record.image for record in batch])
        elapsed = (perf_counter() - started) / max(len(batch), 1)
        if len(outputs) != len(batch):
            raise ValueError("detector outputs must align with input images")
        for record, output in zip(batch, outputs, strict=True):
            predictions[record.image_id] = tuple(output)
            timings[record.image_id] = elapsed
    return predictions, timings


def run_evaluation_matrix(
    records: Sequence[ImageRecord],
    *,
    detectors: Sequence[ComponentSpec] = tuple(
        ComponentSpec(name, {}) for name in DEFAULT_DETECTORS
    ),
    route_discriminators: Sequence[ComponentSpec] = MATRIX_ROUTE_SPECS,
    output_dir: str | Path = "results/evaluation_matrix",
    run_id: str | None = None,
    purge_existing: bool = True,
) -> dict[str, Any]:
    """Run detector-prediction versus route-discriminator combinations."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if purge_existing:
        for path in root.glob("*.json"):
            path.unlink()
    run_id = run_id or datetime.now(UTC).strftime("matrix-%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "record_count": len(records),
        "combinations": [],
        "errors": [],
    }
    for detector_spec in detectors:
        try:
            detector = hold_detector_factory(detector_spec.name, detector_spec.config)
            predictions, timings = _quiet_call(predict_records, detector, records)
            iou_matrices = {}
            for record in records:
                truths = tuple(
                    hold
                    for hold in record.annotations
                    if hold.attributes.get("hold_type") != "volume"
                )
                iou_matrices[record.image_id] = pairwise_hold_iou(
                    predictions[record.image_id],
                    truths,
                    record.image.size,
                    device=getattr(detector, "device", None),
                )
            detector_report = _quiet_call(
                evaluate_hold_detector,
                detector,
                records,
                run_id=f"{run_id}:{detector_spec.name}",
                predictions_by_image=predictions,
                inference_seconds_by_image=timings,
                iou_matrices_by_image=iou_matrices,
            ).to_dict()
            route_records = [
                ImageRecord(
                    r.image_id,
                    r.image,
                    _relabel_predictions(
                        r, predictions[r.image_id], iou_matrix=iou_matrices[r.image_id]
                    ),
                    r.split,
                    r.source_id,
                    r.parent_image_id,
                    r.condition,
                    r.augmentation_metadata,
                )
                for r in records
            ]
        except Exception as error:  # noqa: BLE001 - preserve matrix progress
            manifest["errors"].append(
                {
                    "component_type": "detector",
                    "name": detector_spec.name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        for route_spec in route_discriminators:
            name = f"{detector_spec.name}__{_slug(route_spec)}"
            path = (
                root
                / f"{_slug(ComponentSpec(detector_spec.name, {}))}__{_slug(route_spec)}.json"
            )
            try:
                discriminator = route_discriminator_factory(
                    route_spec.name, route_spec.config
                )
                route_report = _quiet_call(
                    evaluate_route_discriminator,
                    _MappedRouteDiscriminator(discriminator),
                    route_records,
                    run_id=f"{run_id}:{name}",
                ).to_dict()
                payload = {
                    "schema_version": 1,
                    "run_id": f"{run_id}:{name}",
                    "detector": detector_report,
                    "route_discriminator": route_report,
                    "metadata": {
                        "protocol": "detector_predictions_then_route_grouping",
                        "match_threshold": 0.5,
                    },
                }
                path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                    + "\n",
                    encoding="utf-8",
                )
                manifest["combinations"].append(
                    {
                        "name": name,
                        "path": str(path),
                        "status": "completed",
                        "detector": detector_spec.name,
                        "route_discriminator": route_spec.name,
                        "configuration": route_spec.config,
                    }
                )
            except Exception as error:  # noqa: BLE001 - preserve matrix progress
                manifest["errors"].append(
                    {
                        "component_type": "route_discriminator",
                        "name": name,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                manifest["combinations"].append(
                    {"name": name, "path": str(path), "status": "error"}
                )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_independent_evaluations(
    records: Sequence[ImageRecord],
    *,
    detectors: Sequence[ComponentSpec] = tuple(
        ComponentSpec(name, {}) for name in DEFAULT_DETECTORS
    ),
    route_discriminators: Sequence[ComponentSpec] = MATRIX_ROUTE_SPECS,
    output_dir: str | Path = "results/independent",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate detectors and route discriminators independently."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now(UTC).strftime("independent-%Y%m%dT%H%M%SZ")
    detector_result = {
        "schema_version": 1,
        "run_id": run_id,
        "record_count": len(records),
        "detectors": [],
        "errors": [],
    }
    route_result = {
        "schema_version": 1,
        "run_id": run_id,
        "record_count": len(records),
        "route_discriminators": [],
        "errors": [],
    }
    for spec in detectors:
        try:
            detector = hold_detector_factory(spec.name, spec.config)
            predictions, timings = _quiet_call(predict_records, detector, records)
            report = _quiet_call(
                evaluate_hold_detector,
                detector,
                records,
                run_id=f"{run_id}:{spec.name}",
                predictions_by_image=predictions,
                inference_seconds_by_image=timings,
            )
            detector_result["detectors"].append(report.to_dict())
        except Exception as error:  # noqa: BLE001 - preserve independent progress
            detector_result["errors"].append(
                {"name": spec.name, "error": f"{type(error).__name__}: {error}"}
            )
    for spec in route_discriminators:
        try:
            report = _quiet_call(
                evaluate_route_discriminator,
                route_discriminator_factory(spec.name, spec.config),
                records,
                run_id=f"{run_id}:{spec.name}",
            )
            route_result["route_discriminators"].append(report.to_dict())
        except Exception as error:  # noqa: BLE001 - preserve independent progress
            route_result["errors"].append(
                {
                    "name": spec.name,
                    "configuration": spec.config,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    detector_path = root / "detectors.json"
    route_path = root / "route_discriminators.json"
    detector_path.write_text(
        json.dumps(detector_result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    route_path.write_text(
        json.dumps(route_result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"detectors": detector_result, "route_discriminators": route_result}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--output", default="results/evaluation_matrix/manifest.json", type=Path
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--independent",
        action="store_true",
        help="Run detector-only and ground-truth-route evaluations.",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory for independent reports."
    )
    parser.add_argument(
        "--detector",
        action="append",
        dest="detectors",
        help="Factory detector name; repeat to override defaults.",
    )
    parser.add_argument(
        "--route-discriminator",
        action="append",
        dest="route_discriminators",
        help="Factory discriminator name; repeat to override defaults.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records = load_benchmark_records(args.dataset_root, split=args.split)
    detectors = tuple(
        ComponentSpec(name, {}) for name in (args.detectors or DEFAULT_DETECTORS)
    )
    routes = (
        MATRIX_ROUTE_SPECS
        if not args.route_discriminators
        else tuple(ComponentSpec(name, {}) for name in args.route_discriminators)
    )
    if args.independent:
        result = run_independent_evaluations(
            records,
            detectors=detectors,
            route_discriminators=routes,
            output_dir=args.output_dir or Path("results/independent"),
            run_id=args.run_id,
        )
        print(
            f"Saved {len(result['detectors']['detectors'])} detector and {len(result['route_discriminators']['route_discriminators'])} route reports."
        )
    else:
        manifest = run_evaluation_matrix(
            records,
            detectors=detectors,
            route_discriminators=routes,
            output_dir=args.output.parent,
            run_id=args.run_id,
        )
        print(
            f"Saved {len(manifest['combinations'])} matrix combinations to {args.output.parent} ({len(manifest['errors'])} errors)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
