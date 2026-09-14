"""GPU-first, resumable evaluation stages for Slurm arrays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..hold_detector.hold_detector_factory import hold_detector_factory
from ..interfaces.data_models import Coordinate, Hold, ImageRecord, Polygon
from ..route_discriminator.dino_clustering_route_discriminator import (
    DINOClusteringRouteDiscriminator,
)
from ..route_discriminator.route_discriminator_factory import (
    route_discriminator_factory,
)
from ..utility.dinov3 import DINOv3
from .augmented_runner import LIGHTING_ARMS, build_lighting_records
from .hold_detector_evaluator import evaluate_hold_detector
from .metrics import pairwise_hold_iou
from .results import _safe
from .route_discriminator_evaluator import evaluate_route_discriminator
from .run_all_eval import (
    MATRIX_ROUTE_SPECS,
    ComponentSpec,
    _MappedRouteDiscriminator,
    _relabel_predictions,
    _slug,
    load_benchmark_records,
    predict_records,
)

DATASET = Path("data/evaluation/ground_truth_labels")
CONDITIONS = ("clean", *LIGHTING_ARMS)
ARMS = tuple(LIGHTING_ARMS)
DETECTOR_KEYS = ("mask_rcnn", "yolo", "sam_text", "sam_text_exemplar")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _hold_dict(hold: Hold) -> dict[str, Any]:
    attributes = {key: value for key, value in hold.attributes.items() if key != "mask"}
    return _safe(Hold(hold.polygon, attributes))


def _hold(value: dict[str, Any]) -> Hold:
    polygon = Polygon(
        tuple(
            Coordinate(int(point["x"]), int(point["y"]))
            for point in value["polygon"]["points"]
        )
    )
    return Hold(polygon, value.get("attributes", {}))


def _records(condition: str) -> list[ImageRecord]:
    clean = load_benchmark_records(DATASET, split="test")
    if condition == "clean":
        return clean
    if condition not in LIGHTING_ARMS:
        raise ValueError(f"unknown condition: {condition}")
    return [
        record
        for record in build_lighting_records(clean, LIGHTING_ARMS[condition], seed=42)
        if record.condition != "clean"
    ]


def _detector_spec(key: str) -> ComponentSpec:
    if key == "mask_rcnn":
        return ComponentSpec("Mask-RCNN", {"device": "cuda"})
    if key == "yolo":
        return ComponentSpec("YOLO", {"device": "cuda"})
    if key == "sam_text":
        return ComponentSpec("SAM 3", {"device": "cuda"})
    if key == "sam_text_exemplar":
        root = Path("data/evaluation/exemplars")
        exemplars = tuple(
            (
                Image.open(root / f"exemplar_image_{index}.png").convert("RGB"),
                np.asarray(Image.open(root / f"exemplar_mask_{index}.png")) > 0,
            )
            for index in range(1, 6)
        )
        return ComponentSpec("SAM 3", {"device": "cuda", "exemplars": exemplars})
    raise ValueError(f"unknown detector: {key}")


def _detector_artifact(root: Path, condition: str, key: str) -> Path:
    return root / "artifacts" / condition / "detectors" / f"{key}.json"


def _feature_artifact(root: Path, condition: str) -> Path:
    return root / "artifacts" / condition / "route_features.pt"


def _selected_records(
    condition: str, image_ids: list[str] | None = None
) -> list[ImageRecord]:
    records = _records(condition)
    if image_ids is None:
        return records
    by_id = {record.image_id: record for record in records}
    return [by_id[image_id] for image_id in image_ids]


def detector_stage(
    root: Path, condition: str, key: str, *, limit: int | None = None
) -> dict[str, Any]:
    records = _records(condition)
    if limit:
        records = sorted(
            records, key=lambda record: len(record.annotations), reverse=True
        )[:limit]
    spec = _detector_spec(key)
    detector = hold_detector_factory(spec.name, spec.config)
    started = perf_counter()
    predictions, timings = predict_records(detector, records)
    matrices, relabelled = {}, {}
    for record in records:
        truths = tuple(
            hold
            for hold in record.annotations
            if hold.attributes.get("hold_type") != "volume"
        )
        matrix = pairwise_hold_iou(
            predictions[record.image_id],
            truths,
            record.image.size,
            device="cuda",
        )
        matrices[record.image_id] = matrix
        relabelled[record.image_id] = _relabel_predictions(
            record, predictions[record.image_id], iou_matrix=matrix
        )
    report = evaluate_hold_detector(
        detector,
        records,
        run_id=f"gpu-cache:{condition}:{key}",
        predictions_by_image=predictions,
        inference_seconds_by_image=timings,
        iou_matrices_by_image=matrices,
        iou_device="cuda",
    ).to_dict()
    payload = {
        "schema_version": 1,
        "condition": condition,
        "detector_key": key,
        "record_ids": [record.image_id for record in records],
        "elapsed_seconds": perf_counter() - started,
        "report": report,
        "route_holds": {
            image_id: [_hold_dict(hold) for hold in holds]
            for image_id, holds in relabelled.items()
        },
    }
    _atomic_json(_detector_artifact(root, condition, key), payload)
    return payload


def _sources(
    root: Path, condition: str, records: list[ImageRecord]
) -> dict[str, dict[str, tuple[Hold, ...]]]:
    result = {
        "ground_truth": {
            record.image_id: tuple(
                hold
                for hold in record.annotations
                if hold.attributes.get("hold_type") != "volume"
                and isinstance(hold.attributes.get("route_id"), int)
            )
            for record in records
        }
    }
    for key in DETECTOR_KEYS:
        data = json.loads(_detector_artifact(root, condition, key).read_text())
        result[key] = {
            image_id: tuple(_hold(value) for value in values)
            for image_id, values in data["route_holds"].items()
        }
    return result


def feature_stage(root: Path, condition: str) -> dict[str, Any]:
    first = json.loads(
        _detector_artifact(root, condition, DETECTOR_KEYS[0]).read_text()
    )
    records = _selected_records(condition, first["record_ids"])
    sources = _sources(root, condition, records)
    dino = DINOv3(device="cuda")
    colour = DINOClusteringRouteDiscriminator(device="cuda")
    embeddings: dict[str, dict[str, dict[str, torch.Tensor]]] = {
        source: {"weighted": {}, "mean": {}} for source in sources
    }
    lab: dict[str, dict[str, torch.Tensor]] = {source: {} for source in sources}
    rgb: dict[str, dict[str, torch.Tensor]] = {source: {} for source in sources}
    started = perf_counter()
    for record in records:
        tokens = dino.extract_patch_tokens(record.image)
        rgb_image = np.asarray(record.image.convert("RGB"))
        for source, values in sources.items():
            holds = values[record.image_id]
            masks = [colour._hold_mask(record.image, hold) for hold in holds]
            for pooling in ("weighted", "mean"):
                embeddings[source][pooling][record.image_id] = (
                    dino.pool_mask_embeddings(tokens, masks, pooling).cpu()
                )
            lab[source][record.image_id] = torch.from_numpy(
                colour._lab_features(record.image, holds)
            )
            means = []
            for hold in holds:
                mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
                import cv2

                cv2.fillPoly(
                    mask,
                    [np.array([(p.x, p.y) for p in hold.polygon.points], np.int32)],
                    1,
                )
                pixels = rgb_image[mask.astype(bool)]
                means.append(pixels.mean(0) if len(pixels) else np.zeros(3))
            rgb[source][record.image_id] = torch.as_tensor(np.asarray(means))
    payload = {
        "schema_version": 1,
        "condition": condition,
        "record_ids": [record.image_id for record in records],
        "elapsed_seconds": perf_counter() - started,
        "embeddings": embeddings,
        "lab": lab,
        "rgb": rgb,
    }
    _atomic_torch(_feature_artifact(root, condition), payload)
    return payload


def _load_feature_maps(
    root: Path,
    condition_records: list[ImageRecord],
    source: str,
    condition: str,
) -> tuple[
    dict[tuple[int, str], torch.Tensor], dict[int, np.ndarray], dict[int, np.ndarray]
]:
    data = torch.load(
        _feature_artifact(root, condition), map_location="cpu", weights_only=True
    )
    embeddings, lab, rgb = {}, {}, {}
    for record in condition_records:
        selection = (
            slice(None)
            if source == "ground_truth"
            else [
                index
                for index, hold in enumerate(record.annotations)
                if hold.attributes.get("hold_type") != "volume"
                and isinstance(hold.attributes.get("route_id"), int)
            ]
        )
        for pooling in ("weighted", "mean"):
            embeddings[(id(record.image), pooling)] = data["embeddings"][source][
                pooling
            ][record.image_id][selection]
        lab[id(record.image)] = data["lab"][source][record.image_id][selection].numpy()
        rgb[id(record.image)] = data["rgb"][source][record.image_id][selection].numpy()
    return embeddings, lab, rgb


def _route_discriminator(
    spec: ComponentSpec,
    records: list[ImageRecord],
    root: Path,
    source: str,
    arm: str,
):
    clean = [record for record in records if record.condition == "clean"]
    distorted = [record for record in records if record.condition != "clean"]
    maps = [
        _load_feature_maps(root, group, source, condition)
        for group, condition in ((clean, "clean"), (distorted, arm))
        if group
    ]
    embeddings = {key: value for item in maps for key, value in item[0].items()}
    lab = {key: value for item in maps for key, value in item[1].items()}
    rgb = {key: value for item in maps for key, value in item[2].items()}
    config = {**spec.config, "device": "cuda"}
    if spec.name in {"DINO Clustering", "DINO Learning"}:
        config.update(precomputed_embeddings=embeddings, precomputed_colours=lab)
    elif spec.name == "Color Only":
        config["precomputed_features"] = (
            lab if spec.config.get("color_space") == "lab" else rgb
        )
    return route_discriminator_factory(spec.name, config)


def _combined_records(root: Path, arm: str, source: str) -> list[ImageRecord]:
    result = []
    for condition in ("clean", arm):
        artifact = json.loads(_detector_artifact(root, condition, source).read_text())
        records = _selected_records(condition, artifact["record_ids"])
        for record in records:
            holds = tuple(
                _hold(value) for value in artifact["route_holds"][record.image_id]
            )
            result.append(
                ImageRecord(
                    record.image_id,
                    record.image,
                    holds,
                    record.split,
                    record.source_id,
                    record.parent_image_id,
                    record.condition,
                    record.augmentation_metadata,
                )
            )
    return result


def _combine_detector_report(root: Path, arm: str, key: str) -> dict[str, Any]:
    clean = json.loads(_detector_artifact(root, "clean", key).read_text())["report"]
    distorted = json.loads(_detector_artifact(root, arm, key).read_text())["report"]
    clean["run_id"] = f"gpu:{arm}:{key}"
    clean["dataset"]["record_count"] = (
        clean["dataset"]["record_count"] + distorted["dataset"]["record_count"]
    )
    clean["distorted"] = distorted["distorted"]
    return clean


def combination_stage(
    root: Path, arm: str, detector_key: str, route_index: int
) -> Path:
    spec = MATRIX_ROUTE_SPECS[route_index]
    records = _combined_records(root, arm, detector_key)
    discriminator = _route_discriminator(spec, records, root, detector_key, arm)
    report = evaluate_route_discriminator(
        _MappedRouteDiscriminator(discriminator),
        records,
        run_id=f"gpu:{arm}:{detector_key}:{_slug(spec)}",
    ).to_dict()
    path = root / arm / "matrix" / f"{detector_key}__{_slug(spec)}.json"
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "run_id": f"gpu:{arm}:{detector_key}:{_slug(spec)}",
            "detector": _combine_detector_report(root, arm, detector_key),
            "route_discriminator": report,
            "metadata": {
                "protocol": "detector_predictions_then_route_grouping",
                "match_threshold": 0.5,
                "cached_gpu_features": True,
            },
        },
    )
    return path


def independent_route_stage(root: Path, arm: str, route_index: int) -> Path:
    records = _records("clean") + _records(arm)
    spec = MATRIX_ROUTE_SPECS[route_index]
    discriminator = _route_discriminator(spec, records, root, "ground_truth", arm)
    report = evaluate_route_discriminator(
        discriminator, records, run_id=f"gpu:{arm}:ground_truth:{_slug(spec)}"
    ).to_dict()
    path = root / arm / "independent" / "routes" / f"{_slug(spec)}.json"
    _atomic_json(path, report)
    return path


def consolidate_stage(root: Path, arm: str) -> dict[str, Any]:
    detector_paths = []
    for key in DETECTOR_KEYS:
        path = root / arm / "independent" / "detectors" / f"{key}.json"
        _atomic_json(path, _combine_detector_report(root, arm, key))
        detector_paths.append(str(path))
    combinations = [
        root / arm / "matrix" / f"{key}__{_slug(spec)}.json"
        for key in DETECTOR_KEYS
        for spec in MATRIX_ROUTE_SPECS
        if key != "sam_text" or spec.config.get("color_space") != "lab"
    ]
    routes = [
        root / arm / "independent" / "routes" / f"{_slug(spec)}.json"
        for spec in MATRIX_ROUTE_SPECS
    ]
    missing = [str(path) for path in (*combinations, *routes) if not path.is_file()]
    manifest = {
        "schema_version": 1,
        "arm": arm,
        "hardware": "A40",
        "detectors": list(DETECTOR_KEYS),
        "route_configurations": len(MATRIX_ROUTE_SPECS),
        "matrix_combinations": len(combinations),
        "completed_combinations": sum(path.is_file() for path in combinations),
        "detector_reports": detector_paths,
        "route_reports": [str(path) for path in routes if path.is_file()],
        "missing": missing,
        "status": "complete" if not missing else "incomplete",
    }
    _atomic_json(root / arm / "manifest.json", manifest)
    return manifest


def smoke_stage(root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = perf_counter()
    for key in DETECTOR_KEYS:
        detector_stage(root, "clean", key, limit=2)
    feature_stage(root, "clean")
    artifact = json.loads(
        _detector_artifact(root, "clean", DETECTOR_KEYS[0]).read_text()
    )
    records = []
    for record in _selected_records("clean", artifact["record_ids"]):
        records.append(
            ImageRecord(
                record.image_id,
                record.image,
                tuple(
                    _hold(value) for value in artifact["route_holds"][record.image_id]
                ),
                record.split,
                record.source_id,
                record.parent_image_id,
                record.condition,
                record.augmentation_metadata,
            )
        )
    families = (0, 1, 2, 8, 10, 11)
    for route_index in families:
        spec = MATRIX_ROUTE_SPECS[route_index]
        discriminator = _route_discriminator(
            spec, records, root, DETECTOR_KEYS[0], "clean"
        )
        evaluate_route_discriminator(
            _MappedRouteDiscriminator(discriminator),
            records,
            run_id=f"smoke:{_slug(spec)}",
        )
    elapsed = perf_counter() - started
    manifest = {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "elapsed_seconds": elapsed,
        "projected_arm_seconds": elapsed * 7,
        "target_arm_seconds": 1800,
        "within_target": elapsed * 7 < 1800,
    }
    _atomic_json(root / "smoke_manifest.json", manifest)
    if not manifest["within_target"]:
        raise RuntimeError(f"smoke projection exceeds target: {manifest}")
    return manifest


def _run_indexed(stage: str, index: int, root: Path) -> Any:
    if stage == "detector":
        condition_index, detector_index = divmod(index, len(DETECTOR_KEYS))
        return detector_stage(
            root, CONDITIONS[condition_index], DETECTOR_KEYS[detector_index]
        )
    if stage == "features":
        return feature_stage(root, CONDITIONS[index])
    if stage == "combination":
        arm_index, remainder = divmod(
            index, len(DETECTOR_KEYS) * len(MATRIX_ROUTE_SPECS)
        )
        detector_index, route_index = divmod(remainder, len(MATRIX_ROUTE_SPECS))
        return combination_stage(
            root, ARMS[arm_index], DETECTOR_KEYS[detector_index], route_index
        )
    if stage == "independent":
        arm_index, route_index = divmod(index, len(MATRIX_ROUTE_SPECS))
        return independent_route_stage(root, ARMS[arm_index], route_index)
    if stage == "consolidate":
        return consolidate_stage(root, ARMS[index])
    raise ValueError(f"unknown indexed stage: {stage}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "smoke",
            "detector",
            "features",
            "combination",
            "independent",
            "consolidate",
        ),
    )
    parser.add_argument(
        "--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    parser.add_argument("--root", type=Path, default=Path("results/gpu_fast"))
    args = parser.parse_args(argv)
    _ = (
        smoke_stage(args.root)
        if args.stage == "smoke"
        else _run_indexed(args.stage, args.index, args.root)
    )
    print(
        json.dumps({"stage": args.stage, "index": args.index, "status": "completed"}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
