#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "training/yolov8-hold-detector"
DATASET = ROOT / "data/roboflow_dataset"
CHECKPOINT = RUN_DIR / "checkpoints/yolov8s.pt"
MANIFEST = DATASET / "manifest.json"
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_label(path: Path) -> int:
    normalized = []
    for line in path.read_text().splitlines():
        row = line.split()
        if not row:
            continue
        if row[0] != "0" or len(row) < 5 or len(row[1:]) % 2:
            raise ValueError(f"invalid label in {path}")
        values = [float(value) for value in row[1:]]
        if len(values) == 4:
            xc, yc, width, height = values
        else:
            xs, ys = values[::2], values[1::2]
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
            xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
            width, height = xmax - xmin, ymax - ymin
        if any(value < 0 or value > 1 for value in (xc, yc, width, height)):
            raise ValueError(f"coordinates outside [0, 1] in {path}")
        normalized.append(f"0 {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}")
    path.write_text("\n".join(normalized) + ("\n" if normalized else ""))
    return len(normalized)


def create_manifest() -> None:
    records = []
    counts = {split: 0 for split in SPLITS}
    objects = 0
    for split, directory in SPLITS.items():
        image_dir = DATASET / directory / "images"
        label_dir = DATASET / directory / "labels"
        for image in sorted(image_dir.glob("*")):
            if not image.is_file():
                continue
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(label)
            with Image.open(image) as opened:
                width, height = opened.size
            count = normalize_label(label)
            counts[split] += 1
            objects += count
            records.append(
                {
                    "split": split,
                    "image": str(image.relative_to(ROOT)),
                    "label": str(label.relative_to(ROOT)),
                    "width": width,
                    "height": height,
                    "image_bytes": image.stat().st_size,
                    "label_bytes": label.stat().st_size,
                    "image_sha256": sha256(image),
                    "label_sha256": sha256(label),
                    "objects": count,
                }
            )
    MANIFEST.write_text(
        json.dumps(
            {
                "dataset": "hold-detection-havfy",
                "version": 2,
                "format": "yolov8",
                "classes": {"0": "Hold"},
                "counts": counts,
                "objects_by_class": {"0": objects},
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


def prepare_checkpoint() -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.exists():
        YOLO(str(CHECKPOINT))
    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size == 0:
        raise RuntimeError(f"checkpoint was not downloaded: {CHECKPOINT}")


def prepare() -> Path:
    create_manifest()
    prepare_checkpoint()
    return DATASET / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    data_yaml = prepare()
    if args.prepare_only:
        return
    model = YOLO(str(CHECKPOINT))
    model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=1280,
        batch=8,
        device=0,
        project=str(RUN_DIR),
        name="train",
        exist_ok=True,
    )
    best = Path(model.trainer.best)
    evaluator = YOLO(str(best))
    evaluator.val(
        data=str(data_yaml),
        split="val",
        project=str(RUN_DIR / "eval"),
        name="val",
        exist_ok=True,
    )
    evaluator.val(
        data=str(data_yaml),
        split="test",
        project=str(RUN_DIR / "eval"),
        name="test",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
