#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
CATEGORIES = ("bh", "bh-phone", "sm")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        type=Path,
        default=Path("data/indoor-climbing-gym-hold-segmentation"),
    )
    p.add_argument("--output", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delete-source", action="store_true")
    a = p.parse_args()
    root = a.source
    out = a.output or root / "yolov8_data"
    out.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        for k in ("images", "labels"):
            d = out / k / s
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
    from PIL import Image

    records = []
    skipped = []
    for cat in CATEGORIES:
        base = root / "4"
        meta = {
            m["filename"]: m
            for m in json.loads((base / f"{cat}-annotation.json").read_text())
            .get("_via_img_metadata", {})
            .values()
        }
        for image in sorted((base / cat).glob("*.jpg")):
            try:
                with Image.open(image) as im:
                    w, h = im.size
                lines = []
                for r in meta[image.name].get("regions", []):
                    q = r.get("shape_attributes", {})
                    xs = q.get("all_points_x")
                    ys = q.get("all_points_y")
                    if (
                        q.get("name") != "polygon"
                        or not isinstance(xs, list)
                        or not isinstance(ys, list)
                        or len(xs) < 3
                        or len(xs) != len(ys)
                    ):
                        raise ValueError("invalid polygon")
                    if r.get("region_attributes", {}).get("hold_type") != "hold":
                        continue
                    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                        raise ValueError("polygon outside image")
                    lines.append(
                        "0 %.8f %.8f %.8f %.8f"
                        % (
                            ((x0 + x1) / 2) / w,
                            ((y0 + y1) / 2) / h,
                            (x1 - x0) / w,
                            (y1 - y0) / h,
                        )
                    )
                records.append(
                    (
                        cat,
                        image,
                        f"{cat}_{image.name}",
                        chr(10).join(lines) + (chr(10) if lines else ""),
                    )
                )
            except Exception as e:
                skipped.append(f"{cat}/{image.name}: {e}")
    grouped = defaultdict(list)
    for r in records:
        grouped[r[0]].append(r)
    splits = {s: [] for s in SPLITS}
    rng = random.Random(a.seed)
    for cat in CATEGORIES:
        x = grouped[cat]
        rng.shuffle(x)
        nv = round(len(x) * 0.1)
        nt = round(len(x) * 0.1)
        splits["val"] += x[:nv]
        splits["test"] += x[nv : nv + nt]
        splits["train"] += x[nv + nt :]
    for s in SPLITS:
        rng.shuffle(splits[s])
        for _, im, name, labels in splits[s]:
            shutil.copy2(im, out / "images" / s / name)
            (out / "labels" / s / Path(name).with_suffix(".txt")).write_text(labels)
    (out / "data.yaml").write_text(
        chr(10).join(
            [
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                "  0: hold",
                "",
            ]
        )
    )
    with (out / "conversion_report.txt").open("w") as f:
        f.write(
            f"converted: {len(records)}"
            + chr(10)
            + f"skipped: {len(skipped)}"
            + chr(10)
        )
        for s in SPLITS:
            f.write(
                f"{s}: {len(splits[s])} {dict(Counter(r[0] for r in splits[s]))}"
                + chr(10)
            )
        if skipped:
            f.write(
                chr(10) + "Skipped records:" + chr(10) + chr(10).join(skipped) + chr(10)
            )
    for s in SPLITS:
        if len(list((out / "images" / s).glob("*.jpg"))) != len(splits[s]) or len(
            list((out / "labels" / s).glob("*.txt"))
        ) != len(splits[s]):
            raise RuntimeError(s)
    print(f"Converted {len(records)} images; skipped {len(skipped)}. Output: {out}")
    if a.delete_source:
        shutil.rmtree(root / "4")
        print(f"Deleted source dataset: {root / '4'}")


if __name__ == "__main__":
    main()
