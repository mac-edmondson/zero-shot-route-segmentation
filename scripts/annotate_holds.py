#!/usr/bin/env python3
"""Annotate detected holds on an input image."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw

# Add src to sys.path if not installed in editable mode
src_path = str(Path(__file__).resolve().parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pipeline.hold_detector.hold_detector_factory import (
    AVAILABLE_HOLD_DETECTORS,
    hold_detector_factory,
)
from pipeline.interfaces.data_models import Hold


def mark_holds(image: Image.Image, holds: Sequence[Hold]) -> Image.Image:
    """Render hold polygons for visual debugging."""
    overlay = image.convert("RGB").copy()
    fill_overlay = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_overlay)
    for hold in holds:
        points = [(point.x, point.y) for point in hold.polygon.points]
        fill_draw.polygon(points, fill=(255, 80, 0, 102))
    overlay = Image.alpha_composite(overlay.convert("RGBA"), fill_overlay).convert(
        "RGB"
    )

    draw = ImageDraw.Draw(overlay)
    for hold in holds:
        points = [
            (point.x, point.y)
            for point in (*hold.polygon.points, hold.polygon.points[0])
        ]
        draw.line(points, fill=(255, 80, 0), width=5)
    return overlay


def main() -> None:
    # Normalize detector choices for case-insensitive lookup
    detector_map = {name.lower(): name for name in AVAILABLE_HOLD_DETECTORS}
    detector_map["mock"] = "Mock"

    parser = argparse.ArgumentParser(description="Annotate holds in an image.")
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument(
        "--detector",
        "-d",
        default="Mask-RCNN",
        help=f"Hold detector to use (choices: {', '.join(detector_map.keys())}, default: Mask-RCNN)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path for the annotated output image (default: <input_stem>_annotated<input_ext>)",
    )
    args = parser.parse_args()

    key = args.detector.lower()
    if key not in detector_map:
        parser.error(
            f"Unknown detector '{args.detector}'. Choose from: {list(detector_map.keys())}"
        )

    canonical_name = detector_map[key]
    if canonical_name == "Mock":
        from pipeline.hold_detector.mock_hold_detector import MockHoldDetector

        detector = MockHoldDetector()
    elif canonical_name == "YOLO":
        detector = hold_detector_factory(
            canonical_name, config={"score_threshold": 0.4}
        )
    else:
        detector = hold_detector_factory(canonical_name)

    image = Image.open(args.image).convert("RGB")
    holds_batch = list(detector.get_holds([image]))
    holds = holds_batch[0] if holds_batch else []

    annotated = mark_holds(image, holds)

    output_path = (
        args.output
        if args.output is not None
        else args.image.with_name(f"{args.image.stem}_annotated{args.image.suffix}")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    print(f"Detected {len(holds)} holds. Saved annotated image to {output_path}")


if __name__ == "__main__":
    main()
