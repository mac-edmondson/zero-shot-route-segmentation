#!/usr/bin/env python3
"""Annotate detected routes on an input image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

# Add src to sys.path if not installed in editable mode
src_path = str(Path(__file__).resolve().parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pipeline.hold_detector.hold_detector_factory import (
    AVAILABLE_HOLD_DETECTORS,
    hold_detector_factory,
)
from pipeline.route_discriminator.route_discriminator import RouteDiscriminator
from pipeline.route_discriminator.route_discriminator_factory import (
    AVAILABLE_ROUTE_DISCRIMINATORS,
    route_discriminator_factory,
)


def main() -> None:
    hold_detector_map = {name.lower(): name for name in AVAILABLE_HOLD_DETECTORS}
    hold_detector_map["mock"] = "Mock"

    route_discriminator_map = {
        name.lower(): name for name in AVAILABLE_ROUTE_DISCRIMINATORS
    }
    route_discriminator_map["mock"] = "Mock"

    parser = argparse.ArgumentParser(
        description="Detect holds and annotate routes in an image."
    )
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument(
        "--hold-detector",
        "-hd",
        default="Mask-RCNN",
        help=f"Hold detector to use (choices: {', '.join(hold_detector_map.keys())}, default: Mask-RCNN)",
    )
    parser.add_argument(
        "--route-detector",
        "-rd",
        "--route-discriminator",
        dest="route_detector",
        default="Color Only",
        help=f"Route detector/discriminator to use (choices: {', '.join(route_discriminator_map.keys())}, default: Color Only)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path for the annotated output image (default: <input_stem>_routes<input_ext>)",
    )
    args = parser.parse_args()

    hd_key = args.hold_detector.lower()
    if hd_key not in hold_detector_map:
        parser.error(
            f"Unknown hold detector '{args.hold_detector}'. Choose from: {list(hold_detector_map.keys())}"
        )

    rd_key = args.route_detector.lower()
    if rd_key not in route_discriminator_map:
        parser.error(
            f"Unknown route detector '{args.route_detector}'. Choose from: {list(route_discriminator_map.keys())}"
        )

    canonical_hd = hold_detector_map[hd_key]
    if canonical_hd == "Mock":
        from pipeline.hold_detector.mock_hold_detector import MockHoldDetector

        hold_detector = MockHoldDetector()
    elif canonical_hd == "YOLO":
        hold_detector = hold_detector_factory(
            canonical_hd, config={"score_threshold": 0.4}
        )
    else:
        hold_detector = hold_detector_factory(canonical_hd)

    canonical_rd = route_discriminator_map[rd_key]
    if canonical_rd == "Mock":
        from pipeline.route_discriminator.mock_route_discriminator import (
            MockRouteDiscriminator,
        )

        route_discriminator = MockRouteDiscriminator()
    else:
        route_discriminator = route_discriminator_factory(canonical_rd)

    image = Image.open(args.image).convert("RGB")

    holds_batch = list(hold_detector.get_holds([image]))
    holds = holds_batch[0] if holds_batch else []

    routes_batch = list(route_discriminator.get_routes([image], holds_batch))
    routes = routes_batch[0] if routes_batch else []

    annotated_batch = RouteDiscriminator.mark_routes([image], [routes])
    annotated = annotated_batch[0]

    output_path = (
        args.output
        if args.output is not None
        else args.image.with_name(f"{args.image.stem}_routes{args.image.suffix}")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)

    total_holds_in_routes = sum(len(r.holds) for r in routes)
    print(
        f"Detected {len(holds)} holds, grouped into {len(routes)} routes ({total_holds_in_routes} assigned holds)."
    )
    print(f"Saved annotated image to {output_path}")


if __name__ == "__main__":
    main()
