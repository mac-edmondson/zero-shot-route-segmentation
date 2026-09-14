""" Load ground-truth holds from the restructured COCO-style JSON export. """


from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from ..interfaces.data_models import Coordinate, Hold, Image, Polygon, Route

UNKNOWN_ROUTE_LABEL = "Unknown"


def pixel_hash(image: Image) -> str:
    """Fingerprint an image's exact pixel content."""
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def build_routes(holds: Iterable[Hold]) -> list[Route]:
    """Group holds sharing a route_label into Route objects.
    """
    holds_by_label: dict[str, list[Hold]] = {}
    for hold in holds:
        label = hold.attributes.get("route_label")
        if label is None or label == UNKNOWN_ROUTE_LABEL:
            continue
        holds_by_label.setdefault(label, []).append(hold)

    return [
        Route(holds=set(route_holds), route_id=int(label.removeprefix("route_")))
        for label, route_holds in holds_by_label.items()
    ]


def load_coco_restructured(path: str | Path) -> list[list[Hold]]:
    """Load ground-truth holds, grouped per image, from the JSON at `path`."""
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    image_ids = [image["id"] for image in data["images"]]
    holds_by_image_id: dict[int, list[Hold]] = {image_id: [] for image_id in image_ids}

    for annotation in data["annotations"]:
        points = annotation["segmentation"][0]
        polygon = Polygon(
            tuple(
                Coordinate(int(round(x)), int(round(y)))
                for x, y in zip(points[0::2], points[1::2], strict=True)
            )
        )

        route_label = annotation.get("route_label")
        if isinstance(route_label, str) and route_label.lower() == "unknown":
            route_label = UNKNOWN_ROUTE_LABEL

        hold = Hold(polygon, attributes={"route_label": route_label})
        holds_by_image_id[annotation["image_id"]].append(hold)

    return [holds_by_image_id[image_id] for image_id in image_ids]


def load_via_csv(path: str | Path) -> list[list[Hold]]:
    """Load ground-truth holds, grouped per image, from a VIA CSV export."""
    holds_by_filename: dict[str, list[Hold]] = {}

    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            holds_by_filename.setdefault(row["filename"], [])
            shape = json.loads(row["region_shape_attributes"])
            if shape.get("name") != "polygon":
                continue

            polygon = Polygon(
                tuple(
                    Coordinate(int(round(x)), int(round(y)))
                    for x, y in zip(
                        shape["all_points_x"], shape["all_points_y"], strict=True
                    )
                )
            )
            holds_by_filename[row["filename"]].append(
                Hold(polygon, attributes={"route_label": None})
            )

    return list(holds_by_filename.values())
