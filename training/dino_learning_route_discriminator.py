"""Train the learned DINOv3 pairwise route discriminator head."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from PIL import ImageDraw
from torch.utils.data import DataLoader, TensorDataset

from pipeline.interfaces.data_models import Coordinate, Hold, Polygon
from pipeline.route_discriminator.dino_clustering_route_discriminator import (
    DINOClusteringRouteDiscriminator,
)
from pipeline.utility.dino_pairwise import DINOPairwiseHead
from pipeline.utility.dinov3 import DINOv3


@dataclass(frozen=True)
class HoldAnnotation:
    index: int
    points: tuple[tuple[int, int], ...]
    route_id: str


@dataclass(frozen=True)
class ImageAnnotations:
    path: Path
    holds: tuple[HoldAnnotation, ...]


def parse_via(
    data_dir: Path, annotation: str | Path
) -> tuple[list[ImageAnnotations], dict[str, int]]:
    annotation_path = Path(annotation)
    if not annotation_path.is_absolute():
        annotation_path = data_dir / annotation_path
    data = json.loads(annotation_path.read_text())
    images = []
    skipped = {"images": 0, "regions": 0, "missing_files": 0, "invalid_polygons": 0}
    for metadata in data.get("_via_img_metadata", {}).values():
        image_path = data_dir / metadata["filename"]
        if not image_path.is_file():
            skipped["missing_files"] += 1
            skipped["images"] += 1
            continue
        holds = []
        for index, region in enumerate(metadata.get("regions", [])):
            attributes = region.get("region_attributes", {})
            if (
                attributes.get("hold_type") == "volume"
                or attributes.get("route_id") is None
            ):
                skipped["regions"] += 1
                continue
            shape = region.get("shape_attributes", {})
            xs, ys = shape.get("all_points_x"), shape.get("all_points_y")
            if (
                shape.get("name") != "polygon"
                or not isinstance(xs, list)
                or not isinstance(ys, list)
                or len(xs) < 3
                or len(xs) != len(ys)
            ):
                skipped["invalid_polygons"] += 1
                continue
            try:
                points = tuple((int(x), int(y)) for x, y in zip(xs, ys, strict=True))
            except (TypeError, ValueError, OverflowError):
                skipped["invalid_polygons"] += 1
                continue
            holds.append(HoldAnnotation(index, points, str(attributes["route_id"])))
        if holds:
            images.append(ImageAnnotations(image_path, tuple(holds)))
        else:
            skipped["images"] += 1
    return images, skipped


def split_images(
    images: list[ImageAnnotations], val_ratio: float, seed: int
) -> tuple[list[ImageAnnotations], list[ImageAnnotations]]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]


def _masks(
    image: PILImage.Image, holds: tuple[HoldAnnotation, ...]
) -> list[PILImage.Image]:
    masks = []
    for hold in holds:
        mask = PILImage.new("L", image.size, 0)
        ImageDraw.Draw(mask).polygon(hold.points, fill=255)
        masks.append(mask)
    return masks


def _cache_path(cache_dir: Path, image_path: Path) -> Path:
    key = hashlib.sha1(str(image_path.resolve()).encode()).hexdigest()
    return cache_dir / f"{key}.pt"


def _signature(record: ImageAnnotations) -> str:
    source = {
        "path": str(record.path.resolve()),
        "stat": record.path.stat().st_mtime_ns,
        "holds": [(h.index, h.points, h.route_id) for h in record.holds],
    }
    return json.dumps(source, sort_keys=True)


def load_or_extract_embeddings(
    dino: DINOv3, record: ImageAnnotations, cache_dir: Path
) -> torch.Tensor:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, record.path)
    signature = _signature(record)
    if path.is_file():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached.get("signature") == signature:
            return cached["embeddings"]
    try:
        with PILImage.open(record.path) as image:
            image = image.convert("RGB")
            with torch.inference_mode():
                embeddings = dino.extract_mask_embeddings(
                    image, _masks(image, record.holds)
                )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not extract embeddings from '{record.path}'."
        ) from exc
    embeddings = embeddings.detach().cpu()
    torch.save({"signature": signature, "embeddings": embeddings}, path)
    return embeddings


def color_features(record: ImageAnnotations) -> torch.Tensor:
    with PILImage.open(record.path) as image:
        image = image.convert("RGB")
        holds = [
            Hold(Polygon(tuple(Coordinate(x, y) for x, y in annotation.points)))
            for annotation in record.holds
        ]
        colours = np.stack(
            [
                DINOClusteringRouteDiscriminator._lab_feature(image, hold)
                for hold in holds
            ]
        )
    distances = DINOClusteringRouteDiscriminator._ciede2000_distances(colours)
    return torch.from_numpy(
        np.concatenate(
            (
                distances[..., None],
                np.abs(colours[:, None] - colours[None, :]),
                (colours[:, None] + colours[None, :]) / 2,
            ),
            axis=-1,
        ).astype(np.float32)
    )


def make_pairs(
    embeddings: torch.Tensor,
    holds: tuple[HoldAnnotation, ...],
    pair_color_features: torch.Tensor,
    max_pairs_per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_pairs_per_class <= 0:
        raise ValueError("max_pairs_per_class must be positive.")
    positives = [
        (a, b)
        for a, b in itertools.combinations(range(len(holds)), 2)
        if holds[a].route_id == holds[b].route_id
    ]
    negatives = [
        (a, b)
        for a, b in itertools.combinations(range(len(holds)), 2)
        if holds[a].route_id != holds[b].route_id
    ]
    count = min(max_pairs_per_class, len(positives), len(negatives))
    if count == 0:
        empty = torch.empty((0,))
        return (torch.empty((0, 384)), torch.empty((0, 384)), empty, empty)
    rng = random.Random(seed)
    positives, negatives = rng.sample(positives, count), rng.sample(negatives, count)
    pairs = positives + negatives
    labels = torch.cat((torch.ones(count), torch.zeros(count)))
    first = torch.stack([embeddings[a] for a, _ in pairs])
    second = torch.stack([embeddings[b] for _, b in pairs])
    colours = torch.stack([pair_color_features[a, b] for a, b in pairs])
    order = list(range(len(pairs)))
    rng.shuffle(order)
    return first[order], second[order], labels[order], colours[order]


def build_pairs(
    dino: DINOv3,
    images: list[ImageAnnotations],
    cache_dir: Path,
    max_pairs_per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    first, second, labels, colours, skipped = [], [], [], [], 0
    for index, record in enumerate(images):
        try:
            embeddings = load_or_extract_embeddings(dino, record, cache_dir)
            pair_color_features = color_features(record)
        except RuntimeError:
            skipped += 1
            continue
        a, b, y, c = make_pairs(
            embeddings,
            record.holds,
            pair_color_features,
            max_pairs_per_class,
            seed + index,
        )
        if not len(y):
            skipped += 1
            continue
        first.append(a)
        second.append(b)
        labels.append(y)
        colours.append(c)
    if not labels:
        raise RuntimeError("No trainable positive and negative pairs remain.")
    return (
        torch.cat(first),
        torch.cat(second),
        torch.cat(labels),
        torch.cat(colours),
        skipped,
    )


def metrics(
    model: DINOPairwiseHead, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    total_loss = total = correct = tp = fp = fn = 0
    with torch.inference_mode():
        for first, second, labels, colours in loader:
            logits = model(first.to(device), second.to(device), colours.to(device))
            loss = model.loss(logits, labels.to(device))
            predictions = logits >= 0
            total_loss += loss.item() * len(labels)
            total += len(labels)
            correct += (predictions == labels.to(device).bool()).sum().item()
            tp += (predictions & labels.to(device).bool()).sum().item()
            fp += (predictions & ~labels.to(device).bool()).sum().item()
            fn += ((~predictions) & labels.to(device).bool()).sum().item()
    denominator = 2 * tp + fp + fn
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "f1": 2 * tp / denominator if denominator else 0.0,
    }


def train(
    head: DINOPairwiseHead,
    train_data: TensorDataset,
    val_data: TensorDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output: Path,
) -> dict[str, Any]:
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    best = {"f1": -1.0}
    for epoch in range(1, epochs + 1):
        head.train()
        for first, second, labels, colours in loader:
            optimizer.zero_grad()
            logits = head(first.to(device), second.to(device), colours.to(device))
            head.loss(logits, labels.to(device)).backward()
            optimizer.step()
        train_metrics = metrics(head, loader, device)
        val_metrics = metrics(head, val_loader, device)
        print(f"epoch={epoch} train={train_metrics} val={val_metrics}")
        if val_metrics["f1"] > best["f1"]:
            best = {"epoch": epoch, **val_metrics}
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(head.state_dict(), output)
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/bh"))
    parser.add_argument("--annotation", default="annotation.json")
    parser.add_argument("--model-dir", type=Path, default=Path("models/dinov3"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/bh/.dino-cache"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/dino_learning_route_discriminator.pt"),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "mps")
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs-per-class", type=int, default=256)
    parser.add_argument("--clear-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive.")
    if args.clear_cache and args.cache_dir.exists():
        for path in args.cache_dir.glob("*.pt"):
            path.unlink()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    images, skipped = parse_via(args.data_dir, args.annotation)
    train_images, val_images = split_images(images, args.val_ratio, args.seed)
    requested_device = None if args.device == "auto" else args.device
    dino = DINOv3(args.model_dir, requested_device)
    train_first, train_second, train_labels, train_colours, train_skipped = build_pairs(
        dino, train_images, args.cache_dir, args.max_pairs_per_class, args.seed
    )
    val_first, val_second, val_labels, val_colours, val_skipped = build_pairs(
        dino, val_images, args.cache_dir, args.max_pairs_per_class, args.seed + 1
    )
    device = dino.device
    head = DINOPairwiseHead().to(device)
    best = train(
        head,
        TensorDataset(train_first, train_second, train_labels, train_colours),
        TensorDataset(val_first, val_second, val_labels, val_colours),
        device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.output,
    )
    summary = {
        "train_images": len(train_images),
        "val_images": len(val_images),
        "train_pairs": len(train_labels),
        "val_pairs": len(val_labels),
        "skipped": skipped,
        "train_skipped": train_skipped,
        "val_skipped": val_skipped,
        "best": best,
        "seed": args.seed,
        "pair_cap": args.max_pairs_per_class,
        "device": str(device),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
