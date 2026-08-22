"""DINOv3-only climbing-route discriminator."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image as PILImage
from PIL import ImageDraw

from ..interfaces.data_models import Hold, Image, Route
from ..interfaces.errors import BatchAlignmentError, InvalidImageError
from ..utility.dinov3 import DINOv3
from .route_discriminator import (
    InvalidRouteDiscriminatorConfigError,
    RouteDiscriminator,
    _kmeans,
)

Pooling = Literal["weighted", "mean"]


class DINORouteDiscriminator(DINOv3):
    """Group holds by their contextual DINOv3 patch embeddings."""

    implementation_id = "dino_route_discriminator"
    N_CLUSTERS = 5

    def __init__(
        self,
        pooling: Pooling = "weighted",
        random_state: int | None = 0,
        model_dir: str | Path = "models/dinov3",
        device: str | torch.device | None = None,
        color_weight: float = 1.0,
    ) -> None:
        if pooling not in {"weighted", "mean"}:
            raise InvalidRouteDiscriminatorConfigError(
                "pooling must be either 'weighted' or 'mean'."
            )
        if random_state is not None and type(random_state) is not int:
            raise InvalidRouteDiscriminatorConfigError(
                "random_state must be an integer or None."
            )
        if (
            isinstance(color_weight, bool)
            or not isinstance(color_weight, (int, float))
            or not np.isfinite(color_weight)
            or color_weight < 0
        ):
            raise InvalidRouteDiscriminatorConfigError(
                "color_weight must be a finite non-negative number."
            )
        super().__init__(model_dir=model_dir, device=device)
        self.pooling = pooling
        self.random_state = random_state
        self.color_weight = float(color_weight)

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "n_clusters": self.N_CLUSTERS,
            "pooling": self.pooling,
            "random_state": self.random_state,
            "color_weight": self.color_weight,
            "model_dir": str(self.model_dir),
            "device": str(self.device),
        }

    @staticmethod
    def _elbow_cluster_count(
        features: np.ndarray, max_clusters: int, random_state: int | None
    ) -> int:
        """Choose k from the largest distance to the inertia end-point line."""
        upper = min(max_clusters, len(features))
        if upper <= 1:
            return 1
        if upper == 2:
            return 2 if not np.allclose(features[0], features[1]) else 1

        inertias = []
        for n_clusters in range(1, upper + 1):
            labels = _kmeans(features, n_clusters, random_state)
            centers = np.array(
                [
                    features[labels == index].mean(axis=0)
                    if np.any(labels == index)
                    else np.zeros(features.shape[1])
                    for index in range(n_clusters)
                ]
            )
            inertias.append(float(((features - centers[labels]) ** 2).sum()))

        x = np.arange(upper, dtype=np.float64)
        y = np.asarray(inertias, dtype=np.float64)
        denominator = np.hypot(x[-1] - x[0], y[-1] - y[0])
        if denominator == 0:
            return 1
        distances = (
            np.abs((x - x[0]) * (y[-1] - y[0]) - (y - y[0]) * (x[-1] - x[0]))
            / denominator
        )
        return int(np.argmax(distances)) + 1

    def get_routes(
        self, images: Sequence[Image], holds: Sequence[Sequence[Hold]]
    ) -> list[list[Route]]:
        """Return one DINO-clustered route list for every image."""
        self._validate_inputs(images, holds)
        result = []
        for image, image_holds in zip(images, holds, strict=True):
            if not image_holds:
                result.append([])
                continue
            tokens = self.extract_patch_tokens(image)
            features = torch.stack(
                [self._pool_hold(tokens, image, hold) for hold in image_holds]
            )
            features = np.stack(
                [
                    self._combine_features(feature, self._rgb_feature(image, hold))
                    for feature, hold in zip(features, image_holds, strict=True)
                ]
            )
            n_clusters = self._elbow_cluster_count(
                features, self.N_CLUSTERS, self.random_state
            )
            labels = _kmeans(features, n_clusters, self.random_state)
            groups: dict[int, set[Hold]] = {}
            for hold, label in zip(image_holds, labels, strict=True):
                groups.setdefault(int(label), set()).add(hold)
            result.append(
                [
                    Route(group, route_id)
                    for route_id, group in enumerate(groups.values())
                ]
            )
        return result

    @staticmethod
    def _validate_inputs(
        images: Sequence[Image], holds: Sequence[Sequence[Hold]]
    ) -> None:
        if (
            isinstance(images, (str, bytes))
            or not isinstance(images, Sequence)
            or any(not isinstance(image, PILImage.Image) for image in images)
        ):
            raise InvalidImageError("images must be a sequence of PIL images.")
        if isinstance(holds, (str, bytes)) or not isinstance(holds, Sequence):
            raise TypeError("holds must be a sequence of Hold sequences.")
        if len(images) != len(holds):
            raise BatchAlignmentError("images and holds must align.")
        if any(
            isinstance(batch, (str, bytes))
            or not isinstance(batch, Sequence)
            or any(not isinstance(hold, Hold) for hold in batch)
            for batch in holds
        ):
            raise TypeError("holds must be a sequence of Hold sequences.")

    def _pool_hold(
        self, tokens: torch.Tensor, image: Image, hold: Hold
    ) -> torch.Tensor:
        mask = PILImage.new("L", image.size, 0)
        ImageDraw.Draw(mask).polygon(
            [(point.x, point.y) for point in hold.polygon.points], fill=255
        )
        resized = mask.resize(
            (self.IMAGE_SIZE, self.IMAGE_SIZE), PILImage.Resampling.NEAREST
        )
        coverage = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).to(
            device=tokens.device, dtype=tokens.dtype
        )
        coverage = coverage.reshape(40, 16, 40, 16).mean(dim=(1, 3)).flatten()
        weights = coverage if self.pooling == "weighted" else coverage.gt(0)
        weights = weights.to(dtype=tokens.dtype)

        if not torch.any(weights):
            x = max(0, min(39, hold.centroid.x * 40 // image.width))
            y = max(0, min(39, hold.centroid.y * 40 // image.height))
            weights[y * 40 + x] = 1
        embedding = (tokens * weights[:, None]).sum(0) / weights.sum()
        return torch.nn.functional.normalize(embedding, dim=0)

    def _combine_features(
        self, dino_feature: torch.Tensor, rgb_feature: np.ndarray
    ) -> np.ndarray:
        """Fuse independently normalized DINO and mean-RGB hold features."""
        dino = dino_feature.detach().cpu().numpy()
        return np.concatenate((dino, self.color_weight * rgb_feature))

    @staticmethod
    def _rgb_feature(image: Image, hold: Hold) -> np.ndarray:
        """Return the L2-normalized mean RGB value inside a hold polygon."""
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        mask = PILImage.new("L", image.size, 0)
        ImageDraw.Draw(mask).polygon(
            [(point.x, point.y) for point in hold.polygon.points], fill=1
        )
        values = pixels[np.asarray(mask, dtype=bool)]
        if not len(values):
            return np.zeros(3, dtype=np.float32)
        mean = values.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm else mean

    @staticmethod
    def mark_routes(
        images: Sequence[Image], routes: Sequence[Sequence[Route]]
    ) -> list[Image]:
        return RouteDiscriminator.mark_routes(images, routes)
