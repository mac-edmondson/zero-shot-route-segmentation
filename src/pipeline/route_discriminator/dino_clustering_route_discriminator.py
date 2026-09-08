"""DINOv3 climbing-route discriminator with configurable clustering."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import cv2
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
)

Pooling = Literal["weighted", "mean"]
ClusteringMethod = Literal["hdbscan", "dbscan", "agglomerative"]
AgglomerativeLinkage = Literal["average", "complete", "single"]

class DINOClusteringRouteDiscriminator(DINOv3):
    """Group holds by DINO/color distance using a configurable clusterer."""

    implementation_id = "dino_clustering_route_discriminator"

    def __init__(
        self,
        pooling: Pooling = "weighted",
        model_dir: str | Path = "models/dinov3",
        device: str | torch.device | None = None,
        min_cluster_size: int = 2,
        min_samples: int | None = None,
        clustering_method: ClusteringMethod = "hdbscan",
        eps: float = 0.3,
        distance_threshold: float = 0.3,
        linkage: AgglomerativeLinkage = "average",
        color_weight: float = 0.0,
    ) -> None:
        if pooling not in {"weighted", "mean"}:
            raise InvalidRouteDiscriminatorConfigError(
                "pooling must be either 'weighted' or 'mean'."
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
        if clustering_method not in {"hdbscan", "dbscan", "agglomerative"}:
            raise InvalidRouteDiscriminatorConfigError(
                "clustering_method must be 'hdbscan', 'dbscan', or 'agglomerative'."
            )
        for name, value in (("eps", eps), ("distance_threshold", distance_threshold)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise InvalidRouteDiscriminatorConfigError(
                    f"{name} must be a finite positive number."
                )
        if linkage not in {"average", "complete", "single"}:
            raise InvalidRouteDiscriminatorConfigError(
                "linkage must be 'average', 'complete', or 'single'."
            )
        if type(min_cluster_size) is not int or min_cluster_size <= 0:
            raise InvalidRouteDiscriminatorConfigError(
                "min_cluster_size must be a positive integer."
            )
        if min_samples is not None and (
            type(min_samples) is not int or min_samples <= 0
        ):
            raise InvalidRouteDiscriminatorConfigError(
                "min_samples must be a positive integer or None."
            )
        super().__init__(model_dir=model_dir, device=device)
        self.pooling = pooling
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.clustering_method = clustering_method
        self.eps = float(eps)
        self.distance_threshold = float(distance_threshold)
        self.linkage = linkage
        self.color_weight = float(color_weight)

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "pooling": self.pooling,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "clustering_method": self.clustering_method,
            "eps": self.eps,
            "distance_threshold": self.distance_threshold,
            "linkage": self.linkage,
            "color_weight": self.color_weight,
            "model_dir": str(self.model_dir),
            "device": str(self.device),
        }

    def get_routes(
        self, images: Sequence[Image], holds: Sequence[Sequence[Hold]]
    ) -> list[list[Route]]:
        """Return one clustered route list for every image."""
        self._validate_inputs(images, holds)
        result = []
        for image, image_holds in zip(images, holds, strict=True):
            if not image_holds:
                result.append([])
                continue
            masks = [self._hold_mask(image, hold) for hold in image_holds]
            features = self.extract_mask_embeddings(image, masks, self.pooling)
            if len(image_holds) == 1:
                labels = np.array([0])
                distances = None
            else:
                distances = self._dino_distances(features.detach().cpu().numpy())
            if self.color_weight > 0 and distances is not None:
                colours = np.stack(
                    [self._lab_feature(image, hold) for hold in image_holds]
                )
                distances = (
                    distances + self.color_weight * self._ciede2000_distances(colours)
                ) / (1 + self.color_weight)
            if distances is not None:
                labels = self._cluster_labels(distances)

            groups: dict[int, set[Hold]] = {}
            noise: list[Hold] = []
            for hold, label in zip(image_holds, labels, strict=True):
                if label == -1:
                    noise.append(hold)
                else:
                    groups.setdefault(int(label), set()).add(hold)
            ordered_groups: list[set[Hold]] = []
            for label in sorted(groups):
                ordered_groups.append(groups[label])
            ordered_groups.extend({hold} for hold in noise)
            result.append(
                [
                    Route(group, route_id)
                    for route_id, group in enumerate(ordered_groups)
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

    @staticmethod
    def _hold_mask(image: Image, hold: Hold) -> PILImage.Image:
        mask = PILImage.new("L", image.size, 0)
        ImageDraw.Draw(mask).polygon(
            [(point.x, point.y) for point in hold.polygon.points], fill=255
        )
        if not mask.getbbox():
            x = max(0, min(image.width - 1, hold.centroid.x))
            y = max(0, min(image.height - 1, hold.centroid.y))
            ImageDraw.Draw(mask).point((x, y), fill=255)
        return mask

    @staticmethod
    def _lab_feature(image: Image, hold: Hold) -> np.ndarray:
        """Return the median CIELAB colour inside a hold polygon."""
        pixels = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB)
        mask = PILImage.new("L", image.size, 0)
        ImageDraw.Draw(mask).polygon(
            [(point.x, point.y) for point in hold.polygon.points], fill=1
        )
        values = pixels[np.asarray(mask, dtype=bool)]
        if not len(values):
            return np.zeros(3, dtype=np.float32)
        values = values.astype(np.float32)
        values[:, 0] *= 100 / 255
        values[:, 1:] = (values[:, 1:] - 128) * 100 / 255
        return np.median(values, axis=0).astype(np.float32)

    @staticmethod
    def _dino_distances(features: np.ndarray) -> np.ndarray:
        normalized = features / np.maximum(
            np.linalg.norm(features, axis=1, keepdims=True), np.finfo(np.float32).eps
        )
        return np.clip(1 - normalized @ normalized.T, 0, 2).astype(np.float64)

    @staticmethod
    def _ciede2000_distances(lab: np.ndarray) -> np.ndarray:
        first, second = lab[:, None, :].astype(float), lab[None, :, :].astype(float)
        l1, a1, b1 = np.moveaxis(first, -1, 0)
        l2, a2, b2 = np.moveaxis(second, -1, 0)
        c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
        c_bar = (c1 + c2) / 2
        twenty_five_seven = 25**7
        g = 0.5 * (1 - np.sqrt(c_bar**7 / (c_bar**7 + twenty_five_seven)))
        ap1, ap2 = (1 + g) * a1, (1 + g) * a2
        cp1, cp2 = np.hypot(ap1, b1), np.hypot(ap2, b2)
        hp1 = np.degrees(np.arctan2(b1, ap1)) % 360
        hp2 = np.degrees(np.arctan2(b2, ap2)) % 360
        delta_l, delta_c = l2 - l1, cp2 - cp1
        delta_h = hp2 - hp1
        delta_h = np.where(delta_h > 180, delta_h - 360, delta_h)
        delta_h = np.where(delta_h < -180, delta_h + 360, delta_h)
        delta_h = np.where(cp1 * cp2 == 0, 0, delta_h)
        delta_h_term = 2 * np.sqrt(cp1 * cp2) * np.sin(np.radians(delta_h) / 2)
        l_bar, cp_bar = (l1 + l2) / 2, (cp1 + cp2) / 2
        h_bar = np.where(
            cp1 * cp2 == 0,
            hp1 + hp2,
            np.where(
                np.abs(hp1 - hp2) <= 180,
                (hp1 + hp2) / 2,
                np.where(hp1 + hp2 < 360, (hp1 + hp2 + 360) / 2, (hp1 + hp2 - 360) / 2),
            ),
        )
        t = 1 - 0.17 * np.cos(np.radians(h_bar - 30))
        t += 0.24 * np.cos(np.radians(2 * h_bar))
        t += 0.32 * np.cos(np.radians(3 * h_bar + 6))
        t -= 0.20 * np.cos(np.radians(4 * h_bar - 63))
        sl = 1 + 0.015 * (l_bar - 50) ** 2 / np.sqrt(20 + (l_bar - 50) ** 2)
        sc, sh = 1 + 0.045 * cp_bar, 1 + 0.015 * cp_bar * t
        rt = (
            -2
            * np.sqrt(cp_bar**7 / (cp_bar**7 + twenty_five_seven))
            * np.sin(np.radians(60) * np.exp(-(((h_bar - 275) / 25) ** 2)))
        )
        distance = np.sqrt(
            (delta_l / sl) ** 2
            + (delta_c / sc) ** 2
            + (delta_h_term / sh) ** 2
            + rt * (delta_c / sc) * (delta_h_term / sh)
        )
        return np.clip(distance / 100, 0, 1)

    @staticmethod
    def mark_routes(
        images: Sequence[Image], routes: Sequence[Sequence[Route]]
    ) -> list[Image]:
        return RouteDiscriminator.mark_routes(images, routes)

    def _hdbscan_labels(self, distances: np.ndarray) -> np.ndarray:
        try:
            import hdbscan
        except ModuleNotFoundError as exc:
            raise InvalidRouteDiscriminatorConfigError(
                "HDBSCAN clustering requires the 'hdbscan' package."
            ) from exc
        return hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="precomputed",
        ).fit_predict(distances)

    def _cluster_labels(self, distances: np.ndarray) -> np.ndarray:
        if self.clustering_method == "hdbscan":
            return self._hdbscan_labels(distances)
        try:
            from sklearn.cluster import AgglomerativeClustering, DBSCAN
        except ModuleNotFoundError as exc:
            raise InvalidRouteDiscriminatorConfigError(
                "DBSCAN and agglomerative clustering require the 'scikit-learn' package."
            ) from exc
        if self.clustering_method == "dbscan":
            return DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples or 5,
                metric="precomputed",
            ).fit_predict(distances)
        return AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="precomputed",
            linkage=self.linkage,
        ).fit_predict(distances)
