"""Learned pairwise DINOv3 climbing-route discriminator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..interfaces.data_models import Image, Route
from ..utility.dino_pairwise import DINOPairwiseHead
from ..utility.dinov3 import DINOv3
from .dino_clustering_route_discriminator import DINOClusteringRouteDiscriminator
from .route_discriminator import (
    InvalidRouteDiscriminatorConfigError,
)


class DINOLearningRouteDiscriminator(DINOClusteringRouteDiscriminator):
    """Classify hold pairs with a learned head and join positive pairs."""

    implementation_id = "dino_learning_route_discriminator"

    def __init__(
        self,
        weights_path: str | Path,
        pooling: str = "weighted",
        model_dir: str | Path = "models/dinov3",
        device: str | torch.device | None = None,
        pair_threshold: float = 0.5,
    ) -> None:
        if pooling not in {"weighted", "mean"}:
            raise InvalidRouteDiscriminatorConfigError(
                "pooling must be 'weighted' or 'mean'."
            )
        if (
            isinstance(pair_threshold, bool)
            or not isinstance(pair_threshold, (int, float))
            or not np.isfinite(pair_threshold)
            or not 0 <= pair_threshold <= 1
        ):
            raise InvalidRouteDiscriminatorConfigError(
                "pair_threshold must be a number in [0, 1]."
            )
        DINOv3.__init__(self, model_dir=model_dir, device=device)
        self.weights_path = Path(weights_path)
        self.pooling, self.pair_threshold = pooling, float(pair_threshold)
        self.head = DINOPairwiseHead().to(self.device)
        self._head_loaded = False

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "weights_path": str(self.weights_path),
            "pooling": self.pooling,
            "pair_threshold": self.pair_threshold,
            "model_dir": str(self.model_dir),
            "device": str(self.device),
        }

    def _ensure_head_loaded(self) -> None:
        if self._head_loaded:
            return
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"DINO Learning head weights were not found at '{self.weights_path}'."
            )
        state_dict = torch.load(
            self.weights_path, map_location=self.device, weights_only=True
        )
        try:
            self.head.load_state_dict(state_dict)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise InvalidRouteDiscriminatorConfigError(
                "DINO Learning head checkpoint is incompatible."
            ) from exc
        self.head.eval()
        self._head_loaded = True

    def get_routes(self, images, holds) -> list[list[Route]]:
        self._validate_inputs(images, holds)
        self._ensure_head_loaded()
        result = []
        for image, image_holds in zip(images, holds, strict=True):
            if not image_holds:
                result.append([])
                continue
            masks = [self._hold_mask(image, hold) for hold in image_holds]
            embeddings = self.extract_mask_embeddings(image, masks, self.pooling)
            colours = np.stack([self._lab_feature(image, hold) for hold in image_holds])
            color_distances = self._ciede2000_distances(colours)
            color_features = torch.as_tensor(
                np.concatenate(
                    (
                        color_distances[..., None],
                        np.abs(colours[:, None] - colours[None, :]),
                        (colours[:, None] + colours[None, :]) / 2,
                    ),
                    axis=-1,
                ),
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
            groups = [{index} for index in range(len(image_holds))]
            indices = torch.triu_indices(
                len(image_holds), len(image_holds), offset=1, device=embeddings.device
            )
            with torch.inference_mode():
                probabilities = torch.sigmoid(
                    self.head(
                        embeddings[indices[0]],
                        embeddings[indices[1]],
                        color_features[indices[0], indices[1]],
                    )
                )
            for first, second, probability in zip(
                indices[0].tolist(),
                indices[1].tolist(),
                probabilities.tolist(),
                strict=True,
            ):
                if probability >= self.pair_threshold:
                    left = next(group for group in groups if first in group)
                    right = next(group for group in groups if second in group)
                    if left is not right:
                        left.update(right)
                        groups.remove(right)
            groups.sort(key=min)
            result.append(
                [
                    Route({image_holds[index] for index in group}, route_id)
                    for route_id, group in enumerate(groups)
                ]
            )
        return result
