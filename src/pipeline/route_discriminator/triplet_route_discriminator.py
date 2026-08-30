"""Pretrained TripletNet route discriminator."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from .route_discriminator import RouteDiscriminator
from ..interfaces.data_models import Hold, Route
from ..interfaces.errors import BatchAlignmentError


class TripletRouteDiscriminator:
    implementation_id = "triplet_route_discriminator"
    _WEIGHTS = (
        Path(__file__).resolve().parents[3]
        / "models"
        / "mask_rcnn_hold_detector"
        / "triplet_network_final.pt"
    )

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | torch.device | None = None,
        median_threshold: float = 0.7,
        max_threshold: float = 2.65,
        **config: object,
    ):
        self.weights_path = Path(weights_path or self._WEIGHTS)
        self.device = torch.device(
            "cuda" if device is None and torch.cuda.is_available() else device or "cpu"
        )
        self.median_threshold, self.max_threshold = median_threshold, max_threshold
        self.model = None
        self.preprocess = None

    @property
    def configuration(self):
        return {
            "weights_path": str(self.weights_path),
            "device": str(self.device),
            "median_threshold": self.median_threshold,
            "max_threshold": self.max_threshold,
        }

    def load_model(self):
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"TripletNet weights were not found at '{self.weights_path}'."
            )
        try:
            import torchvision
        except Exception as error:
            raise RuntimeError("torchvision is required for TripletNet.") from error
        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )
        self.preprocess = torchvision.models.ResNet50_Weights.DEFAULT.transforms()
        model = torch.nn.Module()
        model.resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
        )

        def forward(x):
            return torch.nn.functional.normalize(
                model.fc(model.resnet(x).flatten(1)), p=2, dim=1
            )

        model.forward = forward
        model.load_state_dict(
            torch.load(self.weights_path, map_location=self.device, weights_only=True)
        )
        self.model = model.to(self.device).eval()

    def get_routes(
        self, images: Sequence[Image.Image], holds: Sequence[Sequence[Hold]]
    ) -> list[list[Route]]:
        if len(images) != len(holds):
            raise BatchAlignmentError("images and holds must align.")
        if self.model is None:
            self.load_model()
        result = []
        for image, batch in zip(images, holds):
            if not batch:
                result.append([])
                continue
            embeds = self._embeddings(image, batch)
            groups = []
            for hold, embedding in zip(batch, embeds):
                for group in groups:
                    distances = [
                        float(
                            torch.nn.functional.pairwise_distance(
                                embedding[None], other[None]
                            ).square()
                        )
                        for other in group[1]
                    ]
                    if (
                        np.median(distances) <= self.median_threshold
                        and max(distances) <= self.max_threshold
                    ):
                        group[0].add(hold)
                        group[1].append(embedding)
                        break
                else:
                    groups.append(({hold}, [embedding]))
            result.append([Route(group[0], i) for i, group in enumerate(groups)])
        return result

    def _embeddings(self, image, holds):
        crops = []
        array = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        for hold in holds:
            mask = np.zeros(array.shape[:2], np.uint8)
            cv2.fillPoly(
                mask, [np.array([(p.x, p.y) for p in hold.polygon.points], np.int32)], 1
            )
            x = [p.x for p in hold.polygon.points]
            y = [p.y for p in hold.polygon.points]
            crop = (array * mask[:, :, None])[
                int(min(y)) : int(max(y)) + 1, int(min(x)) : int(max(x)) + 1
            ]
            crops.append(
                self.preprocess(torch.from_numpy(crop.copy()).permute(2, 0, 1))
            )
        with torch.no_grad():
            return self.model(torch.stack(crops).to(self.device)).cpu()

    @staticmethod
    def mark_routes(images, routes):
        return RouteDiscriminator.mark_routes(images, routes)
