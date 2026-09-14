"""Learned pairwise classification for DINOv3 hold embeddings."""

from __future__ import annotations

import torch


class DINOPairwiseHead(torch.nn.Module):
    """Small binary classifier for whether two holds share a route."""

    EMBEDDING_SIZE = 384
    HIDDEN_SIZE = 128
    COLOR_FEATURE_SIZE = 7
    FEATURE_SIZE = EMBEDDING_SIZE * 2 + COLOR_FEATURE_SIZE

    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.FEATURE_SIZE, self.HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(self.HIDDEN_SIZE, 1),
        )

    @staticmethod
    def pair_features(
        first: torch.Tensor,
        second: torch.Tensor,
        color_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            first.shape != second.shape
            or first.shape[-1] != DINOPairwiseHead.EMBEDDING_SIZE
        ):
            raise ValueError("pair embeddings must have matching shape (..., 384).")
        if color_features is None:
            color_features = torch.zeros(
                (*first.shape[:-1], DINOPairwiseHead.COLOR_FEATURE_SIZE),
                device=first.device,
                dtype=first.dtype,
            )
        if color_features.shape != (
            *first.shape[:-1],
            DINOPairwiseHead.COLOR_FEATURE_SIZE,
        ):
            raise ValueError("color_features must match the pair batch shape (..., 7).")
        color_features = color_features.to(device=first.device, dtype=first.dtype)
        return torch.cat(
            (torch.abs(first - second), first * second, color_features), dim=-1
        )

    def forward(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        color_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.classifier(
            self.pair_features(first, second, color_features)
        ).squeeze(-1)

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.to(dtype=logits.dtype)
        )
