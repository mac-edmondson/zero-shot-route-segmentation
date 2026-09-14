import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from pipeline.interfaces.data_models import Coordinate, Hold, Polygon
from pipeline.interfaces.errors import InvalidImageError
from pipeline.route_discriminator.dino_learning_route_discriminator import (
    DINOLearningRouteDiscriminator,
)
from pipeline.route_discriminator.route_discriminator import (
    InvalidRouteDiscriminatorConfigError,
)
from pipeline.route_discriminator.route_discriminator_factory import (
    route_discriminator_factory,
)
from pipeline.utility.dino_pairwise import DINOPairwiseHead


def make_hold(left, top, right, bottom):
    return Hold(
        Polygon(
            (
                Coordinate(left, top),
                Coordinate(right, top),
                Coordinate(right, bottom),
                Coordinate(left, bottom),
            )
        )
    )


def test_pairwise_head_forward_and_loss():
    head = DINOPairwiseHead()
    logits = head(torch.randn(4, 384), torch.randn(4, 384))
    assert logits.shape == (4,)
    head.loss(logits, torch.tensor([1, 0, 1, 0])).backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_pairwise_head_rejects_wrong_shape():
    with pytest.raises(ValueError):
        DINOPairwiseHead()(torch.zeros(384), torch.zeros(383))


def test_checkpoint_loading_and_configuration(tmp_path):
    path = tmp_path / "head.pt"
    torch.save(DINOPairwiseHead().state_dict(), path)
    discriminator = DINOLearningRouteDiscriminator(path, device="cpu")
    discriminator._ensure_head_loaded()
    assert discriminator._head_loaded
    assert discriminator.configuration["pair_threshold"] == 0.5


def test_missing_checkpoint_is_rejected(tmp_path):
    discriminator = DINOLearningRouteDiscriminator(
        tmp_path / "missing.pt", device="cpu"
    )
    with pytest.raises(FileNotFoundError):
        discriminator._ensure_head_loaded()


def test_connected_components_group_transitive_pairs(tmp_path, monkeypatch):
    path = tmp_path / "head.pt"
    torch.save(DINOPairwiseHead().state_dict(), path)
    discriminator = DINOLearningRouteDiscriminator(path, device="cpu")
    image = Image.new("RGB", (100, 100))
    holds = [
        make_hold(5, 5, 10, 10),
        make_hold(30, 30, 35, 35),
        make_hold(60, 60, 65, 65),
    ]
    monkeypatch.setattr(
        discriminator, "extract_mask_embeddings", lambda *_: torch.ones(3, 384)
    )

    class FakeHead(torch.nn.Module):
        def forward(self, first, second, color_features):
            return torch.tensor([10.0, -10.0, 10.0])

    discriminator.head = FakeHead()
    discriminator._head_loaded = True
    routes = discriminator.get_routes([image], [holds])[0]
    assert len(routes) == 1
    assert routes[0].holds == set(holds)


def test_validation_and_factory(tmp_path):
    with pytest.raises(InvalidRouteDiscriminatorConfigError):
        DINOLearningRouteDiscriminator(tmp_path / "head.pt", pair_threshold=1.1)
    discriminator = DINOLearningRouteDiscriminator(tmp_path / "head.pt", device="cpu")
    with pytest.raises(InvalidImageError):
        discriminator.get_routes(["image"], [[]])
    assert isinstance(
        route_discriminator_factory(
            "DINO Learning", {"weights_path": tmp_path / "head.pt"}
        ),
        DINOLearningRouteDiscriminator,
    )


def test_real_checkpoint_inference():
    root = Path(__file__).parents[2]
    checkpoint = root / "models/dino_learning_route_discriminator.pt"
    annotation_path = root / "data/bh/bh-annotation.json"
    image_path = root / "data/bh/bh/0003.jpg"
    if not all(path.is_file() for path in (checkpoint, annotation_path, image_path)):
        pytest.skip("trained checkpoint and Bh inference data are unavailable")

    metadata = json.loads(annotation_path.read_text())
    record = next(
        item
        for item in metadata["_via_img_metadata"].values()
        if item["filename"] == image_path.name
    )
    holds = []
    for region in record["regions"]:
        attributes = region["region_attributes"]
        shape = region["shape_attributes"]
        if attributes.get("hold_type") == "volume" or "route_id" not in attributes:
            continue
        holds.append(
            Hold(
                Polygon(
                    tuple(
                        Coordinate(int(x), int(y))
                        for x, y in zip(
                            shape["all_points_x"], shape["all_points_y"], strict=True
                        )
                    )
                )
            )
        )

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    discriminator = DINOLearningRouteDiscriminator(
        checkpoint,
        model_dir=root / "models/dinov3",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    routes = discriminator.get_routes([image], [holds[:3]])

    assert len(routes) == 1
    assert routes[0]
    assert set().union(*(route.holds for route in routes[0])) == set(holds[:3])
