import numpy as np
import pytest
import torch
from PIL import Image

from pipeline.interfaces.data_models import Coordinate, Hold, Polygon
from pipeline.interfaces.errors import BatchAlignmentError, InvalidImageError
from pipeline.route_discriminator.dino_route_discriminator import (
    DINORouteDiscriminator,
)
from pipeline.route_discriminator.route_discriminator import (
    InvalidRouteDiscriminatorConfigError,
)
from pipeline.route_discriminator.route_discriminator_factory import (
    route_discriminator_factory,
)


def hold(left: int, top: int, right: int, bottom: int) -> Hold:
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


def patch_tokens() -> torch.Tensor:
    tokens = torch.zeros(40, 40, 384)
    tokens[:, :20, 0] = 1
    tokens[:, 20:, 1] = 1
    return tokens.reshape(1600, 384)


def test_groups_original_holds_from_one_wall(monkeypatch):
    image = Image.new("RGB", (100, 100))
    left, right = hold(10, 10, 20, 20), hold(70, 70, 80, 80)
    discriminator = DINORouteDiscriminator(device="cpu")
    monkeypatch.setattr(discriminator, "extract_patch_tokens", lambda _: patch_tokens())

    routes = discriminator.get_routes([image], [[left, right]])

    assert len(routes) == 1
    assert {frozenset(route.holds) for route in routes[0]} == {
        frozenset({left}),
        frozenset({right}),
    }
    assert {item for route in routes[0] for item in route.holds} == {left, right}
    assert discriminator.configuration["pooling"] == "weighted"
    assert len(discriminator.mark_routes([image], routes)) == 1


def test_pooling_modes_and_centroid_fallback_produce_normalized_features():
    image = Image.new("RGB", (80, 40))
    polygon = hold(1, 1, 5, 5)
    tokens = patch_tokens()
    weighted = DINORouteDiscriminator(pooling="weighted", device="cpu")
    mean = DINORouteDiscriminator(pooling="mean", device="cpu")

    weighted_feature = weighted._pool_hold(tokens, image, polygon)
    mean_feature = mean._pool_hold(tokens, image, polygon)
    outside_feature = weighted._pool_hold(tokens, image, hold(100, 100, 110, 110))

    assert weighted_feature.shape == (384,)
    assert torch.linalg.vector_norm(weighted_feature) == pytest.approx(1.0)
    assert torch.linalg.vector_norm(mean_feature) == pytest.approx(1.0)
    assert torch.linalg.vector_norm(outside_feature) == pytest.approx(1.0)


def test_empty_batches_skip_dino_and_cluster_count_is_capped(monkeypatch):
    image = Image.new("RGB", (100, 100))
    discriminator = DINORouteDiscriminator(device="cpu")
    calls = []
    monkeypatch.setattr(
        discriminator,
        "extract_patch_tokens",
        lambda _: calls.append(True) or patch_tokens(),
    )

    assert discriminator.get_routes([image], [[]]) == [[]]
    routes = discriminator.get_routes([image], [[hold(10, 10, 20, 20)]])
    assert len(routes[0]) == 1
    assert calls == [True]


def test_factory_and_validation_use_ten_cluster_cap():
    factory_discriminator = route_discriminator_factory("DINO")
    assert isinstance(factory_discriminator, DINORouteDiscriminator)
    assert factory_discriminator.configuration["n_clusters"] == 10

    with pytest.raises(InvalidRouteDiscriminatorConfigError):
        DINORouteDiscriminator(pooling="maximum")
    with pytest.raises(InvalidRouteDiscriminatorConfigError):
        DINORouteDiscriminator(color_weight=-1)

    discriminator = DINORouteDiscriminator(device="cpu")
    with pytest.raises(InvalidImageError):
        discriminator.get_routes(["image"], [[]])
    with pytest.raises(BatchAlignmentError):
        discriminator.get_routes([Image.new("RGB", (10, 10))], [])


def test_elbow_selects_three_clusters_for_three_separated_groups():
    features = np.vstack(
        (
            np.zeros((4, 2)),
            np.ones((4, 2)) * 10,
            np.ones((4, 2)) * 20,
        )
    )
    assert DINORouteDiscriminator._elbow_cluster_count(features, 10, 0) == 3


def test_rgb_feature_uses_polygon_pixels_and_is_normalized():
    image = Image.new("RGB", (20, 20), (0, 0, 255))
    for x in range(5, 10):
        for y in range(5, 10):
            image.putpixel((x, y), (255, 0, 0))

    feature = DINORouteDiscriminator(device="cpu")._rgb_feature(image, hold(5, 5, 9, 9))

    assert np.allclose(feature, (1, 0, 0))
    assert np.linalg.norm(feature) == pytest.approx(1.0)


def test_color_weight_controls_fused_feature():
    dino = torch.zeros(384)
    dino[0] = 1
    rgb = np.array([0, 1, 0], dtype=np.float32)

    with_color = DINORouteDiscriminator(color_weight=1, device="cpu")
    dino_only = DINORouteDiscriminator(color_weight=0, device="cpu")

    assert np.array_equal(with_color._combine_features(dino, rgb)[384:], rgb)
    assert np.array_equal(dino_only._combine_features(dino, rgb)[384:], np.zeros(3))
