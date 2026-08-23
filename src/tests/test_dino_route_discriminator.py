import numpy as np
import pytest
import torch
from PIL import Image

from pipeline.interfaces.data_models import Coordinate, Hold, Polygon
from pipeline.interfaces.errors import BatchAlignmentError, InvalidImageError
from pipeline.route_discriminator.dino_route_discriminator import DINORouteDiscriminator
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
    tokens = torch.zeros(80, 80, 384)
    tokens[:, :40, 0] = 1
    tokens[:, 40:, 1] = 1
    return tokens.reshape(6400, 384)


def test_groups_holds_and_preserves_pooling_interface(monkeypatch):
    image = Image.new("RGB", (100, 100))
    left, right = hold(10, 10, 20, 20), hold(70, 70, 80, 80)
    discriminator = DINORouteDiscriminator(device="cpu")
    monkeypatch.setattr(discriminator, "extract_patch_tokens", lambda _: patch_tokens())
    monkeypatch.setattr(
        discriminator, "_hdbscan_labels", lambda distances: np.array([0, 1])
    )

    routes = discriminator.get_routes([image], [[left, right]])

    assert {frozenset(route.holds) for route in routes[0]} == {
        frozenset({left}),
        frozenset({right}),
    }
    assert discriminator.configuration["pooling"] == "weighted"
    assert discriminator.configuration["color_weight"] == 0.0
    assert len(discriminator.mark_routes([image], routes)) == 1


def test_single_hold_is_a_route_without_hdbscan(monkeypatch):
    image = Image.new("RGB", (100, 100))
    discriminator = DINORouteDiscriminator(device="cpu")
    monkeypatch.setattr(discriminator, "extract_patch_tokens", lambda _: patch_tokens())
    monkeypatch.setattr(
        discriminator, "_hdbscan_labels", lambda _: pytest.fail("not called")
    )
    only = hold(10, 10, 20, 20)

    routes = discriminator.get_routes([image], [[only]])
    assert routes
    assert routes[0][0].holds == {only}


def test_pooling_modes_and_centroid_fallback_produce_normalized_features():
    image = Image.new("RGB", (80, 40))
    polygon = hold(1, 1, 5, 5)
    tokens = patch_tokens()
    weighted = DINORouteDiscriminator(pooling="weighted", device="cpu")
    mean = DINORouteDiscriminator(pooling="mean", device="cpu")

    assert torch.linalg.vector_norm(
        weighted._pool_hold(tokens, image, polygon)
    ) == pytest.approx(1.0)
    assert torch.linalg.vector_norm(
        mean._pool_hold(tokens, image, polygon)
    ) == pytest.approx(1.0)
    assert torch.linalg.vector_norm(
        weighted._pool_hold(tokens, image, hold(100, 100, 110, 110))
    ) == pytest.approx(1.0)


def test_dino_distances_are_cosine_distances():
    features = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    distances = DINORouteDiscriminator._dino_distances(features)
    assert np.allclose(np.diag(distances), 0)
    assert distances[0, 1] == pytest.approx(1)
    assert distances[0, 2] == pytest.approx(1 - 1 / np.sqrt(2))


def test_colour_feature_is_median_lab_inside_polygon():
    image = Image.new("RGB", (20, 20), (0, 0, 255))
    for x in range(5, 10):
        for y in range(5, 10):
            image.putpixel((x, y), (255, 0, 0))
    feature = DINORouteDiscriminator._lab_feature(image, hold(5, 5, 9, 9))
    expected = DINORouteDiscriminator._lab_feature(
        Image.new("RGB", (2, 2), (255, 0, 0)), hold(0, 0, 1, 1)
    )
    assert np.allclose(feature, expected)


def test_ciede2000_is_normalized_and_zero_for_identical_colours():
    colours = np.array([[50, 0, 0], [50, 0, 0], [50, 80, 70]], dtype=float)
    distances = DINORouteDiscriminator._ciede2000_distances(colours)
    assert distances[0, 0] == pytest.approx(0)
    assert distances[0, 1] == pytest.approx(0)
    assert np.all((distances >= 0) & (distances <= 1))
    assert np.allclose(distances, distances.T)


def test_colour_weight_combines_pairwise_distances(monkeypatch):
    image = Image.new("RGB", (20, 20))
    holds = [hold(1, 1, 3, 3), hold(5, 5, 7, 7)]
    discriminator = DINORouteDiscriminator(color_weight=3, device="cpu")
    monkeypatch.setattr(discriminator, "extract_patch_tokens", lambda _: patch_tokens())
    monkeypatch.setattr(
        discriminator, "_dino_distances", lambda _: np.array([[0, 0.2], [0.2, 0]])
    )
    monkeypatch.setattr(
        discriminator, "_ciede2000_distances", lambda _: np.array([[0, 0.8], [0.8, 0]])
    )
    captured = []
    monkeypatch.setattr(
        discriminator,
        "_hdbscan_labels",
        lambda distances: captured.append(distances) or np.array([0, 0]),
    )

    discriminator.get_routes([image], [holds])

    assert captured[0][0, 1] == pytest.approx((0.2 + 3 * 0.8) / 4)


def test_hdbscan_uses_precomputed_distances(monkeypatch):
    discriminator = DINORouteDiscriminator(min_cluster_size=2, device="cpu")
    calls = {}

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def fit_predict(self, distances):
            calls["distances"] = distances
            return np.array([0, 0])

    import hdbscan

    monkeypatch.setattr(hdbscan, "HDBSCAN", FakeHDBSCAN)
    distances = np.array([[0.0, 0.4], [0.4, 0.0]])
    assert np.array_equal(discriminator._hdbscan_labels(distances), [0, 0])
    assert calls["metric"] == "precomputed"
    assert np.array_equal(calls["distances"], distances)


def test_hdbscan_noise_holds_are_separate_routes(monkeypatch):
    image = Image.new("RGB", (100, 100))
    discriminator = DINORouteDiscriminator(device="cpu")
    monkeypatch.setattr(discriminator, "extract_patch_tokens", lambda _: patch_tokens())
    monkeypatch.setattr(
        discriminator, "_hdbscan_labels", lambda _: np.array([-1, -1, 0])
    )
    holds = [hold(10, 10, 20, 20), hold(30, 30, 40, 40), hold(70, 70, 80, 80)]

    routes = discriminator.get_routes([image], [holds])[0]

    assert len(routes) == 3
    assert routes[0].holds == {holds[2]}
    assert routes[1].holds == {holds[0]}
    assert routes[2].holds == {holds[1]}


def test_validation_and_factory():
    assert isinstance(route_discriminator_factory("DINO"), DINORouteDiscriminator)
    for kwargs in (
        {"min_cluster_size": 0},
        {"min_samples": 0},
        {"color_weight": float("nan")},
    ):
        with pytest.raises(InvalidRouteDiscriminatorConfigError):
            DINORouteDiscriminator(**kwargs)
    discriminator = DINORouteDiscriminator(device="cpu")
    with pytest.raises(InvalidImageError):
        discriminator.get_routes(["image"], [[]])
    with pytest.raises(BatchAlignmentError):
        discriminator.get_routes([Image.new("RGB", (10, 10))], [])
