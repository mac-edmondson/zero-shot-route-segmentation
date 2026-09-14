import numpy as np
import torch
from PIL import Image, ImageDraw

from pipeline.evaluation.gpu_runner import _load_feature_maps
from pipeline.evaluation.metrics import pairwise_hold_iou, polygon_iou
from pipeline.evaluation.run_all_eval import predict_records
from pipeline.interfaces.data_models import Coordinate, Hold, ImageRecord, Polygon
from pipeline.route_discriminator.dino_clustering_route_discriminator import (
    DINOClusteringRouteDiscriminator,
)


def hold(left, top, right, bottom, **attributes):
    return Hold(
        Polygon(
            (
                Coordinate(left, top),
                Coordinate(right, top),
                Coordinate(right, bottom),
                Coordinate(left, bottom),
            )
        ),
        attributes,
    )


def test_pairwise_hold_iou_matches_reference_polygons_and_prediction_masks():
    size = (32, 24)
    predictions = (hold(1, 2, 10, 12), hold(12, 3, 24, 18))
    truths = (hold(4, 5, 14, 15), hold(20, 10, 29, 20))
    expected = np.array(
        [
            [polygon_iou(prediction.polygon, truth.polygon, size) for truth in truths]
            for prediction in predictions
        ]
    )
    assert np.allclose(
        pairwise_hold_iou(predictions, truths, size, device="cpu"), expected
    )

    mask = Image.new("1", size)
    ImageDraw.Draw(mask).rectangle((0, 0, 7, 7), fill=1)
    masked = hold(20, 15, 25, 20, mask=np.asarray(mask, dtype=bool))
    matrix = pairwise_hold_iou((masked,), (hold(0, 0, 7, 7),), size, device="cpu")
    assert matrix[0, 0] == 1.0


def test_predict_records_batches_supported_detector():
    class Detector:
        implementation_id = "yolov8_hold_detector"

        def __init__(self):
            self.batch_sizes = []

        def get_holds(self, images):
            self.batch_sizes.append(len(images))
            return [[] for _ in images]

    records = [
        ImageRecord(str(index), Image.new("RGB", (8, 8)), ()) for index in range(10)
    ]
    detector = Detector()
    predictions, timings = predict_records(detector, records)
    assert detector.batch_sizes == [8, 2]
    assert set(predictions) == set(timings) == {str(index) for index in range(10)}


def test_dino_clustering_uses_precomputed_embeddings_without_model(monkeypatch):
    image = Image.new("RGB", (20, 20))
    holds = (hold(1, 1, 5, 5), hold(10, 10, 15, 15))
    cached = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).repeat(1, 192)
    discriminator = DINOClusteringRouteDiscriminator(
        device="cpu",
        clustering_method="dbscan",
        min_samples=1,
        precomputed_embeddings={(id(image), "weighted"): cached},
    )
    monkeypatch.setattr(
        discriminator,
        "extract_mask_embeddings",
        lambda *_: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    routes = discriminator.get_routes([image], [holds])
    assert sum(len(route.holds) for route in routes[0]) == 2


def test_detector_feature_cache_is_aligned_to_matched_route_holds(tmp_path):
    image = Image.new("RGB", (20, 20))
    record = ImageRecord(
        "image",
        image,
        (hold(1, 1, 5, 5), hold(10, 10, 15, 15, route_id=1)),
    )
    feature_path = tmp_path / "artifacts" / "clean" / "route_features.pt"
    feature_path.parent.mkdir(parents=True)
    vectors = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    torch.save(
        {
            "embeddings": {
                "yolo": {
                    "weighted": {"image": vectors},
                    "mean": {"image": vectors},
                }
            },
            "lab": {"yolo": {"image": vectors}},
            "rgb": {"yolo": {"image": vectors}},
        },
        feature_path,
    )

    embeddings, lab, rgb = _load_feature_maps(tmp_path, [record], "yolo", "clean")

    assert embeddings[(id(image), "weighted")].tolist() == [[3.0, 4.0]]
    assert lab[id(image)].tolist() == [[3.0, 4.0]]
    assert rgb[id(image)].tolist() == [[3.0, 4.0]]
