import pytest
import torch
from PIL import Image

from pipeline.hold_detector.yolo_hold_detector import YOLOv8HoldDetector
from pipeline.interfaces.errors import InvalidImageError


class Box:
    def __init__(self, xyxy, confidence, class_id=0):
        self.xyxy = torch.tensor([xyxy])
        self.conf = torch.tensor([confidence])
        self.cls = torch.tensor([class_id])


class Result:
    def __init__(self, boxes):
        self.boxes = boxes


class Model:
    def __init__(self, results):
        self.results = results

    def predict(self, images, **kwargs):
        assert [image.size for image in images] == [(16, 12), (16, 12)]
        assert kwargs == {"device": "cpu", "verbose": False}
        return self.results


def test_converts_source_coordinate_boxes_and_preserves_order():
    detector = YOLOv8HoldDetector(device="cpu")
    detector.model = Model(
        [Result([Box([4, 3, 12, 9], 0.9)]), Result([Box([1, 1, 5, 5], 0.8)])]
    )

    holds = detector.get_holds([Image.new("RGB", (16, 12)), Image.new("RGB", (16, 12))])

    assert [len(batch) for batch in holds] == [1, 1]
    assert holds[0][0].centroid.x == 8
    assert holds[1][0].centroid.y == 3
    assert holds[0][0].attributes["bbox"] == (4, 3, 12, 9)
    assert holds[0][0].attributes["confidence"] == pytest.approx(0.9)


def test_empty_and_non_hold_detections_are_ignored():
    detector = YOLOv8HoldDetector(device="cpu")
    detector.model = Model([Result([]), Result([Box([4, 3, 12, 9], 0.9, class_id=1)])])

    assert detector.get_holds([Image.new("RGB", (16, 12)), Image.new("RGB", (16, 12))]) == [[], []]


def test_invalid_input_and_configuration():
    detector = YOLOv8HoldDetector(device="cpu")
    with pytest.raises(InvalidImageError):
        detector.get_holds(["bad"])
    with pytest.raises(ValueError, match="score_threshold"):
        YOLOv8HoldDetector(score_threshold=1.1)
