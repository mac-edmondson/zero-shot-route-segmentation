import numpy as np
import pytest
import torch
from PIL import Image

from pipeline.hold_detector.hold_detector import HoldDetector
from pipeline.hold_detector.mask_rcnn_hold_detector import MaskRCNNHoldDetector
from pipeline.interfaces.errors import BatchAlignmentError, InvalidImageError


class Instances:
    def __init__(self, masks, scores, classes):
        self.pred_masks, self.scores, self.pred_classes = (
            torch.as_tensor(masks),
            torch.as_tensor(scores),
            torch.as_tensor(classes),
        )

    def to(self, device):
        assert device == "cpu"
        return self


class Model:
    def __init__(self, instances):
        self.instances = instances

    def __call__(self, inputs):
        return [{"instances": item} for item in self.instances]


def mask():
    value = np.zeros((12, 16), dtype=bool)
    value[3:9, 4:12] = True
    return value


def test_converts_source_coordinate_masks_and_preserves_order():
    detector = MaskRCNNHoldDetector(device="cpu")
    second = np.zeros((12, 16), dtype=bool)
    second[1:5, 1:5] = True
    detector.model = Model(
        [Instances([mask()], [0.9], [0]), Instances([second], [0.8], [0])]
    )
    holds = detector.get_holds([Image.new("RGB", (16, 12)), Image.new("RGB", (16, 12))])
    assert [len(batch) for batch in holds] == [1, 1]
    assert holds[0][0].centroid.x == 8
    assert holds[1][0].centroid.y == 2
    assert holds[0][0].attributes["mask"].shape == (12, 16)
    assert holds[0][0].attributes["confidence"] == pytest.approx(0.9)


def test_empty_detections_and_volume_filtering():
    detector = MaskRCNNHoldDetector(device="cpu")
    detector.model = Model(
        [
            Instances(np.empty((0, 12, 16), bool), [], []),
            Instances([mask(), mask()], [0.9, 0.8], [0, 1]),
        ]
    )
    result = detector.get_holds(
        [Image.new("RGB", (16, 12)), Image.new("RGB", (16, 12))]
    )
    assert result[0] == [] and [
        hold.attributes["class_name"] for hold in result[1]
    ] == ["hold"]
    detector.include_volumes = True
    assert (
        len(
            detector.get_holds(
                [Image.new("RGB", (16, 12)), Image.new("RGB", (16, 12))]
            )[1]
        )
        == 2
    )


def test_invalid_input_and_overlay_alignment():
    detector = MaskRCNNHoldDetector(device="cpu")
    with pytest.raises(InvalidImageError):
        detector.get_holds(["bad"])
    with pytest.raises(BatchAlignmentError):
        HoldDetector.mark_holds([Image.new("RGB", (1, 1))], [])
