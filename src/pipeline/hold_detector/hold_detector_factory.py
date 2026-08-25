"""Factory and registry for hold detectors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .hold_detector import HoldDetector


################################################################################
# Errors
################################################################################
class UnknownHoldDetectorError(ValueError):
    pass


################################################################################
# Constructor Wrapper Methods
################################################################################
_ConfigArgs = Mapping[str, Any]
_Constructor = Callable[[_ConfigArgs], HoldDetector]


def _create_mask_rcnn_hold_detector(config: Mapping[str, Any] = dict()) -> HoldDetector:
    from .mask_rcnn_hold_detector import MaskRCNNHoldDetector

    return MaskRCNNHoldDetector(**config)


def _create_sam_hold_detector(config: Mapping[str, Any] = dict()) -> HoldDetector:
    from .sam3_hold_detector import SAMHoldDetector

    return SAMHoldDetector(**config)


def _create_mock_hold_detector(config: Mapping[str, Any] = dict()) -> HoldDetector:
    from .mock_hold_detector import MockHoldDetector

    return MockHoldDetector(**config)

  
def _create_yolo_hold_detector(config: Mapping[str, Any] = dict()) -> HoldDetector:
    from .yolo_hold_detector import YOLOv8HoldDetector

    return YOLOv8HoldDetector(**config)


def _create_ground_truth_hold_detector(config: Mapping[str, Any] = dict()) -> HoldDetector:
    from ..ground_truth.ground_truth_hold_detector import GroundTruthHoldDetector

    return GroundTruthHoldDetector(**config)


_AVAILABLE_HOLD_DETECTORS_MAP: Mapping[str, _Constructor] = {
    "Mask-RCNN": _create_mask_rcnn_hold_detector,
    "SAM 3": _create_sam_hold_detector,  # TODO: Consider adding parameters we want? @joswin03
    "YOLO": _create_yolo_hold_detector,
    "Ground Truth": _create_ground_truth_hold_detector,
    "Mock": _create_mock_hold_detector,
}
AVAILABLE_HOLD_DETECTORS = list(_AVAILABLE_HOLD_DETECTORS_MAP.keys())


################################################################################
# The Factory
################################################################################
def hold_detector_factory(hold_detector_name: str, config: _ConfigArgs = dict()):
    if hold_detector_name not in _AVAILABLE_HOLD_DETECTORS_MAP:
        raise UnknownHoldDetectorError(
            f"'{hold_detector_name}' isn't a known hold detector. Try one of {AVAILABLE_HOLD_DETECTORS}"
        )

    return _AVAILABLE_HOLD_DETECTORS_MAP[hold_detector_name](config)
