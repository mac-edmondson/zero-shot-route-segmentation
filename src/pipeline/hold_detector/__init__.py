from .hold_detector import HoldDetector
from .hold_detector_factory import AVAILABLE_HOLD_DETECTORS, hold_detector_factory
from .mask_rcnn_hold_detector import MaskRCNNHoldDetector

# SAMHoldDetector is deliberately NOT re-exported here -- importing it pulls
# in torch/transformers/cv2, which the lightweight backend build
# (src/backend/requirements.txt, this Dockerfile's `base` stage) doesn't
# install. hold_detector_factory() already lazy-imports it only when "SAM3"
# is actually requested; import it directly from .sam3_hold_detector if you
# need the class itself in an environment that has the full deps.
