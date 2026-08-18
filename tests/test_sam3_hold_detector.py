import numpy as np

from pipeline.hold_detector.sam3_hold_detector import SAMHoldDetector


def test_mask_conversion_uses_polygon_backed_hold():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    holds = SAMHoldDetector._to_holds([mask], "text")
    assert len(holds) == 1
    hold = holds[0]
    assert hold.centroid.x == 4
    assert hold.centroid.y == 4
    assert hold.attributes["mode"] == "text"
