from pathlib import Path

import numpy as np

from src.SAM_hold.dataset import largest_mask, load_hold_masks
from src.SAM_hold.evaluate import score_instances


def test_largest_mask() -> None:
    assert np.count_nonzero(largest_mask([np.zeros((2, 2), dtype=bool), np.ones((2, 2), dtype=bool)])) == 4


def test_score_instances() -> None:
    target = np.array([[1, 0], [0, 0]], dtype=bool)
    result = score_instances([target], [target])
    assert result["tp"] == 1
    assert result["f1"] == 1.0


def test_annotation_masks() -> None:
    root = Path(__file__).parent
    masks = load_hold_masks(root / "bh-annotation.csv", "0003.jpg", (2800, 1864))
    assert len(masks) == 7
