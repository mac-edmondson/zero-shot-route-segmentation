"""Small manual smoke test for the local SAM3 wrapper."""

from pathlib import Path

import numpy as np
from PIL import Image

from sam_wrapper import SAMWrapper


HOLD_CLICKS = [[271, 120], [269, 185], [389, 171], [147, 93], [116, 162]]


def run_sam_wrapper_test(image_filename: str) -> tuple[np.ndarray, list[list[list[int]]], list[dict[str, object]]]:
    """Generate one mask for each predefined hold click in ``data/image_filename``."""
    image_path = Path(__file__).parent / "data" / image_filename
    with Image.open(image_path) as image:
        sam = SAMWrapper()
        sam.load_model()
        masks = sam.generate_mask(image, HOLD_CLICKS)
        holds = SAMWrapper.to_polygons(masks)
        rles = SAMWrapper.to_rle(masks)
        return masks, holds, rles


if __name__ == "__main__":
    masks, holds, rles = run_sam_wrapper_test("climbing-wall-climbing-wall-with-colorful-rocks-photo.jpg")
    print(masks.shape, len(holds), len(rles))
