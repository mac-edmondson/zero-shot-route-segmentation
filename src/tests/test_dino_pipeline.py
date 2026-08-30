import os
from pathlib import Path

import pytest
from PIL import Image

from pipeline.hold_detector.mask_rcnn_hold_detector import MaskRCNNHoldDetector
from pipeline.route_discriminator.route_discriminator_factory import (
    route_discriminator_factory,
)
from pipeline.route_discriminator_pipeline import RouteDiscriminatorPipeline

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "evaluation" / "images" / "0007.jpg"
OUTPUT_DIR = ROOT / "data" / "test-dino-pipeline"
DINO_ROUTE_DISCRIMINATOR = os.getenv("DINO_ROUTE_DISCRIMINATOR", "DINO Clustering")
DINO_CONFIG = {
    "DINO Clustering": {"color_weight": 0.0},
    "DINO Learning": {
        "weights_path": ROOT / "models" / "dino_learning_route_discriminator.pt",
        "model_dir": ROOT / "models" / "dinov3",
    },
}


# @pytest.mark.skipif(
#     os.getenv("RUN_REAL_INFERENCE") != "1",
#     reason="Set RUN_REAL_INFERENCE=1 to run model inference",
# )
def test_real_mask_rcnn_dino_pipeline_marks_detected_routes() -> None:
    if DINO_ROUTE_DISCRIMINATOR not in DINO_CONFIG:
        pytest.fail(f"Unknown DINO_ROUTE_DISCRIMINATOR: {DINO_ROUTE_DISCRIMINATOR}")
    image = Image.open(INPUT_PATH).convert("RGB")
    pipeline = RouteDiscriminatorPipeline(
        hold_detector=MaskRCNNHoldDetector(),
        route_discriminator=route_discriminator_factory(
            DINO_ROUTE_DISCRIMINATOR, DINO_CONFIG[DINO_ROUTE_DISCRIMINATOR]
        ),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pipeline.get_routes([image])
    marked_images = [
        ("hold", marked_image)
        for marked_image in pipeline.mark_holds([image], pipeline._last_holds)
    ]
    marked_images.extend(
        ("route", marked_image)
        for marked_image in pipeline.mark_routes([image], routes)
    )

    assert len(routes) == 1
    assert routes[0]
    assert all(route.holds for route in routes[0])

    for kind, marked_image in marked_images:
        assert marked_image.size == image.size
        output_path = OUTPUT_DIR / (
            f"output_{DINO_ROUTE_DISCRIMINATOR.lower().replace(' ', '-')}_"
            f"{kind}_{INPUT_PATH.stem}.jpg"
        )
        marked_image.save(output_path, format="JPEG")
        assert output_path.is_file()
