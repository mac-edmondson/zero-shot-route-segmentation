from pathlib import Path

import pytest
from PIL import Image
from pipeline.hold_detector.yolo_hold_detector import YOLOv8HoldDetector
from pipeline.route_discriminator.color_only_route_discriminator import (
    ColorOnlyRouteDiscriminator,
)
from pipeline.route_discriminator_pipeline import RouteDiscriminatorPipeline

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "evaluation" / "ground_truth_labels" / "Pic_01_jpeg.rf.GHNJG1DT87TvGAILmPUm.jpeg"
OUTPUT_DIR = ROOT / "data" / "test-yolo-pipeline"


#@pytest.mark.skip(reason="Do not run inference every test run")
def test_real_pipeline_marks_detected_routes() -> None:
    image = Image.open(INPUT_PATH).convert("RGB")
    pipeline = RouteDiscriminatorPipeline(
        hold_detector=YOLOv8HoldDetector(),
        route_discriminator=ColorOnlyRouteDiscriminator(n_clusters=6),
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

    assert marked_images
    for kind, marked_image in marked_images:
        assert marked_image.size == image.size
        output_path = OUTPUT_DIR / f"output_{kind}_{INPUT_PATH.stem}.jpg"
        marked_image.save(output_path, format="JPEG")
        assert output_path.is_file()
