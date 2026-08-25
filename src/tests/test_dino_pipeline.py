from pathlib import Path

from PIL import Image

from pipeline.hold_detector.mask_rcnn_hold_detector import MaskRCNNHoldDetector
from pipeline.route_discriminator.dino_route_discriminator import (
    DINORouteDiscriminator,
)
from pipeline.route_discriminator_pipeline import RouteDiscriminatorPipeline

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "evaluation" / "images" / "image.png"
OUTPUT_DIR = ROOT / "data" / "test-dino-pipeline"


@pytest.mark.skip(reason="Do not run inference every test run")
def test_real_mask_rcnn_dino_pipeline_marks_detected_routes() -> None:
    image = Image.open(INPUT_PATH).convert("RGB")
    pipeline = RouteDiscriminatorPipeline(
        hold_detector=MaskRCNNHoldDetector(),
        route_discriminator=DINORouteDiscriminator(color_weight=0.0),
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
        output_path = OUTPUT_DIR / f"output_{kind}_{INPUT_PATH.stem}.jpg"
        marked_image.save(output_path, format="JPEG")
        assert output_path.is_file()
