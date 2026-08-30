from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pipeline.hold_detector.mask_rcnn_hold_detector import MaskRCNNHoldDetector
from pipeline.interfaces.data_models import Hold
from pipeline.preprocessing.augmentation_suite import (
    AugmentationPlan,
    ChalkAugmentation,
    ColorAugmentation,
    LightingAugmentation,
)

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "evaluation" / "images" / "image.png"
OUTPUT_DIR = ROOT / "data" / "test-augmentation-with-images"

# These values were selected through experimentation to make a good test
HOLD_TO_USE_COLOR = 5
HOLD_TO_AUGMENT = 23


def mark_holds(image: Image.Image, holds: Sequence[Hold]) -> Image.Image:
    """Render hold polygons and their detector indexes for visual debugging."""
    overlay = image.convert("RGB").copy()
    fill_overlay = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_overlay)
    for hold in holds:
        points = [(point.x, point.y) for point in hold.polygon.points]
        fill_draw.polygon(points, fill=(255, 80, 0, 102))
    overlay = Image.alpha_composite(overlay.convert("RGBA"), fill_overlay).convert(
        "RGB"
    )

    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=18)
    for index, hold in enumerate(holds):
        points = [
            (point.x, point.y)
            for point in (*hold.polygon.points, hold.polygon.points[0])
        ]
        draw.line(points, fill=(255, 80, 0), width=5)
        draw.text(
            (hold.centroid.x, hold.centroid.y),
            str(index),
            font=font,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
            anchor="mm",
        )
    return overlay


# @pytest.mark.skip(reason="Do not run inference every test run")
def test_real_pipeline_marks_detected_routes() -> None:
    image = Image.open(INPUT_PATH).convert("RGB")
    hold_detector = MaskRCNNHoldDetector(device="cpu")

    # Detect holds in the image and mark the image for debugging
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    holds_l = list(hold_detector.get_holds([image]))
    holds_l.extend(
        [[holds_l[0][HOLD_TO_AUGMENT]], [holds_l[0][HOLD_TO_USE_COLOR]]]
    )  # Use a single hold to mess around with augmenting
    marked_images = [
        (label, mark_holds(image, holds))
        for label, holds in zip(
            ["all_holds", "to_agument", "to_select_color"], holds_l, strict=True
        )
    ]

    for img_num, (kind, marked_image) in enumerate(marked_images):
        assert marked_image.size == image.size
        output_path = OUTPUT_DIR / f"output_{kind}_{INPUT_PATH.stem}_{img_num}.jpg"
        marked_image.save(output_path, format="JPEG")
        assert output_path.is_file()

    # Print the hold indexes for some debugging if needed
    for i, hold in enumerate(holds_l[0]):
        print(f"Hold {i}: {hold.centroid}")

    def _save_augmented_image(name: str, image: Image.Image):
        output_path = OUTPUT_DIR / (name + ".jpg")
        image.save(output_path, format="JPEG")
        assert output_path.is_file()

    # 1. Augment the original image to change the color like what was done previously
    color_picker_hold = holds_l[0][HOLD_TO_USE_COLOR]
    hold_to_augment = holds_l[0][HOLD_TO_AUGMENT]

    color = color_picker_hold.get_color(image)
    color_augmentation = ColorAugmentation((hold_to_augment,), (color,))
    _save_augmented_image(
        "color_augmented_image",
        AugmentationPlan((color_augmentation,), seed=0).apply(image),
    )

    # 2. Augment the original image to add chalk augmentation on a hold
    chalk_augmentation = ChalkAugmentation((hold_to_augment,), (1.0,))
    _save_augmented_image(
        "chalk_augmented_image_1",
        AugmentationPlan((chalk_augmentation,), seed=0).apply(image),
    )
    chalk_augmentation = ChalkAugmentation((color_picker_hold,), (0.7,))
    _save_augmented_image(
        "chalk_augmented_image_2",
        AugmentationPlan((chalk_augmentation,), seed=12).apply(image),
    )

    # 3. Augment the original image to add chalk augmentation on a hold
    lighting_augmentation = LightingAugmentation(0.5)
    _save_augmented_image(
        "brighter_augmented_image",
        AugmentationPlan((lighting_augmentation,), seed=0).apply(image),
    )
    lighting_augmentation = LightingAugmentation(-0.5)
    _save_augmented_image(
        "dimmer_augmented_image",
        AugmentationPlan((lighting_augmentation,), seed=0).apply(image),
    )
