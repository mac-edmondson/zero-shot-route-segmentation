# Pipeline Updates

## `sam3_ui.py`

`SAMWrapper` is unchanged and remains the front-end segmentation API. Load it with `load_model()`, call `generate_mask(input_img, click_coordinates)` for positive clicks, then convert returned masks with `to_polygons()` or `to_rle()`.

## `sam3_hold_detector.py`

`SAMHoldDetector` is the reusable automatic hold detector. It supports these configurations:

- Text only: `SAMHoldDetector(mode="text", text_prompt="colored climbing holds", ...)`.
- One exemplar: `SAMHoldDetector(mode="text_exemplar", exemplar_image=image, exemplar_mask=mask, ...)`.
- Multiple exemplars: `SAMHoldDetector(mode="text_exemplar", exemplars=[(image_a, mask_a), (image_b, mask_b)], nms_iou=0.85, ...)`.

Single-exemplar calls remain compatible. Multiple exemplars are run independently, then their scored masks are merged and deduplicated with confidence-ordered mask-IoU NMS. The detector only removes overlaps at or above `nms_iou` (default 0.85), preserving nearby distinct holds. Masks must be non-empty binary arrays aligned with their exemplar images.

Retained evaluation inputs and completed A40 findings are documented in [evaluation.md](evaluation.md).
