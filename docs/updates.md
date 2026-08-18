# Pipeline Updates

## `sam3_ui.py`

`SAMWrapper` is unchanged and remains the front-end segmentation API. Load it with `load_model()`, call `generate_mask(input_img, click_coordinates)` for positive clicks, then convert returned masks with `to_polygons()` or `to_rle()`.

## `sam3_hold_detector.py`

`SAMHoldDetector` is the reusable automatic hold detector. Configure it with a text prompt and an optional ordered list of `(PIL.Image.Image, binary_mask)` exemplar pairs, then call `get_holds(images)`.

- Text only: `SAMHoldDetector(text_prompt="colored climbing holds", ...)`.
- One exemplar: `SAMHoldDetector(exemplars=[(image, mask)], ...)`.
- Multiple exemplars: `SAMHoldDetector(exemplars=[(image_a, mask_a), (image_b, mask_b)], nms_iou=0.85, ...)`.

An empty or omitted exemplar list selects text-only inference. Each supplied exemplar is run independently; scored masks are merged and deduplicated with confidence-ordered mask-IoU NMS. The detector removes only overlaps at or above `nms_iou` (default 0.85), preserving nearby distinct holds. Exemplar masks must be non-empty binary arrays aligned with their exemplar images.

Retained evaluation inputs and completed A40 findings are documented in [evaluation.md](evaluation.md).
