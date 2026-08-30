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

## Route classifiers

Color-only and pretrained TripletNet route classifiers are available through `RouteClassifierFactory` as `color_only_classifier` and `triplet_route_classifier`.

## DINO Learning training data

The Bh dataset was reduced to the 8 images with route-labeled holds:
`0003.jpg`, `0074.jpg`, `0075.jpg`, `0457.jpg`, `0502.jpg`, `0518.jpg`,
`0530.jpg`, and `0917.jpg`. The other 1,053 images had no `route_id` and
were removed because they cannot produce pairwise training examples.

The retained annotations contain 587 route-labeled holds across 17 routes.
With balanced sampling capped at 256 pairs per class per image, training
produces 3,258 pairs: 1,629 positive and 1,629 negative. The seed-42 split
used 2,234 training pairs and 1,024 validation pairs across 6 and 2 images.
