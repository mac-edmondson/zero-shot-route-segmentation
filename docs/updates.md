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

## Route discriminators

Color-only and pretrained TripletNet route discriminators are available through `RouteDiscriminatorFactory` as `color_only_discriminator` and `triplet_route_discriminator`.

## Preprocessing and augmentation

`pipeline.preprocessing` now provides `AugmentationSuite` and
`DataPreprocessingPipeline` for reproducible clean-versus-augmented evaluation
inputs. `ImageRecord` is the shared immutable image/annotation/provenance
contract; annotations are tuples of existing `Hold` values.

`AugmentationSuite` supports polygon-local seeded chalk, polygon-local color
blending, and global lighting. Plans always apply chalk, then color, then
lighting without mutating caller-owned PIL images. `DatasetSpec` supports
`STATIC` and `ON_THE_FLY` materialization; both preserve source and parent IDs,
annotations, deterministic split assignment, and augmentation metadata.

This layer deliberately does not perform Mask R-CNN or TripletNet tensor
preprocessing. Those adapters retain their validated BGR/config-driven
preprocessing behavior.

## SAM3 adapters

Shared SAM3 lifecycle, device, local-artifact, and binary-mask helpers live in
`utility/sam3.py`. The click-guided `SAMWrapper` now lives in
`hold_detector/sam3_ui.py`; `utility/sam3_ui.py` remains a compatibility
re-export. `sam3_hold_detector.py` remains the automatic text/exemplar
`HoldDetector` and reuses the same utility. Only the automatic detector is
registered in `HoldDetectorFactory` because the UI adapter requires clicks.
