# Inference preprocessing reference

This note records the preprocessing used by the original Indoor Climbing Hold
and Route Segmentation repository, commit `27ba65f`, so that future adapters do
not silently change the model input contract.

## Hold detector (Detectron2)

- Read images with `cv2.imread`, which returns a uint8 **BGR** array.
- Preserve the original image height and width for Detectron2 postprocessing.
- Apply `ResizeShortestEdge([INPUT.MIN_SIZE_TEST, INPUT.MIN_SIZE_TEST],
  INPUT.MAX_SIZE_TEST)` before calling the model. The concrete values come from
  `experiment_config.yml`; do not hard-code a different resize policy.
- Convert the resized BGR array from HWC to float32 CHW. Detectron2 performs
  model-specific normalization internally from the configuration.
- Run the model in evaluation/no-grad mode, then filter `pred_classes == 0` for
  climbing holds. Class 1 is a volume and is only included when explicitly
  requested.

## Route classifier crops

- The reference repository also reads hold crops with OpenCV, so the TripletNet
  receives BGR CHW tensors.
- A crop is masked by its hold polygon before it is resized and normalized by
