# SAM3 Hold-Detection Evaluation

## Retained evaluation data

- `data/evaluation/images/`: nine 2800×1864 wall images (`0000.jpg`–`0008.jpg`) and `bh-annotation.csv`, a VIA polygon annotation file.
- `data/evaluation/exemplars/`: five clean single-hold exemplar image/mask pairs.

The CSV labels `hold` and `volume`; the quantitative evaluation scores `hold` only, treating volumes as false positives.

## Environment

Experiments used the local SAM3 video model at `/home/vault/v123be/v123be56/LIT/models/sam3` in the `lit` Conda environment on one NVIDIA A40 GPU. The final multi-exemplar job ran on A40 node `a0127` and completed in 2m42s.

## Completed evaluation

### Qualitative text and single-exemplar prompt sweep

All nine images were run with five text prompts. Detected polygon counts were:

| Prompt | Text only | Text + one exemplar |
| --- | ---: | ---: |
| `Climbings Holds` | 0 | 5 |
| `colored climbing holds` | 3 | 5 |
| `Climbing holds on the wall` | 3 | 5 |
| `All the climbing holds on the wall` | 2 | 5 |
| `all Bouldering holds on the wall` | 2 | 5 |

Text-only inference was about 0.70–0.76 seconds per image after model loading; text plus one exemplar was about 1.23–1.26 seconds. The exemplar counts were identical across the tested prompt wordings.

### Quantitative 3-versus-5 exemplar comparison

The final prompt was `colored climbing holds`. A longer candidate prompt exceeded SAM3's 32-token text limit and must not be used unchanged. The first three exemplar pairs were compared with all five pairs using the same nine images.

| Exemplars | Detections | TP | FP | FN | Recall | Precision | F1 | Mean matched IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 19 | 2 | 17 | 387 | 0.005 | 0.105 | 0.010 | 0.844 |
| 5 | 29 | 3 | 26 | 386 | 0.008 | 0.103 | 0.014 | 0.812 |

A match uses one-to-one mask IoU ≥0.50. Evaluation used recall-first proposal filtering (minimum confidence 0.05, minimum area 4 pixels, maximum area 25% of image, maximum aspect ratio 50, minimum fill ratio 0.005) and duplicate NMS at IoU 0.85. Five exemplars were **not** a meaningful improvement under the predefined rule: recall needed to rise by at least 3 percentage points with no more than a 1-point F1 loss; observed recall gain was 0.3 points.

SAM3 reported that the optional `kernels` package was unavailable, so its internal NMS, hole filling, and sprinkle removal were skipped during this run.

For current pipeline API usage, see [updates.md](updates.md).
