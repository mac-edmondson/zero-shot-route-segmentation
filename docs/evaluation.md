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

## Color-specific prompt sweep

Ten independent A40 jobs ran the same nine images with five prompts: `red climbing holds`, `blue climbing holds`, `green climbing holds`, `yellow climbing holds`, and `orange climbing holds`. For each prompt, one text-only arm and one text-plus-the-fixed-five-exemplar arm were run. Jobs `4034185`–`4034194` all completed successfully. This is a qualitative count-and-timing sweep; no VIA annotation metrics were computed.

| Prompt | Text-only detections | Text-only s/image | Five-exemplar detections | Five-exemplar s/image |
| --- | ---: | ---: | ---: | ---: |
| `red climbing holds` | 0 | 0.837 | 31 | 5.676 |
| `blue climbing holds` | 1 | 0.856 | 31 | 5.686 |
| `green climbing holds` | 1 | 0.856 | 31 | 5.682 |
| `yellow climbing holds` | 2 | 0.856 | 31 | 5.405 |
| `orange climbing holds` | 2 | 0.607 | 31 | 5.391 |

Text-only detections occurred only in `0003.jpg`: 0 for red, 1 for blue and green, and 2 for yellow and orange. Every five-exemplar arm produced the same per-image counts: `0000` 3, `0001` 4, `0002` 3, `0003` 4, `0004` 1, `0005` 4, `0006` 5, `0007` 3, and `0008` 4. Thus, for this fixed exemplar set, color wording did not change the five-exemplar result. Raw job scripts, logs, and CSV/JSON summaries are intentionally temporary under `tmp/color_prompt_sweep/`.

## Mask R-CNN + TripletNet Kaggle inference validation

An inference-only validation was attempted on 18 August 2026; it was not an accuracy or original-pipeline-equivalence evaluation.

- Dataset: tomasslama/indoor-climbing-gym-hold-segmentation via KaggleHub.
- Intended selection: the first 15 supported image files in sorted relative-path order, with paths and SHA-256 hashes recorded temporarily.
- Compute: A40 Slurm job 4043643 on a0429, using the LIT Conda environment; CUDA was available and temporary runtime dependencies installed successfully.
- Result: KaggleHub timed out while connecting to api.kaggle.com to resolve the dataset. No images were downloaded or selected, and neither Mask R-CNN nor TripletNet model loading or inference was reached.

**Verdict: BLOCKED — dataset access unavailable.** This is a network-access failure, not evidence of an inference or model incompatibility. The temporary job workspace, logs, dependency target, and any cache were deleted after this result was recorded.

Rerun after Kaggle API access is available; retain the same 15-image deterministic selection rule and A40 configuration.
