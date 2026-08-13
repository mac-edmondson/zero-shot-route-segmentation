
## 4. Provenance

Evaluation requires comparison between clean and distorted inputs. Therefore all dataset/image envelopes SHOULD preserve:

- stable source image ID;
- source URI/path when safe to expose;
- split (`train`, `val`, `test`, or custom);
- augmentation recipe/seed;
- parent/source ID for derived images;
- pipeline/configuration IDs used to generate results.

This provenance is formalized in [data-models.md](interfaces/data-models.md) and consumed by [evaluation-suite.md](interfaces/evaluation-suite.md).

## 5. Configuration identity

Every swappable model or algorithm SHOULD expose a stable `implementation_id` and a serializable configuration. Evaluation reports MUST record both, so a result can be tied to the exact detector/classifier variant, including board-mentioned conditions such as color-only, color+spatial, DINO-only, and color+spatial+DINO.
