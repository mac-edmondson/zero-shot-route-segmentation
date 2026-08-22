# DataPreprocessingPipeline Interface

TODO: Port to code and clean

**Depends on:** [Shared Data Models](data-models.md), [image augmentations](augmentation-suite.md)
**Consumed by:** [EvaluationSuite](evaluation-suite.md)

## 1. Responsibility

Turn source images/annotations into reproducible datasets for evaluation. The board indicates that output may include train/validation/test splits and that augmentations may preferably be performed on the fly to avoid rebuilding datasets at the cost of some compute.

## 2. Core interface

```python
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Protocol, Sequence
from .data_models import ImageRecord
from .augmentation_suite import AugmentationPlan

class MaterializationMode(str, Enum):
    STATIC = "static"
    ON_THE_FLY = "on_the_fly"

@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    splits: tuple[str, ...] = ("test",)
    materialization: MaterializationMode = MaterializationMode.ON_THE_FLY
    augmentation_plan: AugmentationPlan | None = None
    metadata: Mapping[str, object] | None = None

class DatasetProvider(Protocol):
    def iter_split(self, split: str) -> Iterable[ImageRecord]: ...
    def describe(self) -> DatasetSpec: ...

class DataPreprocessingPipeline:
    def build(
        self,
        sources: Sequence[ImageRecord],
        spec: DatasetSpec,
    ) -> DatasetProvider: ...
```

## 3. Static vs on-the-fly

### Static

Precompute augmented images once and persist them. Use when evaluation needs frozen artifacts or model training requires a physical dataset.

### On-the-fly

Keep source images and augmentation recipes, applying them when `DatasetProvider.iter_split()` is consumed. This reflects the board's stated preference to avoid repeatedly running a full preprocessing pipeline, while accepting extra compute.

## 4. Source identity

Derived images MUST preserve a link to the original source image through `source_id` / `parent_image_id`. This permits the [EvaluationSuite](evaluation-suite.md) to compare a clean image with its distorted variants rather than treating them as unrelated samples.

Filename preservation was ambiguous on the board. Implementations MAY preserve filenames for convenience, but evaluation SHOULD use stable IDs instead of filenames as the canonical identity.

## 5. Outputs

`DatasetProvider` MUST expose enough metadata for evaluation to know:

- dataset ID/version;
- split;
- clean vs distorted condition;
- augmentation recipe/seed;
- ground-truth annotations, if available;
- source image relationship.

The exact annotation schema is not present on the whiteboard and is tracked in [Open Questions](../decisions/open-questions.md#ground-truth-and-metrics).
