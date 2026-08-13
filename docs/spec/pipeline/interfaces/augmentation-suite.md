# AugmentationSuite Interface

TODO: Port to code and clean

**Depends on:** [Shared Data Models](data-models.md)
**Consumed by:** [DataPreprocessingPipeline](data-preprocessing-pipeline.md), optionally [Dashboard Backend](dashboard-backend.md)

## 1. Responsibility

Describe and apply controlled image distortions so evaluation can compare clean and distorted inputs. The source board specifically calls out lighting changes, chalk contamination, and color changes.

## 2. Core types

```python
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence
from .data_models import Image, Polygon, RGBColor

class Augmentation(Protocol):
    @property
    def name(self) -> str: ...

    def apply(self, image: Image, *, seed: int | None = None) -> Image: ...

@dataclass(frozen=True)
class AugmentationPlan:
    augmentations: tuple[Augmentation, ...]
    seed: int | None = None
```

`Augmentation`/`AugmentationPlan` are a proposed normalization of the whiteboard's “augmentation” values.

## 3. Suite API

```python
class AugmentationSuite:
    def augment(
        self,
        images: Sequence[Image],
        augmentations: Sequence[Augmentation] | None = None,
        *,
        seed: int | None = None,
    ) -> AugmentationPlan: ...

    def change_lighting(
        self,
        polygons: Sequence[Polygon] | None = None,
        *,
        intensity: float | None = None,
    ) -> Augmentation: ...

    def add_chalk(
        self,
        polygons: Sequence[Polygon] | None = None,
        *,
        intensity: float | None = None,
    ) -> Augmentation: ...

    def change_color(
        self,
        polygons: Sequence[Polygon] | None = None,
        *,
        color: RGBColor | None = None,
        intensity: float | None = None,
    ) -> Augmentation: ...

    def materialize(
        self,
        images: Sequence[Image],
        plan: AugmentationPlan,
    ) -> list[Image]: ...
```

The exact `intensity` semantics and whether augmentation targets are polygons, holds, or whole images were not fully specified on the board. They are therefore parameterized but remain an [open question](../decisions/open-questions.md#augmentation-parameters).

## 4. Determinism and provenance

For evaluation, the suite SHOULD support a deterministic seed. Materialized images SHOULD retain source/parent IDs and augmentation metadata when wrapped as `ImageRecord`s from [data-models.md](data-models.md).

Two calls using the same source image, augmentation configuration, and seed SHOULD produce identical outputs unless the augmentation explicitly documents nondeterministic behavior.

## 5. Mutation

Augmentations MUST NOT mutate caller-owned source images in place. unless the concrete image type and implementation clearly document copy-on-write semantics. The safer default is to return new image values.

## 6. Relationship to preprocessing

[DataPreprocessingPipeline](data-preprocessing-pipeline.md) decides **when** augmentations are applied (materialized ahead of time vs. on the fly). `AugmentationSuite` decides **how** a requested augmentation is applied.
