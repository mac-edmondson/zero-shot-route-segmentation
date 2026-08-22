# Image Augmentations

**Depends on:** [Shared Data Models](data-models.md)
**Consumed by:** [DataPreprocessingPipeline](data-preprocessing-pipeline.md), [Dashboard Backend](dashboard-backend.md)

## Responsibility

Apply deterministic evaluation distortions without mutating source images.

## API

`src/pipeline/preprocessing/augmentation_suite.py` owns both recipes and operations:

```python
@dataclass(frozen=True)
class ChalkAugmentation:
    targets: tuple[Polygon | Hold, ...]
    strengths: tuple[float, ...]  # each in [0, 1]

@dataclass(frozen=True)
class ColorAugmentation:
    targets: tuple[Polygon | Hold, ...]
    colors: tuple[RGBColor, ...]

@dataclass(frozen=True)
class LightingAugmentation:
    strength: float  # in [-1, 1]

@dataclass(frozen=True)
class AugmentationPlan:
    augmentations: tuple[ChalkAugmentation | ColorAugmentation | LightingAugmentation, ...]
    seed: int | None = None

    def apply(self, image: Image) -> Image: ...


def add_chalk(
    image: Image,
    targets: Sequence[Polygon | Hold],
    strengths: Sequence[float],
    *,
    seed: int | None = None,
) -> Image: ...


def change_color(
    image: Image,
    targets: Sequence[Polygon | Hold],
    colors: Sequence[RGBColor],
) -> Image: ...


def change_lighting(image: Image, strength: float) -> Image: ...
```

Recipes execute in plan order. A seeded plan derives a distinct deterministic chalk texture for each chalk recipe. Chalk and colour target matching polygons/holds; colour retains per-pixel lightness while applying the requested hue and saturation. Lighting applies globally: `-1` is black, `0` unchanged, and `1` white.

## Provenance and mutation

`DataPreprocessingPipeline` decides when plans run and records their seed/recipes with derived `ImageRecord`s. Every operation returns a new RGB image; inputs are never mutated.
