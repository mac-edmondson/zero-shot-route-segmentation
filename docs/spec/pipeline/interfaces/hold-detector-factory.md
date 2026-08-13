# HoldDetectorFactory Interface

TODO: Port to code and clean

**Produces:** [HoldDetector](hold-detector.md)
**Used by:** [RouteDetectionPipeline](route-detection-pipeline.md), CLI/configuration code, [EvaluationSuite](evaluation-suite.md)

## 1. Responsibility

Create a concrete hold detector by stable method name while preventing higher-level code from importing model-specific classes.

## 2. Protocol

```python
from typing import Any, Mapping
from .hold_detector import HoldDetector

class HoldDetectorFactory:
    def create(
        self,
        detector_method: str,
        config: Mapping[str, Any] | None = None,
    ) -> HoldDetector: ...

    def available_methods(self) -> tuple[str, ...]: ...
```

The board explicitly contained `create(detectorMethod: str) -> HoldDetector`; `config` and `available_methods()` are **proposed extensions**.

## 3. Method IDs

`detector_method` MUST be a stable identifier suitable for configuration files and evaluation reports, for example `"sam_zero_shot"` or `"rcnn"`. These names are illustrative, not mandated by the whiteboard.

Factories SHOULD support registration rather than a growing `if/elif` chain:

```python
factory.register("method-id", constructor)
```

## 4. Failure behavior

Unknown methods MUST fail clearly and include the available IDs in the error.

```python
class UnknownHoldDetectorError(ValueError): ...
class InvalidHoldDetectorConfigError(ValueError): ...
```

## 5. Relationship to evaluation

The [EvaluationSuite](evaluation-suite.md) MAY use this factory to instantiate multiple detector variants under the same harness. Every created detector SHOULD expose a stable `implementation_id` matching or refining the factory method ID.
