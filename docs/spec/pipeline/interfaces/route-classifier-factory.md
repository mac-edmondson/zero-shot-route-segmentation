# RouteClassifierFactory Interface

TODO: Port to code and clean

**Produces:** [RouteClassifier](route-classifier.md)
**Used by:** [RouteDetectionPipeline](route-detection-pipeline.md), configuration code, [EvaluationSuite](evaluation-suite.md)

## 1. Responsibility

Instantiate a route classifier by stable method/configuration identifier while keeping model-specific dependencies out of the pipeline and dashboard layers.

## 2. Protocol

```python
from typing import Any, Mapping
from .route_classifier import RouteClassifier

class RouteClassifierFactory:
    def create(
        self,
        classifier_method: str,
        config: Mapping[str, Any] | None = None,
    ) -> RouteClassifier: ...

    def available_methods(self) -> tuple[str, ...]: ...
```

The board explicitly showed a `RouteClassifierFactory` with `create(...) -> RouteClassifier`; parameter names were not fully legible, so `classifier_method` and `config` are proposed normalization.

## 3. Configuration

Feature-selection conditions (color/spatial/DINO) SHOULD be configuration, for example:

```python
factory.create(
    "route-clusterer",
    {
        "use_color": True,
        "use_spatial": True,
        "use_dino": False,
    },
)
```

Exact implementation IDs are project decisions.

## 4. Errors

```python
class UnknownRouteClassifierError(ValueError): ...
class InvalidRouteClassifierConfigError(ValueError): ...
```

Unknown methods MUST fail before an evaluation run begins.
