# RouteDiscriminatorFactory Interface

TODO: Port to code and clean

**Produces:** [RouteDiscriminator](route-discriminator.md)
**Used by:** [RouteDetectionPipeline](route-detection-pipeline.md), configuration code, [EvaluationSuite](evaluation-suite.md)

## 1. Responsibility

Instantiate a route discriminator by stable method/configuration identifier while keeping model-specific dependencies out of the pipeline and dashboard layers.

## 2. Protocol

```python
from typing import Any, Mapping
from .route_discriminator import RouteDiscriminator

class RouteDiscriminatorFactory:
    def create(
        self,
        discriminator_method: str,
        config: Mapping[str, Any] | None = None,
    ) -> RouteDiscriminator: ...

    def available_methods(self) -> tuple[str, ...]: ...
```

The board explicitly showed a `RouteDiscriminatorFactory` with `create(...) -> RouteDiscriminator`; parameter names were not fully legible, so `discriminator_method` and `config` are proposed normalization.

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
class UnknownRouteDiscriminatorError(ValueError): ...
class InvalidRouteDiscriminatorConfigError(ValueError): ...
```

Unknown methods MUST fail before an evaluation run begins.
