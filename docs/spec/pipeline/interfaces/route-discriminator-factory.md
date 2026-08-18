# RouteDiscriminatorFactory Interface

**Produces:** [RouteDiscriminator](route-discriminator.md)
**Used by:** [RouteDiscriminatorPipeline](route-discriminator-pipeline.md), configuration code, [EvaluationSuite](evaluation-suite.md)

## 1. Responsibility

Instantiate a route discriminator by stable method/configuration identifier while keeping model-specific dependencies out of the pipeline and dashboard layers.

## 2. Interface Contract, self-documenting

Implementation: [/src/pipeline/route_discriminator/route_discriminator_factory.py](/src/pipeline/route_discriminator/route_discriminator_factory.py)
