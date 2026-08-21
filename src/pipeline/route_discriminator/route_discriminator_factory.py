"""Factory and registry for route discriminators."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .route_discriminator import RouteDiscriminator


################################################################################
# Errors
################################################################################
class UnknownRouteDiscriminatorError(ValueError):
    pass


################################################################################
# Constructor Wrapper Methods
################################################################################
_ConfigArgs = Mapping[str, Any]
_Constructor = Callable[[_ConfigArgs], RouteDiscriminator]


def _create_triplet_route_discriminator(
    config: Mapping[str, Any] = dict(),
) -> RouteDiscriminator:
    from .triplet_route_discriminator import TripletRouteDiscriminator

    return TripletRouteDiscriminator(**config)


def _create_mock_route_discriminator(
    config: Mapping[str, Any] = dict(),
) -> RouteDiscriminator:
    from .mock_route_discriminator import MockRouteDiscriminator

    return MockRouteDiscriminator(**config)


_AVAILABLE_ROUTE_DISCRIMINATORS_MAP: Mapping[str, _Constructor] = {
    "Triplet MLP": _create_triplet_route_discriminator,
    "Mock": _create_mock_route_discriminator,
}
AVAILABLE_ROUTE_DISCRIMINATORS = list(_AVAILABLE_ROUTE_DISCRIMINATORS_MAP.keys())


################################################################################
# The Factory
################################################################################
def route_discriminator_factory(hold_detector_name: str, config: _ConfigArgs = dict()):
    if hold_detector_name not in _AVAILABLE_ROUTE_DISCRIMINATORS_MAP:
        raise UnknownRouteDiscriminatorError(
            f"'{hold_detector_name}' isn't a known route discriminator. Try one of {AVAILABLE_ROUTE_DISCRIMINATORS}"
        )

    return _AVAILABLE_ROUTE_DISCRIMINATORS_MAP[hold_detector_name](config)
