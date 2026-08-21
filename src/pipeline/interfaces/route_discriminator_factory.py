from collections.abc import Callable, Mapping
from typing import Any
from .route_discriminator import RouteDiscriminator
class UnknownRouteDiscriminatorError(ValueError): pass
class InvalidRouteDiscriminatorConfigError(ValueError): pass
Constructor=Callable[[Mapping[str,Any]],RouteDiscriminator]
class RouteDiscriminatorFactory:
 def __init__(self):
  self._constructors={}; self.register("color_only_discriminator",self._color); self.register("triplet_route_discriminator",self._triplet)
 def register(self,method_id:str,constructor:Constructor)->None:
  if not isinstance(method_id,str) or not method_id.strip(): raise ValueError("Method ID must be non-empty.")
  if not callable(constructor): raise TypeError("Constructor must be callable.")
  self._constructors[method_id]=constructor
 def available_methods(self)->tuple[str,...]: return tuple(sorted(self._constructors))
 def create(self,discriminator_method:str,config:Mapping[str,Any]|None=None)->RouteDiscriminator:
  if discriminator_method not in self._constructors: raise UnknownRouteDiscriminatorError(f"Unknown route discriminator '{discriminator_method}'. Available methods: {', '.join(self.available_methods())}.")
  if config is not None and not isinstance(config,Mapping): raise InvalidRouteDiscriminatorConfigError("Config must be a mapping.")
  try: return self._constructors[discriminator_method](dict(config or {}))
  except InvalidRouteDiscriminatorConfigError: raise
  except (TypeError,ValueError) as error: raise InvalidRouteDiscriminatorConfigError(str(error)) from error
 @staticmethod
 def _color(config):
  from ..route_discriminator.color_only_route_discriminator import ColorOnlyRouteDiscriminator
  return ColorOnlyRouteDiscriminator(**config)
 @staticmethod
 def _triplet(config):
  from ..route_discriminator.triplet_route_discriminator import TripletRouteDiscriminator
  return TripletRouteDiscriminator(**config)

