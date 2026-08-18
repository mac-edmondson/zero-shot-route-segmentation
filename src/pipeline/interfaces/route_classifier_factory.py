from collections.abc import Callable, Mapping
from typing import Any
from .route_classifier import RouteClassifier
class UnknownRouteClassifierError(ValueError): pass
class InvalidRouteClassifierConfigError(ValueError): pass
Constructor=Callable[[Mapping[str,Any]],RouteClassifier]
class RouteClassifierFactory:
 def __init__(self):
  self._constructors={}; self.register("color_only_classifier",self._color); self.register("triplet_route_classifier",self._triplet)
 def register(self,method_id:str,constructor:Constructor)->None:
  if not isinstance(method_id,str) or not method_id.strip(): raise ValueError("Method ID must be non-empty.")
  if not callable(constructor): raise TypeError("Constructor must be callable.")
  self._constructors[method_id]=constructor
 def available_methods(self)->tuple[str,...]: return tuple(sorted(self._constructors))
 def create(self,classifier_method:str,config:Mapping[str,Any]|None=None)->RouteClassifier:
  if classifier_method not in self._constructors: raise UnknownRouteClassifierError(f"Unknown route classifier '{classifier_method}'. Available methods: {', '.join(self.available_methods())}.")
  if config is not None and not isinstance(config,Mapping): raise InvalidRouteClassifierConfigError("Config must be a mapping.")
  try: return self._constructors[classifier_method](dict(config or {}))
  except InvalidRouteClassifierConfigError: raise
  except (TypeError,ValueError) as error: raise InvalidRouteClassifierConfigError(str(error)) from error
 @staticmethod
 def _color(config):
  from ..route_classifier.color_only_route_classifier import ColorOnlyRouteClassifier
  return ColorOnlyRouteClassifier(**config)
 @staticmethod
 def _triplet(config):
  from ..route_classifier.triplet_route_classifier import TripletRouteClassifier
  return TripletRouteClassifier(**config)

