from typing import Protocol, Sequence
from .data_models import Hold, Image, Route
class RouteClassificationError(RuntimeError): pass
class BatchAlignmentError(RouteClassificationError): pass
class RouteClassifier(Protocol):
 @property
 def implementation_id(self)->str: ...
 def get_routes(self,images:Sequence[Image],holds:Sequence[Sequence[Hold]])->list[list[Route]]: ...
 @staticmethod
 def mark_routes(images:Sequence[Image],routes:Sequence[Sequence[Route]])->list[Image]: ...

