from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vstools import inject_self, vs

__all__ = [
    'Grainer', 'DynamicGrainer',

    'Debander'
]


class Grainer(ABC):
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

        super().__init__()

    def __post_init__(self) -> None:
        ...

    @abstractmethod
    @inject_self
    def grain(self, clip: vs.VideoNode, strength: float | tuple[float, float], **kwargs: Any) -> vs.VideoNode:
        ...


class DynamicGrainer(Grainer):
    @abstractmethod
    @inject_self
    def grain(  # type: ignore[override]
        self, clip: vs.VideoNode, strength: float | tuple[float, float], static: bool = False, **kwargs: Any
    ) -> vs.VideoNode:
        ...


class Debander(DynamicGrainer):
    @abstractmethod
    @inject_self
    def deband(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        ...
