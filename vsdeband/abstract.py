from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from vstools import inject_self, vs

__all__ = [
    'Grainer',

    'Debander'
]


class Grainer(ABC):
    @dataclass
    class SupportsConfig:
        static: bool
        dynamic: bool
        size: bool

    config: SupportsConfig

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

        super().__init__()

    def __post_init__(self) -> None:
        ...

    def _perform_graining(
        self, clip: vs.VideoNode, strength: float | tuple[float, float], dynamic: bool | int = True, **kwargs: Any
    ) -> vs.VideoNode:
        raise NotImplementedError

    @abstractmethod
    @inject_self.cached
    def grain(
        self, clip: vs.VideoNode, strength: float | tuple[float, float], dynamic: bool | int = True, **kwargs: Any
    ) -> vs.VideoNode:
        ...


class Debander(Grainer):
    @abstractmethod
    @inject_self
    def deband(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        ...
