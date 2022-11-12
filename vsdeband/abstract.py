from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vstools import inject_self, vs

__all__ = [
    'Debander',

    'Grainer'
]


class Debander(ABC):
    @abstractmethod
    @inject_self
    def deband(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        ...


class Grainer(ABC):
    @abstractmethod
    @inject_self
    def grain(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        ...
