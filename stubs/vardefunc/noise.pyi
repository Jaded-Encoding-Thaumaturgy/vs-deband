import abc
import vapoursynth as vs
from .deband import dumb3kdb as dumb3kdb
from .mask import FDOG as FDOG
from .placebo import deband as deband
from .util import FormatError as FormatError, get_sample_type as get_sample_type, pick_px_op as pick_px_op
from abc import ABC, abstractmethod
from typing import Any, List, Sequence, Tuple, Union

core: Any

class Grainer(ABC, metaclass=abc.ABCMeta):
    kwargs: Any = ...
    def __init__(self, **kwargs: Any) -> None: ...
    @abstractmethod
    def grain(self, clip: vs.VideoNode, strength: Tuple[float, float]) -> vs.VideoNode: ...

class AddGrain(Grainer):
    def grain(self, clip: vs.VideoNode, strength: Tuple[float, float]) -> vs.VideoNode: ...

class PlaceboGrain(Grainer):
    def grain(self, clip: vs.VideoNode, strength: Tuple[float, float]) -> vs.VideoNode: ...

class F3kdbGrain(Grainer):
    def grain(self, clip: vs.VideoNode, strength: Tuple[float, float]) -> vs.VideoNode: ...

class Graigasm:
    thrs: List[float]
    strengths: List[Tuple[float, float]]
    sizes: List[float]
    sharps: List[float]
    overflows: List[float]
    grainers: List[Grainer]
    def __init__(self, thrs: Sequence[float], strengths: Sequence[Tuple[float, float]], sizes: Sequence[float], sharps: Sequence[float], *, overflows: Union[float, Sequence[float]]=..., grainers: Union[Grainer, Sequence[Grainer]]=...) -> None: ...
    def graining(self, clip: vs.VideoNode, *, prefilter: vs.VideoNode=..., show_masks: bool=...) -> vs.VideoNode: ...
    @staticmethod
    def _m__(x: int, mod: int) -> int: ...

def decsiz(clip: vs.VideoNode, sigmaS: float=..., sigmaR: float=..., min_in: Union[int, float]=..., max_in: Union[int, float]=..., gamma: float=..., protect_mask: vs.VideoNode=..., prefilter: bool=..., planes: List[int]=..., show_mask: bool=...) -> vs.VideoNode: ...
def adaptative_regrain(denoised: vs.VideoNode, new_grained: vs.VideoNode, original_grained: vs.VideoNode, range_avg: Tuple[float, float]=..., luma_scaling: int=...) -> vs.VideoNode: ...
