import vapoursynth as vs
from .util import FormatError as FormatError
from typing import Any, Dict, List, Union

core: Any

class F3kdb:
    radius: int
    thy: int
    thcb: int
    thcr: int
    gry: int
    grc: int
    sample_mode: int
    use_neo: bool
    f3kdb_args: Dict[str, Any]
    def __init__(self, radius: int=..., threshold: Union[int, List[int]]=..., grain: Union[int, List[int]]=..., sample_mode: int=..., use_neo: bool=..., **kwargs: Any) -> None: ...
    def deband(self, clip: vs.VideoNode) -> vs.VideoNode: ...
    def grain(self, clip: vs.VideoNode) -> vs.VideoNode: ...

def dumb3kdb(clip: vs.VideoNode, radius: int=..., threshold: Union[int, List[int]]=..., grain: Union[int, List[int]]=..., sample_mode: int=..., use_neo: bool=..., **kwargs: Any) -> vs.VideoNode: ...
def f3kbilateral(clip: vs.VideoNode, radius: int=..., threshold: Union[int, List[int]]=..., grain: Union[int, List[int]]=..., f3kdb_args: Dict[str, Any]=..., lf_args: Dict[str, Any]=...) -> vs.VideoNode: ...
def lfdeband(clip: vs.VideoNode) -> vs.VideoNode: ...
