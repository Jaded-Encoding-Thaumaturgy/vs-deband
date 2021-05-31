import vapoursynth as vs
from .util import FormatError as FormatError
from typing import Any, List, Union

core: Any

def dumb3kdb(clip: vs.VideoNode, radius: int=..., threshold: Union[int, List[int]]=..., grain: Union[int, List[int]]=..., sample_mode: int=..., use_neo: bool=..., **kwargs: Any) -> vs.VideoNode: ...
