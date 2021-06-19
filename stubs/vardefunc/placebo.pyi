import vapoursynth as vs
from .util import FormatError as FormatError
from typing import Any, List, Union

core: Any

def deband(clip: vs.VideoNode, radius: float=..., threshold: Union[float, List[float]]=..., iterations: int=..., grain: Union[float, List[float]]=..., chroma: bool=..., **kwargs: Any) -> vs.VideoNode: ...
def shader(clip: vs.VideoNode, width: int, height: int, shader_file: str, luma_only: bool=..., **kwargs: Any) -> vs.VideoNode: ...
