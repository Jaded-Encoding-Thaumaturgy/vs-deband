from __future__ import annotations

import warnings
from typing import Any, Callable, Sequence

from vsexprtools import aka_expr_available, expr_func, norm_expr_planes
from vskernels import BicubicAuto
from vstools import (
    depth, disallow_variable_format, disallow_variable_resolution, get_depth, get_neutral_value,
    get_peak_value, mod4, normalize_seq, scale_value, split, vs
)

from .mask import adg_mask
from .abstract import Grainer, Debander
from .grainers import AddGrain, AddNoise

__all__ = [
    'adaptive_grain',
    'sized_grain'
]


GrainerFunc = Callable[[vs.VideoNode], vs.VideoNode]

GrainerFuncGenerator = Callable[[float, float, int, bool], GrainerFunc]


@disallow_variable_format
@disallow_variable_resolution
def adaptive_grain(
    clip: vs.VideoNode, strength: float | list[float] = 0.25,
    size: float = 1, sharp: int = 50, dynamic: bool | int = False,
    luma_scaling: float = 12, grainer: Grainer | type[Grainer] = AddGrain,
    fade_edges: bool = True, tv_range: bool = True,
    lo: int | None = None, hi: int | None = None,
    protect_neutral: bool = True, seed: int = -1,
    show_mask: bool = False, temporal_average: int | tuple[int, int] = (0, 3), **kwargs: Any
) -> vs.VideoNode:
    mask = adg_mask(clip, luma_scaling)

    vdepth = get_depth(clip)
    mask = depth(mask, vdepth)

    if show_mask:
        return mask

    grained = sized_grain(
        clip, strength, size, sharp, dynamic, grainer, fade_edges,
        tv_range, lo, hi, protect_neutral, seed, temporal_average,
        **kwargs
    )

    return clip.std.MaskedMerge(grained, mask)


@disallow_variable_format
@disallow_variable_resolution
def sized_grain(
    clip: vs.VideoNode,
    strength: float | list[float] = 0.25, size: float = 1, sharp: int = 50,
    dynamic: bool | int = True, grainer: Grainer | type[Grainer] = AddGrain,
    fade_edges: bool = True, tv_range: bool = True,
    lo: int | Sequence[int] | None = None, hi: int | Sequence[int] | None = None,
    protect_neutral: bool = True, seed: int = -1, temporal_average: int | tuple[int, int] = (0, 3), **kwargs: Any
) -> vs.VideoNode:
    assert clip.format

    sx, sy = clip.width, clip.height
    vdepth = get_depth(clip)

    if isinstance(temporal_average, tuple):
        temporal_average, temporal_radius = temporal_average
    else:
        temporal_average, temporal_radius = temporal_average, 3

    def scale_val8x(value: int, chroma: bool = False) -> float:
        return scale_value(value, 8, vdepth, scale_offsets=not tv_range, chroma=chroma)

    neutral = [get_neutral_value(clip), get_neutral_value(clip, True)]

    scaler = BicubicAuto(sharp / -50 + 1)

    if not isinstance(strength, list):
        strength = [strength, strength / 2]
    elif len(strength) > 2:
        raise ValueError('sized_grain: Only 2 strength values are supported!')

    if not isinstance(grainer, Grainer):
        grainer = grainer()

    supports_size = sharp == 50 and isinstance(grainer, AddNoise)

    kwargs |= dict(strength=strength, dynamic=dynamic)

    if supports_size:
        kwargs |= dict(xsize=size, ysize=size)

    if not isinstance(grainer, Debander):
        kwargs |= dict(seed=seed)

    if not supports_size:
        if size != 1:
            sx, sy = (mod4(x / size) for x in (sx, sy))

        sxa, sya = mod4((clip.width + sx) / 2), mod4((clip.height + sy) / 2)

    length = clip.num_frames + ((temporal_radius - 1) if not dynamic and temporal_average > 0 else 0)

    blank = clip.std.BlankClip(sx, sy, None, length, color=normalize_seq(neutral, clip.format.num_planes))

    grain = grainer.grain(blank, **kwargs)

    if not supports_size and size != 1 and (sx, sy) != (clip.width, clip.height):
        if size > 1.5:
            grain = scaler.scale(grain, sxa, sya)

        grain = scaler.scale(grain, clip.width, clip.height)

    if dynamic and temporal_average > 0:
        average = grain.std.AverageFrames([1] * temporal_radius)

        cut = (temporal_radius - 1) // 2
        grain = grain.std.Merge(average, temporal_average / 100)[cut:-cut]

    if fade_edges:
        if lo is None:
            lovals = [scale_val8x(16), scale_val8x(16, True)]
        elif not isinstance(lo, Sequence):
            lovals = [scale_val8x(lo), scale_val8x(lo, True)]
        else:
            lovals = list(lo)

        if hi is None:
            hivals = [scale_val8x(235), scale_val8x(240, True)]
        elif not isinstance(hi, Sequence):
            hivals = [scale_val8x(hi), scale_val8x(hi, True)]
        else:
            hivals = list(hi)

        if aka_expr_available:
            limit_expr = ['y {mid} - D! D@ abs DA! x DA@ - {low} < x DA@ + {high} > or x D@ x + ?']

            limit_expr = ['x y {mid} - abs - {low} < x y {mid} - abs + {high} > or x y {mid} - x + ?']

        if clip.format.sample_type == vs.FLOAT:
            limit_expr.append('x y abs + {high} > x abs y - {low} < or x x y + ?')

        grained = expr_func(
            [clip, grain], norm_expr_planes(clip, limit_expr, None, mid=neutral, low=lovals, high=hivals)
        )

        if protect_neutral and strength[1] > 0 and clip.format.color_family == vs.YUV:
            neutral_mask = clip.resize.Bicubic(format=clip.format.replace(subsampling_h=0, subsampling_w=0).id)

            # disable grain if neutral chroma
            neutral_mask = expr_func(
                split(neutral_mask), f'y {neutral[1]} = z {neutral[1]} = and {get_peak_value(clip)} 0 ?'
            )

            grained = grained.std.MaskedMerge(clip, neutral_mask, planes=[1, 2])
    else:
        if lo is not None or hi is not None:
            warnings.warn("sized_grain: setting lo and hi won't do anything with fade_edges=False", Warning)

        grained = clip.std.MergeDiff(grain)

    return grained
