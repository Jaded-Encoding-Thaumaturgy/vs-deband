from __future__ import annotations

import warnings
from typing import Any, Callable, Sequence

from vsexprtools import aka_expr_available, expr_func, norm_expr_planes
from vskernels import Bicubic, BicubicAuto, Kernel, Sinc
from vstools import (
    depth, disallow_variable_format, disallow_variable_resolution, get_depth, get_neutral_value,
    get_peak_value, mod4, normalize_seq, scale_value, split, vs
)
from vsmasktools import adg_mask

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
    size: float = 1.0, sharp: int = 50, dynamic: bool | int = False,
    luma_scaling: float = 12, grainer: Grainer | type[Grainer] = AddGrain,
    fade_edges: bool = True, tv_range: bool = True, kernel: Kernel = Sinc,
    lo: int | None = None, hi: int | None = None,
    protect_neutral: bool = True, seed: int = -1,
    show_mask: bool = False, temporal_average: int | tuple[int, int] = (0, 3),
    **kwargs: Any
) -> vs.VideoNode:
    """
    Adaptive graining using a luminance mask.

    Based on kageru's `adaptive_grain`. For more information, check out his
    `blog post <https://blog.kageru.moe/legacy/adaptivegrain.html>`_ on the subject.

    This function greatly expands the scope of the old `adaptive_grain`,
    adding a lot of useful functionality to further adjust the grain added to the clip.

    :param clip:                Clip to process.
    :param strength:            Graining strength. Added grain is relative to the `grainer` passed.
                                Accepts a list of floats or a single float. The strength is applied
                                to each plane separately. If one float is passed, the same strength
                                is applied to every plane.
                                Default: 0.25.
    :param size:                Relative size of the grain. This parameter allows you to tune the size
                                of the grain. Lower-than-1 values will create smaller, finer pieces of grain,
                                whereas higher-than-1 will increase the size.
                                For more information, see :py:func:`sized_grain`. Default: 1.0.
    :param sharp:               Sharpness of the grain resizing. This affects the `Bicubic` "c" value.
                                Higher values will be sharper, but may allow create more ringing and streaking.
                                Only used if `kernel` is a Bicubic Kernel. Default: 50.
    :param dynamic:             Whether to make the grain dynamic. False makes the grain static. Default: False.
    :param luma_scaling:        Luma scale adjusting. Lower values will catch more darker areas, and vice versa.
                                Recommended values are between 4 minimum and 16 maximum. Default: 12.
    :param grainer:             The graining method to use. Must be a graining class inheriting :py:class:`Grainer`.
                                Default: AddGrain.
    :param fade_edges:          @PLACEHOLDER@
    :param tv_range:            Whether to treat the clip as `TV Range` for internal value scaling.
                                Default: True.
    :param kernel:              Kernel used for scaling if size is greater or lesser than 1.0.
                                Most kernels will create a streaking-like effect,
                                so only the following kernels are recommended:

                                * Sinc
                                * Spline
                                * Lanczos
                                * Blackman

                                If you pass a `Bicubic` kernel, `sharp` will be used to determine the "c" value.

                                Default: Sinc.
    :param lo:                  @PLACEHOLDER@
    :param hi:                  @PLACEHOLDER@
    :param protect_netural:     @PLACEHOLDER@
    :param seed:                Seed for the grain pattern. Set this to make the grain deterministic.
                                Default: -1 (random).
    :param show_mask:           Return the luminosity mask. Default: False.
    :param temporal_average:    Temporal averaging of grain. Higher values will make the grain pattern
                                smoother across multiple frames. Default: (0, 3)
                                @PLACEHOLDER@ # ^ Was this not how it always worked? Or did you change it?

    :return:                    Either a grained clip or a luminosity mask.
    """
    mask = adg_mask(clip, luma_scaling)

    vdepth = get_depth(clip)
    mask = depth(mask, vdepth)

    if show_mask:
        return mask

    grained = sized_grain(
        clip, strength, size, sharp, dynamic, grainer, fade_edges,
        tv_range, kernel, lo, hi, protect_neutral, seed, temporal_average,
        **kwargs
    )

    return clip.std.MaskedMerge(grained, mask)


@disallow_variable_format
@disallow_variable_resolution
def sized_grain(
    clip: vs.VideoNode,
    strength: float | list[float] = 0.25, size: float = 1, sharp: int = 50,
    dynamic: bool | int = True, grainer: Grainer | type[Grainer] = AddGrain,
    fade_edges: bool = True, tv_range: bool = True, kernel: Kernel = Sinc,
    lo: int | Sequence[int] | None = None, hi: int | Sequence[int] | None = None,
    protect_neutral: bool = True, seed: int = -1, temporal_average: int | tuple[int, int] = (0, 3),
    **kwargs: Any
) -> vs.VideoNode:
    """
    Adjust the size of the grains when graining a clip.

    For more information, please refer to :py:func:`adaptive_grain`'s docstring.

    :return:        Grained clip with the grain adjusted.
    """
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

    scaler = BicubicAuto(sharp / -50 + 1) if isinstance(kernel, Bicubic) else kernel

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
            neutral_mask = kernel.resample(clip, format=clip.format.replace(subsampling_h=0, subsampling_w=0).id)

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
