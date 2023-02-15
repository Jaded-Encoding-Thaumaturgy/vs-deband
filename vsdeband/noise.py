from __future__ import annotations

import warnings
from typing import Any, Callable, Sequence

from vsexprtools import aka_expr_available, expr_func, norm_expr_planes
from vskernels import BicubicAuto, Lanczos, ScalerT
from vsmasktools import adg_mask
from vstools import (
    FuncExceptT, copy_signature, get_depth, get_neutral_value, get_peak_value, mod4, normalize_seq, scale_value, split,
    vs
)

from .abstract import Debander, Grainer
from .grainers import AddNoise

__all__ = [
    'adaptive_grain',
    'sized_grain'
]


GrainerFunc = Callable[[vs.VideoNode], vs.VideoNode]

GrainerFuncGenerator = Callable[[float, float, int, bool], GrainerFunc]


class _sized_grain:
    def __call__(
        self, clip: vs.VideoNode, strength: float | list[float] = 0.25,
        size: float = 1, dynamic: bool | int = False,
        grainer: Grainer | type[Grainer] = AddNoise,
        fade_edges: bool = True, tv_range: bool = True,
        lo: int | None = None, hi: int | None = None,
        protect_neutral: bool = True, seed: int = -1,
        temporal_average: int | tuple[int, int] = (0, 1),
        postprocess: Callable[[vs.VideoNode], vs.VideoNode] | None = None,
        scaler: ScalerT | None = None, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        assert clip.format

        func = func or sized_grain

        if 'sharp' in kwargs:
            warnings.warn('The "sharp" parameter is deprecated! Please use scaler=BicubicAuto(sharp / -50 + 1).')
            scaler = BicubicAuto(kwargs.pop('sharp') / -50 + 1)

        sx, sy = clip.width, clip.height
        vdepth = get_depth(clip)

        if isinstance(temporal_average, tuple):
            temporal_average, temporal_radius = temporal_average
        else:
            temporal_average, temporal_radius = temporal_average, 1

        do_taverage = dynamic and temporal_average > 0 and temporal_radius > 0
        temporal_window = temporal_radius * 2 + 1

        def scale_val8x(value: int, chroma: bool = False) -> float:
            return scale_value(value, 8, vdepth, scale_offsets=not tv_range, chroma=chroma)

        neutral = [get_neutral_value(clip), get_neutral_value(clip, True)]

        scaler = Lanczos.ensure_obj(scaler, func)

        if not isinstance(strength, list):
            strength = [strength, strength / 2]
        elif len(strength) > 2:
            raise ValueError('sized_grain: Only 2 strength values are supported!')

        if not isinstance(grainer, Grainer):
            grainer = grainer()

        kwargs |= dict(strength=strength, dynamic=dynamic)

        if not isinstance(grainer, Debander):
            kwargs |= dict(seed=seed)

        if size != 1:
            sx, sy = (mod4(x / size) for x in (sx, sy))

        sxa, sya = mod4((clip.width + sx) / 2), mod4((clip.height + sy) / 2)

        length = clip.num_frames + (temporal_window - 1 if do_taverage else 0)

        blank = clip.std.BlankClip(sx, sy, None, length, color=normalize_seq(neutral, clip.format.num_planes))

        grain = grainer.grain(blank, **kwargs)

        if size != 1 and (sx, sy) != (clip.width, clip.height):
            if size > 1.5:
                grain = scaler.scale(grain, sxa, sya)

            grain = scaler.scale(grain, clip.width, clip.height)

        if postprocess:
            grain = postprocess(grain)

        if do_taverage:
            average = grain.std.AverageFrames([1] * temporal_window)
            grain = grain.std.Merge(average, temporal_average / 100)[temporal_radius:-temporal_radius]

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
            else:
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

    def adaptive(
        self, clip: vs.VideoNode, strength: float | list[float] = 0.25,
        size: float = 1, dynamic: bool | int = False,
        luma_scaling: float = 12, grainer: Grainer | type[Grainer] = AddNoise,
        fade_edges: bool = True, tv_range: bool = True,
        lo: int | None = None, hi: int | None = None,
        protect_neutral: bool = True, seed: int = -1,
        temporal_average: int | tuple[int, int] = (0, 1),
        postprocess: Callable[[vs.VideoNode], vs.VideoNode] | None = None,
        scaler: ScalerT | None = None, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        grained = sized_grain(
            clip, strength, size, dynamic, grainer, fade_edges, tv_range,
            lo, hi, protect_neutral, seed, temporal_average, postprocess,
            scaler, func or self.adaptive, **kwargs
        )

        return clip.std.MaskedMerge(grained, adg_mask(clip, luma_scaling))


sized_grain = _sized_grain()


@copy_signature(sized_grain.adaptive)
def adaptive_grain(*args: Any, **kwargs: Any) -> Any:
    import warnings
    warnings.warn('adaptive_grain is deprecated! Please use sized_grain.adaptive.')

    return sized_grain.adaptive(*args, **kwargs)
