from __future__ import annotations

from functools import partial
from math import ceil
from typing import Any

from vsdenoise import Prefilter
from vsexprtools import ExprOp
from vskernels import Scaler, ScalerT, Spline64
from vsmasktools import Morpho
from vsrgtools import RemoveGrainMode, RemoveGrainModeT, limit_filter, removegrain
from vstools import (
    ColorRange, PlanesT, VSFunction, check_variable, depth, expect_bits, fallback, normalize_planes, normalize_seq,
    scale_value, to_arr, vs
)

from .abstract import Debander
from .f3kdb import F3kdb
from .filters import guided_filter
from .mask import deband_detail_mask
from .types import GuidedFilterMode

__all__ = [
    'mdb_bilateral',

    'masked_deband',

    'pfdeband',

    'guided_deband'
]


def mdb_bilateral(
    clip: vs.VideoNode, radius: int = 16,
    thr: int | list[int] = 65, grains: int | tuple[int, int] = 0,
    lthr: int | tuple[int, int] = (153, 0), elast: float = 3.0,
    bright_thr: int | None = None,
    debander: type[Debander] | Debander = F3kdb
) -> vs.VideoNode:
    """
    Multi stage debanding, bilateral-esque filter.

    This function is more of a last resort for extreme banding.
    Recommend values are ~40-60 for luma and chroma strengths.

    :param clip:        Input clip.
    :param radius:      Banding detection range.
    :param thr:         Banding detection thr(s) for planes.
    :param grains:      Specifies amount of grains added in the last debanding stage.
                        It happens after `vsrgtools.limit_filter`.
    :param lthr:        Threshold of the limiting. Refer to `vsrgtools.limit_filter`.
    :param elast:       Elasticity of the limiting. Refer to `vsrgtools.limit_filter`.
    :param bright_thr:  Limiting over the bright areas. Refer to `vsrgtools.limit_filter`.
    :param debander:    Specify what Debander to use. You can pass an instance with custom arguments.

    :return:            Debanded clip.
    """

    assert check_variable(clip, mdb_bilateral)

    if not isinstance(debander, Debander):
        debander = debander()

    clip, bits = expect_bits(clip, 16)

    rad1, rad2, rad3 = round(radius * 4 / 3), round(radius * 2 / 3), round(radius / 3)

    db1 = debander.deband(clip, radius=rad1, thr=[max(1, th // 2) for th in to_arr(thr)], grains=0)
    db2 = debander.deband(db1, radius=rad2, thr=thr, grains=0)
    db3 = debander.deband(db2, radius=rad3, thr=thr, grains=0)

    limit = limit_filter(db3, db2, clip, thr=lthr, elast=elast, bright_thr=bright_thr)

    if grains:
        limit = debander.grain(limit, strength=grains)

    return depth(limit, bits)


def masked_deband(
    clip: vs.VideoNode, radius: int = 16,
    thr: int | list[int] = 24, grains: int | list[int] = [12, 0],
    sigma: float = 1.25, rxsigma: list[int] = [50, 220, 300],
    pf_sigma: float | None = 1.25, brz: tuple[int, int] = (2500, 4500),
    rg_mode: RemoveGrainModeT = RemoveGrainMode.MINMAX_MEDIAN_OPP,
    debander: type[Debander] | Debander = F3kdb, **kwargs: Any
) -> vs.VideoNode:
    clip, bits = expect_bits(clip, 16)

    if not isinstance(debander, Debander):
        debander = debander()

    deband_mask = deband_detail_mask(clip, sigma, rxsigma, pf_sigma, brz, rg_mode)

    deband = debander.deband(clip, radius=radius, thr=thr, grains=grains, **kwargs)

    masked = deband.std.MaskedMerge(clip, deband_mask)

    return depth(masked, bits)


def pfdeband(
    clip: vs.VideoNode, radius: int = 16,
    thr: int | list[int] = 30, grains: int | list[int] = 0,
    lthr: int | tuple[int, int] = (76, 0), elast: float = 2.5,
    bright_thr: int | None = None, scaler: ScalerT = Spline64,
    prefilter: Prefilter | VSFunction = partial(Prefilter.SCALEDBLUR, scale=1, radius=2),
    debander: type[Debander] | Debander = F3kdb, **kwargs: Any
) -> vs.VideoNode:
    """
    Prefilter and deband a clip.

    :param clip:        Input clip.
    :param radius:      Banding detection range.
    :param thr:         Banding detection thr(s) for planes.
    :param grains:      Specifies amount of grains added in the last debanding stage.
                        It happens after `vsrgtools.limit_filter`.
    :param lthr:        Threshold of the limiting. Refer to `vsrgtools.limit_filter`.
    :param elast:       Elasticity of the limiting. Refer to `vsrgtools.limit_filter`.
    :param bright_thr:  Limiting over the bright areas. Refer to `vsrgtools.limit_filter`.
    :param prefilter:   Prefilter used to blur the clip before debanding.
    :param debander:    Specify what Debander to use. You can pass an instance with custom arguments.

    :return:            Debanded clip.
    """

    assert check_variable(clip, pfdeband)

    if not isinstance(debander, Debander):
        debander = debander()

    scaler = Scaler.ensure_obj(scaler, pfdeband)

    clip, bits = expect_bits(clip, 16)

    blur = prefilter(clip, **kwargs)
    change_res = (blur.width, blur.height) != (clip.width, clip.height)

    deband = debander.deband(blur, radius=radius, thr=thr, grains=grains)

    out = deband if change_res else clip

    diff = out.std.MakeDiff(blur)

    if change_res:
        diff = scaler.scale(diff, clip.width, clip.height)
    else:
        out = limit_filter(deband, blur, thr=lthr, elast=elast, bright_thr=bright_thr)

    out = out.std.MergeDiff(diff)

    return depth(out, bits)


def guided_deband(
    clip: vs.VideoNode, radius: int | list[int] | None = None, strength: float = 0.3,
    thr: float | list[float] | None = None, mode: GuidedFilterMode = GuidedFilterMode.GRADIENT,
    rad: int = 0, bin_thr: float | list[float] | None = 0, planes: PlanesT = None,
    range_in: ColorRange | None = None, **kwargs: Any
) -> vs.VideoNode:
    assert check_variable(clip, guided_deband)

    planes = normalize_planes(clip, planes)

    range_in = ColorRange.from_param(range_in) or ColorRange.from_video(clip)

    rad = fallback(rad, ceil(clip.height / 540))

    if bin_thr is None:
        if clip.format.sample_type is vs.FLOAT:
            bin_thr = 1.5 / 255 if range_in.is_full else [1.5 / 219, 1.5 / 224]
        else:
            bin_thr = scale_value(0.005859375, 32, clip, range_in)

    bin_thr = normalize_seq(bin_thr, clip.format.num_planes)

    deband = guided_filter(clip, None, radius, strength, mode, planes=planes, **kwargs)

    if thr:
        deband = limit_filter(deband, clip, thr=tuple(map(int, to_arr(thr))))  # type: ignore

    if rad:
        morpho = Morpho(planes)
        rmask = ExprOp.SUB.combine(morpho.expand(clip, rad), morpho.inpand(clip, rad), planes=planes)

        if bin_thr and max(bin_thr) > 0:
            rmask = rmask.std.Binarize(threshold=bin_thr, planes=planes)

        rmask = removegrain(rmask, RemoveGrainMode.OPP_CLIP_AVG_FAST)
        rmask = removegrain(rmask, RemoveGrainMode.SQUARE_BLUR)
        rmask = removegrain(rmask, RemoveGrainMode.MIN_SHARP)

        deband = deband.std.MaskedMerge(clip, rmask, planes=planes)

    return deband
