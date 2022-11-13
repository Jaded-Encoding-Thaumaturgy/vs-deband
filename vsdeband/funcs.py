from __future__ import annotations

from typing import Any

from vsrgtools import limit_filter
from vstools import KwargsT, check_variable, core, depth, expect_bits, to_arr, vs

from .abstract import Debander
from .f3kdb import F3kdb

__all__ = [
    'mdb_bilateral', 'f3kpf',

    'lfdeband'
]


def mdb_bilateral(
    clip: vs.VideoNode, radius: int = 16,
    thr: int | list[int] = 65, grains: int | list[int] = 0,
    lthr: int | tuple[int, int] = [153, 0], elast: float = 3.0,
    bright_thr: int | None = None,
    debander: type[Debander] | Debander = F3kdb
) -> vs.VideoNode:
    """
    Multi stage debanding, bilateral-esque filter.

    This function is more of a last resort for extreme banding.
    Recommend values are ~40-60 for luma and chroma strengths.

    :param clip:        Input clip
    :param radius:      Banding detection range.
    :param thr:         Banding detection thr(s) for planes.
    :param grains:      Specifies amount of grains added in the last debanding stage.
                        It happens after `vsrgtools.limit_filter`.
    :lthr:              Threshold of the limiting. Refer to `vsrgtools.limit_filter`.
    :elast:             Elasticity of the limiting. Refer to `vsrgtools.limit_filter`.
    :bright_thr:        Limiting over the bright areas. Refer to `vsrgtools.limit_filter`.
    :debander:          Specify what Debander to use. You can pass an instance with custom arguments.

    :return:            Debanded clip
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
        limit = debander.grain(limit, grains=grains)

    return depth(limit, bits)


def f3kpf(
    clip: vs.VideoNode, radius: int = 16,
    threshold: int | list[int] = 30, grain: int | list[int] = 0,
    f3kdb_args: KwargsT | None = None,
    limflt_args: KwargsT | None = None
) -> vs.VideoNode:
    """
    f3kdb with a simple prefilter by mawen1250 - https://www.nmm-hd.org/newbbs/viewtopic.php?f=7&t=1495#p12163.

    Since the prefilter is a straight gaussian+average blur, f3kdb's effect becomes very strong, very fast.
    Functions more or less like gradfun3 without the detail mask.

    :param clip:        Input clip
    :param radius:      Banding detection range
    :param threshold:   Banding detection thresholds for multiple planes
    :param f3kdb_args:  Arguments passed to F3kdb constructor
    :param limflt_args: Arguments passed to vsrgtools.limit_filter

    :return:            Debanded clip
    """

    assert check_variable(clip, f3kpf)

    f3_args = (f3kdb_args and f3kdb_args.copy()) or {}

    lf_args = KwargsT(thr=0.3, elast=2.5, thrc=None) | (limflt_args or {})

    blur = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Convolution([1] * 9, planes=0)
    diff = core.std.MakeDiff(clip, blur)

    deband = F3kdb(radius, threshold, grain, **f3_args).deband(blur)
    deband = limit_filter(deband, blur, **lf_args)

    return core.std.MergeDiff(deband, diff)


def lfdeband(
    clip: vs.VideoNode, radius: int = 30,
    threshold: int | list[int] = 80, grain: int | list[int] = 0,
    **f3kdb_args: Any
) -> vs.VideoNode:
    """
    A simple debander ported from AviSynth.

    :param clip:        Input clip
    :param radius:      Banding detection range
    :param threshold:   Banding detection thresholds for multiple planes
    :param f3kdb_args:  Arguments passed to F3kdb constructor

    :return:            Debanded clip
    """

    assert check_variable(clip, lfdeband)

    wss, hss = 1 << clip.format.subsampling_w, 1 << clip.format.subsampling_h

    w, h = clip.width, clip.height

    dw, dh = round(w / 2), round(h / 2)

    clip, bits = expect_bits(clip, 16)
    dsc = core.resize.Spline64(clip, dw-dw % wss, dh-dh % hss)

    d3kdb = F3kdb(radius, threshold, grain, **f3kdb_args).deband(dsc)

    ddif = d3kdb.std.MakeDiff(dsc)

    dif = core.resize.Spline64(ddif, w, h)

    out = clip.std.MergeDiff(dif)

    return depth(out, bits)
