from __future__ import annotations

from functools import partial
from typing import Any

from vsrgtools import blur, limit_filter
from vstools import VSFunction, check_variable, core, depth, expect_bits, to_arr, vs

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

    :param clip:        Input clip.
    :param radius:      Banding detection range.
    :param thr:         Banding detection thr(s) for planes.
    :param grains:      Specifies amount of grains added in the last debanding stage.
                        It happens after `vsrgtools.limit_filter`.
    :lthr:              Threshold of the limiting. Refer to `vsrgtools.limit_filter`.
    :elast:             Elasticity of the limiting. Refer to `vsrgtools.limit_filter`.
    :bright_thr:        Limiting over the bright areas. Refer to `vsrgtools.limit_filter`.
    :debander:          Specify what Debander to use. You can pass an instance with custom arguments.

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
        limit = debander.grain(limit, grains=grains)

    return depth(limit, bits)


def pref_deband(
    clip: vs.VideoNode, radius: int = 16,
    thr: int | list[int] = 30, grains: int | list[int] = 0,
    lthr: int | tuple[int, int] = [76, 0], elast: float = 2.5,
    bright_thr: int | None = None, prefilter: VSFunction = partial(blur, radius=2),
    debander: type[Debander] | Debander = F3kdb, **kwargs: Any
) -> vs.VideoNode:
    """
    pref_deband with a simple prefilter `by mawen1250 <https://www.nmm-hd.org/newbbs/viewtopic.php?f=7&t=1495#p12163.`>_

    The default prefilter is a straight gaussian+average blur, so the effect becomes very strong very fast.
    Functions more or less like GradFun3 without the detail mask.

    :param clip:        Input clip.
    :param radius:      Banding detection range.
    :param thr:         Banding detection thr(s) for planes.
    :param grains:      Specifies amount of grains added in the last debanding stage.
                        It happens after `vsrgtools.limit_filter`.
    :lthr:              Threshold of the limiting. Refer to `vsrgtools.limit_filter`.
    :elast:             Elasticity of the limiting. Refer to `vsrgtools.limit_filter`.
    :bright_thr:        Limiting over the bright areas. Refer to `vsrgtools.limit_filter`.
    :prefilter:         Prefilter used to blur the clip before debanding.
    :debander:          Specify what Debander to use. You can pass an instance with custom arguments.

    :return:            Debanded clip.
    """

    assert check_variable(clip, pref_deband)

    if not isinstance(debander, Debander):
        debander = debander()

    try:
        blur = prefilter(clip, planes=0, **kwargs)
    except Exception:
        blur = prefilter(clip, **kwargs)

    diff = core.std.MakeDiff(clip, blur)

    deband = debander.deband(blur, radius=radius, thr=thr, grains=grains)

    limit = limit_filter(deband, blur, thr=lthr, elast=elast, bright_thr=bright_thr)

    return limit.std.MergeDiff(diff)


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
