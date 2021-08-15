"""
    Various functions used for debanding.

    This used to be the `debandshit` module written by Z4ST1N,
    with some functions that were rarely (if ever) used removed because I can't reasonably maintain them.
"""
from typing import Any, Dict, List, Optional, Union

import vapoursynth as vs
from vsutil import depth

from .f3kdb import SAMPLEMODE, F3kdb, SampleMode
from .placebo import Placebo

core = vs.core

__all__ = ['dumb3kdb', 'f3kbilateral', 'f3kpf', 'lfdeband', 'placebo_deband']


def dumb3kdb(clip: vs.VideoNode, radius: int = 16,
             threshold: Union[int, List[int]] = 30, grain: Union[int, List[int]] = 0,
             sample_mode: Union[SAMPLEMODE, SampleMode] = 2, use_neo: bool = False, **kwargs: Any) -> vs.VideoNode:
    """Small convenience function for calling F3kdb().deband()."""
    return F3kdb(radius, threshold, grain, sample_mode, use_neo, **kwargs).deband(clip)


def f3kbilateral(clip: vs.VideoNode, radius: int = 16,
                 threshold: Union[int, List[int]] = 65, grain: Union[int, List[int]] = 0,
                 f3kdb_args: Optional[Dict[str, Any]] = None,
                 limflt_args: Optional[Dict[str, Any]] = None) -> vs.VideoNode:
    """
    f3kbilateral: f3kdb multistage bilateral-esque filter from debandshit.

    This function is more of a last resort for extreme banding.
    Recommend values are ~40-60 for y and c strengths.

    Dependencies:
    * mvsfunc

    :param clip:        Input clip
    :param radius:      Same as F3kdb constructor.
    :param threshold:   Same as F3kdb constructor.
    :param grain:       Same as F3kdb constructor.
                        It happens after mvsfunc.LimitFilter
                        and call another instance of F3kdb if != 0.
    :f3kdb_args:        Same as F3kdb kwargs constructor.
    :lf_args:           Arguments passed to mvsfunc.LimitFilter.

    :return:            Debanded clip
    """
    try:
        from mvsfunc import LimitFilter
    except ModuleNotFoundError as mod_err:
        raise ModuleNotFoundError("f3kbilateral: missing dependency 'mvsfunc'") from mod_err

    if clip.format is None:
        raise ValueError("f3kbilateral: 'Variable-format clips not supported'")

    bits = clip.format.bits_per_sample

    f3_args: Dict[str, Any] = dict()
    if f3kdb_args is not None:
        f3_args |= f3kdb_args

    lf_args: Dict[str, Any] = dict(thr=0.6, elast=3.0, thrc=None)
    if limflt_args is not None:
        lf_args |= limflt_args

    rad1 = round(radius * 4 / 3)
    rad2 = round(radius * 2 / 3)
    rad3 = round(radius / 3)

    db1 = F3kdb(rad1, threshold, 0, **f3_args)
    db2 = F3kdb(rad2, threshold, 0, **f3_args)
    db3 = F3kdb(rad3, threshold, 0, **f3_args)

    # Edit the thr of first f3kdb object
    db1.thy, db1.thcb, db1.thcr = [max(1, th // 2) for th in (db1.thy, db1.thcb, db1.thcr)]

    clip = depth(clip, 16)

    flt1 = db1.deband(clip)
    flt2 = db2.deband(flt1)
    flt3 = db3.deband(flt2)

    limit = LimitFilter(flt3, flt2, ref=clip, **lf_args)

    if grain:
        grained = F3kdb(grain=grain, **f3_args).grain(limit)
    else:
        grained = limit

    return depth(grained, bits)


def f3kpf(clip: vs.VideoNode, radius: int = 16,
          threshold: Union[int, List[int]] = 30, grain: Union[int, List[int]] = 0,
          f3kdb_args: Optional[Dict[str, Any]] = None,
          limflt_args: Optional[Dict[str, Any]] = None) -> vs.VideoNode:
    """
    f3kdb with a simple prefilter by mawen1250 - https://www.nmm-hd.org/newbbs/viewtopic.php?f=7&t=1495#p12163.

    Since the prefilter is a straight gaussian+average blur, f3kdb's effect becomes very strong, very fast.
    Functions more or less like gradfun3 without the detail mask.

    Dependencies:
    * mvsfunc

    :param clip:        Input clip
    :param radius:      Banding detection range
    :param threshold:   Banding detection thresholds for multiple planes
    :param f3kdb_args:  Arguments passed to F3kdb constructor
    :param limflt_args: Arguments passed to mvsfunc.LimitFilter

    :return:            Debanded clip
    """
    try:
        from mvsfunc import LimitFilter
    except ModuleNotFoundError as mod_err:
        raise ModuleNotFoundError("f3kpf: missing dependency 'mvsfunc'") from mod_err

    if clip.format is None:
        raise ValueError("f3kpf: 'Variable-format clips not supported'")

    f3_args: Dict[str, Any] = dict()
    if f3kdb_args is not None:
        f3_args |= f3kdb_args

    lf_args: Dict[str, Any] = dict(thr=0.3, elast=2.5, thrc=None)
    if limflt_args is not None:
        lf_args |= limflt_args

    blur = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Convolution([1] * 9, planes=0)
    diff = core.std.MakeDiff(clip, blur)

    deband = F3kdb(radius, threshold, grain, **f3_args).deband(blur)
    deband = LimitFilter(deband, blur, **lf_args)

    return core.std.MergeDiff(deband, diff)


def lfdeband(clip: vs.VideoNode) -> vs.VideoNode:
    """
    A simple debander ported from AviSynth.

    :param clip:        Input clip

    :return:            Debanded clip
    """
    if clip.format is None:
        raise ValueError("lfdeband: 'Variable-format clips not supported'")

    bits = clip.format.bits_per_sample
    wss, hss = 1 << clip.format.subsampling_w, 1 << clip.format.subsampling_h
    w, h = clip.width, clip.height
    dw, dh = round(w / 2), round(h / 2)

    clip = depth(clip, 16)
    dsc = core.resize.Spline64(clip, dw-dw % wss, dh-dh % hss)

    d3kdb = F3kdb(radius=30, threshold=80, grain=0).deband(clip)

    ddif = core.std.MakeDiff(d3kdb, dsc)

    dif = core.resize.Spline64(ddif, w, h)
    out = core.std.MergeDiff(clip, dif)
    return depth(out, bits)


def placebo_deband(clip: vs.VideoNode, radius: float = 16.0, threshold: Union[float, List[float]] = 4.0,
                   iterations: int = 1, grain: Union[float, List[float]] = 6.0, **kwargs: Any) -> vs.VideoNode:
    """Small convenience function for calling Placebo().deband()."""
    return Placebo(radius, threshold, iterations, grain, **kwargs).deband(clip)
