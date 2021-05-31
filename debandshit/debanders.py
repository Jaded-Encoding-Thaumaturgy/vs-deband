"""
    Various functions used for debanding.

    This used to be the `debandshit` module written by Z4ST1N,
    with some functions that were rarely (if ever) used removed because I can't reasonably maintain them.
"""
from typing import Any, List, Optional, Union

import vapoursynth as vs
from vardefunc.deband import dumb3kdb
from vsutil import depth

core = vs.core


def f3kbilateral(clip: vs.VideoNode,
                 radius: int = 16,
                 y: int = 64, c: int = 1,
                 **kwargs: Any) -> vs.VideoNode:
    """
    f3kbilateral: f3kdb multistage bilateral-esque filter from debandshit.

    This function is more of a last resort for extreme banding.
    Recommend values are ~40-60 for y and c strengths.

    :param clip:        Input clip
    :param radius:      Banding detection range
    :param y:           Banding detection threshold for luma
    :param c:           Banding detection threshold for chroma
    :param kwargs:      Arguments passed to mvsfunc.LimitFilter

    :return:            Debanded clip
    """
    try:
        from mvsfunc import LimitFilter
    except ModuleNotFoundError:
        raise ModuleNotFoundError("f3kbilateral: missing dependency 'mvsfunc'")

    if clip.format is None:
        raise ValueError("f3kbilateral: 'Variable-format clips not supported'")

    bits = clip.format.bits_per_sample

    if bits < 16:
        clip = depth(clip, 16)

    lf_args: Any = dict(thr=0.6, elast=3.0, thrc=None)
    lf_args.update(kwargs)

    r1 = round(radius * 4 / 3)
    r2 = round(radius * 2 / 3)
    r3 = round(radius / 3)
    y1, y2, y3 = y // 2, y, y
    c1, c2, c3 = max(1, c // 2), c, c

    flt1 = dumb3kdb(clip, radius=r1, threshold=[y1, c1, c1])
    flt2 = dumb3kdb(flt1, radius=r2, threshold=[y2, c2, c2])
    flt3 = dumb3kdb(flt2, radius=r3, threshold=[y3, c3, c3])

    lf = LimitFilter(flt3, flt2, ref=clip, **lf_args)
    return depth(lf, bits) if bits < 16 else lf


def f3kpf(clip: vs.VideoNode,
          radius: int = 16,
          threshold: Union[int, List[int]] = 30,
          **kwargs: Any) -> vs.VideoNode:
    """
    f3kdb with a simple prefilter by mawen1250 - https://www.nmm-hd.org/newbbs/viewtopic.php?f=7&t=1495#p12163.

    Since the prefilter is a straight gaussian+average blur, f3kdb's effect becomes very strong, very fast.
    Functions more or less like gradfun3 without the detail mask.

    :param clip:        Input clip
    :param radius:      Banding detection range
    :param threshold:   Banding detection thresholds for multiple planes
    :param kwargs:      Arguments passed to mvsfunc.LimitFilter

    :return:            Debanded clip
    """
    try:
        from mvsfunc import LimitFilter
    except ModuleNotFoundError:
        raise ModuleNotFoundError("f3kpf: missing dependency 'mvsfunc'")

    if clip.format is None:
        raise ValueError("f3kpf: 'Variable-format clips not supported'")

    lf_args: Any = dict(thr=0.3, elast=2.5, thrc=None)
    lf_args.update(kwargs)

    bits = clip.format.bits_per_sample

    if bits != 32:
        clip = depth(clip, 32)

    blur32 = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Convolution([1] * 9, planes=0)
    blur16 = depth(blur32, 16)

    diff = core.std.MakeDiff(clip, blur32)
    f3k = dumb3kdb(blur16, radius, threshold)
    f3k = LimitFilter(f3k, blur16, **lf_args)
    f3k = depth(f3k, 32)

    out = core.std.MergeDiff(f3k, diff)
    return depth(out, bits)


def lfdeband(clip: vs.VideoNode) -> vs.VideoNode:
    """
    A simple debander ported from AviSynth.

    :param clip:        Input clip

    :return:            Debanded clip
    """
    if clip.format is None:
        raise ValueError("lfdeband: 'Variable-format clips not supported'")

    bits = clip.format.bits_per_sample
    wss = 1 << clip.format.subsampling_w
    hss = 1 << clip.format.subsampling_h
    w, h = clip.width, clip.height
    dw, dh = round(w / 2), round(h / 2)

    clip = depth(clip, 32)
    dsc = core.resize.Spline64(clip, dw-dw % wss, dh-dh % hss)

    d3kdb_down = depth(dsc, 16)
    d3kdb = dumb3kdb(d3kdb_down, radius=30, threshold=80, grain=0)
    d3kdb_up = depth(d3kdb, 32)

    ddif = core.std.MakeDiff(d3kdb_up, dsc)

    dif = core.resize.Spline64(ddif, w, h)
    out = core.std.MergeDiff(clip, dif)
    return depth(out, bits)


def dither_bilateral(clip: vs.VideoNode, ref: Optional[vs.VideoNode] = None,
                     radius: Optional[float] = None,
                     thr: float = 0.35, wmin: float = 1.0,
                     subspl: float = 0) -> vs.VideoNode:
    """
    Dither's Gradfun3 mode 2, for Vapoursynth
    Not much different from using core.avsw.Eval with Dither_bilateral_multistage,
    just with normal value rounding and without Dither_limit_dif16

    If you want the rounding to be exactly the same,
    replace the applicable lines from here: https://pastebin.com/raw/gZKHCFkd

    Default setting of 'radius' changes to reflect the resolution

    Radius auto-adjust: 480p  ->  9
                        720p  -> 12
                        810p  -> 13
                        900p  -> 14
                        1080p -> 16
                        2K    -> 20
                        4K    -> 32

    Basic usage: flt = db.Dither_bilateral(clip, thr=1/3)
                    flt = mvf.LimitFilter(flt, clip, thr=1/3)
    """
    if clip.format is None:
        raise ValueError("Dither_bilateral: 'Variable-format clips not supported'")

    if radius is None:
        radius = max((clip.width - 1280) / 160 + 12, (clip.height - 720) / 90 + 12)

    planes = list(range(clip.format.num_planes))
    y, u, v = [3 if x in planes else 1 for x in range(3)]

    thr_1 = round(max(thr * 4.5, 1.25), 1)
    thr_2 = round(max(thr * 9, 5), 1)
    subspl_2 = subspl if subspl in (0, 1) else subspl / 2
    r4 = round(max(radius * 4 / 3, 4))
    r2 = round(max(radius * 2 / 3, 3))
    r1 = round(max(radius / 3, 2))

    clips = [clip.fmtc.nativetostack16()]
    clip_names = ["c"]
    ref_t = "c"
    if ref is not None:
        ref_s = ref.fmtc.nativetostack16()
        clips += [ref_s]
        clip_names += ["ref"]
        ref_t = "ref"

    avs_stuff = "c.Dither_bilateral16(radius={r4}, thr={thr_1}, flat=0.75, wmin={wmin}, ref={ref_t}, subspl={subspl},   y={y}, u={u}, v={v})" # noqa
    avs_stuff += ".Dither_bilateral16(radius={r2}, thr={thr_2}, flat=0.25, wmin={wmin}, ref={ref_t}, subspl={subspl_2}, y={y}, u={u}, v={v})" # noqa
    avs_stuff += ".Dither_bilateral16(radius={r1}, thr={thr_2}, flat=0.50, wmin={wmin}, ref={ref_t}, subspl={subspl_2}, y={y}, u={u}, v={v})" # noqa
    avs_stuff = avs_stuff.format(ref_t=ref_t, r4=r4, r2=r2, r1=r1, thr_1=thr_1, thr_2=thr_2, wmin=wmin, subspl=subspl, subspl_2=subspl_2, y=y, u=u, v=v) # noqa

    return core.avsw.Eval(avs_stuff, clips=clips, clip_names=clip_names).fmtc.stack16tonative()
# TO-DO: Fix this function. Current error: Avisynth 32-bit proxy: command failed

# TO-DO: Port all the other functions in ../legacy/debandshit.py
