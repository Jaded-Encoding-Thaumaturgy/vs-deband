from __future__ import annotations

from vsexprtools import ExprOp, combine
from vsmasktools import Morpho, retinex
from vsrgtools import RemoveGrainMode, RemoveGrainModeT, gauss_blur, removegrain
from vstools import core, depth, expect_bits, get_y, iterate, vs

__all__ = [
    'deband_detail_mask'
]


def deband_detail_mask(
    clip: vs.VideoNode,
    sigma: float = 1.0, rxsigma: list[int] = [50, 200, 350],
    pf_sigma: float | None = 1.0, brz: tuple[int, int] = (2500, 4500),
    rg_mode: RemoveGrainModeT = RemoveGrainMode.MINMAX_MEDIAN_OPP
) -> vs.VideoNode:
    clip_y, bits = expect_bits(get_y(clip), 16)

    pf = gauss_blur(clip_y, pf_sigma) if pf_sigma else clip_y
    ret = retinex(pf, rxsigma, upper_thr=0.005)

    blur_ret = gauss_blur(ret, sigma)
    blur_ret_diff = combine([blur_ret, ret], ExprOp.SUB).std.Deflate()
    blur_ret_brz = iterate(blur_ret_diff, core.std.Inflate, 4)
    blur_ret_brz = Morpho.closing(blur_ret_brz.std.Binarize(brz[0]), coordinates=8)

    prewitt_mask = clip_y.std.Prewitt().std.Binarize(brz[1]).std.Deflate().std.Inflate()
    prewitt_brz = Morpho.closing(prewitt_mask.std.Binarize(brz[1]), coordinates=4)

    merged = combine([blur_ret_brz, prewitt_brz], ExprOp.ADD)
    rm_grain = removegrain(merged, rg_mode)

    return depth(rm_grain, bits)
