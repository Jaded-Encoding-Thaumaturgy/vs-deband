from __future__ import annotations

from vsexprtools import ExprOp, aka_expr_available, combine, norm_expr
from vstools import (
    CustomRuntimeError, core, depth, disallow_variable_format, expect_bits, get_peak_value, get_y, iterate, vs
)
from vsrgtools import gauss_blur, removegrain, RemoveGrainModeT, RemoveGrainMode

__all__ = [
    'deband_detail_mask',

    'adg_mask'
]


def deband_detail_mask(
    clip: vs.VideoNode,
    sigma: float = 1.0, rxsigma: list[int] = [50, 200, 350],
    pf_sigma: float | None = 1.0, brz: tuple[int, int] = (2500, 4500),
    rg_mode: RemoveGrainModeT = RemoveGrainMode.MINMAX_MEDIAN_OPP
) -> vs.VideoNode:
    clip_y, bits = expect_bits(get_y(clip), 16)

    pf = gauss_blur(clip_y, pf_sigma) if pf_sigma else clip_y
    ret = pf.retinex.MSRCP(rxsigma, None, 0.005)

    blur_ret = gauss_blur(ret, sigma)
    blur_ret_diff = combine([blur_ret, ret], ExprOp.SUB).std.Deflate()
    blur_ret_brz = iterate(blur_ret_diff, core.std.Inflate, 4)
    blur_ret_brz = blur_ret_brz.std.Binarize(brz[0]).morpho.Close(size=8)

    prewitt_mask = clip_y.std.Prewitt().std.Binarize(brz[1]).std.Deflate().std.Inflate()
    prewitt_brz = prewitt_mask.std.Binarize(brz[1]).morpho.Close(size=4)

    merged = combine([blur_ret_brz, prewitt_brz], ExprOp.ADD)
    rm_grain = removegrain(merged, rg_mode)

    return depth(rm_grain, bits)


@disallow_variable_format
def adg_mask(clip: vs.VideoNode, luma_scaling: float = 8.0, relative: bool = False) -> vs.VideoNode:
    """
    Re-reimplementation of kageru's adaptive_grain mask as expr, with added `relative` param.
    Really just the same speed and everything (though, *slightly* faster in float32),
    it's just a plugin dep less in your *func if already make use of Akarin's plugin.
    For a description of the math and the general idea, see his article. \n
    https://blog.kageru.moe/legacy/adaptivegrain.html \n
    https://git.kageru.moe/kageru/adaptivegrain
    """

    y = get_y(clip).std.PlaneStats(prop='P')

    if not aka_expr_available:
        if relative:
            raise CustomRuntimeError('You don\'t have akarin plugin, you can\'t use this function!', 'relative=True')

        return y.adg.Mask(luma_scaling)

    assert y.format

    peak = get_peak_value(y)

    is_integer = y.format.sample_type == vs.INTEGER

    x_string, aft_int = (f'x {peak} / ', f' {peak} * 0.5 +') if is_integer else ('x ', '')

    if relative:
        x_string += 'Y! Y@ 0.5 < x.PMin 0 max 0.5 / log Y@ * x.PMax 1.0 min 0.5 / log Y@ * ? '

    x_string += '0 0.999 clamp X!'

    return norm_expr(
        y, f'{x_string} 1 X@ X@ X@ X@ X@ '
        '18.188 * 45.47 - * 36.624 + * 9.466 - * 1.124 + * - '
        f'x.PAverage 2 pow {luma_scaling} * pow {aft_int}'
    )
