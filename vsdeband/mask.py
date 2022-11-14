from __future__ import annotations

from vsexprtools import aka_expr_available, norm_expr
from vstools import disallow_variable_format, get_peak_value, get_y, vs, CustomRuntimeError

__all__ = [
    'adg_mask'
]


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
