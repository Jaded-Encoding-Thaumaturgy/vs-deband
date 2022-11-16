from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from vstools import (
    CustomIntEnum, CustomValueError, DataType, FuncExceptT, VSFunction, check_variable, clamp_arr, core, fallback,
    inject_self, normalize_seq, vs
)

from .abstract import Debander

__all__ = [
    'F3kdbPlugin', 'SampleMode',

    'F3kdb'
]


class SampleMode(CustomIntEnum):
    """Sampling mode for F3kdb."""

    COLUMN = 1
    """Sample by pixel columns. @@@"""

    SQUARE = 2
    """Sample by a 3x3 square around the centre pixel. @@@"""

    ROW = 3
    """Sample by pixel rows. @@@"""

    COL_ROW_MEAN = 4
    """Samples using the mean of COLUMN and ROW."""

    @property
    def step(self) -> int:
        """Steps to take for the sampling."""

        return 16 if self is SampleMode.SQUARE else 32


class F3kdbPlugin(CustomIntEnum):
    """Verify F3kdb settings, namespaces, peaks, etc."""

    OLD = 0
    """
    Use the original flash3kyuu_deband (F3kdb) plugin.
    `Plugin <https://f3kdb.readthedocs.io/en/latest/>_
    """

    NEO = 1
    """
    Use the HomeOfAviSynthPlusEvolution F3kdb implementation.
    `Plugin <https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb>`_
    """

    NEO_NEW = 2
    """Use more accurate thresholding and settings for NEO. Requires >R8."""

    @property
    def is_neo(self) -> bool:
        """Verify whether the plugin is a neo F3kdb implementation."""
        return self is not F3kdbPlugin.OLD

    @property
    def thr_peak(self) -> int:
        """Return peak allowed by F3kdb plugin."""
        return 511 >> (3 if self is F3kdbPlugin.NEO_NEW else 0)

    @property
    def namespace(self) -> vs.Plugin:
        """Return namespace of the used F3kdb plugin."""
        return core.neo_f3kdb if self.is_neo else core.f3kdb

    @inject_self.with_args(None)
    def Deband(
        self, clip: vs.VideoNode, y: int = 0, cb: int = 0, cr: int = 0, *,
        range: int | None = None, grainy: int | None = None, grainc: int | None = None,
        keep_tv_range: bool = True, output_depth: int = 16,
        seed: int | None = None, sample_mode: int | None = None,
        dynamic_grain: int | None = None, dither_algo: int | None = None,
        preset: DataType | None = None, blur_first: int | None = None,
        random_algo_ref: int | None = None, random_algo_grain: int | None = None,
        random_param_ref: float | None = None, random_param_grain: float | None = None
    ) -> vs.VideoNode:
        """
        @@@
        Perform F3kdb debanding, scaling thresholds depending on the F3kdb implementation used.

        For a full list of parameters and usage, see :py:func:`vsdeband.f3kdb.F3kdb.deband`.
        """
        kwargs = dict(
            range=range, grainy=grainy, grainc=grainc, sample_mode=sample_mode, seed=seed,
            blur_first=blur_first, dynamic_grain=dynamic_grain, random_param_grain=random_param_grain,
            keep_tv_range=keep_tv_range, output_depth=output_depth, random_algo_ref=random_algo_ref,
            random_algo_grain=random_algo_grain, random_param_ref=random_param_ref, preset=preset,
            dither_algo=dither_algo
        )

        if self is F3kdbPlugin.NEO_NEW:
            kwargs |= dict(y2=max(1, y >> 3), cb2=max(1, cb >> 3), cr2=max(1, cr >> 3))
        else:
            kwargs |= dict(y=y, cb=cb, cr=cr)

        return cast(VSFunction, self.namespace.Deband)(clip, **kwargs)

    @classmethod
    def from_param(cls, use_neo: bool | None) -> F3kdbPlugin:  # type: ignore[override]
        """Determine what F3kdb implementation gets used based on params passed."""
        if use_neo is None:
            use_neo = hasattr(core, 'neo_f3kdb')

        if not use_neo:
            return F3kdbPlugin.OLD

        if 'y2' in core.neo_f3kdb.Deband.__signature__.parameters:  # type: ignore[attr-defined]
            return F3kdbPlugin.NEO_NEW

        return F3kdbPlugin.NEO

    def check_sample_mode(self, sample_mode: SampleMode, func: FuncExceptT | None = None) -> SampleMode:
        """Check whether a given sample mode can be used with the F3kdb implementation used."""
        if sample_mode > SampleMode.SQUARE and not self.is_neo:
            raise CustomValueError(
                'Regular fk3db doesn\'t support SampleMode.ROW or SampleMode.COL_ROW_MEAN',
                func or self.__class__.check_sample_mode
            )

        return sample_mode


@dataclass
class F3kdb(Debander):
    """Debander wrapper around the f3kdb plugin."""

    radius: int | None = None
    thr: int | tuple[int, int, int] | None = None
    grains: int | list[int] | None = None

    sample_mode: SampleMode | None = None

    seed: int | None = None
    dynamic_grain: int | None = None
    dither_algo: int | None = None

    blur_first: bool | None = None

    use_neo: bool | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.plugin = F3kdbPlugin.from_param(self.use_neo)

    @inject_self
    def deband(
        self, clip: vs.VideoNode,
        radius: int = 16,
        thr: int | tuple[int, int, int] = 30,
        grains: int | list[int] = 0,
        sample_mode: SampleMode = SampleMode.SQUARE,
        seed: int = None,
        dynamic_grain: int = None,
        dither_algo: int = None,
        blur_first: bool | None = None
    ) -> vs.VideoNode:  # type: ignore[override]
        """
        Main deband function

        Handle debanding operations onto a clip using a set of configured parameters.

        Before neo_f3kdb r7, both f3kdb and neo_f3kdb change their strength
        at 1 + 16 * n for SampleMode.SQUARE and 1 + 32 * n for SampleMode.COLUMN,
        SampleMode.ROW or SampleMode.COL_ROW_MEAN.

        This function aimed to average n and n + 1 strength for better debanding accuracy.

        :param radius:          Banding detection range.
        :param thr:             Banding detection threshold(s) per plane.
                                If you pass an int, the threshold will be applied to every plane.
                                If you pass a tuple, different strengths will be applied per plane.
                                If you pass two ints in a tuple, the last value will be copied to the third plane.
        :param grains:          Specifies amount of grain to add during the final graining stage.
        :param sample_mode:     Sampling mode for debanding.
                                * SampleMode.COLUMN: Take 2 pixels as reference pixel.
                                  Reference pixels are in the same column of current pixel.
                                * SampleMode.SQUARE: Take 4 pixels as reference pixel.
                                  Reference pixels are in the square around current pixel.
                                * SampleMode.ROW: Take 2 pixels as reference pixel.
                                  Reference pixels are in the same row of current pixel.
                                  Only `neo_f3kdb` supports it.
                                * SampleMode.COL_ROW_MEAN: Arithmetic mean of COLUMN and ROW.
                                  Reference points are randomly picked within the range.
                                  Only `neo_f3kdb` supports it.
        :param seed:            Seed used for graining. Currently unused.
        :param dynamic_grain:   If True, apply dynamic grain during the graining stage.
                                Else, static grain is applied. Currently unused.
        :param dither_algo:     Dither algorithm to use for the graining stage. Currently unused.
        :param blur_first:      Whether to blur prior to debanding. Currently unused. @@@

        :return:                Debanded clip.
        """

        assert check_variable(clip, self.__class__.deband)

        radius = fallback(self.radius, radius)

        sample_mode = self.plugin.check_sample_mode(fallback(self.sample_mode, sample_mode), self.__class__.deband)

        thrs = cast(tuple[int, int, int], tuple(clamp_arr(normalize_seq(thr), 1, self.plugin.thr_peak)))
        gry, grc = normalize_seq(fallback(self.grains, grains), 2)

        step = sample_mode.step
        kwargs = dict[str, Any](range=radius, grainy=gry, grainc=grc, sample_mode=sample_mode)

        if self.plugin is F3kdbPlugin.NEO_NEW or all(x % sample_mode.step == 1 for x in thrs):
            return self.plugin.Deband(clip, *thrs, **kwargs)

        lows = cast(tuple[int, int, int], tuple(((th - 1) // step * step + 1 for th in thrs)))
        highs = cast(tuple[int, int, int], tuple((lo + step for lo in lows)))

        lo_clip = self.plugin.Deband(clip, *lows, **kwargs)
        hi_clip = self.plugin.Deband(clip, *highs, **kwargs)

        if clip.format.color_family == vs.GRAY:
            weight = [(thrs[0] - lows[0]) / step]
        else:
            weight = [(thr - low) / step for thr, low in zip(thrs, lows)]

        return lo_clip.std.Merge(hi_clip, weight)

    @inject_self
    def grain(
        self, clip: vs.VideoNode, strength: int | list[int] = 4, radius: int = 16,
        sample_mode: SampleMode = SampleMode.SQUARE
    ) -> vs.VideoNode:  # type: ignore[override]
        """
        Add f3kdb grain to a clip.
        This necessitates very weak debanding is applied to the clip.

        :param clip:            Source clip
        :param strength:        Specifies amount of grain to add during graining.
        :param radius:          Radius for F3kdb.
        :param sample_mode:     Sampling mode for debanding.
                                See :py:func:`vsdeband.f3kdb.F3kdb.deband` for more information.

        :return:                Grained clip.
        """

        radius = fallback(self.radius, radius)

        sample_mode = self.plugin.check_sample_mode(fallback(self.sample_mode, sample_mode), self.__class__.deband)

        gry, grc = normalize_seq(fallback(self.grains, strength), 2)

        return self.plugin.Deband(clip, 1, 1, 1, grainy=gry, grainc=grc, range=radius, sample_mode=sample_mode)
