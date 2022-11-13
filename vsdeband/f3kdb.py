from __future__ import annotations

from typing import Any

from vstools import CustomValueError, check_variable, core, inject_self, vs, normalize_seq, CustomIntEnum, clamp_arr

from .abstract import Debander, Grainer

__all__ = [
    'F3kdbPlugin', 'SampleMode',

    'F3kdb'
]


class SampleMode(CustomIntEnum):
    COLUMN = 1
    SQUARE = 2
    ROW = 3
    COL_ROW_MEAN = 4

    @property
    def step(self) -> int:
        return 16 if self is SampleMode.SQUARE else 32


class F3kdbPlugin(CustomIntEnum):
    OLD = 0
    NEO = 1
    NEO_NEW = 2

    @property
    def is_neo(self) -> bool:
        return self is not F3kdbPlugin.OLD

    @property
    def thr_peak(self) -> int:
        return 511 >> (3 if self.plugin is F3kdbPlugin.NEO_NEW else 0)

    @property
    def namespace(self) -> vs.Plugin:
        return core.neo_f3kdb.Deband if self.is_neo else core.f3kdb.Deband

    @inject_self.with_args(None)
    def Deband(self, clip: vs.VideoNode, y: int = 0, cb: int = 0, cr: int = 0, **kwargs: Any) -> vs.VideoNode:
        kwargs |= dict(keep_tv_range=True, output_depth=16)

        if self is F3kdbPlugin.NEO_NEW:
            kwargs |= dict(y2=y >> 3, cb2=cb >> 3, cr2=cr >> 3)
        else:
            kwargs |= dict(y=y, cb=cb, cr=cr)

        return self.namespace.Deband(clip, **kwargs)

    @classmethod
    def from_param(cls, use_neo: bool | None) -> F3kdbPlugin:
        if use_neo is None:
            use_neo = hasattr(core, 'neo_f3kdb')

        if not use_neo:
            return F3kdbPlugin.OLD

        if 'y2' in core.neo_f3kdb.Deband.__signature__.parameters:
            return F3kdbPlugin.NEO_NEW

        return F3kdbPlugin.NEO

    def check_sample_mode(self, sample_mode: SampleMode) -> SampleMode:
        if sample_mode > SampleMode.SQUARE and not self.is_neo:
            raise CustomValueError(
                'Normal fk3db doesn\'t support SampleMode.ROW or SampleMode.COL_ROW_MEAN',
                self.__class__.deband
            )

        return sample_mode


class F3kdb(Debander, Grainer):
    """f3kdb object."""

    radius: int
    thy: int
    thcb: int
    thcr: int
    gry: int
    grc: int
    sample_mode: SampleMode
    seed: int | None = None
    dynamic_grain: int | None = None
    dither_algo: int | None = None
    blur_first: int | None = None

    _step: int

    def __init__(
        self,
        radius: int = 16,
        threshold: int | list[int] = 30, grain: int | list[int] = 0,
        sample_mode: SampleMode = SampleMode.SQUARE, use_neo: bool = False, **kwargs: Any
    ) -> None:
        """
        Handle debanding operations onto a clip using a set of configured parameters.

        Both f3kdb and neo_f3kdb actually change their strength at 1 + 16 * n for SampleMode.SQUARE
        and 1 + 32 * n for SampleMode.COLUMN, SampleMode.ROW or SampleMode.COL_ROW_MEAN.
        This function is aiming to average n and n + 1 strength for a better accuracy.
        Original function written by Z4ST1N, modified by Vardë.

        :param radius:          Banding detection range
        :param threshold:       Banding detection threshold(s) for planes
        :param grain:           Specifies amount of grains added in the last debanding stage
        :param sample_mode:     Valid modes are:
                                * SampleMode.COLUMN or 1: Take 2 pixels as reference pixel
                                  Reference pixels are in the same column of current pixel
                                * SampleMode.SQUARE or 2: Take 4 pixels as reference pixel
                                  Reference pixels are in the square around current pixel
                                * SampleMode.ROW or 3: Take 2 pixels as reference pixel
                                  Reference pixels are in the same row of current pixel
                                  Only `neo_f3kdb.Deband` supports it
                                * SampleMode.COL_ROW_MEAN or 4: Arithmetic mean of 1 and 3
                                  Reference points are randomly picked within the range
                                  Only `neo_f3kdb.Deband` supports it
        :param use_neo:         Use `neo_f3kdb.Deband`
        :param kwargs:          Arguments passed to f3kdb.Deband
                                Default are `keep_tv_range=True, output_depth=16`
                                Read the f3kdb's documentation for more information about them:
                                https://f3kdb.readthedocs.io/en/latest/usage.html#parameters
        """
        self.radius = radius

        self.plugin = F3kdbPlugin.from_param(use_neo)
        self.sample_mode = self.plugin.check_sample_mode(sample_mode)

        self.thrs = tuple[int, int, int](clamp_arr(normalize_seq(threshold), 1, self.plugin.thr_peak))
        self.gry, self.grc = normalize_seq(grain, 2)

    @inject_self
    def deband(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Main deband function

        :param clip:            Source clip
        :return:                Debanded clip
        """

        assert check_variable(clip, self.__class__.deband)

        kwargs = dict[str, Any](
            range=self.radius, grainy=self.gry, grainc=self.grc, sample_mode=self.sample_mode
        ) | self.f3kdb_args

        if self.new_neo or all(x % self._step == 1 for x in self.thrs):
            return self.plugin.Deband(clip, self.thy, self.thcb, self.thcr, **kwargs)

        lows = tuple[int, int, int]((th - 1) // self._step * self._step + 1 for th in self.thrs)
        highs = tuple[int, int, int](lo + self._step for lo in lows)

        lo_clip = self.plugin.Deband(clip, *lows, **kwargs)
        hi_clip = self.plugin.Deband(clip, *highs, **kwargs)

        if clip.format.color_family == vs.GRAY:
            weight = [(self.thrs[0] - lows[0]) / self._step]
        else:
            weight = [(thr - low) / self._step for thr, low in zip(self.thrs, lows)]

        return lo_clip.std.Merge(hi_clip, weight)

    @inject_self
    def grain(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Convenience function that set thresholds to 1 (basically it doesn't deband)

        :param clip:            Source clip
        :return:                Grained clip
        """
        self.thy, self.thcr, self.thcb = (1, ) * 3
        return self.deband(clip)
