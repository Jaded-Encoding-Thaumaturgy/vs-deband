from __future__ import annotations

from enum import IntEnum
from typing import Any

from vstools import CustomValueError, check_variable, core, inject_self, vs, normalize_seq, VSFunction

from .abstract import Debander, Grainer

__all__ = [
    'SampleMode',

    'F3kdb'
]


class SampleMode(IntEnum):
    COLUMN = 1
    SQUARE = 2
    ROW = 3
    COL_ROW_MEAN = 4


class F3kdb(Debander, Grainer):
    """f3kdb object."""
    radius: int
    thy: int
    thcb: int
    thcr: int
    gry: int
    grc: int
    sample_mode: int
    use_neo: bool
    f3kdb_args: dict[str, Any]

    _step: int

    def __init__(
        self,
        radius: int = 16,
        threshold: int | list[int] = 30, grain: int | list[int] = 0,
        sample_mode: SampleMode = 2, use_neo: bool = False, **kwargs: Any
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

        self.thy, self.thcb, self.thcr = [max(1, x) for x in normalize_seq(threshold)]
        self.gry, self.grc = normalize_seq(grain, 2)

        if sample_mode > 2 and not use_neo:
            raise CustomValueError(
                'Normal fk3db doesn\'t support SampleMode.ROW or SampleMode.COL_ROW_MEAN',
                self.__class__.deband
            )

        self.sample_mode = sample_mode
        self.use_neo = use_neo
        self.new_neo = self.use_neo and ('y2' in core.neo_f3kdb.Deband.__signature__.parameters)

        self._step = 16 if sample_mode == 2 else 32

        self.f3kdb_args = dict(keep_tv_range=True, output_depth=16) | kwargs

    @inject_self
    def deband(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Main deband function

        :param clip:            Source clip
        :return:                Debanded clip
        """

        assert check_variable(clip, self.__class__.deband)

        kwargs = dict[str, Any](
            range=self.radius,
            grainy=self.gry,
            grainc=self.grc,
            sample_mode=self.sample_mode,
        ) | self.f3kdb_args

        if self.new_neo:
            deband = self._f3kdb_plugin(clip, y2=self.thy >> 3, cb2=self.thcb >> 3, cr2=self.thcr >> 3, **kwargs)
        elif self.thy % self._step == 1 and self.thcb % self._step == 1 and self.thcr % self._step == 1:
            deband = self._f3kdb_plugin(clip, y=self.thy, cb=self.thcb, cr=self.thcr, **kwargs)
        else:
            loy, locb, locr = [(th - 1) // self._step * self._step + 1 for th in [self.thy, self.thcb, self.thcr]]
            hiy, hicb, hicr = [lo + self._step for lo in [loy, locb, locr]]

            lo_clip = self._f3kdb_plugin(clip, y=loy, cb=locb, cr=locr, **kwargs)
            hi_clip = self._f3kdb_plugin(clip, y=hiy, cb=hicb, cr=hicr, **kwargs)

            if clip.format.color_family == vs.GRAY:
                weight = [
                    (self.thy - loy) / self._step
                ]
            else:
                weight = [
                    (self.thy - loy) / self._step,
                    (self.thcb - locb) / self._step,
                    (self.thcr - locr) / self._step
                ]

            deband = core.std.Merge(lo_clip, hi_clip, weight)

        return deband

    @inject_self
    def grain(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Convenience function that set thresholds to 1 (basically it doesn't deband)

        :param clip:            Source clip
        :return:                Grained clip
        """
        self.thy, self.thcr, self.thcb = (1, ) * 3
        return self.deband(clip)

    @property
    def _f3kdb_plugin(self) -> VSFunction:
        return core.neo_f3kdb.Deband if self.use_neo else core.f3kdb.Deband
