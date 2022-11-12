from enum import IntEnum
from typing import Any, Dict, List, Literal, Union

from vstools import CustomValueError, VariableFormatError, core, vs

__all__ = ['SAMPLEMODE', 'F3kdb', 'SampleMode']


SAMPLEMODE = Literal[1, 2, 3, 4]


class SampleMode(IntEnum):
    COLUMN = 1
    SQUARE = 2
    ROW = 3
    COL_ROW_MEAN = 4


class F3kdb:
    """f3kdb object."""
    radius: int
    thy: int
    thcb: int
    thcr: int
    gry: int
    grc: int
    sample_mode: int
    use_neo: bool
    f3kdb_args: Dict[str, Any]

    _step: int

    def __init__(self,
                 radius: int = 16,
                 threshold: Union[int, List[int]] = 30, grain: Union[int, List[int]] = 0,
                 sample_mode: Union[SAMPLEMODE, SampleMode] = 2, use_neo: bool = False, **kwargs: Any) -> None:
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

        th_s = [threshold] * 3 if isinstance(threshold, int) else threshold + [threshold[-1]] * (3 - len(threshold))
        self.thy, self.thcb, self.thcr = [max(1, x) for x in th_s]

        self.gry, self.grc = [grain] * 2 if isinstance(grain, int) else grain + [grain[-1]] * (2 - len(grain))

        if sample_mode > 2 and not use_neo:
            raise CustomValueError(
                'Normal fk3db doesn\'t support SampleMode.ROW or SampleMode.COL_ROW_MEAN',
                self.__class__.deband
            )

        self.sample_mode = sample_mode
        self.use_neo = use_neo

        self._step = 16 if sample_mode == 2 else 32

        self.f3kdb_args = dict(keep_tv_range=True, output_depth=16)
        self.f3kdb_args |= kwargs

    def deband(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Main deband function

        :param clip:            Source clip
        :return:                Debanded clip
        """
        if clip.format is None:
            raise VariableFormatError(self.__class__.deband, 'Variable format not allowed!')

        if self.thy % self._step == 1 and self.thcb % self._step == 1 and self.thcr % self._step == 1:
            deband = self._pick_f3kdb(self.use_neo,
                                      clip, self.radius,
                                      self.thy, self.thcb, self.thcr,
                                      self.gry, self.grc,
                                      self.sample_mode, **self.f3kdb_args)
        else:
            loy, locb, locr = [(th - 1) // self._step * self._step + 1 for th in [self.thy, self.thcb, self.thcr]]
            hiy, hicb, hicr = [lo + self._step for lo in [loy, locb, locr]]

            lo_clip = self._pick_f3kdb(self.use_neo,
                                       clip, self.radius,
                                       loy, locb, locr,
                                       self.gry, self.grc,
                                       self.sample_mode, **self.f3kdb_args)
            hi_clip = self._pick_f3kdb(self.use_neo,
                                       clip, self.radius,
                                       hiy, hicb, hicr,
                                       self.gry, self.grc,
                                       self.sample_mode, **self.f3kdb_args)

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

    def grain(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Convenience function that set thresholds to 1 (basically it doesn't deband)

        :param clip:            Source clip
        :return:                Grained clip
        """
        self.thy, self.thcr, self.thcb = (1, ) * 3
        return self.deband(clip)

    @staticmethod
    def _pick_f3kdb(neo: bool, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return core.neo_f3kdb.Deband(*args, **kwargs) if neo else core.f3kdb.Deband(*args, **kwargs)
