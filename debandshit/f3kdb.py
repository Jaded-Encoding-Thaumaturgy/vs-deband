"""Docstring"""

from typing import Any, Dict, List, Union

import vapoursynth as vs


core = vs.core


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
                 sample_mode: int = 2, use_neo: bool = False, **kwargs: Any) -> None:
        """ Handle debanding operations onto a clip using a set of configured parameters.

        Args:
            radius (int, optional):
                Banding detection range. Defaults to 16.

            threshold (Union[int, List[int]], optional):
                Banding detection threshold(s) for planes.
                If difference between current pixel and reference pixel is less than threshold,
                it will be considered as banded. Defaults to 30.

            grain (Union[int, List[int]], optional):
                Specifies amount of grains added in the last debanding stage. Defaults to 0.

            sample_mode (int, optional):
                Valid modes are:
                    – 1: Take 2 pixels as reference pixel. Reference pixels are in the same column of current pixel.
                    – 2: Take 4 pixels as reference pixel. Reference pixels are in the square around current pixel.
                    – 3: Take 2 pixels as reference pixel. Reference pixels are in the same row of current pixel.
                    – 4: Arithmetic mean of 1 and 3.
                Reference points are randomly picked within the range. Defaults to 2.

            use_neo (bool, optional):
                Use neo_f3kdb.Deband. Defaults to False.

            kwargs (optional):
                Arguments passed to f3kdb.Deband.

        """
        self.radius = radius

        th_s = [threshold] * 3 if isinstance(threshold, int) else threshold + [threshold[-1]] * (3 - len(threshold))
        self.thy, self.thcb, self.thcr = [max(1, x) for x in th_s]

        self.gry, self.grc = [grain] * 2 if isinstance(grain, int) else grain + [grain[-1]] * (2 - len(grain))

        if sample_mode > 2 and not use_neo:
            raise ValueError('deband: "sample_mode" argument should be less or equal to 2 when "use_neo" is false.')

        self.sample_mode = sample_mode
        self.use_neo = use_neo

        self._step = 16 if sample_mode == 2 else 32

        self.f3kdb_args = dict(keep_tv_range=True, output_depth=16)
        self.f3kdb_args |= kwargs

    def deband(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
            Main deband function.

        Args:
            clip (vs.VideoNode): Source clip.

        Returns:
            vs.VideoNode: Debanded clip.
        """
        if clip.format is None:
            raise ValueError('deband: Variable format not allowed!')

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

        if self.use_neo:
            deband = core.std.ModifyFrame(deband, [deband, clip], selector=self._trf)

        return deband

    def grain(self, clip: vs.VideoNode) -> vs.VideoNode:
        """Convenience function that set thresholds to 1 (basically it doesn't deband)

        Args:
            clip (vs.VideoNode): Source clip.

        Returns:
            vs.VideoNode: Grained clip.
        """
        self.thy, self.thcr, self.thcb = (1, ) * 3
        return self.deband(clip)

    @staticmethod
    def _trf(n: int, f: List[vs.VideoFrame]) -> vs.VideoFrame:  # noqa: PLW0613
        # neo_f3kdb nukes frame props
        (fout := f[0].copy()).props.update(f[1].props)
        return fout

    @staticmethod
    def _pick_f3kdb(neo: bool, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return core.neo_f3kdb.Deband(*args, **kwargs) if neo else core.f3kdb.Deband(*args, **kwargs)


def dumb3kdb(clip: vs.VideoNode, radius: int = 16,
             threshold: Union[int, List[int]] = 30, grain: Union[int, List[int]] = 0,
             sample_mode: int = 2, use_neo: bool = False, **kwargs: Any) -> vs.VideoNode:
    return F3kdb(radius, threshold, grain, sample_mode, use_neo, **kwargs).deband(clip)
