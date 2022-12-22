from __future__ import annotations

from typing import Any

from vskernels import Catrom, Kernel, KernelT
from vsmasktools import adg_mask
from vsscale import gamma2linear, linear2gamma
from vstools import (
    CustomOverflowError, DitherType, Matrix, Transfer, check_variable, core, depth, expect_bits, get_video_format,
    inject_self, join, normalize_seq, plane, vs
)

from .abstract import Grainer

__all__ = [
    'AddGrain',
    'AddNoise',
    'ChickenDream'
]


class AddGrain(Grainer):
    config = Grainer.SupportsConfig(True, True, False)

    @inject_self.cached
    def grain(  # type: ignore[override]
        self, clip: vs.VideoNode, strength: float | tuple[float, float], dynamic: bool | int = True, **kwargs: Any
    ) -> vs.VideoNode:
        luma, chroma = normalize_seq(strength, 2)

        return core.grain.Add(clip, luma, chroma, constant=not dynamic, **(self.kwargs | kwargs))


class AddNoise(Grainer):
    config = Grainer.SupportsConfig(True, True, True)

    @inject_self.cached
    def grain(  # type: ignore[override]
        self, clip: vs.VideoNode, strength: float | tuple[float, float], dynamic: bool | int = True, **kwargs: Any
    ) -> vs.VideoNode:
        luma, chroma = normalize_seq(strength, 2)

        return core.noise.Add(clip, luma, chroma, constant=not dynamic, **(self.kwargs | kwargs))


class ChickenDream(Grainer):
    config = Grainer.SupportsConfig(False, True, False)

    @inject_self.cached
    def grain(  # type: ignore[override]
        self, clip: vs.VideoNode, strength: float | tuple[float, float] = 0.35,
        dynamic: bool | int = True, rad: float = 0.025, res: int = 1024,
        luma_scaling: float = 10, seed: int = 42069, draft: bool = True,
        coarsharp: bool | float = 1.0, matrix: Matrix | int | None = None,
        kernel: KernelT = Catrom, **kwargs: Any
    ) -> vs.VideoNode:
        assert check_variable(clip, self.__class__.grain)

        if not dynamic:
            raise NotImplementedError

        if not isinstance(strength, int | float):
            if clip.format.num_planes > 1:
                str_luma, str_chroma = strength

                args = (dynamic, rad, res, luma_scaling, seed, draft, coarsharp, matrix, kernel)

                if str_luma > 0 and str_chroma > 0:
                    return join(
                        self.grain(plane(clip, 0), str_luma, *args, **kwargs),
                        self.grain(plane(clip, 1), str_chroma, *args, **kwargs),
                        self.grain(plane(clip, 2), str_chroma, *args, **kwargs)
                    )
                elif str_luma > 0:
                    return join(
                        self.grain(plane(clip, 0), str_luma, *args, **kwargs),
                        clip
                    )
                elif str_chroma > 0:
                    return join(
                        clip,
                        self.grain(plane(clip, 1), str_chroma, *args, **kwargs),
                        self.grain(plane(clip, 2), str_chroma, *args, **kwargs)
                    )

                return clip
            else:
                strength = strength[0]

        kwargs |= dict(sigma=strength, rad=rad, seed=seed, res=res, draft=draft)

        kernel = Kernel.ensure_obj(kernel)

        if coarsharp is True:
            coarse, sharp = True, 0.0
        elif coarsharp is False:
            coarse, sharp = False, 1.0
        else:
            coarse, sharp = False, coarsharp

        if not 0 <= sharp <= 1.0:
            raise CustomOverflowError('Sharp must be between 0.0 and 1.0 (inclusive)!', self.__class__.grain, sharp)

        sharp = 1.0 - (sharp / 2)

        clip_32, bits = expect_bits(clip, 32, dither_type=DitherType.ERROR_DIFFUSION)

        targ_matrix = Matrix.from_param(matrix, self.__class__.grain) or Matrix.from_video(clip)
        transfer = Transfer.from_matrix(targ_matrix)

        if clip.format.color_family is vs.YUV:
            input_clip = clip_32 if coarse else gamma2linear(clip_32, transfer, 1, True, sharp)
            input_clip = kernel.resample(input_clip, vs.RGBS, matrix_in=targ_matrix)
            input_clip = input_clip.fmtc.transfer('srgb', 'linear') if coarse else input_clip
        else:
            input_clip = clip_32

        out_clip = input_clip.std.Limiter().chkdr.grain(**kwargs)

        if clip.format.color_family is vs.YUV:
            out_clip = out_clip.fmtc.transfer('linear', 'srgb') if coarse else out_clip
            out_clip = Catrom.resample(out_clip, get_video_format(clip_32), targ_matrix)
            out_clip = out_clip if coarse else linear2gamma(out_clip, transfer, 1, True, sharp)

        if luma_scaling:
            out_clip = clip_32.std.MaskedMerge(out_clip, adg_mask(plane(clip_32, 0), luma_scaling))

        return depth(out_clip, bits)
