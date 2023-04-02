from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from typing import Any

from vsdenoise import Prefilter
from vsexprtools import norm_expr
from vskernels import BicubicAuto, Bilinear, Catrom, Kernel, KernelT, Lanczos, Scaler, ScalerT
from vsmasktools import adg_mask
from vsrgtools import BlurMatrix
from vsscale import gamma2linear, linear2gamma
from vstools import (
    CustomIndexError, CustomOverflowError, CustomValueError, InvalidColorFamilyError, KwargsT, Matrix, MatrixT,
    Transfer, VSFunctionNoArgs, check_variable, core, depth, fallback, get_neutral_values, get_peak_value, get_y,
    inject_self, join, mod_x, plane, split, vs
)

from .f3kdb import F3kdb
from .placebo import Placebo

__all__ = [
    'Grainer',

    'AddGrain', 'AddNoise',

    'F3kdbGrain', 'PlaceboGrain',

    'ChickenDream',

    'multi_graining', 'MultiGrainerT'
]


class Grainer(ABC):
    """Abstract graining interface"""

    def __init__(
        self, strength: float | tuple[float, float] = 0.25,
        size: float | tuple[float, float] = (1.0, 1.0), sharp: float | ScalerT = Lanczos,
        dynamic: bool = True, temporal_average: int | tuple[float, int] = (0.0, 1),
        postprocess: VSFunctionNoArgs | None = None, protect_chroma: bool = False,
        luma_scaling: float | None = None, *,
        matrix: MatrixT | None = None, kernel: KernelT = Catrom, neutral_out: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__()

        self.strength = strength
        self.size = size if isinstance(size, tuple) else (size, size)
        self.neutral_out = neutral_out
        self.dynamic = dynamic
        self.postprocess = postprocess
        self.protect_chroma = protect_chroma
        self.luma_scaling = luma_scaling
        self.kwargs = kwargs

        if isinstance(sharp, float):
            self.scaler = BicubicAuto(sharp)
        else:
            self.scaler = Scaler.ensure_obj(sharp, self.__class__)

        if isinstance(temporal_average, tuple):
            self.temporal_average, self.temporal_radius = temporal_average
        else:
            self.temporal_average, self.temporal_radius = temporal_average, 1

        self.matrix = Matrix.from_param(matrix, self.__class__)
        self.kernel = Kernel.ensure_obj(kernel, self.__class__)

    def _is_input_dependent(self, clip: vs.VideoNode, **kwargs: Any) -> bool:
        return False

    def _get_kw(self, kwargs: KwargsT) -> KwargsT:
        return self.kwargs | kwargs

    @abstractmethod
    def _perform_graining(
        self, clip: vs.VideoNode, strength: tuple[float, float], dynamic: bool = True, **kwargs: Any
    ) -> vs.VideoNode:
        ...

    @inject_self
    def grain(
        self, clip: vs.VideoNode, strength: float | tuple[float, float] | None = None,
        dynamic: bool | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        kwargs = self._get_kw(kwargs)

        if strength is None:
            strength = self.strength

        dynamic = fallback(dynamic, self.dynamic)
        strength = strength if isinstance(strength, tuple) else (strength, strength)

        do_taverage = (
            dynamic
            and self.temporal_average > 0 and self.temporal_radius > 0
            and (clip.num_frames > self.temporal_radius * 2)
        )
        do_protect_chroma = self.protect_chroma and strength[1] > 0.0 and clip.format.color_family is vs.YUV

        def _wrap_implementation(clip: vs.VideoNode, neutral_out: bool) -> vs.VideoNode:
            input_dep = self._is_input_dependent(clip, **kwargs)

            if neutral_out and not input_dep:
                lenght = clip.num_frames + ((self.temporal_radius * 2) if do_taverage else 0)
                base_clip = clip.std.BlankClip(length=lenght, color=get_neutral_values(clip))
            elif do_taverage:
                base_clip = (clip[0] * self.temporal_radius) + clip + (clip[-1] * self.temporal_radius)
            else:
                base_clip = clip

            def _try_grain(src: vs.VideoNode, stre: tuple[float, float] = strength, **args: Any) -> vs.VideoNode:
                args = kwargs | dict(strength=stre, dynamic=dynamic) | args
                try:
                    grained = self._perform_graining(src, **args)
                except NotImplementedError as e:
                    reason, *_ = map(str, e.args)

                    if reason == 'dynamic-only':
                        grained = _try_grain(src[src.num_frames // 2], dynamic=True)
                    elif reason.startswith('bad-depth'):
                        good_depth = int(reason.split('-')[-1])
                        grained = _try_grain(depth(src, good_depth))
                        grained = depth(grained, src)
                    elif reason == 'single-plane':
                        str_luma, str_chroma = strength

                        if str_luma > 0 and str_chroma > 0:
                            return join(
                                _try_grain(plane(src, 0), str_luma, *args, **kwargs),
                                _try_grain(plane(src, 1), str_chroma, *args, **kwargs),
                                _try_grain(plane(src, 2), str_chroma, *args, **kwargs)
                            )
                        elif str_luma > 0:
                            return join(
                                _try_grain(plane(src, 0), str_luma, *args, **kwargs),
                                src
                            )
                        elif str_chroma > 0:
                            return join(
                                src,
                                _try_grain(plane(src, 1), str_chroma, *args, **kwargs),
                                _try_grain(plane(src, 2), str_chroma, *args, **kwargs)
                            )

                        return src
                    else:
                        raise e

                return grained

            grained = _try_grain(base_clip)

            if clip.num_frames != grained.num_frames:
                if grained.num_frames == 1:
                    grained = grained.std.Loop(clip.num_frames)
                elif grained.num_frames > clip.num_frames:
                    grained = grained[:clip.num_frames]
                else:
                    grained = grained + grained[-1].std.Loop(clip.num_frames - grained.num_frames)

            if input_dep and neutral_out:
                grained = clip.std.MakeDiff(grained)

            return grained

        if (
            self.size == (1.0, 1.0) and not do_taverage and not self.postprocess
            and not do_protect_chroma and self.luma_scaling is None
        ):
            return _wrap_implementation(clip, self.neutral_out)

        (sizex, sizey), mod = self.size, max(clip.format.subsampling_w, clip.format.subsampling_h) << 1
        sx, sy = mod_x(clip.width / sizex, mod), mod_x(clip.height / sizey, mod)

        if (sx, sy) != (clip.width, clip.height):
            sxa, sya = mod_x((clip.width + sx) / 2, mod), mod_x((clip.height + sy) / 2, mod)

            grained = _wrap_implementation(self.scaler.scale(clip, sx, sy), True)

            # If the scale is too big, we need to scale it in two passes, else the window
            # will be too big and the grain will be dampened down too much
            if max(self.size) > 1.5:
                grained = self.scaler.scale(grained, sxa, sya)

            grained = self.scaler.scale(grained, clip.width, clip.height)
        else:
            grained = _wrap_implementation(clip, True)

        if do_taverage:
            average = grained.std.AverageFrames(BlurMatrix.gauss_from_radius(self.temporal_radius))
            grained = grained.std.Merge(average, self.temporal_average)
            grained = grained[self.temporal_radius:-self.temporal_radius]

        if self.postprocess:
            grained = self.postprocess(grained)

        neutral = get_neutral_values(clip)

        if not self.neutral_out:
            if clip.format.sample_type is vs.FLOAT:
                grained = norm_expr([clip, grained], ('x y 0.5 - +', 'x y +'))
            else:
                grained = clip.std.MergeDiff(grained)

        merge_clip = grained.std.BlankClip(color=get_neutral_values(grained)) if self.neutral_out else clip

        if do_protect_chroma:
            neutral_mask = Lanczos.resample(clip, clip.format.replace(subsampling_h=0, subsampling_w=0))

            neutral_mask = norm_expr(
                split(neutral_mask), f'y {neutral[1]} = z {neutral[1]} = and {get_peak_value(clip, chroma=True)} 0 ?',
                planes=[1, 2]
            )

            grained = grained.std.MaskedMerge(merge_clip, neutral_mask, [1, 2])

        if self.luma_scaling is not None:
            mask = adg_mask(clip, self.luma_scaling, func=self.grain)

            grained = merge_clip.std.MaskedMerge(grained, mask)

        return grained


class AddGrain(Grainer):
    """Built-in grain.Add plugin. https://github.com/HomeOfVapourSynthEvolution/VapourSynth-AddGrain"""

    def _perform_graining(
        self, clip: vs.VideoNode, strength: tuple[float, float], dynamic: bool = True, **kwargs: Any
    ) -> vs.VideoNode:
        return core.grain.Add(clip, *strength, constant=not dynamic, **kwargs)


class AddNoiseBase(Grainer):
    def _get_kw(self, kwargs: KwargsT) -> KwargsT:
        kwargs = super()._get_kw(kwargs)

        if hasattr(self, '_noise_type'):
            kwargs.update(type=self._noise_type)

        return kwargs

    def _is_input_dependent(self, clip: vs.VideoNode, **kwargs: Any) -> bool:
        return kwargs.get('type', None) == 4

    def _perform_graining(
        self, clip: vs.VideoNode, strength: tuple[float, float], dynamic: bool = True, **kwargs: Any
    ) -> vs.VideoNode:
        is_poisson = kwargs.get('type', None) == 4

        if is_poisson:
            if not dynamic:
                raise NotImplementedError('dynamic-only')

            if clip.format.bits_per_sample > 16:
                raise NotImplementedError('bad-depth-16')

            if min(*strength) <= 0.0 or max(*strength) >= 1.0:
                raise ValueError('Poisson noise strength must be between 0.0 and 1.0 (not inclusive)!')

            scale = ((1 << (clip.format.bits_per_sample - 8)) - 1) if clip.format.bits_per_sample > 8 else 255

            strength = ((1.0 - stre) * scale for stre in strength)

        return core.noise.Add(clip, *strength, constant=not dynamic, **kwargs)


class AddNoise(AddNoiseBase):
    """Built-in noise.Add plugin. https://github.com/wwww-wwww/vs-noise"""

    class GAUSS(AddNoiseBase):
        _noise_type = 0

    class PERLIN(AddNoiseBase):
        _noise_type = 1

    class SIMPLEX(AddNoiseBase):
        _noise_type = 2

    class FBM_SIMPLEX(AddNoiseBase):
        _noise_type = 3

    class POISSON(AddNoiseBase):
        _noise_type = 4


class F3kdbGrain(Grainer):
    """Built-in f3kdb.Deband plugin. https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb"""

    def _perform_graining(
        self, clip: vs.VideoNode, strength: tuple[float, float], dynamic: bool = True, **kwargs: Any
    ) -> vs.VideoNode:
        return F3kdb.deband(clip, 8, 1, list(strength), dynamic_grain=dynamic, **kwargs)


class PlaceboGrain(Grainer):
    """placebo.Deband plugin. https://github.com/Lypheo/vs-placebo"""

    def _perform_graining(
        self, clip: vs.VideoNode, strength: tuple[float, float], dynamic: bool = True, **kwargs: Any
    ) -> vs.VideoNode:
        if not dynamic:
            raise NotImplementedError('dynamic-only')
        return Placebo.deband(clip, 8, 1, 1, list(strength), **kwargs)


class ChickenDream(Grainer):
    """chkdr.grain plugin. https://github.com/EleonoreMizo/chickendream"""

    def _is_input_dependent(self, clip: vs.VideoNode, **kwargs: Any) -> bool:
        return True

    def __init__(
        self, strength: float | tuple[float, float] = 0.00,
        size: float | tuple[float, float] = (1.0, 1.0), sharp: float | ScalerT = Lanczos,
        dynamic: bool = True, temporal_average: int | tuple[float, int] = (0.0, 1),
        postprocess: VSFunctionNoArgs | None = None, protect_chroma: bool = False,
        luma_scaling: float | None = None, *,
        rad: float = 0.025, res: int = 1024, draft: bool = True, gamma: float = 1.0,
        matrix: MatrixT | None = None, kernel: KernelT = Catrom, neutral_out: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(
            strength, size, sharp, dynamic, temporal_average, postprocess, protect_chroma, luma_scaling,
            matrix=matrix, kernel=kernel, neutral_out=neutral_out, rad=rad, res=res, draft=draft, **kwargs
        )

        if not 0.0 <= gamma <= 1.0:
            raise CustomOverflowError('Gamma must be between 0.0 and 1.0 (inclusive)!', self.__class__, gamma)

        self.gamma = gamma

    def _perform_graining(
        self, clip: vs.VideoNode, strength: tuple[float, float] = 0.35, dynamic: bool | int = True, **kwargs: Any
    ) -> vs.VideoNode:
        assert check_variable(clip, self.__class__.grain)

        if clip.format.bits_per_sample != 32:
            raise NotImplementedError('bad-depth-32')

        if not dynamic:
            raise NotImplementedError('dynamic-only')

        if strength[0] != strength[1] and clip.format.num_planes > 1:
            raise NotImplementedError('single-plane')

        kwargs |= dict(sigma=strength[0])

        gamma = 1.0 - (self.gamma / 2)

        targ_matrix = self.matrix or Matrix.from_video(clip)
        transfer = Transfer.from_matrix(targ_matrix)

        if clip.format.color_family is vs.YUV:
            input_clip = gamma2linear(
                self.kernel.resample(clip, vs.RGBS, matrix_in=targ_matrix), transfer, 1, True, gamma
            )
        else:
            input_clip = clip

        out_clip = input_clip.std.Limiter().chkdr.grain(**kwargs)

        if clip.format.color_family is vs.YUV:
            out_clip = self.kernel.resample(
                linear2gamma(out_clip, transfer, 1, True, gamma), clip, targ_matrix
            )

        return out_clip


def multi_graining(
    clip: vs.VideoNode, *grainers: MultiGrainerT, prefilter: vs.VideoNode | Prefilter | None = None
) -> vs.VideoNode:
    """
    Interface for applying multiple grainers to a clip.

    :param clip:        Input clip.
    :param grainers:    Grainers to apply. Can be a grainer, or a tuple of a grainer/None and threshold, overflow.
                        The threshold is the upper threshold of where the grainer will be applied.
                        The overflow is the range of the hard threshold.
                        If a grainer is None, the original clip will be applied in that range.
                        For example:
                            MultiGrainer((None, 0.1), (AddGrain, 0.5), (AddNoise, 0.8))

                        Will apply no grain for values <= 0.1 and > 0.8, AddGrain for values <= 0.5,
                        AddNoise for values <= 0.8.
    :param prefilter:   Clip or prefilter for making theh clip used for the threshold masks.
    """

    assert check_variable(clip, multi_graining)

    InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV))

    length = len(grainers)

    if length < 2:
        raise CustomIndexError('You need to give at least two grainers!')

    norm_grainers = sorted([
        (
            grainer if len(grainer) == 3 else (
                (*grainer, 1 / length) if len(grainer) == 2 else (None, 1.0, 1 / length)
            )
        ) if isinstance(grainer, tuple) else (grainer, 1.0, 1 / length)
        for grainer in grainers
    ], key=lambda x: x[1])

    if all(grainer is None for grainer, *_ in norm_grainers):
        raise CustomValueError('No valid grainers given!')
    elif any(grainer.neutral_out for grainer, *_ in norm_grainers if isinstance(grainer, Grainer)):
        raise CustomValueError('You can\'t set neutral_out in any grainer!')

    if prefilter is None:
        prefilter = get_y(clip)
    elif isinstance(prefilter, Prefilter):
        prefilter = prefilter(get_y(clip))

    prefilter = depth(get_y(prefilter), clip)

    peak = get_peak_value(clip)
    masks = [prefilter.std.BlankClip(color=0)] + [
        norm_expr(
            prefilter,
            'x {min_thr} >= x {max_thr} <= and x {min_thr} - {max_thr} {min_thr} '
            '- / {peak} * {peak} - abs x {min_thr} < {peak} x {max_thr} > 0 x ? ? ?',
            min_thr=f'{thr} {weight} {peak} * 2 / -', max_thr=f'{thr} {weight} {peak} * 2 / +',
            peak=peak
        ) for _, thr, weight in norm_grainers
    ]

    masks = [norm_expr(diffs, 'x y -') for diffs in zip(masks[:-1], masks[1:])]

    if clip.format.num_planes == 3:
        masks = [Bilinear.resample(join(mask, mask, mask), clip) for mask in masks]

    graineds = [grainer.grain(clip) if grainer else clip for grainer, *_ in norm_grainers]

    clips_merge = [
        clip.std.MaskedMerge(grained, mask) for grained, mask in zip(graineds, masks)
    ]

    return reduce(lambda x, y: y.std.MergeDiff(clip.std.MakeDiff(x)), clips_merge, clip)


MultiGrainerT = Grainer | type[Grainer] | tuple[Grainer | type[Grainer] | None, float] | tuple[
    Grainer | type[Grainer] | None, float, float
]
