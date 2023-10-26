from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import reduce, partial
from typing import TYPE_CHECKING, Any, Callable, Iterable, TypeAlias, cast, Sequence, Union, List, Tuple, Optional, Dict

from vsdenoise import Prefilter
from vsexprtools import complexpr_available, norm_expr
from vskernels import BicubicAuto, Bilinear, Catrom, Kernel, KernelT, Lanczos, LinearLight, Scaler, ScalerT
from vsmasktools import adg_mask
from vsrgtools import BlurMatrix
from vstools import (
    CustomIndexError, CustomOverflowError, CustomValueError, InvalidColorFamilyError, KwargsT, Matrix, MatrixT,
    VSFunctionNoArgs, check_variable, core, depth, fallback, get_neutral_values, get_peak_value, get_y, inject_self,
    join, mod_x, normalize_seq, plane, scale_8bit, split, to_arr, get_depth, get_color_family, get_sample_type, get_plane_sizes, vs
)


from .f3kdb import F3kdb
from .placebo import Placebo

__all__ = [
    'Grainer', 'GrainPP',

    'AddGrain', 'AddNoise',

    'F3kdbGrain', 'PlaceboGrain',

    'LinearLightGrainer',

    'ChickenDream', 'FilmGrain',

    'multi_graining', 'MultiGrainerT',
    
    'Graigasm'
]


class _gpp:
    if TYPE_CHECKING:
        from .noise import GrainPP
        Resolver: TypeAlias = Callable[[vs.VideoNode], GrainPP]
    else:
        Resolver: TypeAlias = Callable[[vs.VideoNode], Any]


@dataclass
class GrainPP(_gpp):
    value: str
    kwargs: KwargsT = field(default_factory=lambda: KwargsT())

    @classmethod
    def Bump(cls, strength: float = 0.1) -> GrainPP:
        return cls('x[-1,1] x - {strength} * x +', KwargsT(strength=strength + 1.0))


FadeLimits = tuple[int | Iterable[int] | None, int | Iterable[int] | None]
GrainPostProcessT = VSFunctionNoArgs | str | GrainPP | GrainPP.Resolver
GrainPostProcessesT = GrainPostProcessT | list[GrainPostProcessT]


class Grainer(ABC):
    """Abstract graining interface"""

    def __init__(
        self, strength: float | tuple[float, float] = 0.25,
        size: float | tuple[float, float] = (1.0, 1.0), sharp: float | ScalerT = Lanczos,
        dynamic: bool = True, temporal_average: int | tuple[float, int] = (0.0, 1),
        postprocess: GrainPostProcessesT | None = None, protect_chroma: bool = False,
        luma_scaling: float | None = None, fade_limits: bool | FadeLimits = True, *,
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
        self.fade_limits = fade_limits
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

    def _check_input(
        self, clip: vs.VideoNode, strength: tuple[float, float], dynamic: bool = True, **kwargs: Any
    ) -> None:
        ...

    @inject_self.init_kwargs.clean
    def grain(
        self, clip: vs.VideoNode, strength: float | tuple[float, float] | None = None,
        dynamic: bool | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        kwargs = self._get_kw(kwargs)

        if strength is None:
            strength = self.strength

        dynamic = fallback(dynamic, self.dynamic)
        strength = strength if isinstance(strength, tuple) else (
            strength, strength if clip.format.num_planes > 1 else 0.0
        )

        if max(strength) <= 0.0:
            return clip

        if strength[0] <= 0.0 and strength[1] > 0.0:
            planes = [1, 2]
        elif strength[0] > 0.0 and strength[1] <= 0.0:
            planes = 0
        else:
            planes = None

        do_taverage = (
            dynamic
            and self.temporal_average > 0 and self.temporal_radius > 0
            and (clip.num_frames > self.temporal_radius * 2)
        )
        do_protect_chroma = self.protect_chroma and strength[1] > 0.0 and clip.format.color_family is vs.YUV
        input_dep = self._is_input_dependent(clip, **kwargs)

        def _wrap_implementation(clip: vs.VideoNode, neutral_out: bool) -> vs.VideoNode:
            if input_dep and do_taverage and not kwargs.get('unsafe_graining', False):
                raise CustomValueError(
                    'You can\'t have temporal averaging with input dependent graining as it will create ghosting!'
                )

            if neutral_out and not input_dep:
                length = clip.num_frames + ((self.temporal_radius * 2) if do_taverage else 0)
                base_clip = clip.std.BlankClip(length=length, color=get_neutral_values(clip))
            elif do_taverage:
                base_clip = (clip[0] * self.temporal_radius) + clip + (clip[-1] * self.temporal_radius)
            else:
                base_clip = clip

            def _try_grain(src: vs.VideoNode, stre: tuple[float, float] = strength, **args: Any) -> vs.VideoNode:
                args = kwargs | dict(strength=stre, dynamic=dynamic) | args
                try:
                    self._check_input(src, **args)
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
                                _try_grain(plane(src, 0), str_luma),
                                _try_grain(plane(src, 1), str_chroma),
                                _try_grain(plane(src, 2), str_chroma)
                            )
                        elif str_luma > 0:
                            return join(
                                _try_grain(plane(src, 0), str_luma),
                                src
                            )
                        elif str_chroma > 0:
                            return join(
                                src,
                                _try_grain(plane(src, 1), str_chroma),
                                _try_grain(plane(src, 2), str_chroma)
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
            and not do_protect_chroma and self.luma_scaling is None and not self.fade_limits
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

        if self.fade_limits:
            low, high = (None, None) if self.fade_limits is True else self.fade_limits

            low = [scale_8bit(clip, l, not not i) for i, l in enumerate(normalize_seq(fallback(low, 16)))]
            high = [scale_8bit(clip, h, not not i) for i, h in enumerate(normalize_seq(fallback(high, [235, 240])))]

            if clip.format.sample_type is vs.FLOAT:
                limit_expr = 'x y abs - {low} < x y abs + {high} > or range_diff y ?'
            elif complexpr_available:
                limit_expr = 'y range_diff - abs A! x A@ - {low} < x A@ + {high} > or range_diff y ?'
            else:
                limit_expr = 'x y range_diff - abs - {low} < x y range_diff - abs + {high} > or range_diff y ?'

            grained = norm_expr([clip, grained], limit_expr, planes, low=low, high=high)

        if self.postprocess:
            for postprocess in cast(list[GrainPostProcessT], to_arr(self.postprocess)):
                if callable(postprocess):
                    postprocess = postprocess(grained)

                if isinstance(postprocess, vs.VideoNode):
                    grained = postprocess
                else:
                    if isinstance(postprocess, GrainPP):
                        postprocess, ppkwargs = postprocess.value, postprocess.kwargs
                    else:
                        ppkwargs = KwargsT()

                    # fuck importing re
                    uses_y = ' y ' in postprocess or postprocess.startswith('y ') or postprocess.endswith(' y')
                    grained = norm_expr([grained, clip] if uses_y else grained, postprocess, **ppkwargs)

        neutral = get_neutral_values(clip)

        if self.neutral_out:
            merge_clip = grained.std.MakeDiff(grained)[0].std.Loop(grained.num_frames)
        else:
            merge_clip, grained = clip, clip.std.MergeDiff(grained, planes)

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
        elif 'type' not in kwargs:
            raise ValueError('Type must be specified! Alternatively, you can use a subclass like AddNoise.GAUSS.')

        return kwargs

    def _is_poisson(self, **kwargs: Any) -> bool:
        return kwargs.get('type', None) == 4

    def _is_input_dependent(self, clip: vs.VideoNode, **kwargs: Any) -> bool:
        return self._is_poisson(**kwargs)

    def _check_input(
        self, clip: vs.VideoNode, strength: tuple[float, float], dynamic: bool = True, **kwargs: Any
    ) -> None:
        if self._is_poisson(**kwargs):
            if not dynamic:
                raise NotImplementedError('dynamic-only')

            if clip.format.bits_per_sample > 16:
                raise NotImplementedError('bad-depth-16')

            if min(*strength) < 0.0 or max(*strength) >= 1.0:
                raise ValueError('Poisson noise strength must be between 0.0 and 1.0 (not inclusive)!')

    def _perform_graining(
        self, clip: vs.VideoNode, strength: tuple[float, float], dynamic: bool = True, **kwargs: Any
    ) -> vs.VideoNode:
        if self._is_poisson(**kwargs):
            scale = ((1 << (clip.format.bits_per_sample - 8)) - 1) if clip.format.bits_per_sample > 8 else 255
            strength = (((1.0 - stre) * scale) if stre else 0.0 for stre in strength)

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

    def _check_input(
        self, clip: vs.VideoNode, strength: tuple[float, float], dynamic: bool = True, **kwargs: Any
    ) -> None:
        if not dynamic:
            raise NotImplementedError('dynamic-only')

    def _perform_graining(
        self, clip: vs.VideoNode, strength: tuple[float, float], dynamic: bool = True, **kwargs: Any
    ) -> vs.VideoNode:
        return Placebo.deband(clip, 8, 1, 1, list(strength), **kwargs)


class LinearLightGrainer(Grainer):
    """Base grainer depending on linear RGB clip, input dependent."""

    def __init__(
        self, strength: float | tuple[float, float],
        size: float | tuple[float, float] = (1.0, 1.0), sharp: float | ScalerT = Lanczos,
        dynamic: bool = True, temporal_average: int | tuple[float, int] = (0.0, 1),
        postprocess: GrainPostProcessesT | None = None, protect_chroma: bool = False,
        luma_scaling: float | None = None, fade_limits: bool | FadeLimits = True,
        *, gamma: float = 1.0, matrix: MatrixT | None = None, kernel: KernelT = Catrom, neutral_out: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(
            strength, size, sharp, dynamic, temporal_average, postprocess, protect_chroma, luma_scaling, fade_limits,
            matrix=matrix, kernel=kernel, neutral_out=neutral_out, **kwargs
        )

        if not 0.0 <= gamma <= 1.0:
            raise CustomOverflowError('Gamma must be between 0.0 and 1.0 (inclusive)!', self.__class__, gamma)
        self.gamma = gamma

    def _is_input_dependent(self, clip: vs.VideoNode, **kwargs: Any) -> bool:
        return True

    @abstractmethod
    def _get_inner_kwargs(self, strength: float, **kwargs: Any) -> KwargsT:
        ...

    @abstractmethod
    def _perform_linear_graining(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        ...

    def _check_input(
        self, clip: vs.VideoNode, strength: float | tuple[float, float], dynamic: bool = True, **kwargs: Any
    ) -> None:
        assert check_variable(clip, self.__class__.grain)

        if clip.format.bits_per_sample != 32:
            raise NotImplementedError('bad-depth-32')

        if not dynamic:
            raise NotImplementedError('dynamic-only')

    def _perform_graining(
        self, clip: vs.VideoNode, strength: float | tuple[float, float], dynamic: bool | int = True, **kwargs: Any
    ) -> vs.VideoNode:
        if isinstance(strength, tuple):
            if strength[0] != strength[1]:
                if clip.format.color_family is vs.YUV:
                    return join(
                        self._perform_graining(get_y(clip), strength[0]) if strength[0] else get_y(clip),
                        self._perform_graining(clip, strength[1]) if strength[1] else clip,
                    )

                raise CustomValueError('GRAY/RGB clips can\'t have different graining strength for chroma!')
            else:
                strength = strength[0]

        gamma = 1.0 - (self.gamma / 2)

        with LinearLight(clip, True, (6.5, gamma), self.kernel) as ll:
            kwargs = self._get_inner_kwargs(strength, **kwargs)
            ll.linear = self._perform_linear_graining(ll.linear.std.Limiter(), **kwargs)

        return ll.out


class ChickenDreamBase(LinearLightGrainer):
    """chkdr.grain plugin. https://github.com/EleonoreMizo/chickendream"""

    def __init__(
        self, strength: float | tuple[float, float], draft: bool,
        size: float | tuple[float, float] = (1.0, 1.0), sharp: float | ScalerT = Lanczos,
        dynamic: bool = True, temporal_average: int | tuple[float, int] = (0.0, 1),
        postprocess: GrainPostProcessesT | None = None, protect_chroma: bool = False,
        luma_scaling: float | None = None, fade_limits: bool | FadeLimits = True, *,
        rad: float = 0.25, res: int = 1024, dev: float = 0.0, gamma: float = 1.0,
        matrix: MatrixT | None = None, kernel: KernelT = Catrom, neutral_out: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(
            strength, size, sharp, dynamic, temporal_average, postprocess, protect_chroma, luma_scaling, fade_limits,
            matrix=matrix, kernel=kernel, neutral_out=neutral_out, rad=rad, res=res, dev=dev, gamma=gamma, **kwargs
        )

        self.draft = draft

    def _get_kw(self, kwargs: KwargsT) -> KwargsT:
        return super()._get_kw(kwargs) | dict(draft=self.draft)

    def _get_inner_kwargs(self, strength: float, **kwargs: Any) -> KwargsT:
        return kwargs | dict(sigma=strength, rad=kwargs.get('rad') / 10)

    def _perform_linear_graining(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        return core.chkdr.grain(clip, **kwargs)


class ChickenDreamBox(ChickenDreamBase):
    def __init__(
        self, strength: float | tuple[float, float] = 0.25,
        size: float | tuple[float, float] = (1.0, 1.0), sharp: float | ScalerT = Lanczos,
        dynamic: bool = True, temporal_average: int | tuple[float, int] = (0.0, 1),
        postprocess: GrainPostProcessesT | None = None, protect_chroma: bool = False,
        luma_scaling: float | None = None, fade_limits: bool | FadeLimits = True,
        *, res: int = 1024, dev: float = 0.0, gamma: float = 1.0,
        matrix: MatrixT | None = None, kernel: KernelT = Catrom, neutral_out: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(
            strength, True, size, sharp, dynamic, temporal_average, postprocess, protect_chroma, luma_scaling,
            fade_limits, matrix=matrix, kernel=kernel, neutral_out=neutral_out, res=res, dev=dev, gamma=gamma, **kwargs
        )

    def _get_inner_kwargs(self, strength: float, **kwargs: Any) -> KwargsT:
        return super()._get_inner_kwargs(0.0, **(kwargs | dict(rad=strength)))


class ChickenDreamGauss(ChickenDreamBase):
    def __init__(
        self, strength: float | tuple[float, float] = 0.35,
        size: float | tuple[float, float] = (1.0, 1.0), sharp: float | ScalerT = Lanczos,
        dynamic: bool = True, temporal_average: int | tuple[float, int] = (0.0, 1),
        postprocess: GrainPostProcessesT | None = None, protect_chroma: bool = False,
        luma_scaling: float | None = None, fade_limits: bool | FadeLimits = True,
        *, rad: float = 0.25, res: int = 1024, dev: float = 0.0, gamma: float = 1.0,
        matrix: MatrixT | None = None, kernel: KernelT = Catrom, neutral_out: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(
            strength, False, size, sharp, dynamic, temporal_average, postprocess, protect_chroma, luma_scaling,
            fade_limits, matrix=matrix, kernel=kernel, neutral_out=neutral_out, rad=rad, res=res, dev=dev,
            gamma=gamma, **kwargs
        )


class FilmGrain(LinearLightGrainer):
    """fgrain_cuda.Add plugin. https://github.com/AmusementClub/vs-fgrain-cuda"""

    def __init__(
        self, strength: float | tuple[float, float] = 0.8,
        size: float | tuple[float, float] = (1.0, 1.0), sharp: float | ScalerT = Lanczos,
        dynamic: bool = True, temporal_average: int | tuple[float, int] = (0.0, 1),
        postprocess: GrainPostProcessesT | None = None, protect_chroma: bool = False,
        luma_scaling: float | None = None, fade_limits: bool | FadeLimits = True,
        *, rad: float = 0.1, iterations: int = 800, dev: float = 0.0, gamma: float = 1.0,
        matrix: MatrixT | None = None, kernel: KernelT = Catrom, neutral_out: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(
            strength, size, sharp, dynamic, temporal_average, postprocess, protect_chroma, luma_scaling, fade_limits,
            matrix=matrix, kernel=kernel, neutral_out=neutral_out, gamma=gamma,
            grain_radius_mean=rad, num_iterations=iterations, grain_radius_std=dev, **kwargs
        )

    def _get_inner_kwargs(self, strength: float, **kwargs: Any) -> KwargsT:
        return kwargs | dict(sigma=strength)

    def _perform_linear_graining(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        return join(core.fgrain_cuda.Add(p, **kwargs) for p in split(clip))


class ChickenDream(ChickenDreamBox):
    class BOX(ChickenDreamBox):
        ...

    class GAUSS(ChickenDreamGauss):
        ...


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




def pick_px_op(
    use_expr: bool,
    operations: Tuple[str, Sequence[int] | Sequence[float] | int | float ]
) -> Callable[..., vs.VideoNode]:
    """Pick either std.Lut or std.Expr"""
    expr, lut = operations
    if use_expr:
        func = partial(core.std.Expr, expr=expr)
    else:
        if callable(lut):
            func = partial(core.std.Lut, function=lut)
        elif isinstance(lut, Sequence):
            if all(isinstance(x, int) for x in lut):
                func = partial(core.std.Lut, lut=lut)
            elif all(isinstance(x, float) for x in lut):
                func = partial(core.std.Lut, lutf=lut)
            else:
                raise ValueError('pick_px_operation: operations[1] is not a valid type!')
        elif isinstance(lut, int):
            func = partial(core.std.Lut, lut=lut)
        elif isinstance(lut, float):
            func = partial(core.std.Lut, lutf=lut)
        else:
            raise ValueError('pick_px_operation: operations[1] is not a valid type!')
    return func

class Graigasm():
    """Custom graining interface based on luma values"""
    thrs: List[float]
    strengths: List[Tuple[float, float]]
    sizes: List[float]
    sharps: List[float]
    overflows: List[float]
    grainers: List[Grainer]

    def __init__(self,
                 thrs: Sequence[float], strengths: Sequence[Tuple[float, float]], sizes: Sequence[float], sharps: Sequence[float], *,
                 overflows: Union[float, Sequence[float], None] = None,
                 grainers: Union[Grainer, Sequence[Grainer]] = AddGrain(seed=-1, constant=False)) -> None:
        """Constructor checks and initializes the values.
           Length of thrs must be equal to strengths, sizes and sharps.
           thrs, strengths, sizes and sharps match the same area.

        Args:
            thrs (Sequence[float]):
                Sequence of thresholds defining the grain boundary.
                Below the threshold, it's grained, above the threshold, it's not grained.

            strengths (Sequence[Tuple[float, float]]):
                Sequence of tuple representing the grain strengh of the luma and the chroma, respectively.

            sizes (Sequence[float]):
                Sequence of size of grain.

            sharps (Sequence[float]):
                Sequence of sharpened grain values. 50 is neutral Catmull-Rom (b=0, c=0.5).

            overflows (Union[float, Sequence[float]], optional):
                Percentage value determining by how much the hard limit of threshold will be extended.
                Range 0.0 - 1.0. Defaults to 1 divided by thrs's length for each thr.

            grainers (Union[Grainer, Sequence[Grainer]], optional):
                Grainer used for each combo of thrs, strengths, sizes and sharps.
                Defaults to AddGrain(seed=-1, constant=False).
        """
        self.thrs = list(thrs)
        self.strengths = list(strengths)
        self.sizes = list(sizes)
        self.sharps = list(sharps)

        length = len(self.thrs)
        datas: List[Any] = [self.strengths, self.sizes, self.sharps]
        if all(len(lst) != length for lst in datas):
            raise ValueError('Graigasm: "thrs", "strengths", "sizes" and "sharps" must have the same length!')

        if overflows is None:
            overflows = [1/length]
        if isinstance(overflows, float):
            overflows = [overflows] * length
        else:
            overflows = list(overflows)
            overflows += [overflows[-1]] * (length - len(overflows))
        self.overflows = overflows

        if isinstance(grainers, Grainer):
            grainers = [grainers] * length
        else:
            grainers = list(grainers)
            grainers += [grainers[-1]] * (length - len(grainers))
        self.grainers = grainers

    def graining(self,
                 clip: vs.VideoNode, /, *,
                 prefilter: Optional[vs.VideoNode] = None, show_masks: bool = False) -> vs.VideoNode:
        """Do grain stuff using settings from constructor.

        Args:
            clip (vs.VideoNode): Source clip.

            prefilter (clip, optional):
                Prefilter clip used to compute masks.
                Defaults to None.

            show_masks (bool, optional):
                Returns interleaved masks. Defaults to False.

        Returns:
            vs.VideoNode: Grained clip.
        """

        bits = get_depth(clip)
        is_float = get_sample_type(clip) == vs.FLOAT
        peak = 1.0 if is_float else (1 << bits) - 1
        num_planes = clip.format.num_planes
        neutral = [0.5] + [0.0] * (num_planes - 1) if is_float else [float(1 << (bits - 1))] * num_planes

        pref = prefilter if prefilter is not None else get_y(clip)

        mod = self._get_mod(clip)

        masks = [self._make_mask(pref, thr, ovf, peak, is_float=is_float) for thr, ovf in zip(self.thrs, self.overflows)]
        masks = [pref.std.BlankClip(color=0)] + masks
        masks = [core.std.Expr([masks[i], masks[i-1]], 'x y -') for i in range(1, len(masks))]


        if num_planes == 3:
            if is_float:
                masks_chroma = [mask.resize.Bilinear(*get_plane_sizes(clip, 1)) for mask in masks]
                masks = [join([mask, mask_chroma, mask_chroma]) for mask, mask_chroma in zip(masks, masks_chroma)]
            else:
                masks = [join([mask] * 3).resize.Bilinear(format=clip.format.id) for mask in masks]

        if show_masks:
            return core.std.Interleave(
                [mask.text.Text(f'Threshold: {thr}', 7).text.FrameNum(9)
                 for thr, mask in zip(self.thrs, masks)]
            )


        graineds = [self._make_grained(clip, strength, size, sharp, grainer, neutral, mod)
                    for strength, size, sharp, grainer in zip(self.strengths, self.sizes, self.sharps, self.grainers)]

        clips_adg = [core.std.Expr([grained, clip, mask], f'x z {peak} / * y 1 z {peak} / - * +')
                     for grained, mask in zip(graineds, masks)]


        out = clip
        for clip_adg in clips_adg:
            out = core.std.MergeDiff(clip_adg, core.std.MakeDiff(clip, out))  # type: ignore


        return out

    def _make_grained(self,
                      clip: vs.VideoNode,
                      strength: Tuple[float, float], size: float, sharp: float, grainer: Grainer,
                      neutral: List[float], mod: int) -> vs.VideoNode:
        ss_w = self._m__(round(clip.width / size), mod)
        ss_h = self._m__(round(clip.height / size), mod)
        b = sharp / -50 + 1
        c = (1 - b) / 2

        blank = core.std.BlankClip(clip, ss_w, ss_h, color=neutral)
        grained = grainer.grain(blank, strength=strength).resize.Bicubic(clip.width, clip.height, filter_param_a=b, filter_param_b=c)

        return clip.std.MakeDiff(grained)

    @staticmethod
    def _get_mod(clip: vs.VideoNode) -> int:
        ss_mod: Dict[Tuple[int, int], int] = {
            (0, 0): 1,
            (1, 1): 2,
            (1, 0): 2,
            (0, 1): 2,
            (2, 2): 4,
            (2, 0): 4
        }
        assert clip.format is not None
        try:
            return ss_mod[(clip.format.subsampling_w, clip.format.subsampling_h)]
        except KeyError as kerr:
            raise ValueError('Graigasm: Format unknown!') from kerr

    @staticmethod
    def _make_mask(clip: vs.VideoNode,
                   thr: float, overflow: float, peak: float, *,
                   is_float: bool) -> vs.VideoNode:

        def _func(x: float) -> int:
            min_thr = thr - (overflow * peak) / 2
            max_thr = thr + (overflow * peak) / 2
            if min_thr <= x <= max_thr:
                x = abs(((x - min_thr) / (max_thr - min_thr)) * peak - peak)
            elif x < min_thr:
                x = peak
            elif x > max_thr:
                x = 0.0
            return round(x)

        min_thr = f'{thr} {overflow} {peak} * 2 / -'
        max_thr = f'{thr} {overflow} {peak} * 2 / +'
        # if x >= min_thr and x <= max_thr -> gradient else ...
        expr = f'x {min_thr} >= x {max_thr} <= and x {min_thr} - {max_thr} {min_thr} - / {peak} * {peak} - abs _ ?'
        # ... if x < min_thr -> peak else ...
        expr = expr.replace('_', f'x {min_thr} < {peak} _ ?')
        # ... if x > max_thr -> 0 else x
        expr = expr.replace('_', f'x {max_thr} > 0 x ?')

        return pick_px_op(is_float, (expr, _func))(clip)

    @staticmethod
    def _m__(x: int, mod: int, /) -> int:
        return x - x % mod
