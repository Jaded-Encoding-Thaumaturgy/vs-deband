#!/usr/bin/env python

from functools import partial
import math
from typing import List, Optional

import vapoursynth as vs

core = vs.core


__author__      = "Z4STIN"
__copyright__   = "Copyright 2020, Z4STIN"
__maintainer__  = "LightArrowsEXE"

__description__ = "A collection of functions and wrappers for debanding-related filtering"
__version__     = "0.4.0"


#####################################################
#####################################################
#####################################################
###############                       ###############
###############   Debanding Filters   ###############
###############                       ###############
#####################################################
#####################################################
#####################################################


def Dither_bilateral(clip: vs.VideoNode, ref: vs.VideoNode = None,
                     radius: Optional[float] = None,
                     thr: float = 0.35, wmin: float = 1.0,
                     subspl: float = 0,
                     planes: Optional[List[int]] = None) -> vs.VideoNode:
    """
    Dither's Gradfun3 mode 2, for Vapoursynth
    Not much different from using core.avsw.Eval with Dither_bilateral_multistage, just with normal value rounding and without Dither_limit_dif16
    If you want the rounding to be exactly the same, replace the applicable lines from here: https://pastebin.com/raw/gZKHCFkd
    Default setting of 'radius' changes to reflect the resolution

    Radius auto-adjust: 480p  ->  9
                        720p  -> 12
                        810p  -> 13
                        900p  -> 14
                        1080p -> 16
                        2K    -> 20
                        4K    -> 32

    Basic usage: flt = db.Dither_bilateral(clip, thr=1/3)
                    flt = mvf.LimitFilter(flt, clip, thr=1/3)
    """
    if radius is None:
        radius = max( (clip.width - 1280) / 160 + 12, (clip.height - 720)/ 90 + 12 )

    planes = list(range(clip.format.num_planes)) if planes is None else [planes] if isinstance(planes, int) else planes
    y, u, v = [3 if x in planes else 1 for x in range(3)]

    thr_1 = round(max(thr * 4.5, 1.25), 1)
    thr_2 = round(max(thr * 9, 5), 1)
    subspl_2 = subspl if subspl in (0, 1) else subspl / 2
    r4 = round(max(radius * 4 / 3, 4))
    r2 = round(max(radius * 2 / 3, 3))
    r1 = round(max(radius     / 3, 2))

    clips = [clip.fmtc.nativetostack16()]
    clip_names = ["c"]
    ref_t = "c"
    if ref is not None:
        ref_s = ref.fmtc.nativetostack16()
        clips += [ref_s]
        clip_names += ["ref"]
        ref_t = "ref"

    avs_stuff = "c.Dither_bilateral16(radius={r4}, thr={thr_1}, flat=0.75, wmin={wmin}, ref={ref_t}, subspl={subspl},   y={y}, u={u}, v={v})"
    avs_stuff += ".Dither_bilateral16(radius={r2}, thr={thr_2}, flat=0.25, wmin={wmin}, ref={ref_t}, subspl={subspl_2}, y={y}, u={u}, v={v})"
    avs_stuff += ".Dither_bilateral16(radius={r1}, thr={thr_2}, flat=0.50, wmin={wmin}, ref={ref_t}, subspl={subspl_2}, y={y}, u={u}, v={v})"
    avs_stuff = avs_stuff.format(ref_t=ref_t, r4=r4, r2=r2, r1=r1, thr_1=thr_1, thr_2=thr_2, wmin=wmin, subspl=subspl, subspl_2=subspl_2, y=y, u=u, v=v)

    return core.avsw.Eval(avs_stuff, clips=clips, clip_names=clip_names).fmtc.stack16tonative()
#TODO: rewrite to take arrays for most parameters (should at least save the extra stack16 conversions)


def Deband(clip: vs.VideoNode,
           radius = None, str = None, thr = None, thrc = None, smode = 2,
           mask = None, bin = None, planes = None, **args) -> vs.VideoNode:
    """
    NOTE: GuidedFilter is really, really strong. Using a (much) lower value on 'str' relative to 'thr' is recommended
          I don't even use GF at all anymore but I recall even values like 0.001 having an effect on banding,
          but my memory is shit so no guarantees

    Somewhat like GradFun3, uses GuidedFilter for the debanding
    radius is now an int array
    thr is split into str and thr:
        thr and thrc are for LimitFilter and take floats/integers
            set them to 0 to disable limiting
        str is for GuidedFilter and takes an int array or flt/int
        they try to copy each other if only one is set
    smode sets the regulation mode, but its more or less like how it was in gf3: 0-2 and higher = safer on detail
    mask is also an int array, and uses a normal chroma mask instead of using a luma mask for all 3 planes (can't be changed)
    thr_det is replaced by bin, and is the threshold for std.Binarize. The user is expected to adjust it to the bitdepth and
        sample type, but default behavior should be about the same as gf3
    planes does what it always does
    args is there for various reasons:
        elast, ref, or even src from LimitFilter
        guidance, etc from GuidedFilter
        setting range/range_in = 1 if you have a full range YUV/GRAY source with integer samples (it's important-ish)

    Default settings for 16 bit input (1080p 4:2:0):
    clip = Deband(clip, radius=[16, 10], str=1/3, thr=1/3, thrc=1/3, smode=2, elast=3, mask=2, bin=384, planes=[0, 1, 2])

    P.S. It works in whatever bitdepth and sample you give it so try not to use 8 bit (or set thr and thrc = 0)
            I've also gotten weird results with LimitFilter in 32 bit before, so probably just stick to 16 bit
   """
    try:
       from mvsfunc import LimitFilter
    except ModuleNotFoundError:
        raise ModuleNotFoundError("deband: missing dependency 'mvsfunc'")


    bits = clip.format.bits_per_sample
    isint = clip.format.sample_type == vs.INTEGER
    numplanes = clip.format.num_planes

    planes = list(range(numplanes)) if planes is None else [planes] if isinstance(planes, int) else planes

    str, thr, thrc = parse_params(str, thr, thrc, numplanes)

    mask = [math.ceil(clip.height/540)] if mask is None else mask if isinstance(mask, list) else [mask]
    while len(mask) < 3:
        mask += [mask[-1]]

    bdf = [1.5/255] if args.get('range_in', clip.format.color_family in [vs.RGB,vs.YCOCG]) else [1.5/219, 1.5/224]
    bdi = [math.ceil(0.005859375 * (1<<bits))]
    bin = (bdi if isint else bdf) if bin is None else bin if isinstance(bin, list) else [bin]
    while len(bin) < numplanes:
        bin += [bin[-1]]

    mask = [mask[i] if i in planes else 0 for i in range(3)]
    bin = [bin[i] if i in planes else 0 for i in range(numplanes)]

    bplanes, mplanes = [[]]*2
    for i in range(numplanes):
        if i in planes and mask[i] > 0:
            mplanes += [i]
            if bin[i] > 0:
                bplanes += [i]

    deband = GuidedFilter(clip, radius=radius, thr=thr, regulation_mode=smode, planes=planes, **args)
    if any([thr, thrc]):
        deband = LimitFilter(flt=deband, src=args.get('src', clip), thr=thr, thrc=thrc, **args)

    if max(mask):
        rmask = rangemask(clip, mask[0], mask[1])
        #if max(bin) > 0:
        #    rmask = rmask.std.Binarize(threshold=bin, planes=bplanes)
        rgvs = core.rgvs if isint else core.rgsf
        for i in range(min(3, max(mask))):
            rmask = rgvs.RemoveGrain(rmask, [[22,11,20][i] if mask[p] > i and bin[p] > 0 else 0 for p in range(numplanes)])
        deband = deband.std.MaskedMerge(clip, rmask, planes=planes)

    return deband


def parse_params(str, thr, thrc, numplanes):
    if str is None: # try to take values from thr, then thrc, and if all else fails use 1/3
        str = []
        if thr is not None:
            str += [thr]
        if thrc is not None:
            str += [thrc]
        if len(str)==0:
            str += [1/3]

    while len(str) < numplanes:
        str += [str[-1]]

    # when parsing thrc try thr first before str
    if thrc is None:
        thrc = str[-1] if thr is None else thr
    if thr is None:
        thr = str[0]
    # probably unnecessary but mt_lut was faster with the same expression for all planes so maybe std.Expr is too?
    # Its not a lut though so I doubt it
    if thr == thrc or numplanes==1:
        thrc = None

    return [str, thr, thrc]


# This is flat-out theft, but I wanted "planes" and separable strengths & radii so fuck it

def GuidedFilter(input,
                 guidance = None,
                 radius = None, thr = 1/3,
                 regulation = None, regulation_mode = 2,
                 use_gauss = False, fast = None,
                 subsampling_ratio = 4,
                 use_fmtc1 = False, kernel1 = 'point', kernel1_args = None,
                 use_fmtc2 = False, kernel2 = 'bilinear', kernel2_args = None,
                 planes = None, **depth_args):
    """
    Guided Filter - fast edge-preserving smoothing algorithm
    Author: Kaiming He et al. (http://kaiminghe.com/eccv10/)
    The guided filter computes the filtering output by considering the content of a guidance image.
    It can be used as an edge-preserving smoothing operator like the popular bilateral filter,
    but it has better behaviors near edges.
    The guided filter is also a more generic concept beyond smoothing:
    It can transfer the structures of the guidance image to the filtering output,
    enabling new filtering applications like detail enhancement, HDR compression,
    image matting/feathering, dehazing, joint upsampling, etc.
    All the internal calculations are done at 32-bit float.
    Args:
        input: Input clip.
        guidance: (clip) Guidance clip used to compute the coefficient of the linear translation on 'input'.
            It must has the same clip properties as 'input'.
            If it is None, it will be set to input, with duplicate calculations being omitted.
            Default is None.
        radius: (int[]) Box / Gaussian filter's radius.
            If box filter is used, the range of radius is 1 ~ 12(fast=False) or 1 ~ 12*subsampling_ratio in VapourSynth R38 or older because of the limitation of std.Convolution().
            For gaussian filter, the radius can be much larger, even reaching the width/height of the clip.
            Default for YUV420 is [16, 10] @ 1080p, [12, 8] at 720p and [9, 7] at 480p.
        thr: (float[]) Alternate way to specify 'regulation'.
            Set 'range_in' if you have a full range YUV source so it scales correctly
            Default is 1/3
        regulation: (float[]) A criterion for judging whether a patch has high variance and should be preserved, or is flat and should be smoothed.
            Similar to the range variance in the bilateral filter.
            Default is thr divided by the number of possible 8 bit values for each plane. Set 'range_in' if needed
        regulation_mode: (int) Tweak on regulation.
            It was mentioned in [1] that the local filters such as the Bilateral Filter (BF) or Guided Image Filter (GIF)
            would concentrate the blurring near these edges and introduce halos.
            The author of Weighted Guided Image Filter (WGIF) [3] argued that,
            the Lagrangian factor (regulation) in the GIF is fixed could be another major reason that the GIF produces halo artifacts.
            In [3], a WGIF was proposed to reduce the halo artifacts of the GIF.
            An edge aware factor was introduced to the constraint term of the GIF,
            the factor makes the edges preserved better in the result images and thus reduces the halo artifacts.
            In [4], a gradient domain guided image filter is proposed by incorporating an explicit first-order edge-aware constraint.
            The proposed filter is based on local optimization
            and the cost function is composed of a zeroth order data fidelity term and a first order regularization term.
            So the factors in the new local linear model can represent the images more accurately near edges.
            In addition, the edge-aware factor is multi-scale, which can separate edges of an image from fine details of the image better.
            0: Guided Filter [1]
            1: Weighted Guided Image Filter [3]
            2: Gradient Domain Guided Image Filter [4]
            Default is 2.
        use_gauss: Whether to use gaussian guided filter [1]. This replaces mean filter with gaussian filter.
            Guided filter is rotationally asymmetric and slightly biases to the x/y-axis because a box window is used in the filter design.
            The problem can be solved by using a gaussian weighted window instead. The resulting kernels are rotationally symmetric.
            The authors of [1] suggest that in practice the original guided filter is always good enough.
            Gaussian is performed by core.tcanny.TCanny(mode=-1).
            The sigma is set to r/sqrt(2).
            Default is False.
        fast: (bool) Whether to use fast guided filter [2].
            This method subsamples the filtering input image and the guidance image,
            computes the local linear coefficients, and upsamples these coefficients.
            The upsampled coefficients are adopted on the original guidance image to produce the output.
            This method reduces the time complexity from O(N) to O(N^2) for a subsampling ratio s.
            Default is True if the version number of VapourSynth is less than 39, otherwise is False.
        subsampling_ratio: (float) Only works when fast=True.
            Generally should be no less than 'radius'.
            Default is 4.
        use_fmtc1, use_fmtc2: (bool) Whether to use fmtconv in subsampling / upsampling.
            Default is False.
            Note that fmtconv's point subsampling may causes pixel shift.
        kernel1, kernel2: (string) Subsampling/upsampling kernels.
            Default is 'point'and 'bilinear'.
        kernel1_args, kernel2_args: (dict) Additional parameters passed to resizers in the form of dict.
            Default is {}.
        planes: (int[]) Which planes should be processed
            Default processes all planes
        depth_args: (dict) Additional arguments passed to fvf.Depth() in the form of keyword arguments.
            Default is {}.
    Ref:
        [1] He, K., Sun, J., & Tang, X. (2013). Guided image filtering. IEEE transactions on pattern analysis and machine intelligence, 35(6), 1397-1409.
        [2] He, K., & Sun, J. (2015). Fast guided filter. arXiv preprint arXiv:1505.00996.
        [3] Li, Z., Zheng, J., Zhu, Z., Yao, W., & Wu, S. (2015). Weighted guided image filtering. IEEE Transactions on Image Processing, 24(1), 120-129.
        [4] Kou, F., Chen, W., Wen, C., & Li, Z. (2015). Gradient domain guided image filtering. IEEE Transactions on Image Processing, 24(11), 4528-4539.
    """
    funcName = 'GuidedFilter'

    try:
       from fvsfunc import Depth
    except ModuleNotFoundError:
        raise ModuleNotFoundError("GuidedFilter: missing dependency 'fvsfunc'")

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    # Plane processing stuff
    def _get_expr_array(expr, planes, num_planes):
        return [expr[i] if i in planes else '' for i in range(num_planes)]
    numpl = input.format.num_planes
    planes = list(range(numpl)) if planes is None else [planes] if isinstance(planes, int) else planes
    fmtc_planes = [3 if i in planes else 1 for i in range(numpl)]

    # Get clip's properties
    inbits = input.format.bits_per_sample
    width = input.width
    height = input.height

    if regulation is None:
        thr = [thr] * numpl if not isinstance(thr, list) else thr[:numpl]
        while len(thr) < numpl:
            thr += [thr[-1]]
        full = depth_args.get('range_in', input.format.color_family in (vs.RGB, vs.YCOCG))
        size = [220, 225, 225] if full else [256] * 3
        regulation = [thr[i]/size[i] for i in range(numpl)]

    if radius is None:
        width_c = width / (1 << input.format.subsampling_w)
        height_c= height/ (1 << input.format.subsampling_h)

        rad = max( (width   - 1280) / 160 + 12, (height - 720) / 90 + 12 )
        radc= max( (width_c - 1280) / 160 + 12, (height_c-720) / 90 + 12 )
        radius = [round(rad), round(radc)][:numpl]

    if guidance is not None:
        if not isinstance(guidance, vs.VideoNode):
            raise TypeError(funcName + ': \"guidance\" must be a clip!')
        if input.format.id != guidance.format.id:
            raise TypeError(funcName + ': \"guidance\" must be of the same format as \"input\"!')
        if input.width != guidance.width or input.height != guidance.height:
            raise TypeError(funcName + ': \"guidance\" must be of the same size as \"input\"!')
        if input == guidance: # Remove redundant computation
            guidance = None

    if fast is None:
        fast = False if core.version_number() >= 39 else True

    if kernel1_args is None:
        kernel1_args = {}
    if kernel2_args is None:
        kernel2_args = {}

    # Bitdepth conversion and variable names modification to correspond to the paper
    p = Depth(input, 32, range_in=depth_args.get('range_in'))
    I = Depth(guidance, 32, range_in=depth_args.get('range_in')) if guidance is not None else p
    r = [radius] if isinstance(radius, int) else radius[:numpl]
    eps = [regulation] * numpl if not isinstance(regulation, list) else regulation[:numpl]
    while len(eps) < numpl:
        eps += [eps[-1]]
    s = subsampling_ratio

    # Back up guidance image
    I_src = I

    # Fast guided filter's subsampling
    if fast:
        down_w = round(width / s + 0.5)
        down_h = round(height / s + 0.5)
        if use_fmtc1:
            p = core.fmtc.resample(p, down_w, down_h, kernel=kernel1, planes=fmtc_planes, **kernel1_args)
            I = core.fmtc.resample(I, down_w, down_h, kernel=kernel1, planes=fmtc_planes, **kernel1_args) if guidance is not None else p
        else: # use zimg
            p = eval('core.resize.{kernel}(p, down_w, down_h, **kernel1_args)'.format(kernel=kernel1.capitalize()))
            I = eval('core.resize.{kernel}(I, down_w, down_h, **kernel1_args)'.format(kernel=kernel1.capitalize())) if guidance is not None else p

        r = round(r / s + 0.5)

    # Select the shape of the kernel. As the width of BoxFilter in this module is (radius*2-1) rather than (radius*2+1), radius should be increased by one.
    Filter = partial(core.tcanny.TCanny, sigma=[val/2 * math.sqrt(2) for val in r], mode=-1, planes=planes) if use_gauss else partial(BoxFilter, radius=[val+1 for val in r], planes=planes)
    Filter_r1 = partial(core.tcanny.TCanny, sigma=1/2 * math.sqrt(2), mode=-1, planes=planes) if use_gauss else partial(BoxFilter, radius=1+1, planes=planes)


    # Edge-Aware Weighting, equation (5) in [3], or equation (9) in [4].
    def _FLT(n, f, clip, core, eps0, planes, numpl):
        frameMean = f.props.PlaneStatsAverage

        return core.std.Expr(clip, _get_expr_array(['x {eps0} + {avg} *'.format(avg=frameMean, eps0=eps0)]*numpl, planes, numpl))


    # Compute the optimal value of a of Gradient Domain Guided Image Filter, equation (12) in [4]
    def _FLT2(n, f, cov_Ip, weight_in, weight, var_I, core, eps, planes, numpl):
        frameMean = f.props.PlaneStatsAverage
        frameMin = f.props.PlaneStatsMin

        alpha = frameMean
        kk = -4 / (frameMin - alpha - 1e-6) # Add a small num to prevent divided by 0

        expr = ['x {eps} 1 1 1 {kk} y {alpha} - * exp + / - * z / + a {eps} z / + /'.format(eps=e, kk=kk, alpha=alpha) for e in eps]

        return core.std.Expr([cov_Ip, weight_in, weight, var_I], _get_expr_array(expr, planes, numpl))

    # Compute local linear coefficients.
    mean_p = Filter(p)
    mean_I = Filter(I) if guidance is not None else mean_p
    I_square = core.std.Expr([I], _get_expr_array(['x dup *']*numpl, planes, numpl))
    corr_I = Filter(I_square)
    corr_Ip = Filter(core.std.Expr([I, p], _get_expr_array(['x y *']*numpl, planes, numpl))) if guidance is not None else corr_I

    var_I = core.std.Expr([corr_I, mean_I], _get_expr_array(['x y dup * -']*numpl, planes, numpl))
    cov_Ip = core.std.Expr([corr_Ip, mean_I, mean_p], _get_expr_array(['x y z * -']*numpl, planes, numpl)) if guidance is not None else var_I

    if regulation_mode: # 0: Original Guided Filter, 1: Weighted Guided Image Filter, 2: Gradient Domain Guided Image Filter
        if r != 1:
            mean_I_1 = Filter_r1(I)
            corr_I_1 = Filter_r1(I_square)
            var_I_1 = core.std.Expr([corr_I_1, mean_I_1], _get_expr_array(['x y dup * -']*numpl, planes, numpl))
        else: # r == 1
            var_I_1 = var_I

        if regulation_mode == 1: # Weighted Guided Image Filter
            weight_in = var_I_1
        else: # regulation_mode == 2, Gradient Domain Guided Image Filter
            weight_in = core.std.Expr([var_I, var_I_1], _get_expr_array(['x y * sqrt']*numpl, planes, numpl))

        eps0 = 0.001 ** 2 # Epsilon in [3] and [4]
        denominator = core.std.Expr([weight_in], _get_expr_array(['1 x {} + /'.format(eps0)]*numpl, planes, numpl))

        denominator = core.std.PlaneStats(denominator, plane=[0])
        weight = core.std.FrameEval(denominator, partial(_FLT, clip=weight_in, core=core, eps0=eps0, planes=planes, numpl=numpl), prop_src=[denominator]) # equation (5) in [3], or equation (9) in [4]

        if regulation_mode == 1: # Weighted Guided Image Filter
            a = core.std.Expr([cov_Ip, var_I, weight], _get_expr_array(['x y {eps} z / + /'.format(eps=e) for e in eps], planes, numpl))
        else: # regulation_mode == 2, Gradient Domain Guided Image Filter
            weight_in = core.std.PlaneStats(weight_in, plane=[0])
            a = core.std.FrameEval(weight, partial(_FLT2, cov_Ip=cov_Ip, weight_in=weight_in, weight=weight, var_I=var_I, core=core, eps=eps, planes=planes, numpl=numpl), prop_src=[weight_in])
    else: # regulation_mode == 0, Original Guided Filter
        a = core.std.Expr([cov_Ip, var_I], _get_expr_array(['x y {} + /'.format(e) for e in eps], planes, numpl))

    b = core.std.Expr([mean_p, a, mean_I], _get_expr_array(['x y z * -']*numpl, planes, numpl))


    mean_a = Filter(a)
    mean_b = Filter(b)

    # Fast guided filter's upsampling
    if fast:
        if use_fmtc2:
            mean_a = core.fmtc.resample(mean_a, width, height, kernel=kernel2, planes=fmtc_planes, **kernel2_args)
            mean_b = core.fmtc.resample(mean_b, width, height, kernel=kernel2, planes=fmtc_planes, **kernel2_args)
        else: # use zimg
            mean_a = eval('core.resize.{kernel}(mean_a, {w}, {h}, **kernel2_args)'.format(kernel=kernel2.capitalize(), w=width, h=height))
            mean_b = eval('core.resize.{kernel}(mean_b, {w}, {h}, **kernel2_args)'.format(kernel=kernel2.capitalize(), w=width, h=height))

    # Linear translation
    q = core.std.Expr([mean_a, I_src, mean_b], _get_expr_array(['x y * z +']*numpl, planes, numpl))

    # Final bitdepth conversion
    try:
        return Depth(q, inbits, **depth_args)
    except TypeError:
        return Depth(q, **depth_args)


def BoxFilter(clip, radius, planes):
    try:
       from muvsfunc import BoxFilter
    except ModuleNotFoundError:
        raise ModuleNotFoundError("BoxFilter: missing dependency 'muvsfunc'")

    numpl = clip.format.num_planes

    radius = [radius] * numpl if isinstance(radius, int) else radius
    while len(radius) < numpl:
        radius += [radius[-1]]

    if len(planes) == numpl and min(radius) == max(radius):
        return BoxFilter(clip, radius=radius[0], planes=planes)

    clips = [core.std.ShufflePlanes(clip, i, vs.GRAY) for i in range(numpl)]
    clips = [BoxFilter(clips[i], radius=radius[i], planes=planes) if i in planes else clips[i] for i in range(numpl)]

    return core.std.ShufflePlanes(clips, [0] * 3, vs.YUV)



###############################################################################
# f3kbilateral - f3kdb multistage bilateral-esque filter                      #
###############################################################################
# This thing is more of a last resort for extreme banding                     #
# With that in mind, 40~60 is probably an effective range for y & c strengths #
# I did use range=20, y=160 to scene-filter some horrendous fades, though     #
###############################################################################
def f3kbilateral(clip: vs.VideoNode,
                 range: Optional[int] = None,
                 y: int = 50, c: int = 0,
                 thr: float = 0.6, thrc: Optional[float] = None,
                 elast: float = 3.) -> vs.VideoNode:
    from fvsfunc import Depth
    from mvsfunc import LimitFilter
    from muvsfunc import MergeChroma

    f = clip.format
    bits = f.bits_per_sample
    isGRAY = f.color_family==vs.GRAY

    range = (12 if clip.width < 1800 and clip.height < 1000 else 16) if range is None else range
    r1 = round(range*4/3)
    r2 = round(range*2/3)
    r3 = round(range/3)
    y1 = y // 2
    y2 = y
    y3 = y
    c1 = c // 2
    c2 = c
    c3 = c

    if c==0:
        flt0 = togray(clip, 16)
    else:
        flt0 = Depth(clip, 16)

    flt1 = f3kdb(flt0, r1, y1, c1)
    flt2 = f3kdb(flt1, r2, y2, c2)
    flt3 = f3kdb(flt2, r3, y3, c3)

    flt = LimitFilter(flt3, flt2, ref=flt0, thr=thr, elast=elast, thrc=thrc)
    flt = Depth(flt, bits)

    if c==0 and not isGRAY:
        flt = MergeChroma(flt, clip)

    return flt


######################################################################################################################
# f3kpf - f3kdb with a simple prefilter by mawen1250 - https://www.nmm-hd.org/newbbs/viewtopic.php?f=7&t=1495#p12163 #
######################################################################################################################
# Since the prefilter is a straight gaussian+average blur, f3kdb's effect becomes very strong, very fast             #
# Functions more or less like gradfun3 without the detail mask                                                       #
######################################################################################################################
def f3kpf(clip, range=None, y=40, cb=40, cr=None, thr=0.3, elast=2.5, thrc=None):
    from fvsfunc import Depth
    from mvsfunc import LimitFilter
    from muvsfunc import MergeChroma

    f = clip.format
    bits = f.bits_per_sample
    isGRAY = f.color_family == vs.GRAY

    range = (12 if clip.width < 1800 and clip.height < 1000 else 16) if range is None else range
    cr = cb if cr is None else cr

    if cr == 0 and cb == 0:
        clp = togray(clip, 32)
    else:
        clp = Depth(clip, 32)
    blur32 = clp.std.Convolution([1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Convolution([1] * 9, planes=0)
    blur16 = Depth(blur32, 16)
    diff = clp.std.MakeDiff(blur32)
    f3k = f3kdb(blur16, range, y, cb, cr)
    f3k = LimitFilter(f3k, blur16, thr=thr, elast=elast, thrc=thrc)
    f3k = Depth(f3k, 32)
    out = f3k.std.MergeDiff(diff)
    out = Depth(out, bits)
    if cr == 0 and cb == 0 and not isGRAY:
        out = MergeChroma(out, clip)
    return out


############################################
# lfdeband - some Avisynth filter I ported #
############################################
def lfdeband(clip: vs.VideoNode) -> vs.VideoNode:
    try:
       from fvsfunc import Depth
    except ModuleNotFoundError:
        raise ModuleNotFoundError("lfdeband: missing dependency 'fvsfunc'")

    f =  clip.format
    bits = f.bits_per_sample
    wss = 1 << f.subsampling_w
    hss = 1 << f.subsampling_h
    w = clip.width
    h = clip.height
    dw = round(w / 2)
    dh = round(h / 2)

    clp = Depth(clip, 32)
    dsc = clp.fmtc.resample(dw-dw%wss, dh-dh%hss, kernel = 'spline64')
    ddb = f3kdb(dsc, range = 30, y = 80, cb = 80, cr = 80, grainy = 0, grainc = 0)
    ddif = ddb.std.MakeDiff(dsc)
    dif = ddif.fmtc.resample(w, h, kernel = 'spline64')
    out = clp.std.MergeDiff(dif)
    return Depth(out, bits)


###########################################################################################
# f3kdb - wrapper function                                                                #
# allows 32 bit in/out, and sets most parameters automatically; otherwise it's just f3kdb #
###########################################################################################
def f3kdb(clip, range=None, y=40, cb=None, cr=None, grainy=0, grainc=0, depth=None):
    try:
       from fvsfunc import Depth
    except ModuleNotFoundError:
        raise ModuleNotFoundError("f3kdb: missing dependency 'fvsfunc'")

    f = clip.format
    cf = f.color_family
    bits = f.bits_per_sample
    H16 = f.sample_type == vs.FLOAT and bits == 16
    tv_range = cf == vs.GRAY or cf == vs.YUV

    range = (12 if clip.width < 1800 and clip.height < 1000 else 16) if range is None else range
    cb = (y if cf == vs.RGB else y // 2) if cb is None else cb
    cr = cb if cr is None else cr
    output_depth = bits if depth is None else depth

    if bits > 16:
        clip = Depth(clip, 16)
    else:
        clip = forceint16(clip)

    deb = core.f3kdb.Deband(clip, range, y, cb, cr, grainy, grainc, keep_tv_range=tv_range, output_depth=min(output_depth, 16))

    if output_depth == 32:
        return Depth(deb, 32)
    else:
        return forceint16(deb, undo = H16)


#######################################################
#######################################################
#######################################################
######################           ######################
######################   Masks   ######################
######################           ######################
#######################################################
#######################################################
#######################################################


##########################################################################################################
# rangemask - min/max mask with separate luma/chroma radii                                               #
##########################################################################################################
# rad/radc are the luma/chroma equivalent of gradfun3's "mask" parameter                                 #
# the way gradfun3's mask works is on an 8 bit scale, with rounded dithering of high depth input         #
# As such, when following this filter with a Binarize, use the following conversion steps based on input #
# -  8 bit = Binarize(2) or Binarize(thr_det)                                                            #
# - 16 bit = Binarize(384) or Binarize((thr_det - 0.5) * 256)                                            #
# - floats = Binarize(0.005859375) or Binarize((thr_det - 0.5) / 256)                                    #
##########################################################################################################
# when radii are equal to 1, this filter becomes identical to mt_edge("min/max", 0, 255, 0, 255)         #
##########################################################################################################
def rangemask(clip: vs.VideoNode, rad: int = 2, radc: Optional[int] = None) -> vs.VideoNode:
    isRGB = clip.format.color_family == vs.RGB
    radc = (rad if isRGB else 0) if radc is None else radc
    if radc == 0:
        clip = togray(clip, int16=False)
    ma = maxm(clip, rad, radc)
    mi = minm(clip, rad, radc)

    expr = 'x y -'
    if not rad:
        expr = ['','x y -']
    if not radc:
        expr = ['x y -','']

    return core.std.Expr([ma, mi], expr)


#########################################################################################
# lumamask - mask each pixel according to its luma value                                #
#########################################################################################
# - lo: all pixels below this threshold will be binary                                  #
# - hi: all pixels above this threshold will be binary                                  #
#       all pixels in-between will be scaled from black to white                        #
# - invert: when true, masks dark areas (pixels below lo will be white, and vice versa) #
#########################################################################################
def lumamask(clip, lo, hi, invert=True):
    f = clip.format
    bits = f.bits_per_sample
    isINT = f.sample_type == vs.INTEGER
    peak = (1 << bits) - 1 if isINT else 1

    if invert:
        mexpr = 'x {lo} < {peak} x {hi} > 0 {peak} x {lo} - {peak} {hi} {lo} - / * - ? ?'.format(peak=peak,lo=lo,hi=hi)
    else:
        mexpr = 'x {lo} < 0 x {hi} > {peak} 0 x {lo} - {peak} {lo} {hi} - / * - ? ?'.format(peak=peak,lo=lo,hi=hi)

    clip = togray(clip, bits, 1)
    mask = clip.std.Expr(mexpr)

    return mask




 #####################
 # Utility Functions #
 #####################

def togray(clip: vs.VideoNode,
           bits: Optional[int] = None,
           dmode: int = 3,
           range: Optional[int] = None,
           int16: bool = True) -> vs.VideoNode:
    try:
       from mvsfunc import GetPlane, GetMatrix
    except ModuleNotFoundError:
        raise ModuleNotFoundError("deband: missing dependency 'mvsfunc'")

    f = clip.format
    cf = f.color_family
    isGRAY = cf == vs.GRAY
    isYUV = cf == vs.YUV
    in_st = f.sample_type
    in_bits = f.bits_per_sample

    bits = in_bits if bits==None else bits
    st = (vs.INTEGER if int16 else (in_st if in_bits==16 else vs.INTEGER)) if bits < 32 else vs.FLOAT

    if (in_bits, in_st) == (bits, st) and isGRAY:
        return clip
    elif (in_bits, in_st) == (bits, st) and isYUV:
        return GetPlane(clip)
    else:
        format = core.register_format(vs.GRAY, st, bits, 0, 0)
        matrix = None if isGRAY or isYUV else GetMatrix(clip, id=True)
        dither_type = dmode if isinstance(dmode, str) else 'ordered' if dmode==0 else 'none' if dmode<3 else 'error_diffusion'
        return clip.resize.Spline36(format=format.id, matrix=matrix, dither_type=dither_type, range_in=range, range=range)


def forceint16(clip: vs.VideoNode, undo: bool = False) -> vs.VideoNode:
    f = clip.format
    dst_st = vs.FLOAT if undo else vs.INTEGER

    if f.bits_per_sample!=16 or f.sample_type==dst_st:
        return clip
    else:
        format = core.register_format(f.color_family, dst_st, 16, f.subsampling_w, f.subsampling_h)
        return clip.resize.Spline36(format=format.id)


def maxm(clip: vs.VideoNode, sy: int = 2, sc: int = 2) -> vs.VideoNode:
    yp = sy>=sc
    yiter = 1 if yp else 0
    cp = sc>=sy
    citer = 1 if cp else 0
    planes = [0] if yp and not cp else [1,2] if cp and not yp else [0,1,2]
    coor = [0, 1, 0, 1, 1, 0, 1, 0] if (max(sy,sc) % 3) != 1 else [1] * 8

    if sy>0 or sc>0:
        return maxm(clip.std.Maximum(planes=planes, coordinates=coor), sy=sy-yiter, sc=sc-citer)
    else:
        return clip


def minm(clip: vs.VideoNode, sy: int = 2, sc: int = 2) -> vs.VideoNode:
    yp = sy>=sc
    yiter = 1 if yp else 0
    cp = sc>=sy
    citer = 1 if cp else 0
    planes = [0] if yp and not cp else [1,2] if cp and not yp else [0,1,2]
    coor = [0, 1, 0, 1, 1, 0, 1, 0] if (max(sy,sc) % 3) != 1 else [1] * 8

    if sy>0 or sc>0:
        return minm(clip.std.Minimum(planes=planes, coordinates=coor), sy=sy-yiter, sc=sc-citer)
    else:
        return clip
