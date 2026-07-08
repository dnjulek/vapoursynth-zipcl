"""f16 <-> f32 parity for every vszipcl filter that accepts half-float input.

Every vszipcl filter takes u8 / u16 / f16 / f32 (Multi-format io pattern): the io
type is a per-program `-DBITS`/`-DHALF` bake and the internal math pipeline runs
in f32 regardless. The half path therefore differs from the float path in exactly
two places — the load (`vload_half`, an EXACT f16->f32 widening) and the store
(`vstore_half*`, an f16 rounding of the final f32 value).

pair(make, f16_fmt) builds an f16 clip and its BYTE-IDENTICAL f32 widening (a
Point resize is a pure container change; f16->f32 is lossless). Feeding both to
the same filter, the internal f32 computation is bit-for-bit identical, so the
ONLY thing that can move the two outputs apart is the filter's f16 STORE rounding.
Widening the f16 output back to f32 and asserting max|f16-f32| <= tol (about 2 ULP
of f16 near 1.0 ≈ 1e-3) proves the half path did not diverge from the float path.

These tests are DEVICE-INDEPENDENT: they compare the plugin's own two io paths on
the same device, not an absolute value, so they run unmarked (CI includes them).

Filters that reject f16 outright have their own validation tests; every filter
below (Bilateral, GaussBlur, EEDI3/EEDI3H, NLMeans, Deband) accepts it.

std.BoxBlur / std.Expr have NO f16 support, so every auxiliary clip a joint /
guide / vcheck path needs (ref, rclip, sclip) is built with vszipcl.GaussBlur —
which does take f16 — and widened to f32 so it too is byte-identical across runs.
"""

import pytest
import vapoursynth as vs

from helpers import max_abs_diff

# f16 format -> its matching f32 format (same family/subsampling, exact widening)
F16_TO_F32 = {
    vs.GRAYH: vs.GRAYS,
    vs.YUV420PH: vs.YUV420PS,
    vs.YUV422PH: vs.YUV422PS,
    vs.YUV444PH: vs.YUV444PS,
    vs.RGBH: vs.RGBS,
}


def pair(make, f16_fmt: int):
    """(f16 clip, f32 clip) carrying byte-identical data: build the f16 clip,
    then widen a copy to f32 (Point = container change only, no resampling) so any
    output difference is the filter's f16 path, not input quantization."""
    f16 = make(f16_fmt)
    f32 = f16.resize.Point(format=F16_TO_F32[f16_fmt])
    return f16, f32


def widen(clip: vs.VideoNode, f16_fmt: int) -> vs.VideoNode:
    """Widen an f16 clip to its matching f32 format (exact)."""
    return clip.resize.Point(format=F16_TO_F32[f16_fmt])


def assert_pixel_parity(out_f16: vs.VideoNode, out_f32: vs.VideoNode, *, tol: float, n: int = 0) -> None:
    """Widen the f16 output to f32 and assert max|f16-f32| <= tol on every plane.
    (std.Expr has no half support, so out_f16 must be widened before the diff.)"""
    widened = out_f16.resize.Point(format=out_f32.format.id)
    for p in range(out_f32.format.num_planes):
        d = max_abs_diff(widened, out_f32, plane=p, n=n)
        assert d <= tol, f"plane {p}: max|f16-f32| = {d} > {tol}"


# f16 store rounding is ~1 ULP; near 1.0 on the 0..1 scale that is ~1e-3 (2 ULP).
TOL = 1e-3
# EEDI3's 4-tap cubic legitimately overshoots to nearly |2.0| (ringing, u. clamped);
# there f16 ULP grows to ~2e-3, so the store-rounding bound is ~1e-3 per pixel but
# we give it 2 ULP of the wider range to stay robust across content.
TOL_EEDI3 = 2e-3


# ===========================================================================
# Bilateral
# ===========================================================================


@pytest.mark.parametrize(
    "f16_fmt",
    [vs.GRAYH, vs.YUV420PH, vs.RGBH],
    ids=["GRAYH", "YUV420PH", "RGBH"],
)
def test_f16_parity_bilateral(make_clip, f16_fmt):
    # Broad spatial/range sigmas: real edge-preserving smoothing. Input is
    # byte-identical (f32 widened from f16), so any output difference is the
    # filter's f16 store path (the range-weight accumulation runs in f32 both ways).
    src16, src32 = pair(make_clip, f16_fmt)
    out16 = src16.vszipcl.Bilateral(sigma_spatial=3, sigma_color=0.1)
    out32 = src32.vszipcl.Bilateral(sigma_spatial=3, sigma_color=0.1)
    assert_pixel_parity(out16, out32, tol=TOL)


@pytest.mark.parametrize(
    "f16_fmt",
    [vs.GRAYH, vs.YUV420PH, vs.RGBH],
    ids=["GRAYH", "YUV420PH", "RGBH"],
)
def test_f16_parity_bilateral_small_sigma_color(make_clip, f16_fmt):
    # A tight range sigma (0.02) makes the range weight very selective — the case
    # where an f16 read/store discrepancy would bite hardest — so exercise it
    # separately from the broad-sigma case.
    src16, src32 = pair(make_clip, f16_fmt)
    out16 = src16.vszipcl.Bilateral(sigma_spatial=3, sigma_color=0.02)
    out32 = src32.vszipcl.Bilateral(sigma_spatial=3, sigma_color=0.02)
    assert_pixel_parity(out16, out32, tol=TOL)


@pytest.mark.parametrize(
    "f16_fmt",
    [vs.GRAYH, vs.YUV420PH, vs.RGBH],
    ids=["GRAYH", "YUV420PH", "RGBH"],
)
def test_f16_parity_bilateral_gl(make_clip, f16_fmt):
    # Force the global-memory kernel (use_shared_memory=0): a distinct code path
    # from the tiled sm kernel, so verify its f16 store matches f32 too.
    src16, src32 = pair(make_clip, f16_fmt)
    out16 = src16.vszipcl.Bilateral(sigma_spatial=3, sigma_color=0.1, use_shared_memory=0)
    out32 = src32.vszipcl.Bilateral(sigma_spatial=3, sigma_color=0.1, use_shared_memory=0)
    assert_pixel_parity(out16, out32, tol=TOL)


@pytest.mark.parametrize(
    "f16_fmt",
    [vs.GRAYH, vs.YUV420PH, vs.RGBH],
    ids=["GRAYH", "YUV420PH", "RGBH"],
)
def test_f16_parity_bilateral_ref(make_clip, f16_fmt):
    # Joint (cross) bilateral: a separate ref clip drives the range weights. Build
    # the ref in f16 with vszipcl.GaussBlur (std.BoxBlur has no half support) and
    # widen it to f32, so both the source AND the ref are byte-identical across runs.
    src16, src32 = pair(make_clip, f16_fmt)
    ref16 = src16.vszipcl.GaussBlur(sigma=[3.0])
    ref32 = widen(ref16, f16_fmt)
    out16 = src16.vszipcl.Bilateral(sigma_spatial=3, sigma_color=0.05, ref=ref16)
    out32 = src32.vszipcl.Bilateral(sigma_spatial=3, sigma_color=0.05, ref=ref32)
    assert_pixel_parity(out16, out32, tol=TOL)


# ===========================================================================
# GaussBlur
# ===========================================================================


@pytest.mark.parametrize(
    "f16_fmt",
    [vs.GRAYH, vs.YUV420PH, vs.RGBH],
    ids=["GRAYH", "YUV420PH", "RGBH"],
)
def test_f16_parity_gaussblur_small(make_clip, f16_fmt):
    # sigma=2 -> radius<=32 -> the fused single-kernel smem path. The gaussian
    # accumulation is f32 in both runs; only the final store differs.
    src16, src32 = pair(make_clip, f16_fmt)
    out16 = src16.vszipcl.GaussBlur(sigma=[2.0])
    out32 = src32.vszipcl.GaussBlur(sigma=[2.0])
    assert_pixel_parity(out16, out32, tol=TOL)


@pytest.mark.parametrize(
    "f16_fmt",
    [vs.GRAYH, vs.YUV420PH, vs.RGBH],
    ids=["GRAYH", "YUV420PH", "RGBH"],
)
def test_f16_parity_gaussblur_large(make_clip, f16_fmt):
    # sigma=20 -> radius 60 > 32 -> the two-pass separable large kernel (a
    # different code path). Summing ~120 taps into an f32 accumulator, only the
    # store is quantized to f16, so the f16-vs-f32 gap stays at store precision.
    src16, src32 = pair(make_clip, f16_fmt)
    out16 = src16.vszipcl.GaussBlur(sigma=[20.0])
    out32 = src32.vszipcl.GaussBlur(sigma=[20.0])
    assert_pixel_parity(out16, out32, tol=TOL)


# ===========================================================================
# EEDI3 / EEDI3H
# ===========================================================================


@pytest.mark.parametrize(
    "f16_fmt",
    [vs.GRAYH, vs.YUV420PH, vs.RGBH],
    ids=["GRAYH", "YUV420PH", "RGBH"],
)
def test_f16_parity_eedi3(make_clip, f16_fmt):
    # Default vertical interpolation. The DP + cubic interp run in f32 identically
    # (byte-identical input); the interpolated rows are stored to f16.
    src16, src32 = pair(make_clip, f16_fmt)
    out16 = src16.vszipcl.EEDI3(field=1)
    out32 = src32.vszipcl.EEDI3(field=1)
    assert_pixel_parity(out16, out32, tol=TOL_EEDI3)


@pytest.mark.parametrize(
    "f16_fmt",
    [vs.GRAYH, vs.YUV420PH, vs.RGBH],
    ids=["GRAYH", "YUV420PH", "RGBH"],
)
def test_f16_parity_eedi3_hp_vcheck(make_clip, f16_fmt):
    # Real half-pel (hp) + the reliability check (vcheck): both add f32 work on top
    # of the base DP but still store the result once to f16. Covers the hp precompute
    # + vcheck blend paths.
    src16, src32 = pair(make_clip, f16_fmt)
    args = dict(field=1, hp=True, vcheck=3, nrad=3, mdis=40)
    out16 = src16.vszipcl.EEDI3(**args)
    out32 = src32.vszipcl.EEDI3(**args)
    assert_pixel_parity(out16, out32, tol=TOL_EEDI3)


@pytest.mark.parametrize(
    "f16_fmt",
    [vs.GRAYH, vs.YUV420PH, vs.RGBH],
    ids=["GRAYH", "YUV420PH", "RGBH"],
)
def test_f16_parity_eedi3_dh(make_clip, f16_fmt):
    # dh=True doubles height: source rows are copied VERBATIM (bit-lossless for any
    # io type via the raw_t bit-pattern copy) and only the new rows are interpolated
    # and stored to f16. Both effects must keep the two paths within store precision.
    src16, src32 = pair(make_clip, f16_fmt)
    out16 = src16.vszipcl.EEDI3(field=1, dh=True)
    out32 = src32.vszipcl.EEDI3(field=1, dh=True)
    assert_pixel_parity(out16, out32, tol=TOL_EEDI3)


@pytest.mark.parametrize(
    "f16_fmt",
    [vs.GRAYH, vs.YUV420PH, vs.RGBH],
    ids=["GRAYH", "YUV420PH", "RGBH"],
)
def test_f16_parity_eedi3h(make_clip, f16_fmt):
    # EEDI3H runs the identical vertical pipeline on a transposed copy; the
    # transpose is a bit-pattern move (io-type agnostic), so the f16 store is the
    # only divergence, same as EEDI3.
    src16, src32 = pair(make_clip, f16_fmt)
    out16 = src16.vszipcl.EEDI3H(field=1)
    out32 = src32.vszipcl.EEDI3H(field=1)
    assert_pixel_parity(out16, out32, tol=TOL_EEDI3)


def test_f16_parity_eedi3_sclip(make_clip):
    # vcheck>0 with an explicit sclip (the vcheck override clip): the reliability
    # blend reads sclip. Build it in f16 with vszipcl.GaussBlur and widen so sclip
    # is byte-identical across the two runs; then the only divergence is the store.
    src16, src32 = pair(make_clip, vs.GRAYH)
    sclip16 = src16.vszipcl.GaussBlur(sigma=[3.0])
    sclip32 = widen(sclip16, vs.GRAYH)
    out16 = src16.vszipcl.EEDI3(field=1, vcheck=3, sclip=sclip16)
    out32 = src32.vszipcl.EEDI3(field=1, vcheck=3, sclip=sclip32)
    assert_pixel_parity(out16, out32, tol=TOL_EEDI3)


# ===========================================================================
# NLMeans
# ===========================================================================


@pytest.mark.parametrize(
    "f16_fmt",
    [vs.GRAYH, vs.YUV420PH, vs.RGBH],
    ids=["GRAYH", "YUV420PH", "RGBH"],
)
def test_f16_parity_nlmeans(make_clip, f16_fmt):
    # Spatial (d=0) denoise. The weight/accumulation kernels run in f32/i32; the
    # per-pixel result is quantized once to f16 on store. (Gray/YUV use the default
    # auto channels = Y; RGB auto = RGB.)
    src16, src32 = pair(make_clip, f16_fmt)
    out16 = src16.vszipcl.NLMeans(d=0)
    out32 = src32.vszipcl.NLMeans(d=0)
    assert_pixel_parity(out16, out32, tol=TOL)


@pytest.mark.parametrize(
    "f16_fmt",
    [vs.GRAYH, vs.YUV420PH, vs.RGBH],
    ids=["GRAYH", "YUV420PH", "RGBH"],
)
def test_f16_parity_nlmeans_strong(make_clip, f16_fmt):
    # A large h + wmode 2 exercises the heavier weight-mode arithmetic; still f32
    # internally, f16 on store.
    src16, src32 = pair(make_clip, f16_fmt)
    args = dict(d=0, h=4.0, wmode=2)
    out16 = src16.vszipcl.NLMeans(**args)
    out32 = src32.vszipcl.NLMeans(**args)
    assert_pixel_parity(out16, out32, tol=TOL)


def test_f16_parity_nlmeans_temporal(make_temporal_clip):
    # Temporal window (d=1) on the 3-frame shifted clip: pulls neighbour frames
    # into the f32 accumulation. Widen via Point (preserves the shifted detail).
    src16 = make_temporal_clip(vs.GRAYH)
    src32 = src16.resize.Point(format=vs.GRAYS)
    out16 = src16.vszipcl.NLMeans(d=1)
    out32 = src32.vszipcl.NLMeans(d=1)
    # check the middle frame (its neighbours actually differ from it)
    assert_pixel_parity(out16, out32, tol=TOL, n=1)


def test_f16_parity_nlmeans_rclip(make_clip):
    # Guide/oracle split: the guide (rclip) drives the weights. Build it in f16 with
    # vszipcl.GaussBlur and widen so the guide is byte-identical across the two runs.
    src16, src32 = pair(make_clip, vs.GRAYH)
    guide16 = src16.vszipcl.GaussBlur(sigma=[3.0])
    guide32 = widen(guide16, vs.GRAYH)
    out16 = src16.vszipcl.NLMeans(d=0, rclip=guide16)
    out32 = src32.vszipcl.NLMeans(d=0, rclip=guide32)
    assert_pixel_parity(out16, out32, tol=TOL)


# ===========================================================================
# Deband
# ===========================================================================


@pytest.mark.parametrize(
    "f16_fmt",
    [vs.GRAYH, vs.YUV420PH, vs.YUV422PH, vs.YUV444PH, vs.RGBH],
    ids=["GRAYH", "YUV420PH", "YUV422PH", "YUV444PH", "RGBH"],
)
def test_f16_parity_deband(make_clip, f16_fmt):
    # Full deband + grain on every plane. The pcg3d grain is pure integer
    # (identical across io types) and sampling widens f16->f32 exactly, so the whole
    # smoothing+grain pipeline is f32-identical; only the final store is f16. dither
    # is inert on half (placebo gates it to 8-bit), so it cannot perturb parity.
    src16, src32 = pair(make_clip, f16_fmt)
    planes = list(range(src16.format.num_planes))
    args = dict(iterations=1, threshold=6.0, radius=16.0, grain=4.0, planes=planes)
    out16 = src16.vszipcl.Deband(**args)
    out32 = src32.vszipcl.Deband(**args)
    assert_pixel_parity(out16, out32, tol=TOL)


@pytest.mark.parametrize(
    "f16_fmt",
    [vs.GRAYH, vs.YUV444PH, vs.RGBH],
    ids=["GRAYH", "YUV444PH", "RGBH"],
)
def test_f16_parity_deband_grain_only(make_clip, f16_fmt):
    # iterations=0 (grain only): isolates the grain PRNG + store. Grain is added in
    # f32 identically; the f16 store is the only divergence.
    src16, src32 = pair(make_clip, f16_fmt)
    planes = list(range(src16.format.num_planes))
    args = dict(iterations=0, grain=8.0, planes=planes)
    out16 = src16.vszipcl.Deband(**args)
    out32 = src32.vszipcl.Deband(**args)
    assert_pixel_parity(out16, out32, tol=TOL)
