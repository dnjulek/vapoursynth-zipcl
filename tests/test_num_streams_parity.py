"""num_streams parity — the codebase's SIGNATURE INVARIANT.

`num_streams` only controls how many per-frame OpenCL resource sets (Streams)
run concurrently; it must NEVER change the math. So for every filter and every
format/param combination the plugin guarantees

    filter(num_streams=1)  ==  filter(num_streams=4)   (bit-identical, all planes)

This is the single most important device-INDEPENDENT test in the suite: it holds
on any OpenCL device (it compares the plugin against itself), so it is UNMARKED
(CI runs it). The per-filter test files each carry a light in-file parity check;
this file is the consolidated cross-filter matrix — a representative sweep of
formats (a YUV multi-plane + an f32 + an f16 per filter) and 1-2 param configs
that exercise each filter's distinct code paths (eedi3 hp, nlmeans temporal d>0,
deband grain+dither, bilateral/gaussblur meaningful sigmas, both gaussblur
radius paths). Every plane (and, for temporal clips, every frame) is checked.

f16 note: std.Expr / std.PlaneStats have no half support, so max_abs_diff can't
run on an f16 clip directly. Point-widening f16->f32 is exact and preserves
bit-identity, so `_widen` lets the same 0-diff assertion cover the half formats.
"""

import pytest
import vapoursynth as vs

from helpers import max_abs_diff


# --- parity helper ----------------------------------------------------------


def _widen(clip: vs.VideoNode) -> vs.VideoNode:
    """f16 -> f32 (exact) so std.Expr/PlaneStats can read it; other formats
    pass through unchanged."""
    f = clip.format
    if f.sample_type == vs.FLOAT and f.bits_per_sample == 16:
        return clip.resize.Point(format=f.replace(bits_per_sample=32).id)
    return clip


def assert_stream_parity(one: vs.VideoNode, four: vs.VideoNode, frames=None) -> None:
    """num_streams=1 and =4 output must be bit-identical on every plane (and
    every frame of `frames`, defaulting to the whole clip)."""
    assert one.format.id == four.format.id, "format changed across num_streams"
    assert (one.width, one.height) == (four.width, four.height)
    assert one.num_frames == four.num_frames
    a, b = _widen(one), _widen(four)
    fr = range(a.num_frames) if frames is None else frames
    for n in fr:
        for p in range(a.format.num_planes):
            d = max_abs_diff(a, b, plane=p, n=n)
            assert d == 0.0, f"frame {n} plane {p}: num_streams 1 vs 4 differ (max|Δ|={d})"


# --- Bilateral --------------------------------------------------------------
# A YUV multi-plane, an f32 RGB (all planes processed), an f16, and a large
# sigma that crosses the sm->gl kernel boundary (radius 60 > 48 KB sm tile).
BILATERAL_CASES = [
    (vs.YUV420P16, dict(sigma_spatial=3, sigma_color=0.1)),
    (vs.RGBS, dict(sigma_spatial=3, sigma_color=0.1)),
    (vs.GRAYH, dict(sigma_spatial=3, sigma_color=0.1)),
    (vs.GRAY8, dict(sigma_spatial=20, sigma_color=0.1)),           # large -> gl kernel
    (vs.YUV444P16, dict(sigma_spatial=[3, 1.5], sigma_color=[0.1, 0.05])),  # per-plane arrays
]


@pytest.mark.parametrize("fmt,args", BILATERAL_CASES, ids=lambda v: getattr(v, "name", None))
def test_bilateral_parity(make_clip, fmt, args):
    src = make_clip(fmt)
    one = src.vszipcl.Bilateral(num_streams=1, **args)
    four = src.vszipcl.Bilateral(num_streams=4, **args)
    assert_stream_parity(one, four)


def test_bilateral_ref_parity(make_clip):
    """Joint/cross bilateral (ref clip drives the range weights) is also stream-
    invariant — the per-Stream ref buffer must be race-free."""
    src = make_clip(vs.GRAY16)
    ref = src.std.BoxBlur(hradius=5, vradius=5)
    args = dict(sigma_spatial=3, sigma_color=0.1, ref=ref)
    assert_stream_parity(src.vszipcl.Bilateral(num_streams=1, **args),
                         src.vszipcl.Bilateral(num_streams=4, **args))


# --- GaussBlur --------------------------------------------------------------
# Both code paths (small fused smem radius<=32, large separable radius>32) plus
# a YUV multi-plane, an f32, and an f16.
GAUSSBLUR_CASES = [
    (vs.YUV420P16, dict(sigma=[3.0])),          # multi-plane, small path (chroma default)
    (vs.GRAYS, dict(sigma=[2.0])),              # small fused smem path
    (vs.GRAYS, dict(sigma=[80.0])),             # large separable path
    (vs.GRAYH, dict(sigma=[2.0])),              # f16
    (vs.RGBS, dict(sigma=[3.0])),               # RGB all planes
]


@pytest.mark.parametrize("fmt,args", GAUSSBLUR_CASES, ids=lambda v: getattr(v, "name", None))
def test_gaussblur_parity(make_clip, fmt, args):
    src = make_clip(fmt)
    one = src.vszipcl.GaussBlur(num_streams=1, **args)
    four = src.vszipcl.GaussBlur(num_streams=4, **args)
    assert_stream_parity(one, four)


# --- EEDI3 ------------------------------------------------------------------
# field=1 base + hp (the real half-pel path), a YUV multi-plane, an f32, an f16,
# and an int format; dh doubles height. Full 640x320 geometry is even on both
# axes so dh=False is valid.
EEDI3_CASES = [
    (vs.GRAYS, dict(field=1)),
    (vs.GRAYS, dict(field=1, hp=True, vcheck=3)),     # half-pel + reliability check
    (vs.YUV444PS, dict(field=1)),                     # multi-plane
    (vs.GRAYH, dict(field=1)),                        # f16
    (vs.GRAY8, dict(field=1, dh=True)),               # int + double-height
]


@pytest.mark.parametrize("fmt,args", EEDI3_CASES, ids=lambda v: getattr(v, "name", None))
def test_eedi3_parity(make_clip, fmt, args):
    src = make_clip(fmt)
    one = src.vszipcl.EEDI3(num_streams=1, **args)
    four = src.vszipcl.EEDI3(num_streams=4, **args)
    assert_stream_parity(one, four)


# --- EEDI3H (transposed vertical pipeline) ----------------------------------
EEDI3H_CASES = [
    (vs.GRAYS, dict(field=1, hp=True, vcheck=3)),
    (vs.YUV420PS, dict(field=1)),                     # multi-plane (width 640/320 even)
    (vs.GRAYH, dict(field=1)),                        # f16
]


@pytest.mark.parametrize("fmt,args", EEDI3H_CASES, ids=lambda v: getattr(v, "name", None))
def test_eedi3h_parity(make_clip, fmt, args):
    src = make_clip(fmt)
    one = src.vszipcl.EEDI3H(num_streams=1, **args)
    four = src.vszipcl.EEDI3H(num_streams=4, **args)
    assert_stream_parity(one, four)


# --- NLMeans (spatial d=0) --------------------------------------------------
NLMEANS_CASES = [
    (vs.YUV420P8, dict(d=0)),                          # luma-only (auto), chroma copied
    (vs.YUV420P8, dict(d=0, channels="UV")),          # chroma-only
    (vs.GRAYS, dict(d=0, wmode=3)),                    # f32, alternate weight mode
    (vs.GRAYH, dict(d=0)),                             # f16
    (vs.RGB24, dict(d=0, channels="RGB")),            # RGB joint
]


@pytest.mark.parametrize("fmt,args", NLMEANS_CASES, ids=lambda v: getattr(v, "name", None))
def test_nlmeans_parity(make_clip, fmt, args):
    src = make_clip(fmt)
    one = src.vszipcl.NLMeans(num_streams=1, **args)
    four = src.vszipcl.NLMeans(num_streams=4, **args)
    assert_stream_parity(one, four)


# --- NLMeans (temporal d>0) -------------------------------------------------
# The 3-frame shifted clip drives a real temporal window; all frames checked so
# the per-Stream temporal band upload is proven race-free at the boundaries too.
NLMEANS_TEMPORAL_CASES = [
    (vs.GRAY16, dict(d=1)),
    (vs.GRAY16, dict(d=2)),
    (vs.YUV420P8, dict(d=1)),
    (vs.YUV444P16, dict(d=1, channels="YUV")),        # joint 3-channel temporal
]


@pytest.mark.parametrize("fmt,args", NLMEANS_TEMPORAL_CASES, ids=lambda v: getattr(v, "name", None))
def test_nlmeans_temporal_parity(make_temporal_clip, fmt, args):
    src = make_temporal_clip(fmt)
    one = src.vszipcl.NLMeans(num_streams=1, **args)
    four = src.vszipcl.NLMeans(num_streams=4, **args)
    assert_stream_parity(one, four)


def test_nlmeans_rclip_parity(make_clip):
    """rclip (guide) split is stream-invariant — the extra per-Stream ref buffer
    must not race."""
    src = make_clip(vs.GRAYS)
    guide = src.std.BoxBlur(hradius=5, vradius=5)
    assert_stream_parity(src.vszipcl.NLMeans(d=0, rclip=guide, num_streams=1),
                         src.vszipcl.NLMeans(d=0, rclip=guide, num_streams=4))


# --- Deband -----------------------------------------------------------------
# iterations>=1 + grain>0 is the fully active path; 8-bit engages dither (whose
# blue-noise LUT generation must be deterministic across Streams). A YUV
# multi-plane (planes=[0,1,2]), an f32, and an f16 round out the io coverage.
DEBAND_CASES = [
    (vs.GRAY8, dict(iterations=1, threshold=6.0, radius=16.0, grain=4.0)),    # dither engaged
    (vs.GRAY8, dict(iterations=2, threshold=8.0, grain=6.0, dither_algo=1)),  # bayer LUT dither
    (vs.YUV444PS, dict(planes=[0, 1, 2], iterations=2, threshold=8.0, radius=16.0, grain=6.0)),
    (vs.GRAYS, dict(iterations=1, threshold=6.0, radius=16.0, grain=4.0)),    # f32
    (vs.GRAYH, dict(iterations=1, threshold=6.0, radius=16.0, grain=8.0)),    # f16
    (vs.GRAY16, dict(iterations=2, threshold=8.0, grain=6.0)),                # 16-bit (no dither)
]


@pytest.mark.parametrize("fmt,args", DEBAND_CASES, ids=lambda v: getattr(v, "name", None))
def test_deband_parity(make_clip, fmt, args):
    src = make_clip(fmt)
    one = src.vszipcl.Deband(num_streams=1, **args)
    four = src.vszipcl.Deband(num_streams=4, **args)
    assert_stream_parity(one, four)
