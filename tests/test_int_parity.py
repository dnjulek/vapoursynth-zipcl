"""8-bit vs 16-bit integer bit-depth parity for every vszipcl filter that accepts
both integer depths (Bilateral, GaussBlur, EEDI3, NLMeans, Deband).

For each filter, run it on the SAME normalized content at 8- and 16-bit, divide
every output plane by its format peak (2^bits-1), and assert the two normalized
results agree within a few LSBs of the *lower* depth. This surfaces bit-depth
bugs (a hard-coded 255 peak, 16-bit overflow, scaling that only holds at one
depth) while the inherent lower-depth quantization stays inside tolerance.

vszipcl accepts ONLY 8- and 16-bit integer (NO 10/12-bit), so the depth families
are just two entries each: GRAY8/16, YUV420P8/16, RGB24/48.

To isolate the FILTER's depth handling from the INPUT quantization, content is
built ONCE at the family's 8-bit format and Point-upscaled to 16-bit (exact peak
scaling => identical normalized content at both depths), exactly like the vszip
deband parity test does. If each depth were resized independently from the RGB
source, the 8-bit input would be a 256-level staircase and the 16-bit input a
finer gradient — filtering those differently is expected, not a bug.

Every arg passed here is depth-INDEPENDENT (sigmas in pixels / 0..1 fractions;
deband threshold/grain in placebo's /1000 domain; eedi3 field=1 with f32-domain
alpha/beta/gamma), so the identical kwargs go to both depths. These tests are
device-independent (relations only) and therefore UNMARKED (run in CI).
"""

import pytest
import vapoursynth as vs

from helpers import max_abs_diff

# Depth families: (bits, format). vszipcl takes 8/16-bit int only (no 10-bit),
# so each family is exactly the low/high integer pair.
GRAY = [(8, vs.GRAY8), (16, vs.GRAY16)]
YUV420 = [(8, vs.YUV420P8), (16, vs.YUV420P16)]
RGB = [(8, vs.RGB24), (16, vs.RGB48)]


def peak(bits: int) -> int:
    return (1 << bits) - 1


def normalize(clip: vs.VideoNode, bits: int) -> vs.VideoNode:
    """Integer clip -> f32 in [0,1] (divide every plane by the format peak) so
    outputs at different depths compare on a common scale."""
    out = clip.format.replace(sample_type=vs.FLOAT, bits_per_sample=32).id
    return clip.std.Expr(f"x {peak(bits)} /", format=out)


def assert_pixel_parity(results, *, lsb: float = 4.0, n: int = 0) -> None:
    """results: list of (bits, clip). Normalize each output to [0,1] and compare
    the lower-depth result to the higher-depth one, per plane. Tolerance is `lsb`
    LSBs of the *lower* depth (its inherent quantization), so a correct filter
    passes and a scaling/overflow bug (diff far larger than a few LSB) fails."""
    results = sorted(results, key=lambda r: -r[0])
    hi_bits, hi_clip = results[0]
    refn = normalize(hi_clip, hi_bits)
    for bits, clip in results[1:]:
        cn = normalize(clip, bits)
        tol = lsb / peak(bits)
        for p in range(refn.format.num_planes):
            d = max_abs_diff(refn, cn, plane=p, n=n)
            assert d <= tol, f"{bits}-bit vs {hi_bits}-bit, plane {p}: max|Δnorm| = {d} > {tol} ({lsb} LSB)"


def depth_matched(make_clip, family):
    """Build content ONCE at the family's lowest (8-bit) format, then expand it UP
    to each higher depth with Point (exact peak scaling), so the NORMALIZED content
    is identical at every depth. Returns [(bits, fmt, clip), ...]."""
    _, base_fmt = min(family, key=lambda bf: bf[0])
    base = make_clip(base_fmt)
    out = []
    for bits, fmt in family:
        src = base if fmt == base_fmt else base.resize.Point(format=fmt)
        out.append((bits, fmt, src))
    return out


# --- Bilateral ---------------------------------------------------------------
# sigma_spatial is in pixels; sigma_color is a 0..1 fraction (the range weight
# never depends on bit depth — src/bilateral.zig loads integers as
# convert_float(v)*(1/peak), normalizing to [0,1] before the color Gaussian, and
# sigmas are NEVER depth-rescaled). So the same args go to both depths and the
# normalized outputs must match. A wrong/hard-coded peak would break that.
_BILATERAL_FAMILIES = [("gray", GRAY), ("yuv420", YUV420), ("rgb", RGB)]
_BILATERAL_ARGS = [
    ("smooth", dict(sigma_spatial=3, sigma_color=0.1)),
    ("sharp_range", dict(sigma_spatial=3, sigma_color=0.02)),
]


@pytest.mark.parametrize("family", [f for _, f in _BILATERAL_FAMILIES], ids=[n for n, _ in _BILATERAL_FAMILIES])
@pytest.mark.parametrize("args", [a for _, a in _BILATERAL_ARGS], ids=[n for n, _ in _BILATERAL_ARGS])
def test_int_parity_bilateral(make_clip, family, args):
    results = [(bits, src.vszipcl.Bilateral(**args)) for bits, _fmt, src in depth_matched(make_clip, family)]
    assert_pixel_parity(results)


# --- GaussBlur ---------------------------------------------------------------
# sigma is in pixels; the Gaussian is LINEAR, so integers run in the RAW code
# value domain and the output is a normalized weighted average (per-depth peak
# derived, not hard-coded). Two sigmas exercise BOTH code paths: small fused smem
# (sigma<=~10.5, radius<=32) and large separable (sigma>~10.5). The chroma default
# sigma[0]/sqrt(subW*subH) is depth-independent, so YUV/RGB pass unchanged args.
_GAUSSBLUR_FAMILIES = [("gray", GRAY), ("yuv420", YUV420), ("rgb", RGB)]
_GAUSSBLUR_ARGS = [
    ("small_path", dict(sigma=[3.0])),
    ("large_path", dict(sigma=[20.0])),
]


@pytest.mark.parametrize("family", [f for _, f in _GAUSSBLUR_FAMILIES], ids=[n for n, _ in _GAUSSBLUR_FAMILIES])
@pytest.mark.parametrize("args", [a for _, a in _GAUSSBLUR_ARGS], ids=[n for n, _ in _GAUSSBLUR_ARGS])
def test_int_parity_gaussblur(make_clip, family, args):
    results = [(bits, src.vszipcl.GaussBlur(**args)) for bits, _fmt, src in depth_matched(make_clip, family)]
    assert_pixel_parity(results)


# --- EEDI3 -------------------------------------------------------------------
# field=1 with f32-domain params (alpha/beta/gamma; the internal /255 scaling is
# depth-independent). Integer io is full-range UNORM: load = convert_float(v)/peak,
# store = convert_*_sat_rte(x*peak). Two per-depth artifacts widen the cross-depth
# diff beyond the plain store-rounding LSB — BOTH rounding granularity, not scaling
# bugs: (1) documented (a+b)/2 half-code ties round differently per depth; (2) more
# significantly, NVIDIA evaluates the constant `/peak` as v*RN(1/peak), whose ~1-ULP
# per-depth difference (src/eedi3.zig) occasionally flips a discrete edge-direction
# DP decision at an edge, changing a single interpolated pixel by several codes.
# Measured worst here is ~5.8 LSB@8, so the tolerance is 12 LSB (~2x margin) — a real
# peak bug instead diverges by hundreds of LSB (a mis-scaled plane is ~half the range
# off), so this still fails hard on genuine depth bugs.
_EEDI3_FAMILIES = [("gray", GRAY), ("yuv420", YUV420), ("rgb", RGB)]
_EEDI3_ARGS = [
    ("field1", dict(field=1)),
    ("field1_dh", dict(field=1, dh=True)),
]


@pytest.mark.parametrize("family", [f for _, f in _EEDI3_FAMILIES], ids=[n for n, _ in _EEDI3_FAMILIES])
@pytest.mark.parametrize("args", [a for _, a in _EEDI3_ARGS], ids=[n for n, _ in _EEDI3_ARGS])
def test_int_parity_eedi3(make_clip, family, args):
    # per-depth DP-decision flips + (a+b)/2 tie rounding (see note above): 12-LSB tol.
    results = [(bits, src.vszipcl.EEDI3(**args)) for bits, _fmt, src in depth_matched(make_clip, family)]
    assert_pixel_parity(results, lsb=12.0)


@pytest.mark.parametrize("family", [f for _, f in _EEDI3_FAMILIES], ids=[n for n, _ in _EEDI3_FAMILIES])
def test_int_parity_eedi3h(make_clip, family):
    results = [(bits, src.vszipcl.EEDI3H(field=1)) for bits, _fmt, src in depth_matched(make_clip, family)]
    assert_pixel_parity(results, lsb=12.0)


# --- NLMeans -----------------------------------------------------------------
# h is a depth-independent strength (the patch distance is normalized by the
# depth-independent 255^2 norm in both KNLMeansCL and this port); integer decode
# is convert_float(v)/peak. YUV default channels="Y" denoises luma and copies
# chroma through (copy-through stays exact under the identical normalized content),
# RGB default denoises all planes. Same args at both depths => parity.
_NLMEANS_FAMILIES = [("gray", GRAY), ("yuv420", YUV420), ("rgb", RGB)]
_NLMEANS_ARGS = [
    ("weak", dict(d=0, h=0.6)),
    ("strong", dict(d=0, h=4.0)),
]


@pytest.mark.parametrize("family", [f for _, f in _NLMEANS_FAMILIES], ids=[n for n, _ in _NLMEANS_FAMILIES])
@pytest.mark.parametrize("args", [a for _, a in _NLMEANS_ARGS], ids=[n for n, _ in _NLMEANS_ARGS])
def test_int_parity_nlmeans(make_clip, family, args):
    results = [(bits, src.vszipcl.NLMeans(**args)) for bits, _fmt, src in depth_matched(make_clip, family)]
    assert_pixel_parity(results)


# --- Deband ------------------------------------------------------------------
# threshold/radius/grain are placebo's /1000-domain floats applied in normalized
# space regardless of depth (src/deband.zig: setup_src scale stays 1.0 for UNORM
# input, so the shader runs unchanged); integer decode is convert_float(v)/peak.
# dither is 8-bit-ONLY (and ignored on 16-bit), so it is turned OFF here to isolate
# the shared deband path across depths. grain=0 + the frame-0 seed makes the output
# deterministic; the ~1.6-LSB UNORM store quantization at 8-bit stays under the
# default tolerance. Content is depth-matched (banding built once at 8-bit).
_DEBAND_FAMILIES = [("gray", GRAY), ("yuv420", YUV420), ("rgb", RGB)]
_DEBAND_ARGS = [
    ("light", dict(iterations=1, threshold=6.0, radius=16.0, grain=0.0, dither=0)),
    ("heavy", dict(iterations=2, threshold=10.0, radius=24.0, grain=0.0, dither=0)),
]


@pytest.mark.parametrize("family", [f for _, f in _DEBAND_FAMILIES], ids=[n for n, _ in _DEBAND_FAMILIES])
@pytest.mark.parametrize("args", [a for _, a in _DEBAND_ARGS], ids=[n for n, _ in _DEBAND_ARGS])
def test_int_parity_deband(make_clip, family, args):
    # deband every plane explicitly so chroma/RGB planes are exercised regardless
    # of the default (a fixed plane list also keeps this robust to default changes).
    matched = list(depth_matched(make_clip, family))
    kwargs = dict(planes=list(range(matched[0][2].format.num_planes)), **args)
    results = [(bits, src.vszipcl.Deband(**kwargs)) for bits, _fmt, src in matched]
    assert_pixel_parity(results)
