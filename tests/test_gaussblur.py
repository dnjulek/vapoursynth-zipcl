"""Tests for vszipcl.GaussBlur — a separable per-plane Gaussian blur.

Registration signature (src/gaussglur.zig):
    clip:vnode; sigma:float[]:opt; device_id:int:opt; num_streams:int:opt;

Key semantics exercised here (from the source):
  * `sigma` is a float[] resolved PER PLANE: plane0 default 0.5; chroma default
    sigma[0]/sqrt((1<<subW)*(1<<subH)); plane2 = plane1. sigma[i] < FLT_EPSILON
    => that plane is COPIED THROUGH byte-for-byte. All planes copy-through is a
    create() error.
  * TWO code paths by radius (radius = getGaussKernel(sigma).len/2, taps =
    ceil(6*sigma+1) forced odd): radius <= 32 => small fused smem kernel;
    radius > 32 => large separable two-pass kernel. radius<=32 <=> sigma <~10.5.
    Both paths are covered (small: sigma 2/5/10; large: sigma 20/80).
  * Accepted io: 8/16-bit int, 16-bit half (f16), 32-bit float; Gray/YUV/RGB.
    NO 10/12-bit int.

Golden (device-specific, @pytest.mark.golden) tests snapshot absolute per-plane
stats. Everything else is device-independent: relations (changed/passthrough),
dimensions, format/param rejection, determinism.
"""

import pytest
import vapoursynth as vs

from golden import Case, sweep
from helpers import assert_same_clip, avg, diff, max_abs_diff, plane_stats, repack

# --------------------------------------------------------------------------- #
# Golden snapshot cases (device-specific: @pytest.mark.golden).
#
# Axis-sweep around a small-path base (sigma=2). The `formats` axis covers every
# accepted io/family on the base sigma; the `args` axis sweeps sigma across BOTH
# code paths (small radius<=32 for sigma<=~10.5, large above); the `geometries`
# axis re-runs the base sigma at cropped sizes (kept small so the tiny 64x48
# crop never trips the radius>=dimension gate). Hand-picked cases below add
# per-plane subsets, multi-plane large-path, and integer large-path stores.
#
# 640x320 source => luma min(w,h)=320, so the large `radius <= dim-1` create()
# gate allows sigma up to ~106 (radius = ceil(6*sigma+1)/2). sigma=80 (radius
# 240) is the heavy large-path anchor; sigma=150 would be REJECTED here (radius
# 450 > 319), so it is deliberately not used.
# --------------------------------------------------------------------------- #
CASES = (
    sweep(
        base_fmt=vs.GRAYS,
        base_args=dict(sigma=[2.0]),
        formats=[
            vs.GRAY8, vs.GRAY16, vs.GRAYH, vs.GRAYS,
            vs.YUV420P8, vs.YUV420P16, vs.YUV444PS, vs.RGBS,
        ],
        args=[
            dict(sigma=[0.5]),   # small path (the plane-0 default)
            dict(sigma=[5.0]),   # small path
            dict(sigma=[10.0]),  # small path, just below the radius<=32 crossover
            dict(sigma=[20.0]),  # large path (radius 60)
            dict(sigma=[80.0]),  # large path, heavy (radius 240)
        ],
        geometries=["odd", "tiny"],
    )
    + [
        # per-plane sigma on subsampled YUV: luma blurred, chroma copied through
        Case(vs.YUV420P16, args=dict(sigma=[2.0, 0.0, 0.0])),
        # multi-plane large path (all planes, runtime separable kernels)
        Case(vs.YUV444PS, args=dict(sigma=[20.0])),
        # integer large-path store (STOREI convert_uchar_sat_rte)
        Case(vs.GRAY8, args=dict(sigma=[80.0])),
        # RGB all-planes small path
        Case(vs.RGBS, args=dict(sigma=[3.0])),
    ]
)


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    golden.check("gaussblur", case, src.vszipcl.GaussBlur(**case.args))


# --------------------------------------------------------------------------- #
# Device-independent contract tests.
# --------------------------------------------------------------------------- #

# Formats safe for std.PlaneStats-based numeric assertions (no f16 — PlaneStats
# has no half support; f16 gets its own "runs" test + widened checks).
NUMERIC_GRAY = [vs.GRAY8, vs.GRAY16, vs.GRAYS]


def _minmax(clip: vs.VideoNode, plane: int = 0, n: int = 0):
    s = plane_stats(clip, plane=plane, n=n)
    return s["PlaneStatsMin"], s["PlaneStatsMax"]


@pytest.mark.parametrize("fmt", NUMERIC_GRAY)
@pytest.mark.parametrize("sigma", [2.0, 80.0])  # small path and large path
def test_output_format_and_dims(to_gray, fmt, sigma):
    """Both code paths preserve format and dimensions."""
    src = to_gray(fmt)
    out = src.vszipcl.GaussBlur(sigma=[sigma])
    assert out.format.id == src.format.id
    assert (out.width, out.height) == (src.width, src.height)
    assert out.num_frames == src.num_frames


@pytest.mark.parametrize("fmt", NUMERIC_GRAY)
@pytest.mark.parametrize("sigma", [2.0, 80.0])  # small path and large path
def test_blur_changes_and_compresses_range(to_gray, fmt, sigma):
    """A blur (a) actually changes the plane and (b) is a weighted average with
    weights summing to 1 over in-range (mirror-reflected) samples, so every
    output stays within the source [min, max]. Both hold on any device."""
    src = to_gray(fmt)
    out = src.vszipcl.GaussBlur(sigma=[sigma])
    assert diff(out, src) > 0.0  # output differs from input
    s_min, s_max = _minmax(src)
    o_min, o_max = _minmax(out)
    tol = 1e-3  # float rounding slack; integers are exact
    assert o_min >= s_min - tol
    assert o_max <= s_max + tol


@pytest.mark.parametrize("fmt", NUMERIC_GRAY)
def test_heavier_blur_diverges_more(to_gray, fmt):
    """Detail reduction is monotone in sigma: a heavier blur diverges further
    from the source than a lighter one (both > 0)."""
    src = to_gray(fmt)
    d_light = diff(src.vszipcl.GaussBlur(sigma=[1.0]), src)
    d_heavy = diff(src.vszipcl.GaussBlur(sigma=[6.0]), src)
    assert 0.0 < d_light < d_heavy


def test_f16_runs(to_gray):
    """f16 io runs (both paths) and produces a plausible in-range blur; measured
    after widening to f32 because PlaneStats has no half support."""
    src = to_gray(vs.GRAYH)
    for sigma in (2.0, 80.0):
        out = src.vszipcl.GaussBlur(sigma=[sigma])
        assert out.format.id == vs.GRAYH
        wide = out.resize.Point(format=vs.GRAYS)
        assert 0.0 < avg(wide) < 1.0


# --- per-plane sigma / passthrough ----------------------------------------- #


def test_luma_only_copies_chroma(to_yuv):
    """sigma=[x,0,0]: plane 0 blurred, chroma planes (sigma < FLT_EPSILON) copied
    through byte-for-byte."""
    src = to_yuv(vs.YUV420P16)
    out = src.vszipcl.GaussBlur(sigma=[2.0, 0.0, 0.0])
    assert diff(out, src, plane=0) > 0.0   # luma changed
    assert diff(out, src, plane=1) == 0.0  # chroma copied
    assert diff(out, src, plane=2) == 0.0
    # the blurred luma equals a standalone gray blur of the same plane
    y_out = out.std.ShufflePlanes(0, vs.GRAY)
    y_ref = src.std.ShufflePlanes(0, vs.GRAY).vszipcl.GaussBlur(sigma=[2.0])
    assert_same_clip(y_out, y_ref)


def test_chroma_default_blurs_all_planes(to_yuv):
    """A single sigma blurs chroma too (via the sigma[0]/sqrt(subW*subH)
    default), so all planes change."""
    src = to_yuv(vs.YUV420P16)
    out = src.vszipcl.GaussBlur(sigma=[3.0])
    for p in range(3):
        assert diff(out, src, plane=p) > 0.0


def test_rgb_all_planes_blurred(make_clip):
    src = make_clip(vs.RGBS)  # make_clip routes RGB without a YUV matrix
    out = src.vszipcl.GaussBlur(sigma=[3.0])
    assert out.format.id == vs.RGBS
    for p in range(3):
        assert diff(out, src, plane=p) > 0.0


# --- determinism / stride --------------------------------------------------- #


@pytest.mark.parametrize("fmt", [vs.GRAY8, vs.GRAYS, vs.YUV420P16])
@pytest.mark.parametrize("sigma", [2.0, 80.0])  # small and large paths
def test_deterministic_same_args(to_yuv, fmt, sigma):
    """Same args twice => bit-identical output."""
    src = to_yuv(fmt)
    a = src.vszipcl.GaussBlur(sigma=[sigma])
    b = src.vszipcl.GaussBlur(sigma=[sigma])
    assert_same_clip(a, b)


@pytest.mark.parametrize("sigma", [2.0, 80.0])  # small and large paths
def test_num_streams_parity(to_gray, sigma):
    """num_streams=1 and =4 are bit-identical (a light in-file check; the
    exhaustive parity suite lives elsewhere)."""
    src = to_gray(vs.GRAYS)
    a = src.vszipcl.GaussBlur(sigma=[sigma], num_streams=1)
    b = src.vszipcl.GaussBlur(sigma=[sigma], num_streams=4)
    assert max_abs_diff(a, b) == 0.0


@pytest.mark.parametrize("fmt", [vs.GRAY8, vs.GRAYS])
@pytest.mark.parametrize("sigma", [5.0, 40.0])  # small and large paths
def test_stride_handling(to_gray, fmt, sigma):
    """An odd-width, offset plane pointer blurs identically to a freshly packed
    copy — exercises the strided H2D/D2H against the baked stride."""
    cropped = to_gray(fmt).std.Crop(left=27)  # odd width + offset plane pointer
    a = cropped.vszipcl.GaussBlur(sigma=[sigma])
    b = repack(cropped).vszipcl.GaussBlur(sigma=[sigma])
    assert_same_clip(a, b)


# --- validation / format rejection ----------------------------------------- #


def test_reject_10bit_int(core):
    """No 10/12-bit int (only 8/16-bit integer, 16 half, 32 float)."""
    src = core.std.BlankClip(format=vs.YUV420P10, width=64, height=64, color=[0, 0, 0])
    with pytest.raises(vs.Error, match="input bitdepth must be"):
        src.vszipcl.GaussBlur(sigma=[2.0])


def test_reject_32bit_int(core):
    src = core.std.BlankClip(format=vs.GRAY32, width=64, height=64)
    with pytest.raises(vs.Error, match="input bitdepth must be"):
        src.vszipcl.GaussBlur(sigma=[2.0])


def test_reject_negative_sigma(to_gray):
    with pytest.raises(vs.Error, match="sigma must be a finite value"):
        to_gray(vs.GRAYS).vszipcl.GaussBlur(sigma=[-1.0])


def test_reject_nonfinite_sigma(to_gray):
    with pytest.raises(vs.Error, match="sigma must be a finite value"):
        to_gray(vs.GRAYS).vszipcl.GaussBlur(sigma=[float("inf")])


def test_reject_all_planes_copythrough(to_gray):
    """sigma < FLT_EPSILON on every plane => nothing to process."""
    with pytest.raises(vs.Error, match="nothing to process"):
        to_gray(vs.GRAYS).vszipcl.GaussBlur(sigma=[0.0])


def test_reject_sigma_too_large(core):
    """radius >= plane dimension is rejected (radius ~ 3*sigma; 100 > 64)."""
    src = core.std.BlankClip(format=vs.GRAYS, width=64, height=64, color=0.5)
    with pytest.raises(vs.Error, match="sigma too large for plane"):
        src.vszipcl.GaussBlur(sigma=[100.0])


@pytest.mark.parametrize("ns", [0, 33])
def test_reject_bad_num_streams(to_gray, ns):
    with pytest.raises(vs.Error, match="num_streams must be"):
        to_gray(vs.GRAYS).vszipcl.GaussBlur(sigma=[2.0], num_streams=ns)


def test_reject_negative_device_id(to_gray):
    with pytest.raises(vs.Error, match="invalid device ID"):
        to_gray(vs.GRAYS).vszipcl.GaussBlur(sigma=[2.0], device_id=-1)
