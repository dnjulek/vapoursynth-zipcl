"""Tests for vszipcl.Bilateral — the OpenCL port of VapourSynth-BilateralGPU.

Surface (from src/bilateral.zig create() + the wiki):
    clip.vszipcl.Bilateral(sigma_spatial=3.0, sigma_color=0.02, radius,
                           device_id=0, num_streams, use_shared_memory=1, ref)

Key differences from vszip's Bilateral (the STYLE exemplar, whose params differ
entirely): params are PER-PLANE float/int ARRAYS; there is NO `algorithm` /
`PBFICnum` and NO `planes` param, and NO small-frame rejection. A plane is
"processed" iff BOTH its sigma_spatial and sigma_color are >= FLT_EPSILON — so
copy-through / passthrough is expressed by setting a plane's sigma to 0 (there is
no planes= mask). Chroma sigma_spatial defaults to sigma_spatial[0]/sqrt(subW*subH).

Formats accepted: 8/16-bit int, 16-bit half, 32-bit float; Gray / YUV / RGB.

Golden (absolute-value) tests are @pytest.mark.golden (device-specific — CI runs
-m "not golden"). Everything else asserts device-independent RELATIONS only:
diff>0 (changed), diff==0 (passthrough/copy-through), bit-identity across
num_streams / repack, dimensions/format preservation, and param rejection.
"""

import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, avg, diff, max_abs_diff, repack


# --- golden snapshot cases (device-specific; -m "not golden" in CI) ----------

# Sweep one axis at a time around a GRAY16 base. sigma_color=0.1 keeps the range
# weight loose enough that smoothing is clearly visible in the golden stats.
CASES = (
    sweep(
        base_fmt=vs.GRAY16,
        base_args=dict(sigma_spatial=3, sigma_color=0.1),
        formats=[
            vs.GRAY8, vs.GRAY16, vs.GRAYH, vs.GRAYS,
            vs.YUV420P8, vs.YUV420P16, vs.YUV422P16, vs.YUV444P16,
            vs.RGB24, vs.RGBS,
        ],
        args=grid(sigma_spatial=[1, 3, 8], sigma_color=[0.02, 0.1])
        + [
            # force the global-memory kernel (both must match the sm kernel's math
            # away from borders); it also owns the border semantics vs the sm tile
            dict(sigma_spatial=3, sigma_color=0.1, use_shared_memory=0),
            # explicit radius (decouples the window from sigma_spatial*3)
            dict(sigma_spatial=3, sigma_color=0.1, radius=5),
            # large sigma -> radius 60 -> sm tile > 48 KB -> auto gl-kernel crossover
            dict(sigma_spatial=20, sigma_color=0.1),
        ],
        geometries=["odd", "tiny"],
    )
    + [
        # per-plane argument arrays (luma/chroma split)
        Case(vs.YUV420P16, args=dict(sigma_spatial=[3, 1.5], sigma_color=[0.1, 0.05])),
        Case(vs.YUV444P16, args=dict(sigma_spatial=[3, 3], sigma_color=[0.1, 0.1], radius=[5, 3])),
        # chroma copy-through via sigma 0 (no planes= param in this plugin)
        Case(vs.YUV420P16, args=dict(sigma_spatial=[3, 0], sigma_color=0.1)),
        # ref-clip (joint / cross bilateral): weights from ref, values from clip
        Case(vs.GRAY16, args=dict(sigma_spatial=3, sigma_color=0.1), variant="ref"),
        Case(vs.YUV420P8, args=dict(sigma_spatial=3, sigma_color=0.1), variant="ref"),
    ]
)


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    kwargs = dict(case.args)
    if case.variant == "ref":
        # a blurred reference: same format/dims as src (required by create())
        kwargs["ref"] = src.std.BoxBlur(hradius=5, vradius=5)
    golden.check("bilateral", case, src.vszipcl.Bilateral(**kwargs))


# --- device-independent contract tests ---------------------------------------


def test_blur_changes_output(to_gray):
    """A processed plane must actually change (bilateral smooths detail/noise)."""
    src = to_gray(vs.GRAY16)
    out = src.vszipcl.Bilateral(sigma_spatial=3, sigma_color=0.1)
    assert diff(out, src) > 0.0


def test_dimensions_and_format_preserved(make_clip):
    src = make_clip(vs.YUV420P16)
    out = src.vszipcl.Bilateral(sigma_spatial=3, sigma_color=0.1)
    assert (out.width, out.height) == (src.width, src.height)
    assert out.format.id == src.format.id
    assert out.num_frames == src.num_frames


def test_per_plane_processing(to_yuv):
    """All planes processed by default; a zero chroma sigma copies chroma through
    (this plugin has no planes= mask — sigma 0 is the passthrough switch)."""
    src = to_yuv(vs.YUV420P16)
    out = src.vszipcl.Bilateral(sigma_spatial=3, sigma_color=0.1)
    for p in range(3):
        assert diff(out, src, plane=p) > 0.0

    # sigma_spatial=[3,0] -> chroma (planes 1,2) unprocessed, copied verbatim.
    luma_only = src.vszipcl.Bilateral(sigma_spatial=[3, 0], sigma_color=0.1)
    assert diff(luma_only, src, plane=0) > 0.0
    assert diff(luma_only, src, plane=1) == 0.0
    assert diff(luma_only, src, plane=2) == 0.0


def test_sigma_zero_is_passthrough(to_gray):
    """A plane with sigma_spatial=0 OR sigma_color=0 is copied through bit-exact."""
    src = to_gray(vs.GRAY16)
    assert_same_clip(src.vszipcl.Bilateral(sigma_spatial=0), src)
    assert_same_clip(src.vszipcl.Bilateral(sigma_color=0), src)


def test_ref_changes_output(to_gray):
    """Joint/cross bilateral: supplying a different ref drives the range weights,
    so the output differs from the plain (ref=clip-implicit) bilateral."""
    src = to_gray(vs.GRAY16)
    args = dict(sigma_spatial=3, sigma_color=0.1)
    plain = src.vszipcl.Bilateral(**args)
    joint = src.vszipcl.Bilateral(ref=src.std.BoxBlur(hradius=5, vradius=5), **args)
    assert max_abs_diff(joint, plain) > 0.0


def test_use_shared_memory_both_paths_run(to_gray):
    """Both the sm (tiled) and gl (global) kernels must produce valid output; they
    differ only at borders, so we just assert each changes the image (their exact
    equality is device-specific and lives in the golden set)."""
    src = to_gray(vs.GRAY16)
    sm = src.vszipcl.Bilateral(sigma_spatial=3, sigma_color=0.1, use_shared_memory=1)
    gl = src.vszipcl.Bilateral(sigma_spatial=3, sigma_color=0.1, use_shared_memory=0)
    assert diff(sm, src) > 0.0
    assert diff(gl, src) > 0.0


def test_f16_runs(to_gray):
    """16-bit half is a self-defined extension (bilateralgpu rejects it)."""
    out = to_gray(vs.GRAYH).vszipcl.Bilateral(sigma_spatial=3, sigma_color=0.1)
    assert out.format.id == vs.GRAYH
    # widen to f32 to read stats (PlaneStats has no half support)
    assert 0.0 < avg(out.resize.Point(format=vs.GRAYS)) < 1.0


# --- determinism / stride --------------------------------------------------


def test_deterministic(to_gray):
    """Same args twice -> bit-identical output."""
    src = to_gray(vs.GRAY16)
    args = dict(sigma_spatial=3, sigma_color=0.1)
    assert_same_clip(src.vszipcl.Bilateral(**args), src.vszipcl.Bilateral(**args))


def test_num_streams_parity(to_gray):
    """num_streams=1 vs =4 must be bit-identical (device-independent)."""
    src = to_gray(vs.GRAY16)
    args = dict(sigma_spatial=3, sigma_color=0.1)
    one = src.vszipcl.Bilateral(num_streams=1, **args)
    four = src.vszipcl.Bilateral(num_streams=4, **args)
    assert max_abs_diff(one, four) == 0.0


def test_stride_handling(to_gray):
    """Cropped (non-trivially strided) input vs a compact repack must match —
    exercises the H2D/D2H strided staging."""
    cropped = to_gray(vs.GRAY16).std.Crop(left=27)
    args = dict(sigma_spatial=3, sigma_color=0.1)
    assert_same_clip(
        cropped.vszipcl.Bilateral(**args),
        repack(cropped).vszipcl.Bilateral(**args),
    )


# --- validation / rejection --------------------------------------------------


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(sigma_spatial=-1), r"sigma_spatial must be non-negative"),
        (dict(sigma_spatial=[3, -1]), r"sigma_spatial must be non-negative"),
        (dict(sigma_color=-0.5), r"sigma_color must be non-negative"),
        (dict(radius=0), r"radius must be positive"),
        (dict(radius=-5), r"radius must be positive"),
        (dict(radius=[9, 0]), r"radius must be positive"),
        (dict(device_id=-1), r"invalid device ID"),
        (dict(num_streams=0), r"num_streams must be 1\.\.32"),
        (dict(num_streams=33), r"num_streams must be 1\.\.32"),
    ],
)
def test_validation_errors(to_yuv, args, msg):
    # a 3-plane clip so per-plane chroma-index args (e.g. radius=[9,0]) are valid;
    # scalar-arg cases work on it too.
    src = to_yuv(vs.YUV444P16)
    with pytest.raises(vs.Error, match=msg):
        src.vszipcl.Bilateral(**args)


def test_ref_format_mismatch_rejected(to_gray):
    src = to_gray(vs.GRAY16)
    bad_ref = to_gray(vs.GRAY8)  # different bit depth -> rejected
    with pytest.raises(vs.Error, match=r'"ref" must be of the same format'):
        src.vszipcl.Bilateral(sigma_spatial=3, sigma_color=0.1, ref=bad_ref)


def test_unsupported_format_rejected(core):
    """10-bit integer is not an accepted io format (only 8/16 int, 16 half, 32 float)."""
    clip = core.std.BlankClip(None, 64, 64, vs.GRAY10, length=1, color=200)
    with pytest.raises(vs.Error, match=r"input bitdepth must be"):
        clip.vszipcl.Bilateral()
