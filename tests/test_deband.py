"""Tests for vszipcl.Deband — the OpenCL port of libplacebo's pl_shader_deband
(the shader behind placebo.Deband). Params are COMPLETELY different from the
vszip exemplar's deband (thr/grain/sample_mode): this filter takes
iterations / threshold / radius / grain / planes / dither / dither_algo.

Semantics (from src/deband.zig):
  * iterations=0 disables the deband smoothing (grain-only pass); grain=0
    disables grain. iterations=0 AND grain=0 => exact passthrough of every
    processed plane (f32 store is identity; u8/u16 UNORM store round-trips).
  * planes is a list of plane indices to deband (default [0,1,2] = every plane).
    Selected planes are debanded; unselected planes are copied through (diff==0).
  * dither (default on for 8-bit) applies ONLY to 8-bit input and ONLY to the
    first processed plane; it is ignored for 16-bit/float.

Accepted io: 8/16-bit integer, 16-bit half (f16), 32-bit float; Gray/YUV/RGB.

Device note: golden snapshot tests capture absolute per-plane stats and are
therefore device-specific (@pytest.mark.golden, deselected in CI). Every other
test asserts only device-independent RELATIONS (diff>0 changed / diff==0
copied / passthrough / determinism / num_streams identity / format+param
rejection).
"""

import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, diff, max_abs_diff

# ── Golden sweep ────────────────────────────────────────────────────────────
# Base: a full deband+grain pass on 8-bit (so dither is active and dither_algo
# matters). The per-plane seed is n*P+rank & 0xFF — deterministic at frame 0 —
# so every (iterations, threshold, radius, grain, dither_algo) value yields a
# distinguishable, reproducible golden on the reference device.
BASE = dict(iterations=1, threshold=6.0, radius=16.0, grain=4.0)

CASES = (
    sweep(
        base_fmt=vs.GRAY8,
        base_args=BASE,
        formats=[
            vs.GRAY8, vs.GRAY16, vs.GRAYH, vs.GRAYS,
            vs.YUV420P8, vs.YUV420P16, vs.YUV444PS, vs.YUV422P8,
            vs.RGB24, vs.RGBS,
        ],
        args=grid(iterations=[0, 1, 2, 4])
        + grid(threshold=[1.0, 8.0])
        + grid(radius=[8.0, 32.0])
        + grid(grain=[0.0, 8.0])
        + grid(dither_algo=[0, 1, 2, 3]),  # 8-bit base -> dither engaged
        geometries=["odd", "tiny"],
    )
    + [
        # planes list: all three planes debanded (multi-plane families)
        Case(vs.YUV444PS, args=dict(planes=[0, 1, 2], **BASE)),
        Case(vs.YUV444P8, args=dict(planes=[0, 1, 2], **BASE)),
        Case(vs.YUV420P16, args=dict(planes=[0, 1, 2], **BASE)),
        Case(vs.YUV422P8, args=dict(planes=[0, 1, 2], **BASE)),
        Case(vs.RGB24, args=dict(planes=[0, 1, 2], **BASE)),
        Case(vs.RGBS, args=dict(planes=[0, 1, 2], **BASE)),
        # partial list: planes 0 and 2 (plane 1 unselected)
        Case(vs.YUV444PS, args=dict(planes=[0, 2], **BASE)),
        # pure grain (iterations=0) across an int + a float format
        Case(vs.GRAY8, args=dict(iterations=0, grain=8.0)),
        Case(vs.GRAYS, args=dict(iterations=0, grain=8.0)),
        # 8-bit with dither explicitly off (deband only, no dither bias)
        Case(vs.GRAY8, args=dict(dither=0, **BASE)),
    ]
)


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    golden.check("deband", case, src.vszipcl.Deband(**case.args))


# ── Banded fixtures (device-independent contract tests need visible banding) ─
# make_clip(GRAY8) quantizes the smooth synthetic gradient to 256 levels ->
# strong banding; widening to float/16-bit keeps the 8-bit-step banding the
# deband smoothing reacts to.


@pytest.fixture(scope="module")
def banded_u8(make_clip):
    return make_clip(vs.GRAY8)


@pytest.fixture(scope="module")
def banded_f32(make_clip):
    return make_clip(vs.GRAY8).resize.Point(format=vs.GRAYS)


# ── Passthrough / disable contracts ─────────────────────────────────────────


def test_passthrough_iter0_grain0_float(banded_f32):
    """iterations=0 (no deband) + grain=0 (no grain) => bit-exact passthrough
    of the processed plane (the f32 store is the identity)."""
    out = banded_f32.vszipcl.Deband(iterations=0, grain=0.0)
    assert_same_clip(out, banded_f32)


def test_passthrough_iter0_grain0_u8(banded_u8):
    """Same on 8-bit with dither disabled: the UNORM load->store round-trips an
    exact code, so an unmodified pixel comes back bit-identical."""
    out = banded_u8.vszipcl.Deband(iterations=0, grain=0.0, dither=0)
    assert_same_clip(out, banded_u8)


# ── Deband / grain actually change the output ───────────────────────────────


def test_deband_changes_output(banded_f32):
    """grain=0, iterations>=1: debanding smooths banding -> output differs."""
    out = banded_f32.vszipcl.Deband(iterations=2, threshold=10.0, radius=16.0, grain=0.0)
    assert diff(out, banded_f32) > 0.0


def test_grain_only_changes_output(banded_f32):
    """iterations=0, grain>0: grain-only pass still perturbs the output."""
    out = banded_f32.vszipcl.Deband(iterations=0, grain=8.0)
    assert diff(out, banded_f32) > 0.0


# ── planes list: selected planes change, unselected are copied ──────────────


@pytest.mark.parametrize(
    ("planes", "changed"),
    [
        ([0], (True, False, False)),
        ([1], (False, True, False)),
        ([2], (False, False, True)),
        ([0, 2], (True, False, True)),
        ([0, 1, 2], (True, True, True)),
    ],
)
def test_planes_list(make_clip, planes, changed):
    """Each index in `planes` selects that plane: selected planes are debanded
    (diff>0), unselected planes are bit-exact copies (diff==0)."""
    src = make_clip(vs.YUV444PS)
    out = src.vszipcl.Deband(planes=planes, iterations=1, threshold=8.0, grain=8.0)
    for p, want_change in enumerate(changed):
        d = diff(out, src, plane=p)
        if want_change:
            assert d > 0.0, f"plane {p} in planes={planes} but unchanged"
        else:
            assert d == 0.0, f"plane {p} not in planes={planes} but changed (diff={d})"


# ── dither: 8-bit only, first plane only ────────────────────────────────────


def test_dither_affects_8bit(banded_u8):
    """Dither engages on 8-bit input: on vs off produces a different result."""
    on = banded_u8.vszipcl.Deband(iterations=1, threshold=6.0, grain=0.0, dither=1)
    off = banded_u8.vszipcl.Deband(iterations=1, threshold=6.0, grain=0.0, dither=0)
    assert diff(on, off) > 0.0


def test_dither_ignored_on_16bit(make_clip):
    """placebo forces dither off for anything but 8-bit: on vs off is a no-op
    on 16-bit input (bit-identical)."""
    src = make_clip(vs.GRAY16)
    on = src.vszipcl.Deband(iterations=1, threshold=6.0, grain=0.0, dither=1)
    off = src.vszipcl.Deband(iterations=1, threshold=6.0, grain=0.0, dither=0)
    assert_same_clip(on, off)


# ── format / dimensions preserved ───────────────────────────────────────────


@pytest.mark.parametrize(
    "fmt",
    [vs.GRAY8, vs.GRAY16, vs.GRAYH, vs.GRAYS, vs.YUV420P8, vs.YUV444PS, vs.RGB24, vs.RGBS],
)
def test_format_and_dims_preserved(make_clip, fmt):
    src = make_clip(fmt)
    out = src.vszipcl.Deband(planes=list(range(src.format.num_planes)), **BASE)
    assert out.format.id == src.format.id
    assert (out.width, out.height) == (src.width, src.height)
    assert out.num_frames == src.num_frames


# ── determinism / num_streams identity ──────────────────────────────────────


def test_deterministic(banded_f32):
    """Same args twice => bit-identical output."""
    a = banded_f32.vszipcl.Deband(iterations=2, threshold=8.0, grain=6.0)
    b = banded_f32.vszipcl.Deband(iterations=2, threshold=8.0, grain=6.0)
    assert_same_clip(a, b)


@pytest.mark.parametrize("fmt", [vs.GRAYS, vs.GRAY8])
def test_num_streams_identity(make_clip, fmt):
    """num_streams=1 and =4 must be bit-identical (device-independent). Covers
    the 8-bit dither path (blue-noise LUT generation is deterministic)."""
    src = make_clip(fmt)
    one = src.vszipcl.Deband(iterations=2, threshold=8.0, grain=6.0, num_streams=1)
    four = src.vszipcl.Deband(iterations=2, threshold=8.0, grain=6.0, num_streams=4)
    assert max_abs_diff(one, four) == 0.0


# ── validation ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def clip16(core):
    return core.std.BlankClip(None, 64, 48, vs.GRAY16, length=1, color=30000)


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        # iterations range [0..32]
        (dict(iterations=-1), r"iterations must be 0\.\.32"),
        (dict(iterations=33), r"iterations must be 0\.\.32"),
        # threshold: finite, >= 0 (negatives AND non-finite rejected)
        (dict(threshold=-1.0), r"threshold must be a finite value >= 0"),
        (dict(threshold=float("inf")), r"threshold must be a finite value >= 0"),
        (dict(threshold=float("nan")), r"threshold must be a finite value >= 0"),
        # radius: finite, >= 0
        (dict(radius=-1.0), r"radius must be a finite value >= 0"),
        (dict(radius=float("nan")), r"radius must be a finite value >= 0"),
        (dict(radius=float("inf")), r"radius must be a finite value >= 0"),
        # grain: finite, >= 0
        (dict(grain=-1.0), r"grain must be a finite value >= 0"),
        (dict(grain=float("nan")), r"grain must be a finite value >= 0"),
        # dither_algo range [0..3]
        (dict(dither_algo=-1), r"dither_algo must be 0\.\.3"),
        (dict(dither_algo=4), r"dither_algo must be 0\.\.3"),
        # num_streams range [1..32]
        (dict(num_streams=0), r"num_streams must be 1\.\.32"),
        (dict(num_streams=33), r"num_streams must be 1\.\.32"),
        # device id
        (dict(device_id=-1), r"invalid device ID"),
        # planes: index list, rejected out-of-range or duplicated (clip16 is GRAY16, 1 plane)
        (dict(planes=[1]), r"plane index out of range"),
        (dict(planes=[-1]), r"plane index out of range"),
        (dict(planes=[0, 0]), r"plane specified twice"),
    ],
)
def test_validation_errors(clip16, args, msg):
    with pytest.raises(vs.Error, match=msg):
        clip16.vszipcl.Deband(**args).get_frame(0)


@pytest.mark.parametrize("fmt", [vs.GRAY10, vs.GRAY12, vs.GRAY14, vs.YUV420P10, vs.RGB30])
def test_unsupported_format_rejected(core, fmt):
    """Only 8/16-bit int, 16-bit half and 32-bit float are accepted; 10/12/14-
    bit integer input is rejected at create()."""
    src = core.std.BlankClip(None, 64, 48, fmt, length=1)
    with pytest.raises(vs.Error, match=r"input bitdepth must be"):
        src.vszipcl.Deband().get_frame(0)
