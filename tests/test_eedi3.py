"""Tests for vszipcl.EEDI3 / vszipcl.EEDI3H (OpenCL edge-directed interpolation).

EEDI3 reconstructs a missing field by a DP that finds the best non-crossing
warping between neighbour lines. EEDI3 interpolates rows (vertical); EEDI3H
interpolates columns by running the identical vertical pipeline over a
transposed copy, so EEDI3H is bit-exact to Transpose -> EEDI3 -> Transpose.

Differences from the vszip (CPU) exemplar this file was styled after:
  * `field` is REQUIRED (no default) and is passed on every call.
  * There is NO `mclip` param (vszip's edge-mask arg) — those tests are dropped;
    `sclip` (the vcheck override clip) IS present and tested.
  * vszipcl accepts u8 / u16 / f16 / f32 multi-plane input (Gray/YUV/RGB), not
    just f32 — so int/half formats run and appear in the golden sweeps. Only
    8/16-bit integer, 16-bit half and 32-bit float are legal (10/12-bit int is
    rejected).

DEVICE NOTE: golden snapshots pin the exact output of one OpenCL device and are
@pytest.mark.golden (CI runs -m "not golden"). Everything else is
device-independent: dimensions, frame counts, relational diffs (changed / equal),
determinism, and param/format rejection.

Geometry: dh=False needs an even interpolated axis (height for EEDI3, width for
EEDI3H). The base 640x320 "full" geometry is even on both axes, so goldens use it
exclusively; "odd"/"tiny" would make the relevant axis odd and get rejected.
"""

import numpy as np
import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, diff, max_abs_diff, repack

# Legal io formats (u8 / u16 / f16 / f32), across color families.
FLOAT_FMTS = [vs.GRAYS, vs.YUV420PS, vs.YUV444PS, vs.RGBS]
# Golden sweeps cover the float families plus a couple int formats + the half
# path (golden_stats widens f16 via Point before PlaneStats).
GOLDEN_FMTS = FLOAT_FMTS + [vs.GRAY8, vs.GRAY16, vs.GRAYH]

# --- golden case matrices ---------------------------------------------------

CASES = (
    sweep(
        base_fmt=vs.GRAYS,
        base_args=dict(field=1),
        formats=GOLDEN_FMTS,
        args=(
            grid(field=[0])            # opposite field
            + grid(dh=[True])          # double height (bob to full res)
            + grid(nrad=[0, 3], mdis=[40])  # cost radius / max search distance
            + grid(hp=[True])          # real half-pel (eedi3m ignores hp)
            + grid(vcheck=[0, 1, 3])   # reliability check off / min / max
            + grid(alpha=[0.4], beta=[0.3], gamma=[40.0])
            + grid(gamma=[0.0])
        ),
    )
    + [
        # double-rate (bob): field 2/3 doubles the frame count, frame 0 interpolated
        Case(vs.GRAYS, args=dict(field=2)),
        Case(vs.YUV420PS, args=dict(field=3, dh=False)),
        # strong edge connection (high alpha, low beta)
        Case(vs.GRAYS, args=dict(field=1, alpha=0.9, beta=0.05, gamma=2.0, mdis=30)),
    ]
)

# EEDI3H mirrors EEDI3 on the width axis (640 is even, so dh=False is valid).
CASES_H = (
    sweep(
        base_fmt=vs.GRAYS,
        base_args=dict(field=1),
        formats=FLOAT_FMTS + [vs.GRAY8],
        args=(
            grid(field=[0])
            + grid(dh=[True])
            + grid(nrad=[3], mdis=[40])
            + grid(hp=[True])
            + grid(vcheck=[0, 3])
        ),
    )
    + [Case(vs.GRAYS, args=dict(field=2))]
)


@pytest.fixture(scope="module")
def grays(make_clip):
    return make_clip(vs.GRAYS)


@pytest.fixture(scope="module")
def yuv(make_clip):
    return make_clip(vs.YUV444PS)


# --- golden snapshots (device-specific) -------------------------------------


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_eedi3(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    golden.check("eedi3", case, src.vszipcl.EEDI3(**case.args))


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES_H, ids=str)
def test_golden_eedi3h(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    golden.check("eedi3h", case, src.vszipcl.EEDI3H(**case.args))


# --- behavioral contract (device-independent) -------------------------------


def test_field_doubles_height(grays):
    out = grays.vszipcl.EEDI3(field=1, dh=True)
    assert (out.width, out.height) == (grays.width, grays.height * 2)


def test_dh_false_keeps_dimensions(grays):
    out = grays.vszipcl.EEDI3(field=1)
    assert (out.width, out.height) == (grays.width, grays.height)
    assert out.num_frames == grays.num_frames


def test_double_rate_doubles_frames(grays):
    clip = grays.std.Loop(4)  # 4 frames -> 8 after bobbing
    out = clip.vszipcl.EEDI3(field=2)
    assert out.num_frames == clip.num_frames * 2


def test_field_0_1_keep_frame_count(grays):
    for f in (0, 1):
        assert grays.vszipcl.EEDI3(field=f).num_frames == grays.num_frames


def test_eedi3h_doubles_width(grays):
    out = grays.vszipcl.EEDI3H(field=1, dh=True)
    assert (out.width, out.height) == (grays.width * 2, grays.height)


def test_eedi3h_matches_transpose_eedi3(grays):
    # EEDI3H runs the identical vertical kernel on a transposed copy, so it is
    # bit-exact to Transpose -> EEDI3 -> Transpose for every option combination.
    for kw in (
        dict(field=1),
        dict(field=0, vcheck=0),
        dict(field=1, dh=True),
        dict(field=1, hp=True, vcheck=3),
        dict(field=1, nrad=3, mdis=40),
    ):
        h = grays.vszipcl.EEDI3H(**kw)
        t = grays.std.Transpose().vszipcl.EEDI3(**kw).std.Transpose()
        assert max_abs_diff(h, t) == 0.0, kw


def test_all_planes_processed(yuv):
    out = yuv.vszipcl.EEDI3(field=1)
    for p in range(3):
        assert diff(out, yuv, plane=p) > 0.0  # every plane interpolated


def test_rgb_all_planes_processed(make_clip):
    src = make_clip(vs.RGBS)
    out = src.vszipcl.EEDI3(field=1)
    for p in range(3):
        assert diff(out, src, plane=p) > 0.0


def test_higher_mdis_changes_output(grays):
    lo = grays.vszipcl.EEDI3(field=1, mdis=1)
    hi = grays.vszipcl.EEDI3(field=1, mdis=40)
    assert max_abs_diff(lo, hi) > 0.0


def test_nrad_changes_output(grays):
    assert max_abs_diff(grays.vszipcl.EEDI3(field=1, nrad=0),
                        grays.vszipcl.EEDI3(field=1, nrad=3)) > 0.0


def test_hp_is_implemented(grays):
    # unlike eedi3m (where hp is a no-op), vszipcl actually does half-pel steps
    assert max_abs_diff(grays.vszipcl.EEDI3(field=1, hp=True),
                        grays.vszipcl.EEDI3(field=1, hp=False)) > 0.0


def test_vcheck_changes_output(grays):
    assert max_abs_diff(grays.vszipcl.EEDI3(field=1, vcheck=0),
                        grays.vszipcl.EEDI3(field=1, vcheck=3)) > 0.0


def test_float_output_is_finite(make_clip):
    # the 4-tap cubic legitimately overshoots the nominal range (ringing), like
    # eedi3m, and is intentionally not clamped; just assert nothing blows up.
    out = make_clip(vs.YUV444PS).vszipcl.EEDI3(field=1)
    for p in range(3):
        s = out.std.PlaneStats(plane=p).get_frame(0).props
        assert -2.0 < s["PlaneStatsMin"] <= s["PlaneStatsMax"] < 2.0


def test_kept_rows_are_lossless_with_dh(grays):
    # With dh=True the source rows are copied VERBATIM into the kept output rows;
    # only the newly created rows are interpolated. For field=0 the interpolated
    # rows are the even output rows, so the odd output rows 1,3,5,... reproduce
    # source rows 0,1,2,... exactly (a documented invariant of this filter).
    out = grays.vszipcl.EEDI3(field=0, dh=True)
    with out.get_frame(0) as of, grays.get_frame(0) as sf:
        kept = np.asarray(of[0])[1::2, :]      # odd output rows
        src = np.asarray(sf[0])
    assert np.array_equal(kept, src)


# --- determinism (device-independent) ---------------------------------------


def test_deterministic_same_args(grays):
    a = grays.vszipcl.EEDI3(field=1, mdis=20, vcheck=3)
    b = grays.vszipcl.EEDI3(field=1, mdis=20, vcheck=3)
    assert_same_clip(a, b)


def test_num_streams_parity(grays):
    # num_streams only affects concurrency, never the math: 1 vs 4 is bit-identical.
    one = grays.vszipcl.EEDI3(field=1, hp=True, vcheck=3, num_streams=1)
    four = grays.vszipcl.EEDI3(field=1, hp=True, vcheck=3, num_streams=4)
    assert max_abs_diff(one, four) == 0.0


def test_stride_handling(grays):
    # odd width (cropped) exercises the scalar tail of the per-line kernel; the
    # height stays even so dh=False is valid.
    cropped = grays.std.Crop(left=19)
    out_a = cropped.vszipcl.EEDI3(field=1, mdis=10)
    out_b = repack(cropped).vszipcl.EEDI3(field=1, mdis=10)
    assert_same_clip(out_a, out_b)


# --- sclip (vcheck override clip) -------------------------------------------


def test_sclip_changes_vcheck_output(grays):
    # with vcheck>0 the reliability check blends toward `cint`; sclip supplies a
    # custom cint, so a non-trivial sclip changes the result.
    sclip = grays.std.BoxBlur(hradius=4, vradius=4)
    with_sclip = grays.vszipcl.EEDI3(field=1, vcheck=3, sclip=sclip)
    no_sclip = grays.vszipcl.EEDI3(field=1, vcheck=3)
    assert max_abs_diff(with_sclip, no_sclip) > 0.0


# --- validation / format rejection ------------------------------------------


def test_10bit_input_rejected(make_clip):
    # only 8/16-bit int, 16-bit half and 32-bit float are legal; 10-bit int is not.
    with pytest.raises(vs.Error, match="bitdepth"):
        make_clip(vs.GRAY10).vszipcl.EEDI3(field=1).get_frame(0)


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(field=4), "field must be"),
        (dict(field=-1), "field must be"),
        (dict(field=2, dh=True), r"field must be 0 or 1 when dh"),
        (dict(field=1, alpha=1.5), r"alpha 0\.\.1"),
        (dict(field=1, alpha=-0.1), r"alpha 0\.\.1"),
        (dict(field=1, beta=-0.1), r"beta 0\.\.1"),
        (dict(field=1, beta=1.5), r"beta 0\.\.1"),
        (dict(field=1, alpha=0.8, beta=0.8), r"alpha\+beta"),
        (dict(field=1, gamma=-1.0), r"gamma >= 0"),
        (dict(field=1, nrad=4), "nrad"),
        (dict(field=1, nrad=-1), "nrad"),
        (dict(field=1, mdis=0), "mdis"),
        (dict(field=1, mdis=41), "mdis"),
        (dict(field=1, vcheck=4), "vcheck"),
        (dict(field=1, vcheck=-1), "vcheck"),
        (dict(field=1, vcheck=2, vthresh0=0.0), "vthresh"),
        (dict(field=1, num_streams=0), "num_streams"),
        (dict(field=1, num_streams=33), "num_streams"),
        (dict(field=1, device_id=-1), "device ID"),
    ],
)
def test_param_validation(grays, args, msg):
    with pytest.raises(vs.Error, match=msg):
        grays.vszipcl.EEDI3(**args).get_frame(0)


def test_odd_axis_rejected_without_dh(make_clip):
    # EEDI3 needs even height, EEDI3H needs even width when dh=False; both report
    # the same unified "interpolated axis must be mod 2" error.
    odd_h = make_clip(vs.GRAYS).std.Crop(bottom=1)
    with pytest.raises(vs.Error, match="interpolated axis must be mod 2"):
        odd_h.vszipcl.EEDI3(field=1).get_frame(0)
    odd_w = make_clip(vs.GRAYS).std.Crop(right=1)
    with pytest.raises(vs.Error, match="interpolated axis must be mod 2"):
        odd_w.vszipcl.EEDI3H(field=1).get_frame(0)


def test_sclip_mismatch_rejected(grays):
    wrong = grays.std.AddBorders(right=2)
    with pytest.raises(vs.Error, match="sclip"):
        grays.vszipcl.EEDI3(field=1, vcheck=2, sclip=wrong).get_frame(0)


@pytest.mark.parametrize(
    "fmt",
    [vs.GRAY8, vs.GRAY16, vs.GRAYH, vs.GRAYS,
     vs.YUV420P8, vs.YUV422PS, vs.YUV444PS, vs.RGB24, vs.RGBS],
)
def test_all_io_formats_run(make_clip, fmt):
    make_clip(fmt).vszipcl.EEDI3(field=1).get_frame(0)
