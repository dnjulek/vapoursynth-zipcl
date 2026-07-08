"""Tests for vszipcl.NLMeans — the OpenCL KNLMeansCL-style non-local means denoiser.

Registration (authoritative):
  NLMeans(clip[, d, a, s, h, channels, wmode, wref, rclip, device_id, num_streams])

Layout mirrors the vszip test_bilateral exemplar: a golden snapshot sweep
(@pytest.mark.golden, device-specific) plus device-independent contract and
validation tests (relations, determinism, num_streams parity, format rejection).

Semantics come from src/nlmeans.zig create():
  d 0..16, a 1..64, s 0..8, h > 0, wmode 0..3, wref >= 0, num_streams 1..32.
  channels: Gray -> "Y"/"auto"; YUV -> "Y"/"UV"/"YUV" ("YUV" needs 4:4:4); RGB -> "RGB"/"auto".
  io formats: 8/16-bit int, 16-bit half, 32-bit float only (no 10/12-bit).
  rclip: guide clip; must match clip's format/dims/frame count; rclip==clip reproduces
         the no-rclip output bit-identically (plumbing invariant).
"""

import pytest
import vapoursynth as vs

from golden import Case, grid, sweep
from helpers import assert_same_clip, avg, diff, max_abs_diff, repack


# --- golden snapshot cases --------------------------------------------------
#
# The sweep varies one axis at a time around a spatial (d=0) GRAY16 base so
# every `args` entry stays valid on Gray (channels is left at the auto default).
# channels/temporal/ref variants that need a specific format or clip setup are
# listed explicitly and dispatched in test_golden_cases below.
CASES = (
    sweep(
        base_fmt=vs.GRAY16,
        base_args=dict(d=0),
        formats=[
            vs.GRAY8, vs.GRAY16, vs.GRAYH, vs.GRAYS,
            vs.YUV420P8, vs.YUV420P16, vs.YUV444P16, vs.RGB24, vs.RGBS,
        ],
        args=grid(h=[0.4, 1.2, 4.0])
        + [
            dict(a=1, s=2),
            dict(a=3, s=1),
            dict(wmode=1),
            dict(wmode=2),
            dict(wmode=3),
            dict(wref=0.0),
            dict(wref=0.5),
        ],
        geometries=["odd", "tiny"],
    )
    + [
        # channels variants (each needs a color family the mode is valid for)
        Case(vs.YUV444P16, args=dict(d=0, channels="YUV")),  # joint 3-channel, 4:4:4 only
        Case(vs.YUV420P8, args=dict(d=0, channels="UV")),     # chroma-only (subsampled dims)
        Case(vs.YUV420P16, args=dict(d=0, channels="UV")),
        Case(vs.RGB24, args=dict(d=0, channels="RGB")),
        Case(vs.RGBS, args=dict(d=0, channels="RGB")),
        # temporal (d>0) — checked at the middle frame of the 3-frame shifted clip
        Case(vs.GRAY16, args=dict(d=1), variant="temporal"),
        Case(vs.GRAY16, args=dict(d=2), variant="temporal"),
        Case(vs.YUV420P8, args=dict(d=1), variant="temporal"),
        # (RGB temporal is skipped: make_temporal_clip applies matrix=1, invalid for RGB)
        Case(vs.YUV444P16, args=dict(d=1, channels="YUV"), variant="temporal"),
        # rclip (guide/oracle) split — weights from a blurred guide
        Case(vs.GRAY16, args=dict(d=0), variant="ref"),
        Case(vs.YUV420P8, args=dict(d=0), variant="ref"),
    ]
)


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, make_temporal_clip, case):
    kwargs = dict(case.args)
    if case.variant == "temporal":
        src = make_temporal_clip(case.fmt, case.geometry)
        golden.check("nlmeans", case, src.vszipcl.NLMeans(**kwargs), n=1)
    elif case.variant == "ref":
        src = make_clip(case.fmt, case.geometry)
        kwargs["rclip"] = src.std.BoxBlur(hradius=3, vradius=3)
        golden.check("nlmeans", case, src.vszipcl.NLMeans(**kwargs))
    else:
        src = make_clip(case.fmt, case.geometry)
        golden.check("nlmeans", case, src.vszipcl.NLMeans(**kwargs))


# --- contract tests (device-independent: relations only) --------------------


def test_output_format_and_frames_preserved(to_gray):
    src = to_gray(vs.GRAY16)
    out = src.vszipcl.NLMeans(d=0)
    assert out.format.id == src.format.id
    assert (out.width, out.height) == (src.width, src.height)
    assert out.num_frames == src.num_frames


def test_denoise_changes_output(to_gray):
    # The synthetic content carries seeded fine noise; NLM must actually alter it.
    src = to_gray(vs.GRAY16)
    out = src.vszipcl.NLMeans(d=0)
    assert diff(out, src) > 0.0


def test_higher_h_smooths_more(to_gray):
    # Stronger filtering (larger h) averages neighbours more heavily, so the
    # output moves further from the noisy source. This relation holds on any
    # correct implementation (it is not an absolute-value assertion).
    src = to_gray(vs.GRAYS)
    weak = src.vszipcl.NLMeans(d=0, h=0.3)
    strong = src.vszipcl.NLMeans(d=0, h=4.0)
    assert diff(strong, src) > diff(weak, src)


def test_search_radius_changes_output(to_gray):
    src = to_gray(vs.GRAYS)
    a1 = src.vszipcl.NLMeans(d=0, a=1)
    a4 = src.vszipcl.NLMeans(d=0, a=4)
    assert max_abs_diff(a1, a4) > 0.0


def test_wmode_changes_output(to_gray):
    src = to_gray(vs.GRAYS)
    w0 = src.vszipcl.NLMeans(d=0, wmode=0)
    w3 = src.vszipcl.NLMeans(d=0, wmode=3)
    assert max_abs_diff(w0, w3) > 0.0


def test_channels_luma_only_copies_chroma(to_yuv):
    # Default (auto == Y on YUV): luma denoised, chroma copied through untouched.
    src = to_yuv(vs.YUV420P8)
    out = src.vszipcl.NLMeans(d=0)
    assert diff(out, src, plane=0) > 0.0
    assert diff(out, src, plane=1) == 0.0
    assert diff(out, src, plane=2) == 0.0


def test_channels_uv_only_copies_luma(to_yuv):
    # channels="UV": chroma denoised, luma copied through untouched.
    src = to_yuv(vs.YUV420P8)
    out = src.vszipcl.NLMeans(d=0, channels="UV")
    assert diff(out, src, plane=0) == 0.0
    assert diff(out, src, plane=1) > 0.0
    assert diff(out, src, plane=2) > 0.0


def test_channels_case_insensitive(to_gray):
    # create() resolves channels via eqlIgnoreCase, so "y" == "Y".
    src = to_gray(vs.GRAY16)
    assert_same_clip(src.vszipcl.NLMeans(d=0, channels="y"), src.vszipcl.NLMeans(d=0, channels="Y"))


def test_temporal_uses_neighbours(make_temporal_clip):
    # On the 3-frame shifted clip the middle frame's neighbours differ from it,
    # so a temporal window (d=1) must produce a different result than spatial-only.
    src = make_temporal_clip(vs.GRAY16)
    spatial = src.vszipcl.NLMeans(d=0)
    temporal = src.vszipcl.NLMeans(d=1)
    assert max_abs_diff(spatial, temporal, n=1) > 0.0


# --- rclip (guide clip) -----------------------------------------------------


def test_rclip_self_is_identity(to_gray):
    # Plumbing invariant: rclip == the source must reproduce the no-rclip output
    # bit-for-bit (the ref buffer is bound to the distance input; with rclip==clip
    # the two inputs are byte-identical).
    src = to_gray(vs.GRAY16)
    assert_same_clip(src.vszipcl.NLMeans(d=0), src.vszipcl.NLMeans(d=0, rclip=src))


def test_rclip_self_is_identity_yuv(to_yuv):
    src = to_yuv(vs.YUV420P8)
    assert_same_clip(src.vszipcl.NLMeans(d=0, channels="UV"),
                     src.vszipcl.NLMeans(d=0, channels="UV", rclip=src))


def test_rclip_different_changes_output(to_gray):
    # A guide that differs from the source drives different weights -> different result.
    src = to_gray(vs.GRAYS)
    guide = src.std.BoxBlur(hradius=5, vradius=5)
    noref = src.vszipcl.NLMeans(d=0)
    withref = src.vszipcl.NLMeans(d=0, rclip=guide)
    assert max_abs_diff(noref, withref) > 0.0


# --- determinism / stream parity --------------------------------------------


def test_deterministic(to_gray):
    src = to_gray(vs.GRAY16)
    assert_same_clip(src.vszipcl.NLMeans(d=0, h=1.5), src.vszipcl.NLMeans(d=0, h=1.5))


def test_num_streams_parity(to_yuv):
    # num_streams=1 vs =4 must be bit-identical (documented invariant).
    src = to_yuv(vs.YUV420P8)
    s1 = src.vszipcl.NLMeans(d=0, num_streams=1)
    s4 = src.vszipcl.NLMeans(d=0, num_streams=4)
    assert_same_clip(s1, s4)


def test_num_streams_parity_temporal(make_temporal_clip):
    src = make_temporal_clip(vs.GRAY16)
    s1 = src.vszipcl.NLMeans(d=1, num_streams=1)
    s4 = src.vszipcl.NLMeans(d=1, num_streams=4)
    assert_same_clip(s1, s4)


def test_stride_handling(to_gray):
    # A non-trivially strided (cropped) frame must match a compactly repacked copy,
    # exercising the strided H2D/D2H path.
    cropped = to_gray(vs.GRAY16).std.Crop(left=27)
    assert_same_clip(cropped.vszipcl.NLMeans(d=0), repack(cropped).vszipcl.NLMeans(d=0))


def test_f16_runs(to_gray):
    out = to_gray(vs.GRAYH).vszipcl.NLMeans(d=0)
    assert out.format.id == vs.GRAYH
    assert 0.0 < avg(out.resize.Point(format=vs.GRAYS)) < 1.0


# --- validation errors ------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (dict(d=-1), r"d must be 0\.\.16"),
        (dict(d=17), r"d must be 0\.\.16"),
        (dict(a=0), r"a must be 1\.\.64"),
        (dict(a=65), r"a must be 1\.\.64"),
        (dict(s=-1), r"s must be 0\.\.8"),
        (dict(s=9), r"s must be 0\.\.8"),
        (dict(h=0), r"h must be > 0"),
        (dict(h=-1.0), r"h must be > 0"),
        (dict(wmode=-1), r"wmode must be 0\.\.3"),
        (dict(wmode=4), r"wmode must be 0\.\.3"),
        (dict(wref=-0.5), r"wref must be >= 0"),
        (dict(num_streams=0), r"num_streams must be 1\.\.32"),
        (dict(num_streams=33), r"num_streams must be 1\.\.32"),
        (dict(device_id=-1), r"invalid device ID"),
        (dict(channels="bogus"), r"'channels' must be 'Y' with Gray"),
        (dict(channels="UV"), r"'channels' must be 'Y' with Gray"),  # UV invalid on Gray
    ],
)
def test_validation_errors_gray(to_gray, args, msg):
    with pytest.raises(vs.Error, match=msg):
        to_gray(vs.GRAY16).vszipcl.NLMeans(**args)


def test_channels_yuv_requires_444(to_yuv):
    with pytest.raises(vs.Error, match=r"'channels'='YUV' requires 4:4:4"):
        to_yuv(vs.YUV420P8).vszipcl.NLMeans(channels="YUV")


def test_channels_invalid_on_yuv(to_yuv):
    with pytest.raises(vs.Error, match=r"'channels' must be 'YUV', 'Y' or 'UV' with YUV"):
        to_yuv(vs.YUV420P8).vszipcl.NLMeans(channels="RGB")


def test_channels_invalid_on_rgb(make_clip):
    with pytest.raises(vs.Error, match=r"'channels' must be 'RGB' with RGB"):
        make_clip(vs.RGB24).vszipcl.NLMeans(channels="Y")


def test_reject_unsupported_bitdepth(core):
    # 10-bit integer is not an accepted io format (only 8/16 int, 16 half, 32 float).
    fmt10 = core.query_video_format(vs.GRAY, vs.INTEGER, 10, 0, 0)
    src = core.std.BlankClip(None, 64, 64, fmt10.id, length=1, color=500)
    with pytest.raises(vs.Error, match=r"input bitdepth must be 8/16"):
        src.vszipcl.NLMeans()


def test_reject_search_window_larger_than_frame(core):
    # 2*a+1 must fit the frame; a=10 -> 21 > 8. (a passes its own 1..64 bound first.)
    src = core.std.BlankClip(None, 8, 8, vs.GRAY16, length=1, color=100)
    with pytest.raises(vs.Error, match=r"research window \(2\*a\+1\) larger than the frame"):
        src.vszipcl.NLMeans(a=10)


def test_reject_rclip_format_mismatch(to_gray):
    src = to_gray(vs.GRAY16)
    bad = to_gray(vs.GRAY8)  # different bit depth
    with pytest.raises(vs.Error, match=r"'rclip' must match the source clip"):
        src.vszipcl.NLMeans(rclip=bad)


def test_reject_rclip_dimension_mismatch(to_gray):
    src = to_gray(vs.GRAY16)
    bad = src.std.Crop(left=16)  # different width
    with pytest.raises(vs.Error, match=r"'rclip' must match the source clip"):
        src.vszipcl.NLMeans(rclip=bad)
