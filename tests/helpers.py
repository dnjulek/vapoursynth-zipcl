"""Stdlib-only helpers for the vszipcl test suite: frame/prop access and clip
comparison. (Unlike the vszip suite there are no PNG/BMP encoders — vszipcl has
no ImageRead; synthetic content is generated in content.py instead.)"""

import vapoursynth as vs


# --- frame access -----------------------------------------------------------


def props(clip: vs.VideoNode, n: int = 0) -> dict:
    with clip.get_frame(n) as f:
        return dict(f.props)


def pix(clip: vs.VideoNode, x: int, y: int, plane: int = 0, n: int = 0):
    with clip.get_frame(n) as f:
        return f[plane][y, x]


def plane_stats(clip: vs.VideoNode, ref: vs.VideoNode | None = None, plane: int = 0, n: int = 0) -> dict:
    return props(clip.std.PlaneStats(ref, plane=plane), n)


def avg(clip: vs.VideoNode, plane: int = 0, n: int = 0) -> float:
    return plane_stats(clip, plane=plane, n=n)["PlaneStatsAverage"]


def diff(a: vs.VideoNode, b: vs.VideoNode, plane: int = 0, n: int = 0) -> float:
    return plane_stats(a, b, plane=plane, n=n)["PlaneStatsDiff"]


def max_abs_diff(a: vs.VideoNode, b: vs.VideoNode, plane: int = 0, n: int = 0) -> float:
    """Largest per-pixel absolute difference, in pixel-value units.
    std.Expr has no half-float support, so don't call this on F16 clips
    (widen to F32 with resize.Point first)."""
    d = vs.core.std.Expr([a, b], "x y - abs")
    return plane_stats(d, plane=plane, n=n)["PlaneStatsMax"]


# --- clip assertions --------------------------------------------------------


def assert_same_clip(a: vs.VideoNode, b: vs.VideoNode, n: int | None = None) -> None:
    """a and b have identical format, dimensions and bit-identical pixels
    (frame props are deliberately not compared)."""
    assert a.format.id == b.format.id, f"format mismatch: {a.format.name} != {b.format.name}"
    assert (a.width, a.height) == (b.width, b.height)
    assert a.num_frames == b.num_frames
    frames = range(a.num_frames) if n is None else [n]
    for fn in frames:
        for plane in range(a.format.num_planes):
            d = diff(a, b, plane=plane, n=fn)
            assert d == 0.0, f"frame {fn} plane {plane}: PlaneStatsDiff={d}"


def repack(clip: vs.VideoNode) -> vs.VideoNode:
    """Bit-identical copy in freshly allocated, compactly strided frames.
    Two flips force real copies (a no-op resize may pass frames through), so
    `filter(cropped)` vs `filter(repack(cropped))` exercises the plugin's
    stride/offset handling (the H2D/D2H strided uploads especially)."""
    return clip.std.FlipVertical().std.FlipVertical()
