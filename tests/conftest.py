"""Shared fixtures for the vszipcl (OpenCL) test suite.

The suite drives the freshly built plugin in-process (no vspipe). Build it first
with `zig build`, or point VSZIPCL_LIB at an existing library. Every filter runs
on an OpenCL device: if none is usable (e.g. a CI runner with no GPU/pocl), the
whole suite skips rather than fails.

Content is synthetic (numpy, see content.py) because vszipcl — unlike vszip —
has no ImageRead. The RGB base is converted to any format/geometry via resize,
matching the vszip fixtures so the per-filter tests read the same way.
"""

import os
import sys
from pathlib import Path

import pytest
import vapoursynth as vs

import content

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="regenerate tests/goldens/*.json from the current build instead of comparing",
    )


def pytest_configure(config):
    from golden import GoldenStore

    config.addinivalue_line("markers", "golden: device-specific snapshot test (deselect in CI with -m 'not golden')")
    config._golden_store = GoldenStore(config.getoption("--update-goldens"))


def pytest_sessionfinish(session, exitstatus):
    store = getattr(session.config, "_golden_store", None)
    if store is not None:
        store.save()


@pytest.fixture(scope="session")
def golden(request):
    return request.config._golden_store


def _plugin_path() -> Path:
    env = os.environ.get("VSZIPCL_LIB")
    if env:
        return Path(env)
    if sys.platform == "win32":
        return REPO_ROOT / "zig-out" / "bin" / "vszipcl.dll"
    suffix = "dylib" if sys.platform == "darwin" else "so"
    return REPO_ROOT / "zig-out" / "lib" / f"libvszipcl.{suffix}"


def _load_vszipcl() -> vs.Core:
    core = vs.core
    if not hasattr(core, "vszipcl"):
        path = _plugin_path()
        if not path.is_file():
            pytest.exit(
                f"vszipcl plugin not found at {path}; run `zig build` first or set VSZIPCL_LIB",
                returncode=2,
            )
        core.std.LoadPlugin(str(path))
    # Probe an OpenCL device: every filter needs one, so skip the whole suite
    # cleanly on a runner without a usable device (rather than erroring in every
    # test). A CI Linux box with pocl passes this; Windows/macOS runners without
    # an OpenCL runtime skip.
    try:
        probe = core.std.BlankClip(None, 16, 16, vs.GRAYS, length=1, color=0.5)
        probe.vszipcl.Deband(iterations=1, threshold=0.05, grain=0.0).get_frame(0)
    except vs.Error as e:
        # Skip from within the (session-scoped) fixture: the Skipped outcome is cached
        # for the session, so every test that needs `core` skips cleanly on a runner with
        # no usable OpenCL device (e.g. GHA Windows), rather than erroring in each test.
        pytest.skip(f"no usable OpenCL device (vszipcl probe failed: {e})")
    return core


@pytest.fixture(scope="session")
def core() -> vs.Core:
    return _load_vszipcl()


@pytest.fixture(scope="session")
def src_rgb(core: vs.Core) -> vs.VideoNode:
    """Single-frame 640x320 RGB24 synthetic image (gradients + edges + noise)."""
    return content.rgb_clip(core)


@pytest.fixture(scope="session")
def to_gray(src_rgb: vs.VideoNode):
    """Factory: the source as a GRAY clip in the given format."""

    def convert(fmt: int) -> vs.VideoNode:
        return src_rgb.resize.Bilinear(format=fmt, matrix=1).std.RemoveFrameProps("_Matrix")

    return convert


@pytest.fixture(scope="session")
def to_yuv(src_rgb: vs.VideoNode):
    """Factory: the source as a YUV clip in the given format."""

    def convert(fmt: int) -> vs.VideoNode:
        return src_rgb.resize.Bilinear(format=fmt, matrix=1)

    return convert


def _convert(src_rgb: vs.VideoNode, fmt: int) -> vs.VideoNode:
    f = vs.core.get_video_format(fmt)
    if f.color_family == vs.GRAY:
        return src_rgb.resize.Bilinear(format=fmt, matrix=1).std.RemoveFrameProps("_Matrix")
    if f.color_family == vs.YUV:
        return src_rgb.resize.Bilinear(format=fmt, matrix=1)
    return src_rgb if fmt == src_rgb.format.id else src_rgb.resize.Bilinear(format=fmt)


def _geometry(clip: vs.VideoNode, geometry: str) -> vs.VideoNode:
    """Geometry variants for golden cases. `odd` shaves the subsampling-mod
    minimum off each axis so width/height stop being multiples of the workgroup
    tile; `tiny` is a small interior crop that forces boundary/tail handling.
    (Filters with even-dimension or minimum-size requirements pick the geometries
    that are valid for them.)"""
    f = clip.format
    wmod, hmod = 1 << f.subsampling_w, 1 << f.subsampling_h
    if geometry == "full":
        return clip
    if geometry == "odd":
        return clip.std.Crop(right=wmod, bottom=hmod)
    if geometry == "tiny":
        return clip.std.CropAbs(width=64 - 64 % wmod, height=48 - 48 % hmod, left=200, top=100)
    raise ValueError(f"unknown geometry {geometry!r}")


@pytest.fixture(scope="session")
def make_clip(src_rgb: vs.VideoNode):
    """Factory: the source image in any format/geometry, cached per session."""
    cache: dict[tuple, vs.VideoNode] = {}

    def make(fmt: int, geometry: str = "full") -> vs.VideoNode:
        key = (int(fmt), geometry)
        if key not in cache:
            cache[key] = _geometry(_convert(src_rgb, fmt), geometry)
        return cache[key]

    return make


@pytest.fixture(scope="session")
def temporal_rgb(core: vs.Core) -> vs.VideoNode:
    """3-frame 640x320 RGB24 clip; each frame shifts the pattern down 2 rows,
    giving deterministic inter-frame motion for temporal NLMeans (d>0)."""
    frames = [content.rgb_clip(core, shift=2 * n) for n in range(3)]
    return frames[0] + frames[1] + frames[2]


@pytest.fixture(scope="session")
def make_temporal_clip(temporal_rgb: vs.VideoNode):
    """Factory: the 3-frame shifted clip in any format/geometry, cached per
    session (Point preserves the shifted detail temporal filters react to)."""
    cache: dict[tuple, vs.VideoNode] = {}

    def make(fmt: int, geometry: str = "full") -> vs.VideoNode:
        key = (int(fmt), geometry)
        if key not in cache:
            f = vs.core.get_video_format(fmt)
            clip = temporal_rgb.resize.Point(format=fmt, matrix=1)
            if f.color_family == vs.GRAY:
                clip = clip.std.RemoveFrameProps("_Matrix")
            cache[key] = _geometry(clip, geometry)
        return cache[key]

    return make
