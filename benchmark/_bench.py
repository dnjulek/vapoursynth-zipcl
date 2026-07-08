"""Shared source + config for the vspipe speed-test scripts in benchmark/.

Each *.vpy imports this, builds a filter-bound 1080p YUV420 source (a flat BlankClip — the source
is free, so the filter dominates the timing; none of these filters early-out on flat content) and
sets one output node. Run one directly with:

    vspipe benchmark/deband_vszipcl.vpy .

`.` discards the output (vspipe still renders every frame and prints the fps). Environment knobs
(run.ps1 sweeps these; the defaults make every file runnable on its own):

    BENCH_FMT      u16 | f16 | f32   (default u16)  -> YUV420P16 / YUV420PH / YUV420PS
    BENCH_NS       num_streams for the vszipcl-family filters (default 1)
    BENCH_FRAMES   frames to render / time over      (default 2000)
    BENCH_THREADS  core.num_threads = render parallelism (default 8) — REQUIRED for num_streams to
                   scale: vspipe pulls frames concurrently across these threads, so a single stream
                   only ever runs one frame at a time.
"""

import os
import pathlib
import vapoursynth as vs

core = vs.core
core.num_threads = int(os.environ.get("BENCH_THREADS", "8"))

_REPO = pathlib.Path(__file__).resolve().parents[1]
_FMT = {"u16": vs.YUV420P16, "f16": vs.YUV420PH, "f32": vs.YUV420PS}


def load_vszipcl():
    if not hasattr(core, "vszipcl"):
        core.std.LoadPlugin(str(_REPO / "zig-out" / "bin" / "vszipcl.dll"))


def ns() -> int:
    return int(os.environ.get("BENCH_NS", "1"))


def source(w: int = 1920, h: int = 1080) -> vs.VideoNode:
    fmt = _FMT[os.environ.get("BENCH_FMT", "u16")]
    frames = int(os.environ.get("BENCH_FRAMES", "2000"))
    return core.std.BlankClip(None, w, h, fmt, length=frames)
