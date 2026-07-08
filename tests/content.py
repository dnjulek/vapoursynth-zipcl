"""Deterministic synthetic test content (numpy), since vszipcl has no ImageRead.

Builds a single 640x320 RGB24 frame combining smooth gradients (a banding source
for Deband), high-frequency sinusoids + hard rectangles (edges for EEDI3 /
Bilateral) and seeded fine noise (texture for NLMeans). Everything is seeded, so
the content — and therefore every golden — is reproducible run to run and across
machines. conftest converts this RGB base to any format/geometry via resize,
exactly like the vszip suite converts its ImageRead source.
"""

import numpy as np
import vapoursynth as vs

WIDTH = 640
HEIGHT = 320


def rgb_array(width: int = WIDTH, height: int = HEIGHT, seed: int = 1234, shift: int = 0) -> np.ndarray:
    """HxWx3 uint8 RGB. `shift` rolls the pattern down `shift` rows, giving
    deterministic inter-frame motion for temporal filters."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    rng = np.random.default_rng(seed)
    r = xx / (width - 1) * 255.0                                          # horizontal gradient -> banding
    g = yy / (height - 1) * 255.0                                         # vertical gradient
    b = 128.0 + 96.0 * np.sin(2 * np.pi * xx / 37.0) * np.cos(2 * np.pi * yy / 29.0)  # detail -> edges
    noise = rng.normal(0.0, 8.0, (height, width))                        # texture -> denoise
    arr = np.stack([r, g + noise, b + noise], axis=-1)
    arr[40:120, 80:200] = (230, 40, 40)                                  # hard edges (edge-directed interp)
    arr[180:260, 300:480] = (30, 220, 60)
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if shift:
        arr = np.roll(arr, shift, axis=0)
    return arr


def rgb_clip(core: vs.Core, width: int = WIDTH, height: int = HEIGHT, seed: int = 1234, shift: int = 0) -> vs.VideoNode:
    """The synthetic RGB array as a single-frame RGB24 clip."""
    arr = rgb_array(width, height, seed, shift)
    blank = core.std.BlankClip(None, width, height, vs.RGB24, length=1)

    def _fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        fout = f.copy()
        for p in range(3):
            np.asarray(fout[p])[:] = arr[:, :, p]
        return fout

    return blank.std.ModifyFrame(blank, _fill)
