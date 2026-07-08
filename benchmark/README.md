# benchmark/ — vszipcl speed tests

One `*.vpy` per (filter, implementation). Each renders a **filter-bound 1080p YUV420** source
(a flat `BlankClip`, so decode ≈ 0 and the filter dominates — none of these filters early-out on
flat content) and sets one output node, so you can time it with:

```powershell
vspipe benchmark/deband_vszipcl.vpy .      # `.` discards output; vspipe prints "Output N frames in T s (F fps)"
```

`run.ps1` sweeps every script over the sample formats and `num_streams`, and prints a table.

## Run everything

```powershell
pwsh benchmark/run.ps1                       # u16/f16/f32 x num_streams {1,2}, 2000 frames each
pwsh benchmark/run.ps1 -Frames 1000          # more frames = steadier fps (GPU boost + startup ramp)
pwsh benchmark/run.ps1 -Formats u16,f32      # subset of formats
pwsh benchmark/run.ps1 -Streams 1,2,4        # more stream counts
pwsh benchmark/run.ps1 -Csv benchmark/out.csv    # also write results to CSV
```

`run.ps1` runs `num_streams` 1 and 2 only for implementations that expose the knob
(`vszipcl`, `bilateralgpu`, `nlm_cuda`); the others (`placebo`, `knlm`, CPU `vszip`) run once
(`NS = -`). A combo the reference rejects (e.g. `bilateralgpu`/`nlm_cuda` on f16, or the f32-only
CPU `vszip` EEDI3 on u16/f16) is shown as **N/A** with the error reason.

## Scripts

| Script | Filter | vs |
|--------|--------|----|
| `bilateral_vszipcl.vpy` / `bilateral_bilateralgpu.vpy` | Bilateral | `bilateralgpu` (CUDA) |
| `deband_vszipcl.vpy` / `deband_placebo.vpy` | Deband | `placebo` (libplacebo/Vulkan) |
| `gaussblur1_vszipcl.vpy` / `gaussblur100_vszipcl.vpy` / `gaussblur250_vszipcl.vpy` | GaussBlur (σ = 1 / 100 / 250) | — (no reference; compares the sigma / code paths) |
| `eedi3_vszipcl.vpy` / `eedi3_vszip.vpy` | EEDI3 | `vszip` (CPU) |
| `eedi3h_vszipcl.vpy` / `eedi3h_vszip.vpy` | EEDI3H | `vszip` (CPU) |
| `nlmeans_vszipcl.vpy` / `nlmeans_nlm_cuda.vpy` / `nlmeans_knlm.vpy` | NLMeans | `nlm_cuda` (CUDA), `knlm` (OpenCL) |

Matched params (equivalent work on both sides): Bilateral `sigma_spatial=3, sigma_color=0.02`;
Deband `iterations=1, threshold=4, radius=16, grain=6, planes=[0]`; GaussBlur `sigma ∈ {1, 100, 250}`
(σ=1 is the fused small-radius path, σ=100/250 the large separable path); EEDI3(H)
`field=1, mdis=40, nrad=3`; NLMeans `d=1, a=2, s=4, h=1.2`.

## Environment knobs (set by `run.ps1`; scripts also run standalone with these defaults)

| Var | Default | Meaning |
|-----|---------|---------|
| `BENCH_FMT` | `u16` | `u16` / `f16` / `f32` → YUV420P16 / YUV420PH / YUV420PS |
| `BENCH_NS` | `1` | `num_streams` for the vszipcl-family filters |
| `BENCH_FRAMES` | `2000` | frames rendered / timed over |
| `BENCH_THREADS` | `8` | `core.num_threads` — **required for `num_streams` to scale**: vspipe pulls frames concurrently across these threads, so a single stream only ever runs one frame at a time |

## Notes

- Build the plugin first (`zig build -Doptimize=ReleaseFast`); the scripts load `zig-out/bin/vszipcl.dll`.
- Keep `BENCH_FRAMES` high (the default is 2000) — at low counts the per-run startup/JIT + GPU
  boost ramp dominate and the fps is noisy (a filter at ~800 fps renders 40 frames in 0.05 s).
- Pre-formatted result tables (this machine, RTX 3070 Ti) live in the wiki: **Benchmarks**.
