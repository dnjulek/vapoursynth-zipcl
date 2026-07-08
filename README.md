# vapoursynth-zipcl

`vszipcl` is a VapourSynth plugin written in [Zig](https://ziglang.org/) that provides
**OpenCL-accelerated** video filters. Each filter is a faithful port of an established
reference, verified bit-exact where the reference allows, and shares one fast architecture
(a shared `cl_context`, a prewarmed per-stream resource pool, and a `num_streams` knob that
actually scales on the GPU).

All filters accept **8/16-bit integer, 16-bit half, and 32-bit float** input and are
**multi-plane** (Gray / YUV / RGB, subsampled chroma). Every filter also takes `device_id`
and `num_streams`.

[READ THE DOCS](https://github.com/dnjulek/vapoursynth-zipcl/wiki)

# FILTERS
- [Bilateral](https://github.com/dnjulek/vapoursynth-zipcl/wiki/Bilateral): OpenCL port of [VapourSynth-BilateralGPU](https://github.com/WolframRhodium/VapourSynth-BilateralGPU) — bit-exact on integer input.
- [Deband](https://github.com/dnjulek/vapoursynth-zipcl/wiki/Deband): OpenCL port of `placebo.Deband` ([libplacebo](https://code.videolan.org/videolan/libplacebo)'s `pl_shader_deband`).
- [EEDI3](https://github.com/dnjulek/vapoursynth-zipcl/wiki/EEDI3): Edge-directed interpolation (`EEDI3` vertical, `EEDI3H` horizontal).
- [GaussBlur](https://github.com/dnjulek/vapoursynth-zipcl/wiki/GaussBlur): Separable Gaussian blur.
- [NLMeans](https://github.com/dnjulek/vapoursynth-zipcl/wiki/NLMeans): Non-local means denoiser, matching [KNLMeansCL](https://github.com/Khanattila/KNLMeansCL).

# BENCHMARK

1080p YUV420, filter-bound, RTX 3070 Ti (`num_streams=2`). vszipcl leads every head-to-head, and
integer/half wire formats are often faster still (they move fewer bytes over PCIe). Full per-format
tables (u16/f16/f32 × `num_streams` 1 & 2) with methodology:
**[Benchmarks](https://github.com/dnjulek/vapoursynth-zipcl/wiki/Benchmarks)**.

| Filter | Reference | vszipcl vs reference |
|--------|-----------|----------------------|
| Bilateral | `bilateralgpu` (CUDA) | **2.2×** (u16), 1.2× (f32); f16 vszipcl-only |
| Deband | `placebo` (libplacebo / Vulkan) | **1.35–1.55×** (u16/f16/f32) |
| EEDI3 / EEDI3H | `vszip` (CPU) | **1.35×** (f32); u16/f16 vszipcl-only |
| NLMeans | `nlm_cuda` / `knlm` | **1.7×** / **2.0×**; f16 vs `knlm` only |
| GaussBlur | none installed | σ=1 ~522, σ=100 ~266, σ=250 ~148 fps (u16) |
