# vszipcl test suite

---

## Quick start

```sh
# from the repo root
zig build                                  # build the Debug plugin -> zig-out/
pytest -v                                  # run the whole suite
pytest tests/test_bilateral.py             # one file
pytest -m "not golden"                     # device-independent subset (see below)
```

The suite needs `vapoursynth`, `numpy` and `pytest` importable, a built plugin
in `zig-out/` (or `VSZIPCL_LIB` pointing at one), **and a usable OpenCL device**.
With no device the whole suite skips (it does not fail), so CI runners without a
GPU/pocl stay green-but-skipped.

---

## OpenCL & goldens (read this)

vszipcl runs every filter on the GPU, so results depend on the OpenCL device.
Two consequences shape the suite:

- **Golden snapshot tests** (`@pytest.mark.golden`) store exact per-plane output
  stats. On one device the plugin is deterministic — `num_streams=1` and `=4` are
  bit-identical, and the kernels are JIT-compiled from source so Zig's Debug vs
  ReleaseFast does not change the math — so a committed golden is a stable
  regression anchor **for the reference GPU it was captured on** (an NVIDIA RTX
  3070 Ti). It is *not* portable to another device/driver (a GPU's SFU trig and
  rounding differ from pocl's CPU device). Regenerate goldens on the reference
  GPU and **always review `git diff tests/goldens/`** afterwards.
- **Everything else is device-independent** and runs anywhere with a device:
  behavioral contracts (dimensions, passthrough, determinism, all-planes),
  validation/format-rejection errors, and the parity suites — `num_streams` 1-vs-4
  bit-identity (`test_num_streams_parity.py`), integer u8/u16↔f32
  (`test_int_parity.py`), and f16↔f32 (`test_f16_parity.py`). CI runs
  `pytest -m "not golden"` so it exercises these on pocl without needing the
  reference GPU.

### Regenerating goldens

```sh
pytest --update-goldens          # rewrite tests/goldens/*.json from the CURRENT build/device
git diff tests/goldens/          # ALWAYS review
```

---

## Notes & gotchas

- **Build before testing.** The suite tests `zig-out/`, not the source — rebuild
  after any Zig change.
- **Content is synthetic** (numpy, `content.py`): gradients (banding), edges and
  seeded noise. vszipcl has no ImageRead, so there is no bundled photo.
- **Parity tolerances** are documented per test (integer parity is in LSBs of the
  lower depth; f16 parity is ~2 ULP of f16). A real depth/precision bug diverges
  by orders of magnitude past these bounds.
- **Sensitivity check** when adding goldens: tweak one stored value by ~1% and
  confirm the matching test fails, then revert.
