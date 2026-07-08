"""Smoke test: render one frame per filter using the installed wheel plugin.

Expects `pip install wheelhouse/*.whl` before this runs. VapourSynth auto-loads plugins
from site-packages — do not call LoadPlugin here.
"""
import glob
import os
import site
import sys

import vapoursynth as vs

library_suffixes = {".so", ".dll", ".dylib"}


def installed_plugin_path() -> str:
    for site_dir in site.getsitepackages() + ([site.getusersitepackages()] if site.getusersitepackages() else []):
        plugin_dir = os.path.join(site_dir, "vapoursynth", "plugins", "vszipcl")
        for path in sorted(glob.glob(os.path.join(plugin_dir, "*"))):
            if os.path.isfile(path) and os.path.splitext(path)[1] in library_suffixes:
                return path
    sys.exit("no vszipcl plugin in installed wheel (vapoursynth/plugins/vszipcl/)")


plugin_path = installed_plugin_path()
core = vs.core

if not hasattr(core, "vszipcl"):
    sys.exit(f"vszipcl not auto-loaded from installed wheel ({plugin_path})")

print(f"  using installed wheel: {plugin_path}")

src = core.std.BlankClip(width=256, height=256, format=vs.GRAYS, length=2, color=0.5)
u8 = src.resize.Point(format=vs.GRAY8)

checks = [
    ("Bilateral", lambda: core.vszipcl.Bilateral(src)),
    ("GaussBlur", lambda: core.vszipcl.GaussBlur(src, sigma=[2])),
    ("EEDI3", lambda: core.vszipcl.EEDI3(src, field=1)),
    ("EEDI3H", lambda: core.vszipcl.EEDI3H(src, field=1)),
    ("NLMeans", lambda: core.vszipcl.NLMeans(src, d=0)),
    ("Deband", lambda: core.vszipcl.Deband(u8, dither=1)),
]

for name, mk in checks:
    mk().get_frame(0)
    print(f"  ok: {name}")

print("smoke test passed")