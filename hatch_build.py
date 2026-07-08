import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging import tags

# One entry per shipped wheel. `zig_target` pins glibc 2.17 (manylinux) / the macOS 11 minos so the
# wheel is broadly compatible regardless of the runner; `platform_tag` is the wheel's platform tag;
# `os` selects the shared-library name that `zig build` emits (see LIBRARY_NAME).
TARGETS = {
    "x86_64-linux-gnu":  {"zig_target": "x86_64-linux-gnu.2.17",  "platform_tag": "manylinux_2_17_x86_64",  "os": "linux"},
    "aarch64-linux-gnu": {"zig_target": "aarch64-linux-gnu.2.17", "platform_tag": "manylinux_2_17_aarch64", "os": "linux"},
    "aarch64-macos":     {"zig_target": "aarch64-macos.11.0",     "platform_tag": "macosx_11_0_arm64",      "os": "macos"},
    "x86_64-macos":      {"zig_target": "x86_64-macos.11.0",      "platform_tag": "macosx_11_0_x86_64",     "os": "macos"},
    "x86_64-windows":    {"zig_target": "x86_64-windows",         "platform_tag": "win_amd64",              "os": "windows"},
}

LIBRARY_NAME = {
    "windows": "vszipcl.dll",
    "macos": "libvszipcl.dylib",
    "linux": "libvszipcl.so",
}

# Host OS -> TARGETS `os`, for a plain (non-ZTARGET) local build.
HOST_OS = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")


class CustomHook(BuildHookInterface[Any]):
    """Compile the plugin with Zig and place its shared library in the wheel's plugin dir."""

    source_dir = Path("zig-out")
    target_dir = Path("vapoursynth/plugins/vszipcl")

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        build_data["pure_python"] = False
        self.target_dir.mkdir(parents=True, exist_ok=True)

        zig_cmd = [sys.executable, "-m", "ziglang", "build", "-Doptimize=ReleaseFast"]

        # CI passes ZTARGET (e.g. "x86_64-linux-gnu") to build + tag a specific wheel; a bare local
        # build targets this machine and tags with the host platform.
        ztarget = os.environ.get("ZTARGET")
        if ztarget is not None:
            try:
                target = TARGETS[ztarget]
            except KeyError:
                raise ValueError(
                    f"Unsupported ZTARGET {ztarget!r}; expected one of: {', '.join(TARGETS)}"
                ) from None
            build_data["tag"] = f"py3-none-{target['platform_tag']}"
            zig_cmd.append(f"-Dtarget={target['zig_target']}")
            os_name = target["os"]
        else:
            build_data["tag"] = f"py3-none-{next(tags.platform_tags())}"
            os_name = HOST_OS

        subprocess.run(zig_cmd, check=True)

        # Copy exactly this OS's plugin library into the wheel. A missing file fails the build
        # loudly (e.g. the OpenCL loader didn't link) instead of silently shipping an empty,
        # unloadable wheel.
        lib_name = LIBRARY_NAME[os_name]
        matches = sorted(self.source_dir.rglob(lib_name))
        if not matches:
            raise RuntimeError(
                f"Zig build produced no {lib_name} under {self.source_dir}/ — the plugin failed to "
                f"compile/link (missing OpenCL loader for {os_name}?)."
            )
        shutil.copy2(matches[0], self.target_dir)

        manifest = self.target_dir / "manifest.vs"
        manifest.write_text(f"[VapourSynth Manifest V1]\n{Path(lib_name).stem}\n")

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        # The wheel is already assembled here; drop the whole staged tree (vapoursynth/…) so the
        # source checkout stays clean. parents[1] is "vapoursynth/" (parents[0] is ".../plugins").
        shutil.rmtree(self.target_dir.parents[1], ignore_errors=True)
