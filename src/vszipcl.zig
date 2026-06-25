pub const cl = @import("opencl");
pub const vapoursynth = @import("vapoursynth");
const std = @import("std");
const builtin = @import("builtin");
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;
const zon = @import("zon");

const bilateral = @import("bilateral.zig");
const gaussglur = @import("gaussglur.zig");
const eedi3 = @import("eedi3.zig");
const nlmeans = @import("nlmeans.zig");

const eedi3_sig = "clip:vnode;field:int;dh:int:opt;mdis:int:opt;nrad:int:opt;alpha:float:opt;beta:float:opt;gamma:float:opt;hp:int:opt;vcheck:int:opt;vthresh0:float:opt;vthresh1:float:opt;vthresh2:float:opt;sclip:vnode:opt;";

export fn VapourSynthPluginInit2(plugin: *vs.Plugin, vspapi: *const vs.PLUGINAPI) void {
    ZAPI.Plugin.config("com.julek.vszipcl", "vszipcl", "VapourSynth Zig Image Process OpenCL", zon.version, plugin, vspapi);
    ZAPI.Plugin.function("Bilateral", "clip:vnode;sigma_spatial:float[]:opt;sigma_color:float[]:opt;radius:int[]:opt;", "clip:vnode;", bilateral.create, plugin, vspapi);
    ZAPI.Plugin.function("GaussBlur", "clip:vnode;sigma:float[]:opt;", "clip:vnode;", gaussglur.create, plugin, vspapi);
    ZAPI.Plugin.function("EEDI3", eedi3_sig, "clip:vnode;", eedi3.createEEDI3, plugin, vspapi);
    ZAPI.Plugin.function("EEDI3H", eedi3_sig, "clip:vnode;", eedi3.createEEDI3H, plugin, vspapi);
    ZAPI.Plugin.function("NLMeans", "clip:vnode;d:int:opt;a:int:opt;s:int:opt;h:float:opt;wmode:int:opt;wref:float:opt;rclip:vnode:opt;", "clip:vnode;", nlmeans.create, plugin, vspapi);
}

/// [luma, chroma] stride
pub fn strideFromVi(vi: *const vs.VideoInfo) [2]u32 {
    const n: u32 = @divExact(vsFrameAlignment(), @as(u32, @intCast(vi.format.bytesPerSample)));
    const ssw: u3 = @intCast(vi.format.subSamplingW);
    return .{
        @intCast(vsh.ceilN(@intCast(vi.width), n)),
        @intCast(vsh.ceilN(@intCast(vi.width >> ssw), n)),
    };
}

/// VapourSynth aligns every frame's plane rows to `VSFrame::alignment` bytes,
/// which it picks once at startup from the *running* CPU: 64 bytes when AVX-512F
/// is available, otherwise 32 (and 32 on non-x86) — see `alignmentHelper()` in vscore.cpp.
pub fn vsFrameAlignment() u32 {
    if (comptime (builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .x86)) {
        // Replicates cpufeatures.cpp's `avx512_f` derivation exactly.
        // 1. CPUID.1:ECX must report OSXSAVE(27) + AVX(28).
        const leaf1 = cpuid(1, 0);
        const osxsave_avx: u32 = (1 << 27) | (1 << 28);
        if ((leaf1.ecx & osxsave_avx) != osxsave_avx) return 32;

        // 2. The OS must have enabled AVX state (XCR0 XMM+YMM bits).
        const xcr0 = getXCR0();
        if ((xcr0 & 0x06) != 0x06) return 32;

        // 3. AVX-512F (CPUID.7.0:EBX bit 16) AND OS-enabled AVX-512 state (XCR0
        //    opmask+ZMM bits) => 64-byte frames, matching VSFrame::alignment.
        const leaf7 = cpuid(7, 0);
        if ((leaf7.ebx & (1 << 16)) != 0 and (xcr0 & 0xE0) == 0xE0) return 64;
    }
    return 32;
}

const CpuidLeaf = struct { eax: u32, ebx: u32, ecx: u32, edx: u32 };

fn cpuid(leaf: u32, subleaf: u32) CpuidLeaf {
    var eax: u32 = undefined;
    var ebx: u32 = undefined;
    var ecx: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile ("cpuid"
        : [_] "={eax}" (eax),
          [_] "={ebx}" (ebx),
          [_] "={ecx}" (ecx),
          [_] "={edx}" (edx),
        : [_] "{eax}" (leaf),
          [_] "{ecx}" (subleaf),
    );
    return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
}

/// Read XCR0 (low 32 bits hold every AVX / AVX-512 state bit we test).
fn getXCR0() u32 {
    return asm volatile (
        \\ xor %%ecx, %%ecx
        \\ xgetbv
        : [_] "={eax}" (-> u32),
        :
        : .{ .edx = true, .ecx = true });
}
