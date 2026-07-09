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
const deband = @import("deband.zig");

pub const io: std.Io = std.Io.Threaded.global_single_threaded.io();

pub const std_options: std.Options = .{ .log_level = .warn };
const eedi3_sig = "clip:vnode;field:int;dh:int:opt;mdis:int:opt;nrad:int:opt;alpha:float:opt;beta:float:opt;gamma:float:opt;hp:int:opt;vcheck:int:opt;vthresh0:float:opt;vthresh1:float:opt;vthresh2:float:opt;sclip:vnode:opt;device_id:int:opt;num_streams:int:opt;tune:int[]:opt;";

export fn VapourSynthPluginInit2(plugin: *vs.Plugin, vspapi: *const vs.PLUGINAPI) void {
    ZAPI.Plugin.config("com.julek.vszipcl", "vszipcl", "VapourSynth Zig Image Process OpenCL", zon.version, plugin, vspapi);
    ZAPI.Plugin.function("Bilateral", "clip:vnode;sigma_spatial:float[]:opt;sigma_color:float[]:opt;radius:int[]:opt;device_id:int:opt;num_streams:int:opt;use_shared_memory:int:opt;ref:vnode:opt;tune:int[]:opt;", "clip:vnode;", bilateral.create, plugin, vspapi);
    ZAPI.Plugin.function("GaussBlur", "clip:vnode;sigma:float[]:opt;device_id:int:opt;num_streams:int:opt;tune:int[]:opt;", "clip:vnode;", gaussglur.create, plugin, vspapi);
    ZAPI.Plugin.function("EEDI3", eedi3_sig, "clip:vnode;", eedi3.createEEDI3, plugin, vspapi);
    ZAPI.Plugin.function("EEDI3H", eedi3_sig, "clip:vnode;", eedi3.createEEDI3H, plugin, vspapi);
    ZAPI.Plugin.function("NLMeans", "clip:vnode;d:int:opt;a:int:opt;s:int:opt;h:float:opt;channels:data:opt;wmode:int:opt;wref:float:opt;rclip:vnode:opt;device_id:int:opt;num_streams:int:opt;tune:int[]:opt;", "clip:vnode;", nlmeans.create, plugin, vspapi);
    ZAPI.Plugin.function("Deband", "clip:vnode;iterations:int:opt;threshold:float:opt;radius:float:opt;grain:float:opt;planes:int[]:opt;dither:int:opt;dither_algo:int:opt;device_id:int:opt;num_streams:int:opt;tune:int[]:opt;", "clip:vnode;", deband.create, plugin, vspapi);
}

pub fn initContext(d: anytype, device_id: usize) !void {
    const a = std.heap.c_allocator;
    const platforms = try cl.getPlatforms(a);
    defer a.free(platforms);
    if (platforms.len == 0) return error.NoPlatforms;
    const platform = platforms[0];
    const devices = try platform.getDevices(a, cl.DeviceType.all);
    defer a.free(devices);
    if (devices.len == 0) return error.NoDevices;
    if (device_id >= devices.len) return error.InvalidDeviceID;
    d.device = devices[device_id];
    d.platform = platform;
    d.context = try cl.createContext(&.{d.device}, .{ .platform = d.platform });
}

pub fn tuneEntry(map_in: anytype, idx: usize) ?i64 {
    const v = map_in.getInt2(i64, "tune", idx) orelse return null;
    return if (v < 0) null else v;
}
pub fn deviceLocalMemSize(device: cl.Device) usize {
    var lmem: cl.c.cl_ulong = 0;
    if (cl.c.clGetDeviceInfo(device.id, cl.c.CL_DEVICE_LOCAL_MEM_SIZE, @sizeOf(cl.c.cl_ulong), &lmem, null) == cl.c.CL_SUCCESS and lmem > 0) {
        return @intCast(lmem);
    }
    return 32 * 1024;
}
pub fn deviceMaxWG(device: cl.Device) usize {
    var wg: usize = 0;
    if (cl.c.clGetDeviceInfo(device.id, cl.c.CL_DEVICE_MAX_WORK_GROUP_SIZE, @sizeOf(usize), &wg, null) == cl.c.CL_SUCCESS and wg > 0) {
        return wg;
    }
    return 256;
}
pub fn ceilTo(n: usize, m: usize) usize {
    return ((n + m - 1) / m) * m;
}
pub fn ndr(s: anytype, k: cl.Kernel, gws: []const usize, lws: []const usize) !void {
    if (cl.c.clEnqueueNDRangeKernel(s.queue.handle, k.handle, @intCast(gws.len), null, gws.ptr, lws.ptr, 0, null, null) != cl.c.CL_SUCCESS)
        return error.EnqueueKernel;
}

pub fn enqWrite(queue: cl.CommandQueue, mem: cl.c.cl_mem, offset: usize, src: []const u8) !void {
    if (cl.c.clEnqueueWriteBuffer(queue.handle, mem, cl.c.CL_FALSE, offset, src.len, src.ptr, 0, null, null) != cl.c.CL_SUCCESS)
        return error.EnqueueWrite;
}

pub fn enqRead(queue: cl.CommandQueue, mem: cl.c.cl_mem, offset: usize, dst: []u8) !void {
    if (cl.c.clEnqueueReadBuffer(queue.handle, mem, cl.c.CL_FALSE, offset, dst.len, dst.ptr, 0, null, null) != cl.c.CL_SUCCESS)
        return error.EnqueueRead;
}

pub fn strideFromVi(vi: *const vs.VideoInfo) [2]u32 {
    const n: u32 = @divExact(vsFrameAlignment(), @as(u32, @intCast(vi.format.bytesPerSample)));
    const ssw: u3 = @intCast(vi.format.subSamplingW);
    return .{
        @intCast(vsh.ceilN(@intCast(vi.width), n)),
        @intCast(vsh.ceilN(@intCast(vi.width >> ssw), n)),
    };
}

pub fn vsFrameAlignment() u32 {
    if (comptime (builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .x86)) {
        const leaf1 = cpuid(1, 0);
        const osxsave_avx: u32 = (1 << 27) | (1 << 28);
        if ((leaf1.ecx & osxsave_avx) != osxsave_avx) return 32;

        const xcr0 = getXCR0();
        if ((xcr0 & 0x06) != 0x06) return 32;

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

fn getXCR0() u32 {
    return asm volatile (
        \\ xor %%ecx, %%ecx
        \\ xgetbv
        : [_] "={eax}" (-> u32),
        :
        : .{ .edx = true, .ecx = true });
}
