const std = @import("std");
const vszipcl = @import("vszipcl.zig");
const clpool = @import("clpool.zig");
const clframecache = @import("clframecache.zig");

const cl = vszipcl.cl;
const vapoursynth = vszipcl.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
const math = std.math;

const allocator = std.heap.c_allocator;

const kernels_text = @embedFile("bm3d_kernels.cl");

const FLT_EPSILON: f32 = 1.19209290e-7;

const MAX_RADIUS: i32 = 16;
const MAX_TW: usize = 2 * @as(usize, MAX_RADIUS) + 1;

const DEFAULT_WARPS: u32 = 2;

const ModKey = struct {
    w: i32,
    h: i32,
    stride: i32,
    sigma: [3]f32,
    block_step: i32,
    bm_range: i32,
    ps_num: i32,
    ps_range: i32,
    proc_mask: u32,
    bm_error: BmError,
    t2d: Transform,
    t1d: Transform,
};

const BmError = enum {
    ssd,
    sad,
    zssd,
    zsad,
    ssd_norm,

    fn parse(s: []const u8) ?BmError {
        if (eqlLower(s, "ssd")) return .ssd;
        if (eqlLower(s, "sad")) return .sad;
        if (eqlLower(s, "zssd")) return .zssd;
        if (eqlLower(s, "zsad")) return .zsad;
        if (eqlLower(s, "ssd/norm")) return .ssd_norm;
        return null;
    }
};

const Transform = enum {
    dct,
    haar,
    wht,
    bior1_5,

    fn parse(s: []const u8) ?Transform {
        if (eqlLower(s, "dct")) return .dct;
        if (eqlLower(s, "haar")) return .haar;
        if (eqlLower(s, "wht")) return .wht;
        if (eqlLower(s, "bior1.5")) return .bior1_5;
        return null;
    }
};

fn eqlLower(s: []const u8, lit: []const u8) bool {
    if (s.len != lit.len) return false;
    for (s, lit) |a, b| {
        if (std.ascii.toLower(a) != b) return false;
    }
    return true;
}

const Entry = struct {
    key: ModKey,
    mod_idx: usize,

    n_planes: u32,
    planes: [3]u8,
    plane_extent: usize,

    cache_off: usize,
    src_off: usize,
    src_elems: usize,
    res_off: usize,
    res_elems: usize,
    dst_off: usize,
    dst_elems: usize,
};

const Data = struct {
    node: ?*vs.Node = null,
    ref_node: ?*vs.Node = null,
    vi: *const vs.VideoInfo = undefined,
    vi_out: vs.VideoInfo = undefined,

    radius: i32 = 0,
    tw: usize = 1,

    extractor: f32 = 0,
    warps: u32 = DEFAULT_WARPS,
    chroma: bool = false,
    final: bool = false,
    zero_init: bool = true,
    process: [3]bool = .{ false, false, false },

    n_entries: usize = 0,
    entries: [3]Entry = undefined,

    sum_src: usize = 0,
    sum_res: usize = 0,
    sum_dst: usize = 0,

    use_pinned: bool = false,
    stage_down_off: usize = 0,
    stage_elems: usize = 0,

    scache: bool = false,
    cache: clframecache.FrameCache = .{},
    cache_elems: usize = 0,
    n_clips: usize = 1,

    n_mods: usize = 0,
    mod_src: [3][:0]u8 = undefined,

    platform: cl.Platform = undefined,
    device: cl.Device = undefined,
    context: cl.Context = undefined,

    pool: clpool.Pool(Stream, Data) = .{},

    fn normalOut(self: *const Data) bool {
        return self.radius == 0;
    }
};

const Stream = struct {
    programs: [3]cl.Program,
    n_prog: usize,
    queue: cl.CommandQueue,
    k_bm3d: [3]cl.Kernel,
    k_agg: [3]cl.Kernel,
    n_kern: usize,
    d_src: cl.Buffer(f32),
    d_res: cl.Buffer(f32),
    d_dst: cl.Buffer(f32),
    stage: ?cl.Buffer(u8),
    host: ?[]u8,

    pub fn init(self: *Stream, d: *Data) !void {
        self.n_prog = 0;
        self.n_kern = 0;
        errdefer self.unwind();

        self.queue = try cl.createCommandQueue(d.context, d.device, .{});
        errdefer self.queue.release();

        const fp_ok = deviceHasCrDivSqrt(d.device);
        const opts: [:0]const u8 = if (fp_ok)
            "-cl-std=CL1.2 -cl-denorms-are-zero -cl-fp32-correctly-rounded-divide-sqrt"
        else
            "-cl-std=CL1.2 -cl-denorms-are-zero";
        for (0..d.n_mods) |mi| {
            const prog = try cl.createProgramWithSource(d.context, d.mod_src[mi]);
            self.programs[mi] = prog;
            self.n_prog += 1;
            prog.build(&.{d.device}, opts) catch |err| {
                if (err == error.BuildProgramFailure) {
                    const log = try prog.getBuildLog(allocator, d.device);
                    defer allocator.free(log);
                    std.log.err("BM3D OpenCL build failed: {s}", .{log});
                }
                return err;
            };
        }
        for (0..d.n_mods) |mi| {
            self.k_bm3d[mi] = try cl.createKernel(self.programs[mi], "bm3d");
            self.n_kern += 1;
            errdefer {
                self.k_bm3d[mi].release();
                self.n_kern -= 1;
            }
            self.k_agg[mi] = try cl.createKernel(self.programs[mi], "aggregate");
        }

        self.d_src = try cl.createBuffer(f32, d.context, .{ .read_write = true }, d.sum_src);
        errdefer self.d_src.release();
        self.d_res = try cl.createBuffer(f32, d.context, .{ .read_write = true }, d.sum_res);
        errdefer self.d_res.release();
        self.d_dst = try cl.createBuffer(f32, d.context, .{ .read_write = true }, if (d.sum_dst > 0) d.sum_dst else 1);
        errdefer self.d_dst.release();

        self.stage = null;
        self.host = null;
        if (d.use_pinned and d.stage_elems > 0) blk: {
            const bytes = d.stage_elems * 4;
            const stage = cl.createBuffer(u8, d.context, .{ .alloc_host_ptr = true }, bytes) catch {
                std.log.warn("vszipcl BM3D: pinned staging alloc failed ({d} MB); using pageable transfers.", .{bytes >> 20});
                break :blk;
            };
            var map_err: cl.c.cl_int = cl.c.CL_SUCCESS;
            const map_ptr = cl.c.clEnqueueMapBuffer(self.queue.handle, stage.handle, cl.c.CL_TRUE, cl.c.CL_MAP_READ | cl.c.CL_MAP_WRITE, 0, bytes, 0, null, null, &map_err);
            if (map_err != cl.c.CL_SUCCESS or map_ptr == null) {
                stage.release();
                std.log.warn("vszipcl BM3D: pinned staging map failed; using pageable transfers.", .{});
                break :blk;
            }
            self.stage = stage;
            self.host = @as([*]u8, @ptrCast(map_ptr.?))[0..bytes];
        }
    }

    fn unwind(self: *Stream) void {
        var k = self.n_kern;
        while (k > 0) {
            k -= 1;
            self.k_agg[k].release();
            self.k_bm3d[k].release();
        }
        self.n_kern = 0;
        var p = self.n_prog;
        while (p > 0) {
            p -= 1;
            self.programs[p].release();
        }
        self.n_prog = 0;
    }

    pub fn deinit(self: *Stream) void {
        _ = cl.c.clFinish(self.queue.handle);
        if (self.host) |h| {
            _ = cl.c.clEnqueueUnmapMemObject(self.queue.handle, self.stage.?.handle, h.ptr, 0, null, null);
            _ = cl.c.clFinish(self.queue.handle);
        }
        if (self.stage) |st| st.release();
        self.d_dst.release();
        self.d_res.release();
        self.d_src.release();
        var k = self.n_kern;
        while (k > 0) {
            k -= 1;
            self.k_agg[k].release();
            self.k_bm3d[k].release();
        }
        self.queue.release();
        var p = self.n_prog;
        while (p > 0) {
            p -= 1;
            self.programs[p].release();
        }
    }
};

fn deviceHasCrDivSqrt(device: cl.Device) bool {
    var cfg: cl.c.cl_device_fp_config = 0;
    if (cl.c.clGetDeviceInfo(device.id, cl.c.CL_DEVICE_SINGLE_FP_CONFIG, @sizeOf(cl.c.cl_device_fp_config), &cfg, null) != cl.c.CL_SUCCESS) return false;
    return (cfg & cl.c.CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) != 0;
}

const ZFrame = @typeInfo(@TypeOf(ZAPI.initZFrame)).@"fn".return_type.?;
const ZFrameW = @typeInfo(@TypeOf(ZFrame.newVideoFrame)).@"fn".return_type.?;

const Window = struct {
    f: [2][MAX_TW]ZFrame,
    idx: [MAX_TW]i64,
    n_clips: usize,
    tw: usize,

    fn deinit(self: *const Window) void {
        for (0..self.n_clips) |c| {
            for (0..self.tw) |t| self.f[c][t].deinit();
        }
    }
};

const CacheWin = struct {
    n: usize = 0,
    keys: [2 * MAX_TW]i64 = undefined,
    idx: [2 * MAX_TW]usize = undefined,
    load: [2 * MAX_TW]bool = undefined,
    published: [2 * MAX_TW]bool = undefined,
};

const ndr = vszipcl.ndr;

fn fillZeroF32(queue: cl.CommandQueue, buf: cl.Buffer(f32), off_elems: usize, count_elems: usize) !void {
    var pat: f32 = 0;
    if (cl.c.clEnqueueFillBuffer(queue.handle, buf.handle, &pat, @sizeOf(f32), off_elems * 4, count_elems * 4, 0, null, null) != cl.c.CL_SUCCESS)
        return error.EnqueueFillBuffer;
}

fn uploadPlane(s: *Stream, buf: cl.c.cl_mem, dst_off: usize, stage_off: usize, want: usize, src: []const u8, ev_out: ?*cl.c.cl_event) !void {
    if (s.host != null and src.len == want and (stage_off * 4 + src.len) <= s.host.?.len) {
        const dstb = s.host.?[stage_off * 4 ..][0..src.len];
        @memcpy(dstb, src);
        if (cl.c.clEnqueueWriteBuffer(s.queue.handle, buf, cl.c.CL_FALSE, dst_off * 4, src.len, dstb.ptr, 0, null, ev_out) != cl.c.CL_SUCCESS)
            return error.EnqueueWrite;
    } else {
        if (cl.c.clEnqueueWriteBuffer(s.queue.handle, buf, cl.c.CL_FALSE, dst_off * 4, src.len, src.ptr, 0, null, ev_out) != cl.c.CL_SUCCESS)
            return error.EnqueueWrite;
    }
}

fn launchBm3d(d: *Data, s: *Stream, e: *const Entry, res_buf: cl.Buffer(f32), res_off: usize) !void {
    const k = s.k_bm3d[e.mod_idx];
    try k.setArg(@TypeOf(res_buf), 0, res_buf);
    try k.setArg(@TypeOf(s.d_src), 1, s.d_src);
    try k.setArg(c_int, 2, @intCast(res_off));
    try k.setArg(c_int, 3, @intCast(e.src_off));
    const w: usize = @intCast(e.key.w);
    const h: usize = @intCast(e.key.h);
    const bs: usize = @intCast(e.key.block_step);
    const groups_x = (w + 4 * bs - 1) / (4 * bs);
    const wgs_x = (groups_x + d.warps - 1) / d.warps;
    const gws = [2]usize{ wgs_x * 32 * d.warps, (h + bs - 1) / bs };
    const lws = [2]usize{ 32 * d.warps, 1 };
    try ndr(s, k, &gws, &lws);
}

fn process(d: *Data, s: *Stream, win: *const Window, dst: ZFrameW, cw: ?*CacheWin) !void {
    errdefer _ = cl.c.clFinish(s.queue.handle);

    const tw = d.tw;
    const n_clips = win.n_clips;

    if (cw) |cwin| {
        for (0..cwin.n) |ki| {
            if (!cwin.load[ki]) continue;
            const c = ki / tw;
            const t = ki % tw;
            const slot = &d.cache.slots[cwin.idx[ki]];
            if (slot.ev) |old| {
                _ = cl.c.clReleaseEvent(old);
                slot.ev = null;
            }
            var ev: cl.c.cl_event = null;
            for (0..d.n_entries) |ei| {
                const e = &d.entries[ei];
                for (0..e.n_planes) |i| {
                    const srcp = win.f[c][t].getReadSlice(e.planes[i]);
                    const coff = e.cache_off + i * e.plane_extent;
                    const last = ei == d.n_entries - 1 and i == e.n_planes - 1;
                    const hoff = e.src_off + ((c * e.n_planes + i) * tw + t) * e.plane_extent;
                    try uploadPlane(s, slot.buf.handle, coff, hoff, e.plane_extent * 4, srcp, if (last) &ev else null);
                }
            }
            slot.ev = ev;
            if (cl.c.clFlush(s.queue.handle) != cl.c.CL_SUCCESS) return error.Flush;
            cwin.published[ki] = true;
            d.cache.publish(cwin.idx[ki]);
        }
        {
            var hit_evs: [2 * MAX_TW]cl.c.cl_event = undefined;
            var n_hit: cl.c.cl_uint = 0;
            for (0..cwin.n) |ki| {
                if (cwin.load[ki]) continue;
                if (d.cache.slots[cwin.idx[ki]].ev) |ev| {
                    hit_evs[n_hit] = ev;
                    n_hit += 1;
                }
            }
            if (n_hit > 0) {
                if (cl.c.clEnqueueBarrierWithWaitList(s.queue.handle, n_hit, &hit_evs, null) != cl.c.CL_SUCCESS)
                    return error.EnqueueBarrier;
            }
        }
        for (0..d.n_entries) |ei| {
            const e = &d.entries[ei];
            for (0..cwin.n) |ki| {
                const c = ki / tw;
                const t = ki % tw;
                const slot = &d.cache.slots[cwin.idx[ki]];
                for (0..e.n_planes) |i| {
                    const dst_off = e.src_off + ((c * e.n_planes + i) * tw + t) * e.plane_extent;
                    const coff = e.cache_off + i * e.plane_extent;
                    if (cl.c.clEnqueueCopyBuffer(s.queue.handle, slot.buf.handle, s.d_src.handle, coff * 4, dst_off * 4, e.plane_extent * 4, 0, null, null) != cl.c.CL_SUCCESS)
                        return error.EnqueueCopy;
                }
            }
        }
    } else {
        for (0..d.n_entries) |ei| {
            const e = &d.entries[ei];
            for (0..n_clips) |c| {
                for (0..e.n_planes) |i| {
                    const plane = e.planes[i];
                    for (0..tw) |t| {
                        const srcp = win.f[c][t].getReadSlice(plane);
                        const off = e.src_off + ((c * e.n_planes + i) * tw + t) * e.plane_extent;
                        try uploadPlane(s, s.d_src.handle, off, off, e.plane_extent * 4, srcp, null);
                    }
                }
            }
        }
    }

    for (0..d.n_entries) |ei| {
        const e = &d.entries[ei];
        try fillZeroF32(s.queue, s.d_res, e.res_off, e.res_elems);
        try launchBm3d(d, s, e, s.d_res, e.res_off);

        if (d.radius == 0) {
            const k = s.k_agg[e.mod_idx];
            try k.setArg(@TypeOf(s.d_dst), 0, s.d_dst);
            try k.setArg(@TypeOf(s.d_res), 1, s.d_res);
            try k.setArg(c_int, 2, @intCast(e.dst_off));
            try k.setArg(c_int, 3, @intCast(e.res_off));
            const w: usize = @intCast(e.key.w);
            const h: usize = @intCast(e.key.h);
            const gws = [3]usize{ vszipcl.ceilTo(w, 32), vszipcl.ceilTo(h, 8), e.n_planes };
            const lws = [3]usize{ 32, 8, 1 };
            try ndr(s, k, &gws, &lws);
        }
    }

    try download(d, s, dst);
}

fn dlSkip(e: *const Entry, i: usize) bool {
    return (e.key.proc_mask >> @intCast(i)) & 1 == 0;
}

fn dlOff(d: *Data, e: *const Entry, i: usize) usize {
    return if (d.normalOut())
        e.dst_off + i * e.plane_extent
    else
        e.res_off + i * d.tw * 2 * e.plane_extent;
}

fn download(d: *Data, s: *Stream, dst: ZFrameW) !void {
    const src_buf = if (d.normalOut()) s.d_dst.handle else s.d_res.handle;

    if (s.host != null) {
        if (cl.c.clFinish(s.queue.handle) != cl.c.CL_SUCCESS) return error.Finish;
        const hs = s.host.?;
        const tw_mult: usize = if (d.normalOut()) 1 else d.tw * 2;
        for (0..d.n_entries) |ei| {
            const e = &d.entries[ei];
            for (0..e.n_planes) |i| {
                if (dlSkip(e, i)) continue;
                const off = dlOff(d, e, i);
                const len = dst.getWriteSlice(e.planes[i]).len;
                if (len != e.plane_extent * tw_mult * 4) return error.PlaneLayoutMismatch;
                try vszipcl.enqRead(s.queue, src_buf, off * 4, hs[(d.stage_down_off + off) * 4 ..][0..len]);
            }
        }
        if (cl.c.clFinish(s.queue.handle) != cl.c.CL_SUCCESS) return error.Finish;
        for (0..d.n_entries) |ei| {
            const e = &d.entries[ei];
            for (0..e.n_planes) |i| {
                if (dlSkip(e, i)) continue;
                const dstp = dst.getWriteSlice(e.planes[i]);
                @memcpy(dstp, hs[(d.stage_down_off + dlOff(d, e, i)) * 4 ..][0..dstp.len]);
            }
        }
    } else {
        for (0..d.n_entries) |ei| {
            const e = &d.entries[ei];
            for (0..e.n_planes) |i| {
                if (dlSkip(e, i)) continue;
                const dstp = dst.getWriteSlice(e.planes[i]);
                try vszipcl.enqRead(s.queue, src_buf, dlOff(d, e, i) * 4, dstp);
            }
        }
        if (cl.c.clFinish(s.queue.handle) != cl.c.CL_SUCCESS) return error.Finish;
    }
}

fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    const r: i32 = d.radius;
    const ni: i32 = @intCast(n);
    const nf: i32 = d.vi.numFrames;

    if (activation_reason == .Initial) {
        var i: i32 = @max(ni - r, 0);
        const end: i32 = @min(ni + r, nf - 1);
        while (i <= end) : (i += 1) {
            zapi.requestFrameFilter(@intCast(i), d.node);
            if (d.ref_node) |rn| zapi.requestFrameFilter(@intCast(i), rn);
        }
    } else if (activation_reason == .AllFramesReady) {
        var win: Window = undefined;
        win.tw = d.tw;
        win.n_clips = if (d.final) 2 else 1;

        var t: usize = 0;
        while (t < d.tw) : (t += 1) {
            const idx: c_int = @intCast(@min(@max(ni - r + @as(i32, @intCast(t)), 0), nf - 1));
            win.idx[t] = idx;
            if (d.final) {
                win.f[0][t] = zapi.initZFrame(d.ref_node, idx);
                win.f[1][t] = zapi.initZFrame(d.node, idx);
            } else {
                win.f[0][t] = zapi.initZFrame(d.node, idx);
            }
        }
        defer win.deinit();

        const center = win.f[win.n_clips - 1][@intCast(r)];

        const dst = if (d.radius == 0)
            center.newVideoFrame2(d.process)
        else
            center.newVideoFrame3(.{ .height = d.vi_out.height });

        if (d.radius > 0) {
            if (d.zero_init) {
                const np: u32 = @intCast(d.vi.format.numPlanes);
                var p: u32 = 0;
                while (p < np) : (p += 1) {
                    if (!d.process[p]) @memset(dst.getWriteSlice(p), 0);
                }
            }
            const props = dst.getPropertiesRW();
            props.setInt("BM3D_V_radius", d.radius, .Replace);
            props.setIntArray("BM3D_V_process", &[_]i64{
                @intFromBool(d.process[0]), @intFromBool(d.process[1]), @intFromBool(d.process[2]),
            });
        }

        const s = d.pool.acquire();
        defer d.pool.release(s);

        var cwin = CacheWin{};
        if (d.scache) {
            for (0..win.n_clips) |c| {
                for (0..d.tw) |ti| {
                    cwin.keys[c * d.tw + ti] = (@as(i64, @intCast(c)) << 40) | win.idx[ti];
                }
            }
            cwin.n = win.n_clips * d.tw;
            @memset(cwin.published[0..cwin.n], false);
            d.cache.acquire(cwin.keys[0..cwin.n], cwin.idx[0..cwin.n], cwin.load[0..cwin.n]);
        }
        defer if (d.scache) {
            for (0..cwin.n) |ki| {
                if (cwin.load[ki] and !cwin.published[ki]) d.cache.abandon(cwin.idx[ki]);
            }
            d.cache.release(cwin.idx[0..cwin.n]);
        };

        process(d, s, &win, dst, if (d.scache) &cwin else null) catch |err| {
            zapi.setFilterError("BM3D: process frame failed.");
            std.log.err("vszipcl BM3D process frame failed: {}", .{err});
            dst.deinit();
            return null;
        };

        return dst.frame;
    }

    return null;
}

fn freeModSrc(d: *Data) void {
    var i = d.n_mods;
    while (i > 0) {
        i -= 1;
        allocator.free(d.mod_src[i]);
    }
    d.n_mods = 0;
}

fn free(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    d.pool.deinit();
    d.cache.deinit();
    d.context.release();
    freeModSrc(d);
    vsapi.?.freeNode.?(d.node);
    if (d.ref_node) |rn| vsapi.?.freeNode.?(rn);
    allocator.destroy(d);
}

fn perPlane(comptime T: type, map_in: anytype, comptime key: [:0]const u8, base: T) [3]T {
    var v: [3]T = undefined;
    for (0..3) |i| {
        if (map_in.getValue2(T, key, i)) |given| {
            v[i] = given;
        } else {
            v[i] = if (i == 0) base else v[i - 1];
        }
    }
    return v;
}

fn genDefines(d: *const Data, key: *const ModKey) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\#define WIDTH {d}
        \\#define HEIGHT {d}
        \\#define STRIDE {d}
        \\#define SIGMA_Y {x}f
        \\#define SIGMA_U {x}f
        \\#define SIGMA_V {x}f
        \\#define BLOCK_STEP {d}
        \\#define BM_RANGE {d}
        \\#define RADIUS {d}
        \\#define PS_NUM {d}
        \\#define PS_RANGE {d}
        \\#define TEMPORAL {d}
        \\#define CHROMA {d}
        \\#define FINAL {d}
        \\#define EXTRACTOR {x}f
        \\#define WARPS {d}
        \\#define PROC_MASK {d}
        \\#define BM_ERROR {s}
        \\#define TRANSFORM_2D {s}
        \\#define TRANSFORM_1D {s}
        \\
    , .{
        key.w,                  key.h,               key.stride,
        key.sigma[0],           key.sigma[1],        key.sigma[2],
        key.block_step,         key.bm_range,        d.radius,
        key.ps_num,             key.ps_range,        @intFromBool(d.radius > 0),
        @intFromBool(d.chroma), @intFromBool(d.final), d.extractor,
        d.warps,                key.proc_mask,
        @tagName(key.bm_error), @tagName(key.t2d),   @tagName(key.t1d),
    });
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    createInner(in, out, core, vsapi);
}

fn createInner(in: ?*const vs.Map, out: ?*vs.Map, core: ?*vs.Core, vsapi: ?*const vs.API) void {
    var d: Data = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, d.vi = map_in.getNodeVi("clip").?;

    var keep = false;
    defer if (!keep) {
        zapi.freeNode(d.node);
        if (d.ref_node) |rn| zapi.freeNode(rn);
    };

    const fmt = d.vi.format;
    if (d.vi.width <= 0 or d.vi.height <= 0 or fmt.sampleType != .Float or fmt.bitsPerSample != 32) {
        return map_out.setError("BM3D: only constant format 32 bit float input supported.");
    }

    if (map_in.getNode("ref")) |rn| {
        d.ref_node = rn;
        const rvi = zapi.getVideoInfo(rn);
        if (rvi.format.colorFamily != fmt.colorFamily or rvi.format.sampleType != fmt.sampleType or
            rvi.format.bitsPerSample != fmt.bitsPerSample or rvi.format.subSamplingW != fmt.subSamplingW or
            rvi.format.subSamplingH != fmt.subSamplingH)
        {
            return map_out.setError("BM3D: \"ref\" must be of the same format as \"clip\".");
        }
        if (rvi.width != d.vi.width or rvi.height != d.vi.height) {
            return map_out.setError("BM3D: \"ref\" must be of the same dimensions as \"clip\".");
        }
        if (rvi.numFrames != d.vi.numFrames) {
            return map_out.setError("BM3D: \"ref\" must be of the same number of frames as \"clip\".");
        }
        d.final = true;
    }

    var sigma = perPlane(f32, map_in, "sigma", 3.0);
    for (sigma) |sv| {
        if (!(sv >= 0)) return map_out.setError("BM3D: \"sigma\" must be non-negative.");
    }
    for (0..3) |i| d.process[i] = !(sigma[i] < FLT_EPSILON);

    const sigma_factor: f32 = if (d.final)
        @bitCast(@as(u32, 0x3e40c0c1))
    else
        @bitCast(@as(u32, 0x3f021bb6));
    for (&sigma) |*sv| sv.* *= sigma_factor;

    const block_step = perPlane(i32, map_in, "block_step", 8);
    for (block_step) |v| {
        if (v <= 0 or v > 8) return map_out.setError("BM3D: \"block_step\" must be in range [1, 8].");
    }
    const bm_range = perPlane(i32, map_in, "bm_range", 9);
    for (bm_range) |v| {
        if (v <= 0) return map_out.setError("BM3D: \"bm_range\" must be positive.");
    }
    const ps_num = perPlane(i32, map_in, "ps_num", 2);
    for (ps_num) |v| {
        if (v <= 0 or v > 8) return map_out.setError("BM3D: \"ps_num\" must be in range [1, 8].");
    }
    const ps_range = perPlane(i32, map_in, "ps_range", 4);
    for (ps_range) |v| {
        if (v <= 0) return map_out.setError("BM3D: \"ps_range\" must be positive.");
    }

    d.radius = map_in.getValue(i32, "radius") orelse 0;
    if (d.radius < 0) return map_out.setError("BM3D: \"radius\" must be non-negative.");
    if (d.radius > MAX_RADIUS) return map_out.setError("BM3D: \"radius\" must be <= 16.");
    d.tw = @intCast(2 * d.radius + 1);

    d.chroma = (map_in.getValue(i32, "chroma") orelse 0) != 0;
    if (d.chroma and (fmt.colorFamily != .YUV or fmt.subSamplingW != 0 or fmt.subSamplingH != 0)) {
        return map_out.setError("BM3D: clip format must be YUV444 when \"chroma\" is true.");
    }

    d.zero_init = (map_in.getValue(i32, "zero_init") orelse 1) != 0;

    const extractor_exp = map_in.getValue(i32, "extractor_exp") orelse 0;
    d.extractor = if (extractor_exp != 0) math.ldexp(@as(f32, 1.0), extractor_exp) else 0.0;

    var bm_error: [3]BmError = .{ .ssd, .ssd, .ssd };
    var t2d: [3]Transform = .{ .dct, .dct, .dct };
    var t1d: [3]Transform = .{ .dct, .dct, .dct };
    for (0..3) |i| {
        const idx: i32 = @intCast(i);
        if (map_in.getData("bm_error_s", idx)) |sv| {
            bm_error[i] = BmError.parse(sv) orelse return map_out.setError("BM3D: invalid \"bm_error_s\".");
        } else if (i > 0) bm_error[i] = bm_error[i - 1];
        if (map_in.getData("transform_2d_s", idx)) |sv| {
            t2d[i] = Transform.parse(sv) orelse return map_out.setError("BM3D: invalid \"transform_2d_s\".");
        } else if (i > 0) t2d[i] = t2d[i - 1];
        if (map_in.getData("transform_1d_s", idx)) |sv| {
            t1d[i] = Transform.parse(sv) orelse return map_out.setError("BM3D: invalid \"transform_1d_s\".");
        } else if (i > 0) t1d[i] = t1d[i - 1];
    }

    const device_id = map_in.getValue(i32, "device_id") orelse 0;
    if (device_id < 0) return map_out.setError("BM3D: invalid device ID.");
    const platform_id = map_in.getValue(i32, "platform_id") orelse 0;
    if (platform_id < 0) return map_out.setError("BM3D: invalid platform ID.");
    const ns_req = map_in.getValue(i32, "num_streams");
    if (ns_req) |ns| if (ns < 1 or ns > 32) {
        return map_out.setError("BM3D: num_streams must be 1..32.");
    };
    const num_streams: usize = if (ns_req) |ns| @intCast(ns) else 4;

    var cache_req = true;
    var pin_min: usize = 2;
    {
        var terr: ?[:0]const u8 = null;
        if (map_in.numElements("tune")) |ne| {
            if (ne > 3) terr = "BM3D: tune expects at most 3 entries [warps, cache, pin_min_streams].";
        }
        if (vszipcl.tuneEntry(map_in, 0)) |v| {
            if (v < 1 or v > 8) terr = "BM3D: tune[0] (warps) must be 1..8." else d.warps = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 1)) |v| {
            if (v > 1) terr = "BM3D: tune[1] (cache) must be 0 or 1." else cache_req = v != 0;
        }
        if (vszipcl.tuneEntry(map_in, 2)) |v| {
            if (v < 1 or v > 33) terr = "BM3D: tune[2] (pin_min_streams) must be 1..33." else pin_min = @intCast(v);
        }
        if (terr) |msg| return map_out.setError(msg);
    }

    const num_planes: usize = @intCast(fmt.numPlanes);
    var any_proc = false;
    for (0..num_planes) |i| {
        if (d.process[i]) any_proc = true;
    }
    if (!any_proc) return map_out.setError("BM3D: all planes have sigma < FLT_EPSILON (nothing to process).");

    const strides = vszipcl.strideFromVi(d.vi);
    const subW: u5 = @intCast(fmt.subSamplingW);
    const subH: u5 = @intCast(fmt.subSamplingH);

    d.n_entries = 0;
    var sum_src: usize = 0;
    var sum_res: usize = 0;
    var sum_dst: usize = 0;
    var cache_elems: usize = 0;

    if (d.chroma) {
        const w: i32 = d.vi.width;
        const h: i32 = d.vi.height;
        const stride: i32 = @intCast(strides[0]);
        const pe: usize = @as(usize, @intCast(stride)) * @as(usize, @intCast(h));
        var pm: u32 = 0;
        for (0..3) |i| pm |= @as(u32, @intFromBool(d.process[i])) << @intCast(i);
        d.entries[0] = .{
            .key = .{
                .w = w,
                .h = h,
                .stride = stride,
                .sigma = sigma,
                .block_step = block_step[0],
                .bm_range = bm_range[0],
                .ps_num = ps_num[0],
                .ps_range = ps_range[0],
                .proc_mask = pm,
                .bm_error = bm_error[0],
                .t2d = t2d[0],
                .t1d = t1d[0],
            },
            .mod_idx = 0,
            .n_planes = 3,
            .planes = .{ 0, 1, 2 },
            .plane_extent = pe,
            .cache_off = 0,
            .src_off = 0,
            .src_elems = (if (d.final) @as(usize, 2) else 1) * 3 * d.tw * pe,
            .res_off = 0,
            .res_elems = 3 * d.tw * 2 * pe,
            .dst_off = 0,
            .dst_elems = if (d.normalOut()) 3 * pe else 0,
        };
        sum_src = d.entries[0].src_elems;
        sum_res = d.entries[0].res_elems;
        sum_dst = d.entries[0].dst_elems;
        cache_elems = 3 * pe;
        d.n_entries = 1;
        if (d.vi.width < 8 or d.vi.height < 8) {
            return map_out.setError("BM3D: every processed plane must be at least 8x8.");
        }
    } else {
        for (0..num_planes) |p| {
            if (!d.process[p]) continue;
            const w: i32 = if (p == 0) d.vi.width else d.vi.width >> subW;
            const h: i32 = if (p == 0) d.vi.height else d.vi.height >> subH;
            const stride: i32 = @intCast(if (p == 0) strides[0] else strides[1]);
            if (w < 8 or h < 8) return map_out.setError("BM3D: every processed plane must be at least 8x8.");
            const pe: usize = @as(usize, @intCast(stride)) * @as(usize, @intCast(h));
            const ei = d.n_entries;
            d.entries[ei] = .{
                .key = .{
                    .w = w,
                    .h = h,
                    .stride = stride,
                    .sigma = .{ sigma[p], sigma[p], sigma[p] },
                    .block_step = block_step[p],
                    .bm_range = bm_range[p],
                    .ps_num = ps_num[p],
                    .ps_range = ps_range[p],
                    .proc_mask = 1,
                    .bm_error = bm_error[p],
                    .t2d = t2d[p],
                    .t1d = t1d[p],
                },
                .mod_idx = 0,
                .n_planes = 1,
                .planes = .{ @intCast(p), 0, 0 },
                .plane_extent = pe,
                .cache_off = cache_elems,
                .src_off = sum_src,
                .src_elems = (if (d.final) @as(usize, 2) else 1) * d.tw * pe,
                .res_off = sum_res,
                .res_elems = d.tw * 2 * pe,
                .dst_off = sum_dst,
                .dst_elems = if (d.normalOut()) pe else 0,
            };
            sum_src += d.entries[ei].src_elems;
            sum_res += d.entries[ei].res_elems;
            sum_dst += d.entries[ei].dst_elems;
            cache_elems += pe;
            d.n_entries += 1;
        }
    }
    d.sum_src = sum_src;
    d.sum_res = sum_res;
    d.sum_dst = sum_dst;
    d.cache_elems = cache_elems;

    if (sum_src >= (1 << 31) or sum_res >= (1 << 31)) {
        return map_out.setError("BM3D: frame/radius too large (a device region exceeds 2^31 samples).");
    }
    for (0..d.n_entries) |ei| {
        const e = &d.entries[ei];
        if (e.plane_extent * d.tw * 2 * 4 >= (1 << 32)) {
            return map_out.setError("BM3D: stacked output plane too large (exceeds 4 GiB).");
        }
    }

    d.use_pinned = num_streams >= pin_min;
    d.scache = d.radius > 0 and cache_req;
    d.n_clips = if (d.final) 2 else 1;

    d.stage_down_off = d.sum_src;
    d.stage_elems = d.sum_src + (if (d.normalOut()) d.sum_dst else d.sum_res);

    d.vi_out = d.vi.*;
    if (d.radius > 0) {
        const mult: i64 = 2 * (2 * @as(i64, d.radius) + 1);
        if (@as(i64, d.vi.height) * mult > math.maxInt(i32)) {
            return map_out.setError("BM3D: clip too tall for the stacked output.");
        }
        d.vi_out.height = @intCast(@as(i64, d.vi.height) * mult);
    }

    for (0..d.n_entries) |ei| {
        const e = &d.entries[ei];
        var mi: usize = 0;
        while (mi < d.n_mods) : (mi += 1) {
            if (std.meta.eql(d.entries[mi].key, e.key)) break;
        }
        if (mi < d.n_mods) {
            e.mod_idx = mi;
            continue;
        }
        const defines = genDefines(&d, &e.key) catch {
            freeModSrc(&d);
            return map_out.setError("BM3D: out of memory.");
        };
        defer allocator.free(defines);
        d.mod_src[d.n_mods] = std.mem.concatWithSentinel(allocator, u8, &.{ defines, kernels_text }, 0) catch {
            freeModSrc(&d);
            return map_out.setError("BM3D: out of memory.");
        };
        e.mod_idx = d.n_mods;
        d.n_mods += 1;
    }

    vszipcl.initContext(&d, @intCast(device_id), @intCast(platform_id)) catch |err| {
        freeModSrc(&d);
        map_out.setError(if (err == error.InvalidDeviceID) "BM3D: invalid device ID." else if (err == error.InvalidPlatformID) "BM3D: invalid platform ID." else "BM3D: OpenCL init failed.");
        std.log.err("BM3D OpenCL init failed: {}", .{err});
        return;
    };

    const data: *Data = allocator.create(Data) catch {
        d.context.release();
        freeModSrc(&d);
        return map_out.setError("BM3D: out of memory.");
    };
    data.* = d;
    keep = true;

    data.pool.prime(data, num_streams);
    data.pool.prewarm(num_streams) catch |err| {
        map_out.setError("BM3D: OpenCL stream init failed.");
        std.log.err("BM3D stream init failed: {}", .{err});
        data.pool.deinit();
        data.context.release();
        freeModSrc(data);
        allocator.destroy(data);
        keep = false;
        return;
    };

    if (data.scache) blk: {
        var core_info: vs.CoreInfo = .{};
        zapi.getCoreInfo(core, &core_info);
        const n_threads: usize = @intCast(@max(core_info.numThreads, 1));
        const min_slots = data.n_clips * data.tw;
        const want = min_slots + @max(num_streams, n_threads) + 2 * @as(usize, @intCast(data.radius));
        const slots = allocator.alloc(clframecache.CacheSlot, want) catch {
            data.scache = false;
            break :blk;
        };
        var n_ok: usize = 0;
        for (slots) |*slot| {
            slot.* = .{};
            slot.buf = cl.createBuffer(u8, data.context, .{ .read_write = true }, data.cache_elems * 4) catch break;
            slot.has_buf = true;
            n_ok += 1;
        }
        if (n_ok < min_slots) {
            var i: usize = 0;
            while (i < n_ok) : (i += 1) slots[i].buf.release();
            allocator.free(slots);
            data.scache = false;
            std.log.warn("vszipcl BM3D: not enough device memory for the source cache; running uncached.", .{});
        } else {
            data.cache.slots = if (n_ok == want) slots else blk2: {
                const shrunk = allocator.realloc(slots, n_ok) catch slots[0..n_ok];
                break :blk2 shrunk;
            };
            if (n_ok < want)
                std.log.warn("vszipcl BM3D: source cache shrunk to {d}/{d} slots (low device memory).", .{ n_ok, want });
        }
    }

    var dep_buf: [2]vs.FilterDependency = undefined;
    var ndeps: usize = 0;
    const rp: vs.RequestPattern = if (d.radius > 0) .General else .StrictSpatial;
    dep_buf[ndeps] = .{ .source = d.node, .requestPattern = rp };
    ndeps += 1;
    if (d.ref_node) |rn| {
        dep_buf[ndeps] = .{ .source = rn, .requestPattern = rp };
        ndeps += 1;
    }
    zapi.createVideoFilter(out, "BM3D", &data.vi_out, getFrame, free, .Parallel, dep_buf[0..ndeps], data);
}

const V = std.simd.suggestVectorLength(f32) orelse 8;
const Vec = @Vector(V, f32);

const AggData = struct {
    node: ?*vs.Node = null,
    src_node: ?*vs.Node = null,
    src_vi: *const vs.VideoInfo = undefined,
    radius: i32 = 0,
    process: [3]bool = .{ false, false, false },
};

fn aggZ(i: usize, n: i32, nframes: i32, radius: i32) i32 {
    return @min(@max(2 * radius - @as(i32, @intCast(i)), n - nframes + 1 + radius), n + radius);
}

fn aggPlane(dstp: []f32, srcps: []const []const f32, w: usize, h: usize, stride: usize, radius: i32, n: i32, nframes: i32) void {
    const tw: usize = @intCast(2 * radius + 1);
    for (0..h) |y| {
        var rows: [MAX_TW][]const f32 = undefined;
        var wts: [MAX_TW][]const f32 = undefined;
        for (0..tw) |i| {
            const z = aggZ(i, n, nframes, radius);
            const base = (@as(usize, @intCast(z)) * 2 * h + y) * stride;
            rows[i] = srcps[i][base..][0..w];
            wts[i] = srcps[i][base + h * stride ..][0..w];
        }

        const drow = dstp[y * stride ..][0..w];
        var x: usize = 0;
        while (x + V <= w) : (x += V) {
            var acc: Vec = @splat(0);
            var acw: Vec = @splat(0);
            for (0..tw) |i| {
                acc += @as(Vec, rows[i][x..][0..V].*);
                acw += @as(Vec, wts[i][x..][0..V].*);
            }
            drow[x..][0..V].* = acc / acw;
        }
        while (x < w) : (x += 1) {
            var acc: f32 = 0;
            var acw: f32 = 0;
            for (0..tw) |i| {
                acc += rows[i][x];
                acw += wts[i][x];
            }
            drow[x] = acc / acw;
        }
    }
}

fn aggGetFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *AggData = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    const r = d.radius;
    const ni: i32 = @intCast(n);
    const nf: i32 = d.src_vi.numFrames;

    if (activation_reason == .Initial) {
        var i: i32 = @max(ni - r, 0);
        const end: i32 = @min(ni + r, nf - 1);
        while (i <= end) : (i += 1) zapi.requestFrameFilter(@intCast(i), d.node);
        zapi.requestFrameFilter(n, d.src_node);
    } else if (activation_reason == .AllFramesReady) {
        const src = zapi.initZFrame(d.src_node, n);
        defer src.deinit();

        const tw: usize = @intCast(2 * r + 1);
        var stack: [MAX_TW]ZFrame = undefined;
        for (0..tw) |i| {
            const idx: c_int = @intCast(@min(@max(ni - r + @as(i32, @intCast(i)), 0), nf - 1));
            stack[i] = zapi.initZFrame(d.node, idx);
        }
        defer for (0..tw) |i| stack[i].deinit();

        const dst = src.newVideoFrame2(d.process);

        const num_planes: u32 = @intCast(d.src_vi.format.numPlanes);
        var p: u32 = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) continue;
            const w, const h, const stride_b = src.getDimensions(p);
            const stride: usize = stride_b / 4;

            var srcps: [MAX_TW][]const f32 = undefined;
            for (0..tw) |i| srcps[i] = stack[i].getReadSlice2(f32, p);

            aggPlane(dst.getWriteSlice2(f32, p), srcps[0..tw], w, h, stride, r, ni, nf);
        }

        return dst.frame;
    }

    return null;
}

fn aggFree(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *AggData = @ptrCast(@alignCast(instance_data));
    vsapi.?.freeNode.?(d.node);
    vsapi.?.freeNode.?(d.src_node);
    allocator.destroy(d);
}

pub fn createVAggregate(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: AggData = .{};

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    var keep = false;
    defer if (!keep) {
        zapi.freeNode(d.node);
        zapi.freeNode(d.src_node);
    };

    const node, const vi = map_in.getNodeVi("clip").?;
    d.node = node;
    const src_node, const src_vi = map_in.getNodeVi("src").?;
    d.src_node = src_node;
    d.src_vi = src_vi;

    if (src_vi.height <= 0 or src_vi.format.sampleType != .Float or src_vi.format.bitsPerSample != 32) {
        return map_out.setError("VAggregate: \"src\" must be 32 bit float.");
    }
    if (vi.format.colorFamily != src_vi.format.colorFamily or
        vi.format.sampleType != src_vi.format.sampleType or
        vi.format.bitsPerSample != src_vi.format.bitsPerSample or
        vi.format.subSamplingW != src_vi.format.subSamplingW or
        vi.format.subSamplingH != src_vi.format.subSamplingH or
        vi.numFrames != src_vi.numFrames)
    {
        return map_out.setError("VAggregate: \"clip\" must have the same format and frame count as \"src\".");
    }
    const ratio = @divTrunc(vi.height, src_vi.height);
    if (ratio < 6 or @rem(ratio - 2, 4) != 0 or @rem(vi.height, src_vi.height) != 0 or
        vi.width != src_vi.width)
    {
        return map_out.setError("VAggregate: \"clip\" is not a BM3D stacked clip of \"src\" (radius >= 1).");
    }
    d.radius = @divTrunc(ratio - 2, 4);
    if (d.radius > MAX_RADIUS) return map_out.setError("VAggregate: radius too large.");

    var i: usize = 0;
    while (map_in.getValue2(i32, "planes", i)) |pl| : (i += 1) {
        if (pl < 0 or pl > 2) return map_out.setError("VAggregate: \"planes\" must be 0..2.");
        d.process[@intCast(pl)] = true;
    }
    if (i == 0) return map_out.setError("VAggregate: \"planes\" is required.");

    const data = allocator.create(AggData) catch {
        return map_out.setError("VAggregate: out of memory.");
    };
    data.* = d;
    keep = true;

    var deps = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .General },
        .{ .source = d.src_node, .requestPattern = .StrictSpatial },
    };
    zapi.createVideoFilter(out, "VAggregate", d.src_vi, aggGetFrame, aggFree, .Parallel, &deps, data);
}

pub fn createV2(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    var proc: [3]bool = .{ true, true, true };
    for (0..3) |i| {
        if (map_in.getValue2(f32, "sigma", i)) |sv| {
            proc[i] = !(sv < FLT_EPSILON);
        } else if (i > 0) {
            proc[i] = proc[i - 1];
        }
    }

    const src, const src_vi = map_in.getNodeVi("clip").?;
    defer zapi.freeNode(src);

    var skip = true;
    for (0..@intCast(src_vi.format.numPlanes)) |i| {
        if (proc[i]) skip = false;
    }
    if (skip) {
        _ = map_out.setNode("clip", src, .Replace);
        return;
    }

    const radius = map_in.getValue(i32, "radius") orelse 0;

    if (radius > 0 and (map_in.getValue(i32, "fast_fused") orelse 0) != 0) {
        return map_out.setError("BM3Dv2: fast_fused is not supported by vszipcl (the CUDA sibling's device-side aggregation needs pointer tables OpenCL 1.2 cannot express). Use fast_fused=False.");
    }

    const plugin = zapi.getPluginByID("com.julek.vszipcl") orelse {
        return map_out.setError("BM3Dv2: could not find the vszipcl plugin.");
    };

    const bm3d_ret = map_in.invoke(plugin, "BM3D");
    defer bm3d_ret.free();
    if (bm3d_ret.getError()) |msg| {
        return map_out.setError(msg);
    }

    if (radius == 0) {
        const node = bm3d_ret.getNode("clip");
        defer zapi.freeNode(node);
        _ = map_out.setNode("clip", node, .Replace);
        return;
    }

    const agg_args = zapi.createZMap();
    defer agg_args.free();
    const bm3d_node = bm3d_ret.getNode("clip");
    _ = agg_args.consumeNode("clip", bm3d_node, .Replace);
    _ = agg_args.setNode("src", src, .Replace);
    for (0..3) |i| {
        if (proc[i]) agg_args.setInt("planes", @intCast(i), .Append);
    }

    const agg_ret = agg_args.invoke(plugin, "VAggregate");
    defer agg_ret.free();
    if (agg_ret.getError()) |msg| {
        return map_out.setError(msg);
    }

    const node = agg_ret.getNode("clip");
    defer zapi.freeNode(node);
    _ = map_out.setNode("clip", node, .Replace);
}
