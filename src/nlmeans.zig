//! Non-local-means denoiser — an OpenCL (buffer-based) port of KNLMeansCL's
//! GRAYS (32-bit float, single plane / luma) SPATIAL path (temporal d=0).
//!
//! KNLMeansCL routes all state through cl_image / image2d_array_t, which PoCL's
//! CUDA backend does NOT support (CL_DEVICE_IMAGE_SUPPORT = false). This port
//! reimplements the exact algorithm on plain cl_mem buffers, mirroring eedi3.zig's
//! design (Data/Stream split, per-frame Stream pool, in-order queue, -D
//! specialization, `restrict`). Edge handling reproduces the original's
//! CLK_ADDRESS_CLAMP = BORDER COLOR 0 (out-of-frame taps read 0.0), via a
//! zero-prepadded input buffer + explicit zero tests in the box-sum.

const std = @import("std");
const vszipcl = @import("vszipcl.zig");
const clpool = @import("clpool.zig");

const cl = vszipcl.cl;
const vapoursynth = vszipcl.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;

// One displacement -> distance -> horizontal -> vertical -> accumulation.
// After the half-window sweep, a single finish. Edge = zero border throughout.
const kernel_src =
    \\#define NLM_NORM        (255.0f*255.0f)
    \\#define NLM_LEGACY      (3.0f)
    \\#define NLM_S_SIZE      ((2*NLM_S+1)*(2*NLM_S+1))
    \\#define NLM_H2_INV_NORM (NLM_NORM/(NLM_LEGACY*NLM_H*NLM_H*NLM_S_SIZE))
    \\
    \\// Copy the input into a zero-padded buffer (margin = PAD = a) so the q-shifted
    \\// reads in distance/accumulation land in the zeroed margin with no branch.
    \\kernel void nlmZeropad(global const float *restrict in, global float *restrict u1) {
    \\    const int x = get_global_id(0), y = get_global_id(1);
    \\    if (x >= VI_DIM_X || y >= VI_DIM_Y) return;
    \\    u1[(y+PAD)*PSTRIDE + (x+PAD)] = in[y*STRIDE + x];
    \\}
    \\
    \\// Fused distance + horizontal box-sum: the per-tap luma squared difference for
    \\// displacement q is recomputed on-chip from the (zero-prepadded) input instead
    \\// of round-tripping a separate distance buffer — fewer launches, no U4a write.
    \\// Bit-exact: same ascending sum of 3*(diff)^2, OOB taps skipped (== zero border).
    \\kernel void nlmDistHorz(global const float *restrict u1, global float *restrict u4b,
    \\                        const int qx, const int qy) {
    \\    const int x = get_global_id(0), y = get_global_id(1);
    \\    if (x >= VI_DIM_X || y >= VI_DIM_Y) return;
    \\    const int cy = y+PAD;
    \\    float sum = 0.0f;
    \\    for (int dj = -NLM_S; dj <= NLM_S; ++dj) {
    \\        const int xx = x + dj;
    \\        if (xx < 0 || xx >= VI_DIM_X) continue;
    \\        const int cxx = xx+PAD;
    \\        const float a = u1[cy*PSTRIDE + cxx];
    \\        const float b = u1[(cy+qy)*PSTRIDE + (cxx+qx)];
    \\        const float diff = a - b;
    \\        sum += 3.0f * diff * diff;
    \\    }
    \\    u4b[y*STRIDE + x] = sum;
    \\}
    \\
    \\// Vertical box-sum, then the weight transform; u4a is reused to hold the weight.
    \\kernel void nlmVertical(global const float *restrict u4b, global float *restrict u4a) {
    \\    const int x = get_global_id(0), y = get_global_id(1);
    \\    if (x >= VI_DIM_X || y >= VI_DIM_Y) return;
    \\    float sum = 0.0f;
    \\    for (int dj = -NLM_S; dj <= NLM_S; ++dj) {
    \\        const int yy = y + dj;
    \\        sum += (yy < 0 || yy >= VI_DIM_Y) ? 0.0f : u4b[yy*STRIDE + x];
    \\    }
    \\    const float arg = sum * NLM_H2_INV_NORM;
    \\    float w;
    \\#if   WMODE == 0
    \\    w = exp(-arg);
    \\#elif WMODE == 1
    \\    w = fdim(1.0f, arg);
    \\#elif WMODE == 2
    \\    { float c = fdim(1.0f, arg); w = c*c; }
    \\#else
    \\    { float c = fdim(1.0f, arg); c = c*c; c = c*c; w = c*c; }
    \\#endif
    \\    u4a[y*STRIDE + x] = w;
    \\}
    \\
    \\// Accumulate both +q and -q at p, exploiting weight(p,p+q)==weight(p,p-q).
    \\// u2 is float2 interleaved: [2g]=weighted-sum, [2g+1]=weight-sum. u5=max weight.
    \\kernel void nlmAccumulation(global const float *restrict u1, global float *restrict u2,
    \\                            global const float *restrict u4a, global float *restrict u5,
    \\                            const int qx, const int qy) {
    \\    const int x = get_global_id(0), y = get_global_id(1);
    \\    if (x >= VI_DIM_X || y >= VI_DIM_Y) return;
    \\    const int g = y*STRIDE + x;
    \\    const int cx = x+PAD, cy = y+PAD;
    \\    const float u4 = u4a[g];
    \\    const int xm = x-qx, ym = y-qy;
    \\    const float u4_mq = (xm < 0 || xm >= VI_DIM_X || ym < 0 || ym >= VI_DIM_Y)
    \\                        ? 0.0f : u4a[ym*STRIDE + xm];
    \\    u5[g] = fmax(u4, fmax(u4_mq, u5[g]));
    \\    const float u1_pq = u1[(cy+qy)*PSTRIDE + (cx+qx)];
    \\    const float u1_mq = u1[(cy-qy)*PSTRIDE + (cx-qx)];
    \\    u2[2*g+0] += (u4*u1_pq) + (u4_mq*u1_mq);
    \\    u2[2*g+1] += (u4 + u4_mq);
    \\}
    \\
    \\// out = (center*m + weighted_sum) / (m + weight_sum), m = wref*maxweight.
    \\// KNL does not guard den==0 (U5 seeded with FLT_EPS keeps den>0); match it.
    \\kernel void nlmFinish(global const float *restrict u1, global float *restrict u1z,
    \\                      global const float *restrict u2, global const float *restrict u5) {
    \\    const int x = get_global_id(0), y = get_global_id(1);
    \\    if (x >= VI_DIM_X || y >= VI_DIM_Y) return;
    \\    const int g = y*STRIDE + x;
    \\    const float m  = NLM_WREF * u5[g];
    \\    const float u  = u1[(y+PAD)*PSTRIDE + (x+PAD)];
    \\    const float den = m + u2[2*g+1];
    \\    u1z[g] = (u*m + u2[2*g+0]) / den;
    \\}
;

const FLT_EPS: f32 = 1.1920929e-7;

const Data = struct {
    node: ?*vs.Node = null,
    vi: vs.VideoInfo = undefined,

    d: u8 = 0,
    a: u8 = 0,
    s: u8 = 0,
    h: f32 = 0,
    wref: f32 = 0,
    wmode: u8 = 0,

    w: u32 = 0,
    h_: u32 = 0,
    stride: u32 = 0, // ceilN(w,8)  — matches the VS GRAYS frame stride
    pad: u32 = 0, // = a
    pstride: u32 = 0, // ceilN(w+2a,8)
    ph: u32 = 0, // h + 2a

    context: cl.Context = undefined,
    device: cl.Device = undefined,
    program: cl.Program = undefined,
    pool: clpool.Pool(Stream, Data) = .{},
};

const Stream = struct {
    queue: cl.CommandQueue,
    d_in: cl.Buffer(f32), // raw upload (stride*h)
    d_u1: cl.Buffer(f32), // zero-prepadded input (pstride*ph)
    d_u1z: cl.Buffer(f32), // output (stride*h)
    d_u2: cl.Buffer(f32), // float2/pixel (stride*h*2)
    d_u4a: cl.Buffer(f32), // squared-diff -> weight (stride*h)
    d_u4b: cl.Buffer(f32), // horizontal box-sum scratch (stride*h)
    d_u5: cl.Buffer(f32), // running max weight (stride*h)
    k_zeropad: cl.Kernel,
    k_disthorz: cl.Kernel,
    k_vertical: cl.Kernel,
    k_accumulation: cl.Kernel,
    k_finish: cl.Kernel,

    pub fn init(self: *Stream, d: *Data) !void {
        const npix = d.stride * d.h_;
        self.queue = try cl.createCommandQueue(d.context, d.device, .{});
        errdefer self.queue.release();
        self.d_in = try cl.createBuffer(f32, d.context, .{ .read_write = true }, npix);
        errdefer self.d_in.release();
        self.d_u1 = try cl.createBuffer(f32, d.context, .{ .read_write = true }, d.pstride * d.ph);
        errdefer self.d_u1.release();
        self.d_u1z = try cl.createBuffer(f32, d.context, .{ .read_write = true }, npix);
        errdefer self.d_u1z.release();
        self.d_u2 = try cl.createBuffer(f32, d.context, .{ .read_write = true }, npix * 2);
        errdefer self.d_u2.release();
        self.d_u4a = try cl.createBuffer(f32, d.context, .{ .read_write = true }, npix);
        errdefer self.d_u4a.release();
        self.d_u4b = try cl.createBuffer(f32, d.context, .{ .read_write = true }, npix);
        errdefer self.d_u4b.release();
        self.d_u5 = try cl.createBuffer(f32, d.context, .{ .read_write = true }, npix);
        errdefer self.d_u5.release();
        self.k_zeropad = try cl.createKernel(d.program, "nlmZeropad");
        errdefer self.k_zeropad.release();
        self.k_disthorz = try cl.createKernel(d.program, "nlmDistHorz");
        errdefer self.k_disthorz.release();
        self.k_vertical = try cl.createKernel(d.program, "nlmVertical");
        errdefer self.k_vertical.release();
        self.k_accumulation = try cl.createKernel(d.program, "nlmAccumulation");
        errdefer self.k_accumulation.release();
        self.k_finish = try cl.createKernel(d.program, "nlmFinish");
        errdefer self.k_finish.release();
        try self.setStaticArgs();
    }

    // All buffer args are fixed per Stream; only q varies per dispatch.
    fn setStaticArgs(self: *Stream) !void {
        try self.k_zeropad.setArg(@TypeOf(self.d_in), 0, self.d_in);
        try self.k_zeropad.setArg(@TypeOf(self.d_u1), 1, self.d_u1);
        try self.k_disthorz.setArg(@TypeOf(self.d_u1), 0, self.d_u1);
        try self.k_disthorz.setArg(@TypeOf(self.d_u4b), 1, self.d_u4b);
        try self.k_vertical.setArg(@TypeOf(self.d_u4b), 0, self.d_u4b);
        try self.k_vertical.setArg(@TypeOf(self.d_u4a), 1, self.d_u4a);
        try self.k_accumulation.setArg(@TypeOf(self.d_u1), 0, self.d_u1);
        try self.k_accumulation.setArg(@TypeOf(self.d_u2), 1, self.d_u2);
        try self.k_accumulation.setArg(@TypeOf(self.d_u4a), 2, self.d_u4a);
        try self.k_accumulation.setArg(@TypeOf(self.d_u5), 3, self.d_u5);
        try self.k_finish.setArg(@TypeOf(self.d_u1), 0, self.d_u1);
        try self.k_finish.setArg(@TypeOf(self.d_u1z), 1, self.d_u1z);
        try self.k_finish.setArg(@TypeOf(self.d_u2), 2, self.d_u2);
        try self.k_finish.setArg(@TypeOf(self.d_u5), 3, self.d_u5);
    }

    pub fn deinit(self: *Stream) void {
        self.k_finish.release();
        self.k_accumulation.release();
        self.k_vertical.release();
        self.k_disthorz.release();
        self.k_zeropad.release();
        self.d_u5.release();
        self.d_u4b.release();
        self.d_u4a.release();
        self.d_u2.release();
        self.d_u1z.release();
        self.d_u1.release();
        self.d_in.release();
        self.queue.release();
    }
};

fn fillF32(queue: cl.CommandQueue, buf: cl.Buffer(f32), value: f32, count: usize) !void {
    var pat = value;
    if (cl.c.clEnqueueFillBuffer(queue.handle, buf.handle, &pat, @sizeOf(f32), 0, count * @sizeOf(f32), 0, null, null) != cl.c.CL_SUCCESS) {
        return error.EnqueueFillBuffer;
    }
}

fn initOpenCL(d: *Data) !void {
    const platforms = try cl.getPlatforms(allocator);
    defer allocator.free(platforms);
    if (platforms.len == 0) return error.NoPlatforms;
    const platform = platforms[0];
    const devices = try platform.getDevices(allocator, cl.DeviceType.all);
    defer allocator.free(devices);
    if (devices.len == 0) return error.NoDevices;
    d.device = devices[0];
    d.context = try cl.createContext(&.{d.device}, .{ .platform = platform });
    d.program = try cl.createProgramWithSource(d.context, kernel_src);
    // Per-instance specialization: bake every create-time constant into the program
    // (KNL's build-option approach == EEDI3's -D pattern). exp() stays precise.
    const opts = try std.fmt.allocPrintSentinel(allocator,
        \\-cl-std=CL3.0 -DVI_DIM_X={d} -DVI_DIM_Y={d} -DSTRIDE={d} -DPSTRIDE={d} -DPAD={d} -DNLM_S={d} -DWMODE={d} -DNLM_H={e}f -DNLM_WREF={e}f
    , .{ d.w, d.h_, d.stride, d.pstride, d.pad, d.s, d.wmode, d.h, d.wref }, 0);
    defer allocator.free(opts);
    d.program.build(&.{d.device}, opts) catch |err| {
        if (err == error.BuildProgramFailure) {
            const log = try d.program.getBuildLog(allocator, d.device);
            defer allocator.free(log);
            std.log.err("NLMeans OpenCL build failed: {s}", .{log});
        }
        return err;
    };
}

fn process(d: *Data, s: *Stream, dstp: []f32, srcp: []const f32) !void {
    const none: []const cl.Event = &.{};
    const npix = d.stride * d.h_;
    const gws: [2]usize = .{ vsh.ceilN(@as(usize, d.w), 16), vsh.ceilN(@as(usize, d.h_), 16) };
    const lws: [2]usize = .{ 16, 16 };

    // upload + build zero-padded input (in-order queue chains the passes)
    (try s.queue.enqueueWriteBuffer(f32, s.d_in, false, 0, srcp, none)).release();
    try fillF32(s.queue, s.d_u1, 0.0, d.pstride * d.ph);
    (try s.queue.enqueueNDRangeKernel(s.k_zeropad, null, &gws, &lws, none)).release();

    // per-frame accumulator inits (U5 seeded with FLT_EPS, NOT zero)
    try fillF32(s.queue, s.d_u2, 0.0, npix * 2);
    try fillF32(s.queue, s.d_u5, FLT_EPS, npix);

    // half-window sweep: each accumulation handles both +q and -q (i innermost).
    const a: i32 = @intCast(d.a);
    const side: i32 = 2 * a + 1;
    var j: i32 = -a;
    while (j <= a) : (j += 1) {
        var i: i32 = -a;
        while (i <= a) : (i += 1) {
            if (j * side + i < 0) {
                try s.k_disthorz.setArg(c_int, 2, i);
                try s.k_disthorz.setArg(c_int, 3, j);
                (try s.queue.enqueueNDRangeKernel(s.k_disthorz, null, &gws, &lws, none)).release();
                (try s.queue.enqueueNDRangeKernel(s.k_vertical, null, &gws, &lws, none)).release();
                try s.k_accumulation.setArg(c_int, 4, i);
                try s.k_accumulation.setArg(c_int, 5, j);
                (try s.queue.enqueueNDRangeKernel(s.k_accumulation, null, &gws, &lws, none)).release();
            }
        }
    }

    (try s.queue.enqueueNDRangeKernel(s.k_finish, null, &gws, &lws, none)).release();

    const rd = try s.queue.enqueueReadBuffer(f32, s.d_u1z, false, 0, dstp, none);
    try cl.waitForEvents(&.{rd});
    rd.release();
}

fn getFrame(n: c_int, ar: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core_ptr, frame_ctx);

    if (ar == .Initial) {
        zapi.requestFrameFilter(n, d.node);
    } else if (ar == .AllFramesReady) {
        const src = zapi.initZFrame(d.node, n);
        defer src.deinit();
        const dst = src.newVideoFrame();

        const s = d.pool.acquire();
        defer d.pool.release(s);

        const srcp = src.getReadSlice2(f32, 0);
        const dstp = dst.getWriteSlice2(f32, 0);
        process(d, s, dstp, srcp) catch |err| {
            zapi.setFilterError("NLMeans: process failed.");
            std.log.err("NLMeans process failed: {}", .{err});
            dst.deinit();
            return null;
        };
        return dst.frame;
    }
    return null;
}

fn free(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    d.pool.deinit();
    d.program.release();
    d.context.release();
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = .{};
    const zapi = ZAPI.init(vsapi, core_ptr, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, const vi_in = map_in.getNodeVi("clip").?;
    d.vi = vi_in.*;

    var keep = false;
    defer if (!keep) zapi.freeNode(d.node);

    const fmt = d.vi.format;
    if (fmt.colorFamily != .Gray or fmt.numPlanes != 1 or fmt.sampleType != .Float or fmt.bitsPerSample != 32) {
        return map_out.setError("NLMeans: only 32-bit float single-plane (GrayS) is supported.");
    }
    if (d.vi.width <= 0 or d.vi.height <= 0) {
        return map_out.setError("NLMeans: clip must have constant dimensions.");
    }
    if (d.vi.width > 8192 or d.vi.height > 8192) {
        return map_out.setError("NLMeans: 8192x8192 is the highest supported resolution.");
    }

    const dd = map_in.getValue(i32, "d") orelse 0;
    const a = map_in.getValue(i32, "a") orelse 2;
    const ss = map_in.getValue(i32, "s") orelse 4;
    d.h = map_in.getValue(f32, "h") orelse 1.2;
    const wmode = map_in.getValue(i32, "wmode") orelse 0;
    d.wref = map_in.getValue(f32, "wref") orelse 1.0;

    if (dd < 0) return map_out.setError("NLMeans: d must be >= 0.");
    if (dd > 0) return map_out.setError("NLMeans: temporal (d > 0) is not implemented yet; use d=0.");
    if (map_in.getNode("rclip") != null) return map_out.setError("NLMeans: rclip is not implemented yet.");
    if (a < 1) return map_out.setError("NLMeans: a must be >= 1.");
    if (ss < 0 or ss > 8) return map_out.setError("NLMeans: s must be 0..8.");
    if (d.h <= 0) return map_out.setError("NLMeans: h must be > 0.");
    if (wmode < 0 or wmode > 3) return map_out.setError("NLMeans: wmode must be 0..3.");
    if (d.wref < 0) return map_out.setError("NLMeans: wref must be >= 0.");
    if (2 * a + 1 > d.vi.width or 2 * a + 1 > d.vi.height) {
        return map_out.setError("NLMeans: research window (2*a+1) larger than the frame.");
    }

    d.d = @intCast(dd);
    d.a = @intCast(a);
    d.s = @intCast(ss);
    d.wmode = @intCast(wmode);
    d.w = @intCast(d.vi.width);
    d.h_ = @intCast(d.vi.height);
    d.stride = @intCast(vsh.ceilN(@as(usize, d.w), 8));
    d.pad = @intCast(a);
    d.pstride = @intCast(vsh.ceilN(@as(usize, d.w) + 2 * @as(usize, d.pad), 8));
    d.ph = d.h_ + 2 * d.pad;

    initOpenCL(&d) catch |err| {
        map_out.setError("NLMeans: OpenCL init failed.");
        std.log.err("NLMeans OpenCL init failed: {}", .{err});
        return;
    };

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;
    keep = true;

    var info: vs.CoreInfo = .{};
    zapi.getCoreInfo(core_ptr, &info);
    const threads: usize = if (info.numThreads > 0) @intCast(info.numThreads) else 1;
    const streams: usize = @min(@as(usize, 4), threads);
    data.pool.prime(data, streams);
    data.pool.prewarm(streams) catch |err| {
        map_out.setError("NLMeans: OpenCL stream init failed.");
        std.log.err("NLMeans stream init failed: {}", .{err});
        data.pool.deinit();
        data.program.release();
        data.context.release();
        allocator.destroy(data);
        keep = false;
        return;
    };

    var dep = [_]vs.FilterDependency{.{ .source = d.node, .requestPattern = .StrictSpatial }};
    zapi.createVideoFilter(out, "NLMeans", &d.vi, getFrame, free, .Unordered, dep[0..], data);
}
