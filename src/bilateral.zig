const std = @import("std");
const vszipcl = @import("vszipcl.zig");
const clpool = @import("clpool.zig");

const cl = vszipcl.cl;
const math = std.math;
const vapoursynth = vszipcl.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;

const FLT_EPSILON: f32 = 1.19209290e-7;

const Config = struct {
    radius: i32,
    w: i32,
    h: i32,
    stride: i32,
    sp: f32,
    sc: f32,
    use_sm: bool,
};

const Data = struct {
    node: ?*vs.Node,
    ref_node: ?*vs.Node,
    vi: *const vs.VideoInfo,
    sig_sp_scaled: [3]f32,
    sig_col_scaled: [3]f32,
    radius: [3]i32,
    process: [3]bool,
    configs: [3]Config,
    n_cfg: usize,
    plane_cfg: [3]usize,
    has_ref: bool,
    use_shared_memory: bool,
    bits: i32,
    half: bool,
    bytes: u32,
    platform: cl.Platform,
    device: cl.Device,
    context: cl.Context,
    buff_src: usize,
    buff_dst: usize,
    stage_off: [3]usize,
    stage_elems: [3]usize,
    buff_stage: usize,
    pool: clpool.Pool(Stream, Data),
};

const Stream = struct {
    program: cl.Program,
    kernels: [3]cl.Kernel,
    n_kern: usize,
    cin: [3]cl.Kernel,
    n_cin: usize,
    cout: [3]cl.Kernel,
    n_cout: usize,
    queue: cl.CommandQueue,
    src: cl.Buffer(f32),
    dst: cl.Buffer(f32),
    raw: cl.Buffer(u8),
    has_raw: bool,
    stage: cl.Buffer(u8),
    host: []u8,

    pub fn init(self: *Stream, d: *Data) !void {
        self.n_kern = 0;
        self.n_cin = 0;
        self.n_cout = 0;
        self.has_raw = false;
        self.program = try cl.createProgramWithSource(d.context, kernel_src);
        errdefer self.program.release();
        const opts = try std.fmt.allocPrintSentinel(allocator, "-cl-std=CL3.0 -DHAS_REF={d} -DBITS={d} -DHALF={d}", .{ @intFromBool(d.has_ref), d.bits, @intFromBool(d.half) }, 0);
        defer allocator.free(opts);
        self.program.build(&.{d.device}, opts) catch |err| {
            if (err == error.BuildProgramFailure) {
                const log = try self.program.getBuildLog(allocator, d.device);
                defer allocator.free(log);
                std.log.err("Bilateral OpenCL build failed: {s}", .{log});
            }
            return err;
        };
        self.queue = try cl.createCommandQueue(d.context, d.device, .{});
        errdefer self.queue.release();
        self.src = try cl.createBuffer(f32, d.context, .{ .read_only = true }, d.buff_src);
        errdefer self.src.release();
        self.dst = try cl.createBuffer(f32, d.context, .{ .read_write = true }, d.buff_dst);
        errdefer self.dst.release();
        if (d.bits != 32) {
            self.raw = try cl.createBuffer(u8, d.context, .{ .read_write = true }, d.buff_src * d.bytes);
            self.has_raw = true;
        }
        errdefer if (self.has_raw) self.raw.release();
        self.stage = try cl.createBuffer(u8, d.context, .{ .alloc_host_ptr = true }, d.buff_stage * d.bytes);
        errdefer self.stage.release();
        var map_err: cl.c.cl_int = cl.c.CL_SUCCESS;
        const map_ptr = cl.c.clEnqueueMapBuffer(
            self.queue.handle,
            self.stage.handle,
            cl.c.CL_TRUE,
            cl.c.CL_MAP_READ | cl.c.CL_MAP_WRITE,
            0,
            d.buff_stage * d.bytes,
            0,
            null,
            null,
            &map_err,
        );
        if (map_err != cl.c.CL_SUCCESS or map_ptr == null) return error.MapStaging;
        self.host = @as([*]u8, @ptrCast(map_ptr.?))[0 .. d.buff_stage * d.bytes];
        errdefer {
            _ = cl.c.clEnqueueUnmapMemObject(self.queue.handle, self.stage.handle, map_ptr, 0, null, null);
            _ = cl.c.clFinish(self.queue.handle);
        }

        errdefer self.releaseKernels();
        for (d.configs[0..d.n_cfg], 0..) |cfg, ci| {
            const k = try cl.createKernel(self.program, if (cfg.use_sm) "bilateral_sm" else "bilateral_gl");
            self.kernels[ci] = k;
            errdefer k.release();
            try k.setArg(@TypeOf(self.dst), 0, self.dst);
            try k.setArg(@TypeOf(self.src), 1, self.src);
            try k.setArg(c_int, 2, cfg.w);
            try k.setArg(c_int, 3, cfg.h);
            try k.setArg(c_int, 4, cfg.stride);
            try k.setArg(f32, 5, cfg.sp);
            try k.setArg(f32, 6, cfg.sc);
            try k.setArg(c_int, 7, cfg.radius);
            if (cfg.use_sm) {
                const rr: usize = @intCast(cfg.radius);
                const smem_bytes: usize = (1 + @as(usize, @intFromBool(d.has_ref))) * (2 * rr + 8) * (2 * rr + 16) * @sizeOf(f32);
                if (cl.c.clSetKernelArg(k.handle, 8, smem_bytes, null) != cl.c.CL_SUCCESS)
                    return error.SetKernelArg;
            }
            self.n_kern = ci + 1;
        }

        if (d.bits != 32) {
            for (d.configs[0..d.n_cfg], 0..) |cfg, ci| {
                const n_out: usize = @as(usize, @intCast(cfg.h)) * @as(usize, @intCast(cfg.stride));
                const n_in: usize = (1 + @as(usize, @intFromBool(d.has_ref))) * n_out;
                {
                    const kin = try cl.createKernel(self.program, "convert_in");
                    self.cin[ci] = kin;
                    errdefer kin.release();
                    try kin.setArg(@TypeOf(self.src), 0, self.src);
                    try kin.setArg(@TypeOf(self.raw), 1, self.raw);
                    try kin.setArg(c_int, 2, @as(c_int, @intCast(n_in)));
                    self.n_cin = ci + 1;
                }
                {
                    const kout = try cl.createKernel(self.program, "convert_out");
                    self.cout[ci] = kout;
                    errdefer kout.release();
                    try kout.setArg(@TypeOf(self.raw), 0, self.raw);
                    try kout.setArg(@TypeOf(self.dst), 1, self.dst);
                    try kout.setArg(c_int, 2, @as(c_int, @intCast(n_out)));
                    self.n_cout = ci + 1;
                }
            }
        }
    }

    fn releaseKernels(self: *Stream) void {
        var i: usize = self.n_kern;
        while (i > 0) {
            i -= 1;
            self.kernels[i].release();
        }
        i = self.n_cin;
        while (i > 0) {
            i -= 1;
            self.cin[i].release();
        }
        i = self.n_cout;
        while (i > 0) {
            i -= 1;
            self.cout[i].release();
        }
        self.n_kern = 0;
        self.n_cin = 0;
        self.n_cout = 0;
    }

    pub fn deinit(self: *Stream) void {
        _ = cl.c.clEnqueueUnmapMemObject(self.queue.handle, self.stage.handle, self.host.ptr, 0, null, null);
        _ = cl.c.clFinish(self.queue.handle);
        self.releaseKernels();
        self.stage.release();
        if (self.has_raw) self.raw.release();
        self.dst.release();
        self.src.release();
        self.queue.release();
        self.program.release();
    }
};

const kernel_src =
    \\#ifndef HAS_REF
    \\#define HAS_REF 0
    \\#endif
    \\#ifndef BITS
    \\#define BITS 32
    \\#endif
    \\#ifndef HALF
    \\#define HALF 0
    \\#endif
    \\#define BLOCK_X 16
    \\#define BLOCK_Y 8
    \\
    \\#if BITS != 32
    \\// io wire type. u8/u16 LOAD = the shipped bilateralgpu Windows binary's exact reciprocal
    \\// MULTIPLY (0x3B808081 = fl32(1/255), 0x37800080 = fl32(1/65535) — MSVC /fp:fast folds
    \\// the /peak divide); STORE = (int)round(x*peak), round-half-AWAY-from-zero, NO clamp
    \\// (convex-combination output: x*peak stays in [0,peak]; peak*fl32(1/peak) == 1.0f). Do
    \\// NOT use convert_*_sat_rte (that is upstream's AVX2/Linux path, ties differ). f16 is
    \\// self-defined (no reference): exact widening load, RTE store.
    \\#if BITS == 16 && HALF
    \\typedef half io_t;
    \\#define LOADI(p, i) vload_half((size_t)(i), p)
    \\#define STOREI(p, i, x) vstore_half_rte((x), (size_t)(i), p)
    \\#elif BITS == 16
    \\typedef ushort io_t;
    \\#define LOADI(p, i) (convert_float((p)[i]) * as_float(0x37800080u))
    \\#define STOREI(p, i, x) ((p)[i] = (ushort)(int)round((x) * 65535.0f))
    \\#else
    \\typedef uchar io_t;
    \\#define LOADI(p, i) (convert_float((p)[i]) * as_float(0x3B808081u))
    \\#define STOREI(p, i, x) ((p)[i] = (uchar)(int)round((x) * 255.0f))
    \\#endif
    \\
    \\// Boundary converters (== upstream's CPU h_buffer conversion, moved on-device). The
    \\// global store in convert_in forces the RN rounding of the conversion multiply, so the
    \\// bilateral kernels consume EXACTLY the values upstream's kernel sees in h_buffer.
    \\__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
    \\void convert_in(__global float *restrict dstf, __global const io_t *restrict srcr, const int n)
    \\{
    \\    const int i = get_global_id(0);
    \\    if (i < n) dstf[i] = LOADI(srcr, i);
    \\}
    \\
    \\__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
    \\void convert_out(__global io_t *restrict dstr, __global const float *restrict srcf, const int n)
    \\{
    \\    const int i = get_global_id(0);
    \\    if (i < n) STOREI(dstr, i, srcf[i]);
    \\}
    \\#endif
    \\
    \\__kernel __attribute__((reqd_work_group_size(BLOCK_X, BLOCK_Y, 1)))
    \\void bilateral_sm(
    \\    __global float *restrict dst, __global const float *restrict src,
    \\    const int width, const int height, const int stride,
    \\    const float sigma_spatial_scaled, const float sigma_color_scaled, const int radius,
    \\    __local float *buffer)
    \\{
    \\    const int tx = get_local_id(0);
    \\    const int ty = get_local_id(1);
    \\    const int x  = get_global_id(0);
    \\    const int y  = get_global_id(1);
    \\    const int bx = get_group_id(0) * BLOCK_X;
    \\    const int by = get_group_id(1) * BLOCK_Y;
    \\
    \\    const int tile_w = 2 * radius + BLOCK_X;
    \\    const int tile_h = 2 * radius + BLOCK_Y;
    \\
    \\    for (int cy = ty; cy < tile_h; cy += BLOCK_Y) {
    \\        const int sy = min(max(cy - radius + by, 0), height - 1);
    \\        for (int cx = tx; cx < tile_w; cx += BLOCK_X) {
    \\            const int sx = min(max(cx - radius + bx, 0), width - 1);
    \\            buffer[cy * tile_w + cx] = src[sy * stride + sx];
    \\#if HAS_REF
    \\            buffer[(tile_h + cy) * tile_w + cx] = src[(height + sy) * stride + sx];
    \\#endif
    \\        }
    \\    }
    \\    barrier(CLK_LOCAL_MEM_FENCE);
    \\
    \\    if (x >= width || y >= height) return;
    \\
    \\#if HAS_REF
    \\    const float center = buffer[(tile_h + radius + ty) * tile_w + (radius + tx)];
    \\#else
    \\    const float center = buffer[(radius + ty) * tile_w + (radius + tx)];
    \\#endif
    \\
    \\    float num = 0.0f;
    \\    float den = 0.0f;
    \\    for (int cy = -radius; cy <= radius; ++cy) {
    \\        const int sy = cy + radius + ty;
    \\        for (int cx = -radius; cx <= radius; ++cx) {
    \\            const int sx = cx + radius + tx;
    \\#if HAS_REF
    \\            float value = buffer[(tile_h + sy) * tile_w + sx];
    \\#else
    \\            float value = buffer[sy * tile_w + sx];
    \\#endif
    \\            const float space = (float)(cy * cy + cx * cx);
    \\            const float range = (value - center) * (value - center);
    \\            const float weight = native_exp2(space * sigma_spatial_scaled + range * sigma_color_scaled);
    \\#if HAS_REF
    \\            value = buffer[sy * tile_w + sx];
    \\#endif
    \\            num += weight * value;
    \\            den += weight;
    \\        }
    \\    }
    \\    dst[y * stride + x] = num / den;
    \\}
    \\
    \\__kernel __attribute__((reqd_work_group_size(BLOCK_X, BLOCK_Y, 1)))
    \\void bilateral_gl(
    \\    __global float *restrict dst, __global const float *restrict src,
    \\    const int width, const int height, const int stride,
    \\    const float sigma_spatial_scaled, const float sigma_color_scaled, const int radius)
    \\{
    \\    const int x = get_global_id(0);
    \\    const int y = get_global_id(1);
    \\    if (x >= width || y >= height) return;
    \\
    \\#if HAS_REF
    \\    const float center = src[(height + y) * stride + x];
    \\#else
    \\    const float center = src[y * stride + x];
    \\#endif
    \\
    \\    float num = 0.0f;
    \\    float den = 0.0f;
    \\    for (int cy = max(y - radius, 0); cy <= min(y + radius, height - 1); ++cy) {
    \\        for (int cx = max(x - radius, 0); cx <= min(x + radius, width - 1); ++cx) {
    \\#if HAS_REF
    \\            float value = src[(height + cy) * stride + cx];
    \\#else
    \\            float value = src[cy * stride + cx];
    \\#endif
    \\            const float space = (float)((y - cy) * (y - cy) + (x - cx) * (x - cx));
    \\            const float range = (value - center) * (value - center);
    \\            const float weight = native_exp2(space * sigma_spatial_scaled + range * sigma_color_scaled);
    \\#if HAS_REF
    \\            value = src[cy * stride + cx];
    \\#endif
    \\            num += weight * value;
    \\            den += weight;
    \\        }
    \\    }
    \\    dst[y * stride + x] = num / den;
    \\}
;

const ndr = vszipcl.ndr;
fn writeBuf(s: *Stream, mem: cl.c.cl_mem, src: []const u8) !void {
    return vszipcl.enqWrite(s.queue, mem, 0, src);
}
fn readBuf(s: *Stream, mem: cl.c.cl_mem, dst: []u8) !void {
    return vszipcl.enqRead(s.queue, mem, 0, dst);
}

fn process(d: *Data, s: *Stream, src: ZFrame, ref: ?ZFrame, dst: ZFrameW) !void {
    const has_ref = d.has_ref;
    errdefer _ = cl.c.clFinish(s.queue.handle);

    const numPlanes: u32 = @intCast(d.vi.format.numPlanes);
    var p: u32 = 0;
    while (p < numPlanes) : (p += 1) {
        if (!d.process[p]) continue;
        const cfg = d.configs[d.plane_cfg[p]];
        std.debug.assert(src.getStride2(u8, p) == @as(u32, @intCast(cfg.stride)) * d.bytes);
        std.debug.assert(src.getHeightSigned(p) == cfg.h);
        const plane_elems: usize = @as(usize, @intCast(cfg.h)) * @as(usize, @intCast(cfg.stride));
        const plane_bytes: usize = plane_elems * d.bytes;
        std.debug.assert((1 + @as(usize, @intFromBool(has_ref))) * plane_elems == d.stage_elems[p]);

        const reg = s.host[d.stage_off[p] * d.bytes ..][0 .. d.stage_elems[p] * d.bytes];
        @memcpy(reg[0..plane_bytes], src.getReadSlice(p));
        if (has_ref) @memcpy(reg[plane_bytes..][0..plane_bytes], ref.?.getReadSlice(p));

        const lws: [2]usize = .{ 16, 8 };
        const gws: [2]usize = .{
            vsh.ceilN(@as(usize, @intCast(cfg.w)), 16),
            vsh.ceilN(@as(usize, @intCast(cfg.h)), 8),
        };
        const ci = d.plane_cfg[p];

        if (d.bits != 32) {
            const lws1: [1]usize = .{256};
            const n_out: usize = plane_elems;
            const n_in: usize = (1 + @as(usize, @intFromBool(has_ref))) * n_out;
            try writeBuf(s, s.raw.handle, reg);
            try ndr(s, s.cin[ci], &.{vsh.ceilN(n_in, 256)}, &lws1);
            try ndr(s, s.kernels[ci], &gws, &lws);
            try ndr(s, s.cout[ci], &.{vsh.ceilN(n_out, 256)}, &lws1);
            try readBuf(s, s.raw.handle, reg[0..plane_bytes]);
        } else {
            try writeBuf(s, s.src.handle, reg);
            try ndr(s, s.kernels[ci], &gws, &lws);
            try readBuf(s, s.dst.handle, reg[0..plane_bytes]);
        }
    }

    if (cl.c.clFinish(s.queue.handle) != cl.c.CL_SUCCESS) return error.Finish;

    p = 0;
    while (p < numPlanes) : (p += 1) {
        if (!d.process[p]) continue;
        const plane_bytes = (d.stage_elems[p] / (1 + @as(usize, @intFromBool(has_ref)))) * d.bytes;
        @memcpy(dst.getWriteSlice(p)[0..plane_bytes], s.host[d.stage_off[p] * d.bytes ..][0..plane_bytes]);
    }
}

const ZFrame = @typeInfo(@TypeOf(ZAPI.initZFrame)).@"fn".return_type.?;
const ZFrameW = @typeInfo(@TypeOf(ZFrame.newVideoFrame)).@"fn".return_type.?;

fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    if (activation_reason == .Initial) {
        zapi.requestFrameFilter(n, d.node);
        if (d.has_ref) zapi.requestFrameFilter(n, d.ref_node);
    } else if (activation_reason == .AllFramesReady) {
        const src = zapi.initZFrame(d.node, n);
        defer src.deinit();
        const ref: ?ZFrame = if (d.has_ref) zapi.initZFrame(d.ref_node, n) else null;
        defer if (ref) |r| r.deinit();

        const dst = src.newVideoFrame();
        const numPlanes: u32 = @intCast(d.vi.format.numPlanes);
        var p: u32 = 0;
        while (p < numPlanes) : (p += 1) {
            if (!d.process[p]) @memcpy(dst.getWriteSlice(p), src.getReadSlice(p));
        }

        const s = d.pool.acquire();
        defer d.pool.release(s);

        process(d, s, src, ref, dst) catch |err| {
            zapi.setFilterError("Bilateral: process frame failed.");
            std.log.err("Bilateral process frame failed: {}", .{err});
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
    d.context.release();
    vsapi.?.freeNode.?(d.ref_node);
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

fn sameFormat(a: *const vs.VideoInfo, b: *const vs.VideoInfo) bool {
    const fa = a.format;
    const fb = b.format;
    return fa.colorFamily == fb.colorFamily and fa.sampleType == fb.sampleType and
        fa.bitsPerSample == fb.bitsPerSample and fa.subSamplingW == fb.subSamplingW and
        fa.subSamplingH == fb.subSamplingH and a.width == b.width and a.height == b.height;
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = undefined;
    d.ref_node = null;

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, d.vi = map_in.getNodeVi("clip").?;
    d.ref_node = map_in.getNode("ref");
    d.has_ref = d.ref_node != null;
    var keep = false;
    defer if (!keep) {
        zapi.freeNode(d.ref_node);
        zapi.freeNode(d.node);
    };
    const fmt = d.vi.format;
    const bits: i32 = fmt.bitsPerSample;
    const depth_ok = (fmt.sampleType == .Float and (bits == 32 or bits == 16)) or
        (fmt.sampleType == .Integer and (bits == 8 or bits == 16));
    if (!depth_ok or d.vi.width <= 0 or d.vi.height <= 0 or
        (fmt.colorFamily != .Gray and fmt.colorFamily != .YUV and fmt.colorFamily != .RGB))
    {
        return map_out.setError("Bilateral: input bitdepth must be 8/16 (integer), 16 (half) or 32 (float), Gray/YUV/RGB.");
    }
    d.bits = bits;
    d.half = fmt.sampleType == .Float and bits == 16;
    d.bytes = @intCast(fmt.bytesPerSample);

    if (d.has_ref) {
        const ref_vi = zapi.getVideoInfo(d.ref_node);
        if (!sameFormat(d.vi, ref_vi) or d.vi.numFrames != ref_vi.numFrames) {
            return map_out.setError("Bilateral: \"ref\" must be of the same format as \"clip\".");
        }
    }

    const subW: u5 = @intCast(fmt.subSamplingW);
    const subH: u5 = @intCast(fmt.subSamplingH);
    var sigma_spatial: [3]f32 = undefined;
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        if (map_in.getValue2(f32, "sigma_spatial", i)) |given| {
            if (given < 0) return map_out.setError("Bilateral: sigma_spatial must be non-negative.");
            sigma_spatial[i] = given;
        } else if (i == 0) {
            sigma_spatial[i] = 3.0;
        } else if (i == 1) {
            const prod = (@as(u32, 1) << subH) * (@as(u32, 1) << subW);
            const sub_factor = @sqrt(@as(f64, @floatFromInt(prod)));
            sigma_spatial[i] = @floatCast(@as(f64, sigma_spatial[0]) / sub_factor);
        } else {
            sigma_spatial[i] = sigma_spatial[i - 1];
        }
    }

    var sigma_color: [3]f32 = undefined;
    i = 0;
    while (i < 3) : (i += 1) {
        if (map_in.getValue2(f32, "sigma_color", i)) |given| {
            if (given < 0) return map_out.setError("Bilateral: sigma_color must be non-negative.");
            sigma_color[i] = given;
        } else if (i == 0) {
            sigma_color[i] = 0.02;
        } else {
            sigma_color[i] = sigma_color[i - 1];
        }
    }

    i = 0;
    while (i < 3) : (i += 1) {
        d.process[i] = sigma_spatial[i] >= FLT_EPSILON and sigma_color[i] >= FLT_EPSILON;
    }

    const log2e: f32 = math.log2e;
    i = 0;
    while (i < 3) : (i += 1) {
        d.sig_sp_scaled[i] = -0.5 / (sigma_spatial[i] * sigma_spatial[i]) * log2e;
        d.sig_col_scaled[i] = if (sigma_color[i] >= FLT_EPSILON)
            (-0.5 / (sigma_color[i] * sigma_color[i])) * log2e
        else
            0;
    }

    i = 0;
    while (i < 3) : (i += 1) {
        if (map_in.getValue2(i32, "radius", i)) |given| {
            if (given <= 0) return map_out.setError("Bilateral: radius must be positive.");
            d.radius[i] = given;
        } else {
            const r_f = @min(@round(sigma_spatial[i] * 3.0), 1_000_000.0);
            d.radius[i] = @max(1, @as(i32, @intFromFloat(r_f)));
        }
    }

    const device_id = map_in.getValue(i32, "device_id") orelse 0;
    if (device_id < 0) return map_out.setError("Bilateral: invalid device ID.");

    const ns_req = map_in.getValue(i32, "num_streams"); // ?i32; null -> 1 (single stream)
    if (ns_req) |ns| if (ns < 1 or ns > 32) {
        return map_out.setError("Bilateral: num_streams must be 1..32.");
    };

    d.use_shared_memory = (map_in.getValue(i32, "use_shared_memory") orelse 1) != 0;
    const strides = vszipcl.strideFromVi(d.vi);
    const luma_stride: usize = strides[0];
    const luma_h: usize = @intCast(d.vi.height);
    d.buff_dst = luma_stride * luma_h;
    d.buff_src = (1 + @as(usize, @intFromBool(d.has_ref))) * luma_stride * luma_h;

    if (d.buff_src >= (1 << 31)) {
        return map_out.setError("Bilateral: frame too large (a plane exceeds 2^31 samples).");
    }

    {
        var off: usize = 0;
        var pi: usize = 0;
        while (pi < @as(usize, @intCast(fmt.numPlanes))) : (pi += 1) {
            d.stage_off[pi] = off;
            d.stage_elems[pi] = 0;
            if (!d.process[pi]) continue;
            const ph: usize = if (pi == 0) luma_h else luma_h >> subH;
            const pstride: usize = if (pi == 0) strides[0] else strides[1];
            d.stage_elems[pi] = (1 + @as(usize, @intFromBool(d.has_ref))) * ph * pstride;
            off += d.stage_elems[pi];
        }
        d.buff_stage = @max(1, off);
    }

    d.n_cfg = 0;
    {
        var pi: usize = 0;
        while (pi < @as(usize, @intCast(fmt.numPlanes))) : (pi += 1) {
            if (!d.process[pi]) continue;
            const rr: usize = @intCast(d.radius[pi]);
            const smem_bytes: usize = (1 + @as(usize, @intFromBool(d.has_ref))) * (2 * rr + 8) * (2 * rr + 16) * @sizeOf(f32);
            const cfg: Config = .{
                .radius = d.radius[pi],
                .w = if (pi == 0) d.vi.width else d.vi.width >> subW,
                .h = if (pi == 0) d.vi.height else d.vi.height >> subH,
                .stride = @intCast(if (pi == 0) strides[0] else strides[1]),
                .sp = d.sig_sp_scaled[pi],
                .sc = d.sig_col_scaled[pi],
                .use_sm = d.use_shared_memory and smem_bytes < 48 * 1024,
            };
            var ci: usize = 0;
            while (ci < d.n_cfg) : (ci += 1) {
                if (std.meta.eql(cfg, d.configs[ci])) break;
            }
            if (ci == d.n_cfg) {
                d.configs[ci] = cfg;
                d.n_cfg += 1;
            }
            d.plane_cfg[pi] = ci;
        }
    }

    vszipcl.initContext(&d, @intCast(device_id)) catch |err| {
        map_out.setError(if (err == error.InvalidDeviceID) "Bilateral: invalid device ID." else "Bilateral: OpenCL initialization failed.");
        std.log.err("Bilateral OpenCL init failed: {}", .{err});
        return;
    };

    d.pool = .{};
    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    const streams: usize = if (ns_req) |ns| @intCast(ns) else 1;
    data.pool.prime(data, streams);
    data.pool.prewarm(streams) catch |err| {
        map_out.setError("Bilateral: OpenCL stream init failed.");
        std.log.err("Bilateral stream init failed: {}", .{err});
        data.pool.deinit();
        data.context.release();
        allocator.destroy(data);
        return;
    };

    keep = true;

    var dep = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
        .{ .source = d.ref_node, .requestPattern = .StrictSpatial },
    };
    const deps: []const vs.FilterDependency = if (d.has_ref) dep[0..2] else dep[0..1];
    zapi.createVideoFilter(out, "Bilateral", d.vi, getFrame, free, .Parallel, deps, data);
}
