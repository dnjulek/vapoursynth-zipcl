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

const Mode = enum { small, large };

const large_path_radius_threshold: i32 = 32;

const large_blk_x: usize = 16;
const large_blk_y: usize = 8;
const large_r: usize = 8;

const tune_len = 7;

const Config = struct {
    key: Key,
    ksize: i32,
    radius: i32,
    mode: Mode,
    weights: []const f32,

    const Key = struct { w: i32, h: i32, stride: i32, sigma: f32 };

    fn extent(self: *const Config) usize {
        return @as(usize, @intCast(self.key.stride)) * @as(usize, @intCast(self.key.h));
    }
};

const Data = struct {
    node: ?*vs.Node,
    vi: *const vs.VideoInfo,

    platform: cl.Platform,
    device: cl.Device,
    context: cl.Context,

    bits: i32,
    half: bool,
    bytes: u32,

    process: [3]bool,
    plane_cfg: [3]usize,
    configs: [3]Config,
    n_cfg: usize,
    any_large: bool,

    buff_elems: usize,

    stage_off: [3]usize,
    stage_sum: usize,
    bake_large: bool,
    use_pinned: bool,
    memcpy_sem: std.Io.Semaphore,

    blk_x: usize,
    blk_y: usize,
    vrt: usize,
    large_r: usize,

    pool: clpool.Pool(Stream, Data),
};

const CfgRes = struct {
    program: ?cl.Program,
    weights: cl.Buffer(f32),
    k1: cl.Kernel,
    k2: ?cl.Kernel,
};

const Stream = struct {
    queue: cl.CommandQueue,
    src: cl.Buffer(u8),
    dst: cl.Buffer(u8),
    tmp: ?cl.Buffer(f32),
    large_prog: ?cl.Program,
    cfgs: [3]CfgRes,
    n_cfg_ready: usize,
    stage: ?cl.Buffer(u8),
    host: []u8,

    pub fn init(self: *Stream, d: *Data) !void {
        self.n_cfg_ready = 0;
        self.tmp = null;
        self.large_prog = null;
        self.stage = null;

        self.queue = try cl.createCommandQueue(d.context, d.device, .{});
        errdefer self.queue.release();
        self.src = try cl.createBuffer(u8, d.context, .{ .read_only = true }, d.buff_elems * d.bytes);
        errdefer self.src.release();
        self.dst = try cl.createBuffer(u8, d.context, .{ .read_write = true }, d.buff_elems * d.bytes);
        errdefer self.dst.release();

        if (d.use_pinned) blk: {
            const st = cl.createBuffer(u8, d.context, .{ .alloc_host_ptr = true }, 2 * d.stage_sum) catch break :blk;
            var map_err: cl.c.cl_int = cl.c.CL_SUCCESS;
            const map_ptr = cl.c.clEnqueueMapBuffer(self.queue.handle, st.handle, cl.c.CL_TRUE, cl.c.CL_MAP_READ | cl.c.CL_MAP_WRITE, 0, 2 * d.stage_sum, 0, null, null, &map_err);
            if (map_err != cl.c.CL_SUCCESS or map_ptr == null) {
                st.release();
                std.log.warn("GaussBlur: pinned staging unavailable; this stream runs pageable transfers.", .{});
                break :blk;
            }
            self.stage = st;
            self.host = @as([*]u8, @ptrCast(map_ptr.?))[0 .. 2 * d.stage_sum];
        }
        errdefer if (self.stage) |st| {
            _ = cl.c.clEnqueueUnmapMemObject(self.queue.handle, st.handle, self.host.ptr, 0, null, null);
            _ = cl.c.clFinish(self.queue.handle);
            st.release();
        };
        if (d.any_large)
            self.tmp = try cl.createBuffer(f32, d.context, .{ .read_write = true }, d.buff_elems);
        errdefer if (self.tmp) |t| t.release();

        const idx_t: []const u8 = if (d.buff_elems < (1 << 31)) "int" else "long";

        if (d.any_large and !d.bake_large) {
            const opts = try std.fmt.allocPrintSentinel(allocator,
                \\-cl-std=CL1.2 -DBX={d} -DBY={d} -DR={d} -DIDX={s} -DBITS={d} -DHALF={d}
            , .{ d.blk_x, d.blk_y, d.large_r, idx_t, d.bits, @intFromBool(d.half) }, 0);
            defer allocator.free(opts);
            self.large_prog = try buildProgram(d, io_src ++ vertical_blur_src ++ horizontal_blur_src, opts);
        }
        errdefer if (self.large_prog) |pr| pr.release();

        errdefer self.releaseCfgs();
        var ci: usize = 0;
        while (ci < d.n_cfg) : (ci += 1) {
            const cfg = &d.configs[ci];
            const cr = &self.cfgs[ci];
            cr.program = null;
            cr.k2 = null;
            cr.weights = try cl.createBufferWithData(f32, d.context, .{ .read_only = true }, cfg.weights);
            errdefer cr.weights.release();
            switch (cfg.mode) {
                .small => {
                    const opts = try std.fmt.allocPrintSentinel(allocator,
                        \\-cl-std=CL1.2 -DW={d} -DH={d} -DSTRIDE={d} -DKLEN={d} -DRAD={d} -DBLK_X={d} -DBLK_Y={d} -DVRT={d} -DIDX={s} -DBITS={d} -DHALF={d}
                    , .{ cfg.key.w, cfg.key.h, cfg.key.stride, cfg.ksize, cfg.radius, d.blk_x, d.blk_y, d.vrt, idx_t, d.bits, @intFromBool(d.half) }, 0);
                    defer allocator.free(opts);
                    const prog = try buildProgram(d, io_src ++ gauss_blur_src, opts);
                    cr.program = prog;
                    errdefer prog.release();
                    cr.k1 = try cl.createKernel(prog, "gauss_blur");
                    errdefer cr.k1.release();
                    try cr.k1.setArg(@TypeOf(self.dst), 0, self.dst);
                    try cr.k1.setArg(@TypeOf(self.src), 1, self.src);
                    try cr.k1.setArg(@TypeOf(cr.weights), 2, cr.weights);
                },
                .large => {
                    const prog = if (d.bake_large) blk: {
                        const opts = try std.fmt.allocPrintSentinel(allocator,
                            \\-cl-std=CL1.2 -DBX={d} -DBY={d} -DR={d} -DIDX={s} -DBITS={d} -DHALF={d} -DBAKE_KLEN={d} -DBAKE_W={d} -DBAKE_H={d} -DBAKE_S={d}
                        , .{ d.blk_x, d.blk_y, d.large_r, idx_t, d.bits, @intFromBool(d.half), cfg.ksize, cfg.key.w, cfg.key.h, cfg.key.stride }, 0);
                        defer allocator.free(opts);
                        const pr = try buildProgram(d, io_src ++ vertical_blur_src ++ horizontal_blur_src, opts);
                        cr.program = pr;
                        break :blk pr;
                    } else self.large_prog.?;
                    const tmp = self.tmp.?;
                    cr.k1 = try cl.createKernel(prog, "vertical_blur");
                    errdefer cr.k1.release();
                    const k2 = try cl.createKernel(prog, "horizontal_blur");
                    cr.k2 = k2;
                    errdefer k2.release();
                    try cr.k1.setArg(@TypeOf(tmp), 0, tmp);
                    try cr.k1.setArg(@TypeOf(self.src), 1, self.src);
                    try cr.k1.setArg(@TypeOf(cr.weights), 2, cr.weights);
                    try cr.k1.setArg(c_int, 3, cfg.ksize);
                    try cr.k1.setArg(c_int, 4, cfg.key.w);
                    try cr.k1.setArg(c_int, 5, cfg.key.h);
                    try cr.k1.setArg(c_int, 6, cfg.key.stride);
                    try k2.setArg(@TypeOf(self.dst), 0, self.dst);
                    try k2.setArg(@TypeOf(tmp), 1, tmp);
                    try k2.setArg(@TypeOf(cr.weights), 2, cr.weights);
                    try k2.setArg(c_int, 3, cfg.ksize);
                    try k2.setArg(c_int, 4, cfg.key.w);
                    try k2.setArg(c_int, 5, cfg.key.h);
                    try k2.setArg(c_int, 6, cfg.key.stride);
                },
            }
            self.n_cfg_ready = ci + 1;
        }
    }

    fn releaseCfgs(self: *Stream) void {
        var i: usize = self.n_cfg_ready;
        while (i > 0) {
            i -= 1;
            const cr = &self.cfgs[i];
            if (cr.k2) |k| k.release();
            cr.k1.release();
            cr.weights.release();
            if (cr.program) |pr| pr.release();
        }
    }

    pub fn deinit(self: *Stream) void {
        if (self.stage) |st| _ = cl.c.clEnqueueUnmapMemObject(self.queue.handle, st.handle, self.host.ptr, 0, null, null);
        _ = cl.c.clFinish(self.queue.handle);
        self.releaseCfgs();
        if (self.large_prog) |pr| pr.release();
        if (self.stage) |st| st.release();
        if (self.tmp) |t| t.release();
        self.dst.release();
        self.src.release();
        self.queue.release();
    }
};

fn buildProgram(d: *Data, src_txt: [:0]const u8, opts: [:0]const u8) !cl.Program {
    var prog = try cl.createProgramWithSource(d.context, src_txt);
    errdefer prog.release();
    prog.build(&.{d.device}, opts) catch |err| {
        if (err == error.BuildProgramFailure) {
            const log = try prog.getBuildLog(allocator, d.device);
            defer allocator.free(log);
            std.log.err("GaussBlur OpenCL build failed: {s}", .{log});
        }
        return err;
    };
    return prog;
}

const io_src =
    \\#if BITS == 32
    \\typedef float io_t;
    \\#define LOADI(p, i) ((p)[i])
    \\#define STOREI(p, i, x) ((p)[i] = (x))
    \\#elif BITS == 16 && HALF
    \\typedef half io_t;
    \\#define LOADI(p, i) vload_half((size_t)(i), p)
    \\#define STOREI(p, i, x) vstore_half_rte((x), (size_t)(i), p)
    \\#elif BITS == 16
    \\typedef ushort io_t;
    \\#define LOADI(p, i) convert_float((p)[i])
    \\#define STOREI(p, i, x) ((p)[i] = convert_ushort_sat_rte(x))
    \\#else
    \\typedef uchar io_t;
    \\#define LOADI(p, i) convert_float((p)[i])
    \\#define STOREI(p, i, x) ((p)[i] = convert_uchar_sat_rte(x))
    \\#endif
    \\
;

const gauss_blur_src =
    \\kernel __attribute__((reqd_work_group_size(BLK_X, BLK_Y, 1)))
    \\void gauss_blur(global io_t *restrict dst, global const io_t *restrict src,
    \\                global const float *restrict blur) {
    \\    local float vblur[VRT*BLK_Y][BLK_X + 2*RAD];
    \\    const int lx = get_local_id(0), ly = get_local_id(1);
    \\    const int gx0 = get_group_id(0) * BLK_X;            // first output column
    \\    const int gy0 = get_group_id(1) * (VRT*BLK_Y);      // first output row
    \\
    \\    // Phase 1 — vertical blur of the tile columns + a +-RAD X-halo, into local memory.
    \\    for (int ry = ly; ry < VRT*BLK_Y; ry += BLK_Y) {
    \\        const int y = gy0 + ry;
    \\        if (y >= H) continue;                            // row past frame: Phase 2 never reads it
    \\        for (int cj = lx; cj < BLK_X + 2*RAD; cj += BLK_X) {
    \\            int cx = gx0 - RAD + cj;                      // raw (maybe OOB) column
    \\            if (cx < 0) cx = -cx;                         // left mirror  (== scalar -src_x)
    \\            else if (cx >= W) cx = 2*(W-1) - cx;          // right mirror (== 2*(w-1)-src_x)
    \\            cx = min(max(cx, 0), W-1);                    // guard the unused over-staged tail
    \\                                                         // (tiny/odd W); a no-op for read cells
    \\            float vsum = 0.0f;
    \\            for (int k = 0; k < KLEN; ++k) {              // ascending k
    \\                int sy = y + k - RAD;
    \\                if (sy < 0) sy = -sy;                     // top mirror    (== scalar -src_y)
    \\                else if (sy >= H) sy = 2*(H-1) - sy;      // bottom mirror (== 2*(h-1)-src_y)
    \\                vsum += LOADI(src, (IDX)sy*STRIDE + cx) * blur[k];
    \\            }
    \\            vblur[ry][cj] = vsum;
    \\        }
    \\    }
    \\    barrier(CLK_LOCAL_MEM_FENCE);                        // every thread reaches it (returns are after)
    \\
    \\    // Phase 2 — horizontal blur from vblur -> dst. Output col x, VRT output rows/thread.
    \\    const int x = gx0 + lx;
    \\    if (x >= W) return;
    \\    for (int r = 0; r < VRT; ++r) {
    \\        const int y = gy0 + ly + r*BLK_Y;
    \\        if (y >= H) return;                              // strided y increasing -> safe to bail
    \\        const int lc = ly + r*BLK_Y;                     // local tile row for this output row
    \\        float sum = 0.0f;
    \\        for (int k = 0; k < KLEN; ++k)                   // ascending k
    \\            sum += vblur[lc][lx + k] * blur[k];
    \\        STOREI(dst, (IDX)y*STRIDE + x, sum);
    \\    }
    \\}
;

const vertical_blur_src =
    \\#ifdef BAKE_KLEN
    \\#define kernel_len (BAKE_KLEN)
    \\#define w (BAKE_W)
    \\#define h (BAKE_H)
    \\#define stride (BAKE_S)
    \\#define LB_ARGS int kernel_len_a, const int w_a, const int h_a, const int stride_a
    \\#else
    \\#define LB_ARGS int kernel_len, const int w, const int h, const int stride
    \\#endif
    \\kernel __attribute__((reqd_work_group_size(BX, BY, 1)))
    \\void vertical_blur(global float *restrict dst, global const io_t *restrict src,
    \\                   global const float *restrict blur_kernel, LB_ARGS) {
    \\    const int x = get_global_id(0);
    \\    const int y0 = get_global_id(1) * R;   // first of this thread's R output rows
    \\    if (x >= w || y0 >= h) return;
    \\    const int radius = kernel_len / 2;
    \\    float sum[R], wreg[R];
    \\    for (int j = 0; j < R; ++j) { sum[j] = 0.0f; wreg[j] = 0.0f; }
    \\    for (int k = 0; k < kernel_len + (R - 1); ++k) {
    \\        for (int j = R - 1; j >= 1; --j) wreg[j] = wreg[j - 1];   // slide the window
    \\        wreg[0] = (k < kernel_len) ? blur_kernel[k] : 0.0f;
    \\        int sy = y0 + k - radius;
    \\        if (sy < 0) sy = -sy;                       // top mirror    (== scalar -src_y)
    \\        else if (sy >= h) sy = 2 * (h - 1) - sy;    // bottom mirror (== 2*(h-1)-src_y)
    \\        sy = min(max(sy, 0), h - 1);                // guards unused over-range taps only
    \\        const float v = LOADI(src, (IDX)sy * stride + x);
    \\        for (int j = 0; j < R; ++j) {
    \\            if (k - j >= 0 && k - j < kernel_len)   // uniform: exactly kk = 0..kernel_len-1
    \\                sum[j] += v * wreg[j];
    \\        }
    \\    }
    \\    for (int j = 0; j < R; ++j) {
    \\        if (y0 + j < h) dst[(IDX)(y0 + j) * stride + x] = sum[j];
    \\    }
    \\}
;

const horizontal_blur_src =
    \\kernel __attribute__((reqd_work_group_size(BX, BY, 1)))
    \\void horizontal_blur(global io_t *restrict dst, global const float *restrict src,
    \\                     global const float *restrict blur_kernel, LB_ARGS) {
    \\    const int x0 = get_global_id(0) * R;   // first of this thread's R output columns
    \\    const int y = get_global_id(1);
    \\    if (x0 >= w || y >= h) return;
    \\    const int radius = kernel_len / 2;
    \\    const IDX row = (IDX)y * stride;
    \\    float sum[R], wreg[R];
    \\    for (int j = 0; j < R; ++j) { sum[j] = 0.0f; wreg[j] = 0.0f; }
    \\    for (int k = 0; k < kernel_len + (R - 1); ++k) {
    \\        for (int j = R - 1; j >= 1; --j) wreg[j] = wreg[j - 1];   // slide the window
    \\        wreg[0] = (k < kernel_len) ? blur_kernel[k] : 0.0f;
    \\        int sx = x0 + k - radius;
    \\        if (sx < 0) sx = -sx;                       // left mirror  (== scalar -src_x)
    \\        else if (sx >= w) sx = 2 * (w - 1) - sx;    // right mirror (== 2*(w-1)-src_x)
    \\        sx = min(max(sx, 0), w - 1);                // guards unused over-range taps only
    \\        const float v = src[row + sx];
    \\        for (int j = 0; j < R; ++j) {
    \\            if (k - j >= 0 && k - j < kernel_len)   // uniform: exactly kk = 0..kernel_len-1
    \\                sum[j] += v * wreg[j];
    \\        }
    \\    }
    \\    for (int j = 0; j < R; ++j) {
    \\        if (x0 + j < w) STOREI(dst, row + x0 + j, sum[j]);
    \\    }
    \\}
;

const ndr = vszipcl.ndr;
fn writeBuf(s: *Stream, buf: cl.Buffer(u8), src: []const u8) !void {
    return vszipcl.enqWrite(s.queue, buf.handle, 0, src);
}

fn readBuf(s: *Stream, buf: cl.Buffer(u8), dst: []u8) !void {
    return vszipcl.enqRead(s.queue, buf.handle, 0, dst);
}

const ZFrame = @typeInfo(@TypeOf(ZAPI.initZFrame)).@"fn".return_type.?;
const ZFrameW = @typeInfo(@TypeOf(ZFrame.newVideoFrame)).@"fn".return_type.?;

fn process(d: *Data, s: *Stream, src: ZFrame, dst: ZFrameW) !void {
    errdefer _ = cl.c.clFinish(s.queue.handle);

    const num_planes: u32 = @intCast(d.vi.format.numPlanes);
    var p: u32 = 0;
    while (p < num_planes) : (p += 1) {
        if (!d.process[p]) continue;

        const cfg = &d.configs[d.plane_cfg[p]];
        const cr = &s.cfgs[d.plane_cfg[p]];
        const srcp = src.getReadSlice(p);
        const dstp = dst.getWriteSlice(p);
        if (srcp.len != cfg.extent() * d.bytes or dstp.len != srcp.len)
            return error.PlaneLayoutMismatch;

        if (s.stage != null) {
            d.memcpy_sem.waitUncancelable(vszipcl.io);
            @memcpy(s.host[d.stage_off[p]..][0..srcp.len], srcp);
            d.memcpy_sem.post(vszipcl.io);
            try vszipcl.enqWrite(s.queue, s.src.handle, 0, s.host[d.stage_off[p]..][0..srcp.len]);
        } else {
            try writeBuf(s, s.src, srcp);
        }

        switch (cfg.mode) {
            .small => {
                const BX: usize = d.blk_x;
                const BY: usize = d.blk_y;
                const VR: usize = d.vrt;
                const lws: [2]usize = .{ BX, BY };
                const gws: [2]usize = .{
                    vszipcl.ceilTo(@as(usize, @intCast(cfg.key.w)), BX),
                    ((@as(usize, @intCast(cfg.key.h)) + VR * BY - 1) / (VR * BY)) * BY,
                };
                try ndr(s, cr.k1, &gws, &lws);
            },
            .large => {
                const bx = d.blk_x;
                const by = d.blk_y;
                const r_ = d.large_r;
                const lws: [2]usize = .{ bx, by };
                const w_: usize = @intCast(cfg.key.w);
                const h_: usize = @intCast(cfg.key.h);
                const gws_v: [2]usize = .{
                    ((w_ + bx - 1) / bx) * bx,
                    (((h_ + r_ - 1) / r_ + by - 1) / by) * by,
                };
                const gws_h: [2]usize = .{
                    (((w_ + r_ - 1) / r_ + bx - 1) / bx) * bx,
                    ((h_ + by - 1) / by) * by,
                };
                try ndr(s, cr.k1, &gws_v, &lws);
                try ndr(s, cr.k2.?, &gws_h, &lws);
            },
        }
        if (s.stage != null) {
            try vszipcl.enqRead(s.queue, s.dst.handle, 0, s.host[d.stage_sum + d.stage_off[p] ..][0..dstp.len]);
        } else {
            try readBuf(s, s.dst, dstp);
        }
    }

    if (cl.c.clFinish(s.queue.handle) != cl.c.CL_SUCCESS) return error.Finish;

    if (s.stage != null) {
        d.memcpy_sem.waitUncancelable(vszipcl.io);
        var q: u32 = 0;
        while (q < num_planes) : (q += 1) {
            if (!d.process[q]) continue;
            const dstp = dst.getWriteSlice(q);
            @memcpy(dstp, s.host[d.stage_sum + d.stage_off[q] ..][0..dstp.len]);
        }
        d.memcpy_sem.post(vszipcl.io);
    }
}

fn getFrame(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    if (activation_reason == .Initial) {
        zapi.requestFrameFilter(n, d.node);
    } else if (activation_reason == .AllFramesReady) {
        const src = zapi.initZFrame(d.node, n);
        defer src.deinit();
        const dst = src.newVideoFrame();

        const num_planes: u32 = @intCast(d.vi.format.numPlanes);
        var p: u32 = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) @memcpy(dst.getWriteSlice(p), src.getReadSlice(p));
        }

        const s = d.pool.acquire();
        defer d.pool.release(s);

        process(d, s, src, dst) catch |err| {
            zapi.setFilterError("GaussBlur: process frame failed.");
            std.log.err("OpenCL process frame failed: {}", .{err});
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
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = undefined;

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, d.vi = map_in.getNodeVi("clip").?;

    var keep = false;
    defer if (!keep) zapi.freeNode(d.node);

    const fmt = d.vi.format;
    const bits: i32 = fmt.bitsPerSample;
    const depth_ok = (fmt.sampleType == .Float and (bits == 32 or bits == 16)) or
        (fmt.sampleType == .Integer and (bits == 8 or bits == 16));
    if (!depth_ok or d.vi.width <= 0 or d.vi.height <= 0 or
        (fmt.colorFamily != .Gray and fmt.colorFamily != .YUV and fmt.colorFamily != .RGB))
    {
        return map_out.setError("GaussBlur: input bitdepth must be 8/16 (integer), 16 (half) or 32 (float), Gray/YUV/RGB.");
    }
    d.bits = bits;
    d.half = fmt.sampleType == .Float and bits == 16;
    d.bytes = @intCast(fmt.bytesPerSample);

    const device_id = map_in.getValue(i32, "device_id") orelse 0;
    if (device_id < 0) return map_out.setError("GaussBlur: invalid device ID.");
    const platform_id = map_in.getValue(i32, "platform_id") orelse 0;
    if (platform_id < 0) return map_out.setError("GaussBlur: invalid platform ID.");
    const ns_req = map_in.getValue(i32, "num_streams");
    if (ns_req) |ns| if (ns < 1 or ns > 32) {
        return map_out.setError("GaussBlur: num_streams must be 1..32.");
    };

    const subW: u5 = @intCast(fmt.subSamplingW);
    const subH: u5 = @intCast(fmt.subSamplingH);
    var sigma: [3]f32 = undefined;
    {
        var i: usize = 0;
        while (i < 3) : (i += 1) {
            if (map_in.getValue2(f32, "sigma", i)) |given| {
                if (!math.isFinite(given) or given < 0)
                    return map_out.setError("GaussBlur: sigma must be a finite value >= 0.");
                sigma[i] = given;
            } else if (i == 0) {
                sigma[i] = 0.5;
            } else if (i == 1) {
                const prod = (@as(u32, 1) << subH) * (@as(u32, 1) << subW);
                const sub_factor = @sqrt(@as(f64, @floatFromInt(prod)));
                sigma[i] = @floatCast(@as(f64, sigma[0]) / sub_factor);
            } else {
                sigma[i] = sigma[i - 1];
            }
        }
    }

    const num_planes: usize = @intCast(fmt.numPlanes);
    var any_proc = false;
    {
        var i: usize = 0;
        while (i < 3) : (i += 1) {
            d.process[i] = i < num_planes and sigma[i] >= FLT_EPSILON;
            if (d.process[i]) any_proc = true;
        }
    }
    if (!any_proc) {
        return map_out.setError("GaussBlur: all planes have sigma < FLT_EPSILON (nothing to process).");
    }

    const strides = vszipcl.strideFromVi(d.vi);
    d.n_cfg = 0;
    defer {
        var wi: usize = 0;
        while (wi < d.n_cfg) : (wi += 1) allocator.free(d.configs[wi].weights);
    }
    {
        var pi: usize = 0;
        while (pi < num_planes) : (pi += 1) {
            if (!d.process[pi]) continue;
            const key: Config.Key = .{
                .w = if (pi == 0) d.vi.width else d.vi.width >> subW,
                .h = if (pi == 0) d.vi.height else d.vi.height >> subH,
                .stride = @intCast(if (pi == 0) strides[0] else strides[1]),
                .sigma = sigma[pi],
            };
            var ci: usize = 0;
            while (ci < d.n_cfg) : (ci += 1) {
                if (std.meta.eql(key, d.configs[ci].key)) break;
            }
            if (ci == d.n_cfg) {
                if (key.sigma > @as(f32, @floatFromInt(@min(key.w, key.h)))) {
                    return map_out.setError("GaussBlur: sigma too large for plane (radius >= dimension).");
                }
                const weights = getGaussKernel(key.sigma) catch unreachable;
                const ksize: i32 = @intCast(weights.len);
                const radius = @divTrunc(ksize, 2);
                if (radius > key.w - 1 or radius > key.h - 1) {
                    allocator.free(weights);
                    return map_out.setError("GaussBlur: sigma too large for plane (radius >= dimension).");
                }
                d.configs[ci] = .{
                    .key = key,
                    .ksize = ksize,
                    .radius = radius,
                    .mode = .small,
                    .weights = weights,
                };
                d.n_cfg += 1;
            }
            d.plane_cfg[pi] = ci;
        }
    }

    vszipcl.initContext(&d, @intCast(device_id), @intCast(platform_id)) catch |err| {
        map_out.setError(if (err == error.InvalidDeviceID) "GaussBlur: invalid device ID." else if (err == error.InvalidPlatformID) "GaussBlur: invalid platform ID." else "GaussBlur: OpenCL initialization failed.");
        std.log.err("GaussBlur OpenCL init failed: {}", .{err});
        return;
    };

    d.blk_x = large_blk_x;
    d.blk_y = large_blk_y;
    d.vrt = 3;
    d.large_r = large_r;
    var thr: i32 = large_path_radius_threshold;
    const streams: usize = if (ns_req) |ns| @intCast(ns) else 1;
    d.use_pinned = streams >= 2;
    d.bake_large = true;
    {
        var terr: ?[:0]const u8 = null;
        if (map_in.numElements("tune")) |n| {
            if (n > tune_len) terr = "GaussBlur: tune expects at most 7 entries [blk_x, blk_y, vrt, large_r, threshold, pinned, bake].";
        }
        if (vszipcl.tuneEntry(map_in, 0)) |v| {
            if (v < 1 or v > 64) terr = "GaussBlur: tune[0] (blk_x) must be 1..64." else d.blk_x = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 1)) |v| {
            if (v < 1 or v > 64) terr = "GaussBlur: tune[1] (blk_y) must be 1..64." else d.blk_y = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 2)) |v| {
            if (v < 1 or v > 8) terr = "GaussBlur: tune[2] (vrt) must be 1..8." else d.vrt = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 3)) |v| {
            if (v < 1 or v > 32) terr = "GaussBlur: tune[3] (large_r) must be 1..32." else d.large_r = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 4)) |v| {
            if (v > 256) terr = "GaussBlur: tune[4] (threshold) must be 0..256 (0 = always large path)." else thr = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 5)) |v| {
            if (v > 1) terr = "GaussBlur: tune[5] (pinned) must be 0 or 1." else d.use_pinned = v != 0;
        }
        if (vszipcl.tuneEntry(map_in, 6)) |v| {
            if (v > 1) terr = "GaussBlur: tune[6] (bake) must be 0 or 1." else d.bake_large = v != 0;
        }
        if (d.blk_x * d.blk_y > vszipcl.deviceMaxWG(d.device)) terr = "GaussBlur: tune blk_x*blk_y exceeds the device max work-group size.";
        if (terr) |msg| {
            map_out.setError(msg);
            d.context.release();
            return;
        }
    }

    d.buff_elems = 0;
    d.any_large = false;
    {
        const local_mem = vszipcl.deviceLocalMemSize(d.device);
        var ci: usize = 0;
        while (ci < d.n_cfg) : (ci += 1) {
            const cfg = &d.configs[ci];
            const tile_bytes = d.vrt * d.blk_y * (d.blk_x + 2 * @as(usize, @intCast(cfg.radius))) * @sizeOf(f32);
            cfg.mode = if (thr == 0 or cfg.radius > thr or tile_bytes > local_mem) .large else .small;
            d.buff_elems = @max(d.buff_elems, cfg.extent());
            if (cfg.mode == .large) d.any_large = true;
        }
        std.debug.assert(d.buff_elems > 0);
    }

    d.stage_sum = 0;
    for (0..3) |sp| {
        d.stage_off[sp] = d.stage_sum;
        if (sp < num_planes and d.process[sp])
            d.stage_sum += d.configs[d.plane_cfg[sp]].extent() * d.bytes;
    }
    d.stage_sum = @max(d.stage_sum, 1);

    d.pool = .{};
    var nproc: usize = 0;
    for (0..3) |pp| {
        if (pp < num_planes and d.process[pp]) nproc += 1;
    }
    d.memcpy_sem = .{ .permits = if (streams <= 2 or nproc > 1) streams else 1 };

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    data.pool.prime(data, streams);
    data.pool.prewarm(streams) catch |err| {
        map_out.setError("GaussBlur: OpenCL stream init failed.");
        std.log.err("OpenCL stream init failed: {}", .{err});
        data.pool.deinit();
        data.context.release();
        allocator.destroy(data);
        return;
    };

    keep = true;

    var dep = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    zapi.createVideoFilter(out, "GaussBlur", d.vi, getFrame, free, .Parallel, &dep, data);
}

fn getGaussKernel(sigma: f32) ![]f32 {
    var taps: usize = @intFromFloat(@ceil(sigma * 6 + 1));
    if (taps % 2 == 0) {
        taps += 1;
    }

    var kernel = try std.ArrayList(f64).initCapacity(allocator, taps);
    defer kernel.deinit(allocator);

    const half_taps = @divFloor(taps, 2);
    var x: usize = 0;
    while (x < half_taps) : (x += 1) {
        const x_f64 = @as(f64, @floatFromInt(x));
        const value = 1.0 / (@sqrt(2.0 * math.pi) * sigma) *
            @exp(-(x_f64 * x_f64) / (2 * sigma * sigma));
        try kernel.append(allocator, value);
    }

    const first_value = kernel.items[0];
    for (kernel.items[1..]) |*item| {
        item.* *= 1 / first_value;
    }
    kernel.items[0] = 1;

    var full_kernel = try std.ArrayList(f64).initCapacity(allocator, taps);
    defer full_kernel.deinit(allocator);

    var i: usize = kernel.items.len;
    while (i > 0) : (i -= 1) {
        try full_kernel.append(allocator, kernel.items[i - 1]);
    }
    try full_kernel.appendSlice(allocator, kernel.items[1..]);

    var sum: f64 = 0;
    for (full_kernel.items) |v| sum += v;

    const out_kernel = try allocator.alloc(f32, full_kernel.items.len);
    for (out_kernel, full_kernel.items) |*s, f| {
        s.* = @floatCast(f / sum);
    }

    return out_kernel;
}
