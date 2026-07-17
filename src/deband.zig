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

const max_iterations: i32 = 32;

const tune_len = 3;

const dlut_sizeb = 6;
const dlut_size = 1 << dlut_sizeb;
const dlut_len = dlut_size * dlut_size;

const CrtRand = struct {
    state: u32 = 1,
    fn next(self: *CrtRand) u32 {
        self.state = self.state *% 214013 +% 2531011;
        return (self.state >> 16) & 0x7fff;
    }
};

const libm = struct {
    extern fn exp(x: f64) f64;
    extern fn log(x: f64) f64;
};

const u64_max_f: f64 = 18446744073709551616.0;

const bn_radius = dlut_size / 2 - 1;
const bn_middle = bn_radius | (bn_radius << dlut_sizeb);
const bn_gsize = bn_radius * 2 + 1;
const bn_gsize2 = bn_gsize * bn_gsize;

fn bnXY(x: usize, y: usize) usize {
    return x | (y << dlut_sizeb);
}

const BnCtx = struct {
    gauss: [dlut_len]u64,
    randomat: [dlut_len]u32,
    calcmat: [dlut_len]bool,
    gaussmat: [dlut_len]u64,
    unimat: [dlut_len]u32,
    rng: CrtRand,
};

fn bnMakegauss(k: *BnCtx) void {
    @memset(&k.gauss, 0);
    const sigma = -libm.log(1.5 / u64_max_f * @as(f64, bn_gsize2)) / @as(f64, bn_radius);
    var gy: usize = 0;
    while (gy <= bn_radius) : (gy += 1) {
        var gx: usize = 0;
        while (gx <= gy) : (gx += 1) {
            const cx: i64 = @as(i64, @intCast(gx)) - bn_radius;
            const cy: i64 = @as(i64, @intCast(gy)) - bn_radius;
            const sq: f64 = @floatFromInt(cx * cx + cy * cy);
            const e = libm.exp(-@sqrt(sq) * sigma);
            const v: u64 = @intFromFloat(e / @as(f64, bn_gsize2) * u64_max_f);
            const g1 = bn_gsize - 1;
            k.gauss[bnXY(gx, gy)] = v;
            k.gauss[bnXY(gy, gx)] = v;
            k.gauss[bnXY(gx, g1 - gy)] = v;
            k.gauss[bnXY(gy, g1 - gx)] = v;
            k.gauss[bnXY(g1 - gx, gy)] = v;
            k.gauss[bnXY(g1 - gy, gx)] = v;
            k.gauss[bnXY(g1 - gx, g1 - gy)] = v;
            k.gauss[bnXY(g1 - gy, g1 - gx)] = v;
        }
    }
}

fn bnSetbit(k: *BnCtx, c: usize) void {
    if (k.calcmat[c]) return;
    k.calcmat[c] = true;
    var m: usize = 0;
    var g: usize = (bn_middle + dlut_len - c) & (dlut_len - 1);
    while (g < dlut_len) : (g += 1) {
        k.gaussmat[m] += k.gauss[g];
        m += 1;
    }
    g = 0;
    while (m < dlut_len) : (m += 1) {
        k.gaussmat[m] += k.gauss[g];
        g += 1;
    }
}

fn bnGetmin(k: *BnCtx) usize {
    var min: u64 = math.maxInt(u64);
    var resnum: u32 = 0;
    for (0..dlut_len) |c| {
        if (k.calcmat[c]) continue;
        const total = k.gaussmat[c];
        if (total <= min) {
            if (total != min) {
                min = total;
                resnum = 0;
            }
            k.randomat[resnum] = @intCast(c);
            resnum += 1;
        }
    }
    if (resnum == 1) return k.randomat[0];
    if (resnum == dlut_len) return dlut_len / 2;
    return k.randomat[k.rng.next() % resnum];
}

fn generateBlueNoise(data: []f32) !void {
    const k = try allocator.create(BnCtx);
    defer allocator.destroy(k);
    k.rng = .{};
    @memset(&k.calcmat, false);
    @memset(&k.gaussmat, 0);
    bnMakegauss(k);
    for (0..dlut_len) |c| {
        const r = bnGetmin(k);
        bnSetbit(k, r);
        k.unimat[r] = @intCast(c);
    }
    for (0..dlut_size) |y| {
        for (0..dlut_size) |x|
            data[x + y * dlut_size] = @as(f32, @floatFromInt(k.unimat[bnXY(x, y)])) / @as(f32, dlut_len);
    }
}

fn generateBayer(data: []f32) void {
    data[0] = 0;
    var sz: usize = 1;
    while (sz < dlut_size) : (sz *= 2) {
        for (0..sz) |y| {
            for (0..sz) |x| {
                const pos = y * dlut_size + x;
                const offs = [3]usize{ sz * dlut_size + sz, sz, sz * dlut_size };
                for (offs, 1..) |off, i| {
                    const inc = @as(f64, @floatFromInt(i)) / (4.0 * @as(f64, @floatFromInt(sz * sz)));
                    data[pos + off] = @floatCast(@as(f64, data[pos]) + inc);
                }
            }
        }
    }
}

const Data = struct {
    node: ?*vs.Node,
    vi: *const vs.VideoInfo,

    platform: cl.Platform,
    device: cl.Device,
    context: cl.Context,

    pw: [3]i32,
    ph: [3]i32,
    pstride: [3]i32,
    process: [3]bool,
    rank: [3]i32,
    rank_plane: [3]i32,
    n_proc: i32,

    off_reg: [3]usize,
    buff_size: usize,
    bits: i32,
    half: bool,
    bytes: u32,

    use_pinned: bool,
    memcpy_sem: std.Io.Semaphore,

    iterations: [3]i32,
    threshold_s: [3]f32,
    radius: [3]f32,
    grain_s: [3]f32,
    grain_on: [3]bool,
    fused: bool,
    n_prog: usize,
    plane_prog: [3]usize,
    prog_iter: [3]i32,
    prog_grain: [3]bool,

    dither_on: bool,
    dmode: i32,
    dlut: ?[]f32,

    blk_x: usize,
    blk_y: usize,

    gws_w: usize,
    gws_h: usize,

    pool: clpool.Pool(Stream, Data),

};

const Stream = struct {
    programs: [3]cl.Program,
    n_prog: usize,
    queue: cl.CommandQueue,
    src: cl.Buffer(u8),
    dst: cl.Buffer(u8),
    geom: cl.Buffer(i32),
    dlut: ?cl.Buffer(f32),
    stage: ?cl.Buffer(u8),
    host: []u8,
    kerns: [3]cl.Kernel,
    n_kern: usize,

    pub fn init(self: *Stream, d: *Data) !void {
        self.n_prog = 0;
        self.n_kern = 0;
        self.dlut = null;
        self.stage = null;
        errdefer {
            var m: usize = 0;
            while (m < self.n_prog) : (m += 1) self.programs[m].release();
        }
        var blk_buf: [48]u8 = undefined;
        const blk_opt: []const u8 = if (d.blk_x != 16 or d.blk_y != 8)
            std.fmt.bufPrint(&blk_buf, " -DDB_BX={d} -DDB_BY={d}", .{ d.blk_x, d.blk_y }) catch unreachable
        else
            "";
        for (0..d.n_prog) |m| {
            self.programs[m] = try cl.createProgramWithSource(d.context, deband_src);
            self.n_prog = m + 1;
            const opts = try std.fmt.allocPrintSentinel(allocator, "-cl-std=CL1.2 -DITER={d} -DGRAIN_ON={d} -DBITS={d} -DHALF={d} -DDITHERK={d} -DDMODE={d}{s}", .{ d.prog_iter[m], @intFromBool(d.prog_grain[m]), d.bits, @intFromBool(d.half), @intFromBool(d.dither_on), d.dmode, blk_opt }, 0);
            defer allocator.free(opts);
            self.programs[m].build(&.{d.device}, opts) catch |err| {
                if (err == error.BuildProgramFailure) {
                    const log = try self.programs[m].getBuildLog(allocator, d.device);
                    defer allocator.free(log);
                    std.log.err("Deband OpenCL build failed: {s}", .{log});
                }
                return err;
            };
        }

        self.queue = try cl.createCommandQueue(d.context, d.device, .{});
        errdefer self.queue.release();
        self.src = try cl.createBuffer(u8, d.context, .{ .read_only = true }, d.buff_size * d.bytes);
        errdefer self.src.release();
        self.dst = try cl.createBuffer(u8, d.context, .{ .read_write = true }, d.buff_size * d.bytes);
        errdefer self.dst.release();

        self.geom = try cl.createBuffer(i32, d.context, .{ .read_only = true }, 12);
        errdefer self.geom.release();
        {
            var g: [12]i32 = @splat(0);
            var r: usize = 0;
            while (r < @as(usize, @intCast(d.n_proc))) : (r += 1) {
                const p: usize = @intCast(d.rank_plane[r]);
                g[r * 4 + 0] = d.pw[p];
                g[r * 4 + 1] = d.ph[p];
                g[r * 4 + 2] = d.pstride[p];
                g[r * 4 + 3] = 0;
            }
            if (cl.c.clEnqueueWriteBuffer(self.queue.handle, self.geom.handle, cl.c.CL_TRUE, 0, 12 * @sizeOf(i32), &g, 0, null, null) != cl.c.CL_SUCCESS)
                return error.EnqueueWrite;
        }

        if (d.dlut != null)
            self.dlut = try cl.createBuffer(f32, d.context, .{ .read_only = true }, dlut_len);
        errdefer if (self.dlut) |b| b.release();
        if (d.dlut) |lut| {
            if (cl.c.clEnqueueWriteBuffer(self.queue.handle, self.dlut.?.handle, cl.c.CL_TRUE, 0, dlut_len * @sizeOf(f32), lut.ptr, 0, null, null) != cl.c.CL_SUCCESS)
                return error.EnqueueWrite;
        }

        if (d.use_pinned) blk: {
            const st = cl.createBuffer(u8, d.context, .{ .alloc_host_ptr = true }, 2 * d.buff_size * d.bytes) catch break :blk;
            var map_err: cl.c.cl_int = cl.c.CL_SUCCESS;
            const map_ptr = cl.c.clEnqueueMapBuffer(self.queue.handle, st.handle, cl.c.CL_TRUE, cl.c.CL_MAP_READ | cl.c.CL_MAP_WRITE, 0, 2 * d.buff_size * d.bytes, 0, null, null, &map_err);
            if (map_err != cl.c.CL_SUCCESS or map_ptr == null) {
                st.release();
                std.log.warn("Deband: pinned staging unavailable; this stream runs pageable transfers.", .{});
                break :blk;
            }
            self.stage = st;
            self.host = @as([*]u8, @ptrCast(map_ptr.?))[0 .. 2 * d.buff_size * d.bytes];
        }
        errdefer if (self.stage) |st| {
            _ = cl.c.clEnqueueUnmapMemObject(self.queue.handle, st.handle, self.host.ptr, 0, null, null);
            _ = cl.c.clFinish(self.queue.handle);
            st.release();
        };

        errdefer {
            var kk: usize = 0;
            while (kk < self.n_kern) : (kk += 1) self.kerns[kk].release();
        }
        const nk: usize = if (d.fused) 1 else @intCast(d.n_proc);
        for (0..nk) |r| {
            const p: usize = @intCast(d.rank_plane[r]);
            self.kerns[r] = try cl.createKernel(self.programs[d.plane_prog[p]], "deband");
            self.n_kern = r + 1;
            try self.kerns[r].setArg(@TypeOf(self.dst), 0, self.dst);
            try self.kerns[r].setArg(@TypeOf(self.src), 1, self.src);
            try self.kerns[r].setArg(@TypeOf(self.geom), 2, self.geom);
            try self.kerns[r].setArg(f32, 3, d.threshold_s[p]);
            try self.kerns[r].setArg(f32, 4, d.radius[p]);
            try self.kerns[r].setArg(f32, 5, d.grain_s[p]);
            if (self.dlut != null) try self.kerns[r].setArg(cl.Buffer(f32), 7, self.dlut.?);
        }
    }

    pub fn deinit(self: *Stream) void {
        if (self.stage) |st| _ = cl.c.clEnqueueUnmapMemObject(self.queue.handle, st.handle, self.host.ptr, 0, null, null);
        _ = cl.c.clFinish(self.queue.handle);
        var kk: usize = 0;
        while (kk < self.n_kern) : (kk += 1) self.kerns[kk].release();
        if (self.dlut) |b| b.release();
        if (self.stage) |st| st.release();
        self.geom.release();
        self.dst.release();
        self.src.release();
        self.queue.release();
        var m: usize = 0;
        while (m < self.n_prog) : (m += 1) self.programs[m].release();
    }
};

const deband_src =
    \\#define TWO_PI 6.283185f
    \\#define INV_U32 0x1p-32f
    \\
    \\#ifndef DB_BX
    \\#define DB_BX 16
    \\#endif
    \\#ifndef DB_BY
    \\#define DB_BY 8
    \\#endif
    \\
    \\// LOADI/STOREI take (pointer, index) because the HALF path cannot dereference a half*
    \\// (cl_khr_fp16 is not exposed on NVIDIA; pointers + vload/vstore_half are core CL).
    \\#if BITS == 32
    \\typedef float io_t;
    \\#define LOADI(p, i) ((p)[i])
    \\#define STOREI(p, i, x) ((p)[i] = (x))
    \\#elif BITS == 16 && HALF
    \\// r16f texture parity: sampling widens f16->f32 EXACTLY (every half is a float), so
    \\// the float pipeline stays bit-identical to BITS==32. The render-target store is
    \\// MEASURED to be RTZ (truncate toward zero), NOT round-to-nearest: an RTE store
    \\// disagreed with placebo on ~50% of grain pixels by exactly 1 half-ulp, while RTZ
    \\// matches 0/2M on both positive and signed probes (RTN excluded by the signed one).
    \\typedef half io_t;
    \\#define LOADI(p, i) vload_half((size_t)(i), p)
    \\#define STOREI(p, i, x) vstore_half_rtz((x), (size_t)(i), p)
    \\#elif BITS == 16
    \\typedef ushort io_t;
    \\#define LOADI(p, i) (convert_float((p)[i]) / 65535.0f)
    \\#define STOREI(p, i, x) ((p)[i] = db_store16(x))
    \\#else
    \\typedef uchar io_t;
    \\#define LOADI(p, i) (convert_float((p)[i]) / 255.0f)
    \\#define STOREI(p, i, x) ((p)[i] = db_store8(x))
    \\#endif
    \\
    \\#if BITS != 32 && !HALF
    \\// NVIDIA's Vulkan UNORM render-target store, MEASURED (diag_store2..4 + the k=2048 tie
    \\// probe): the clamped float goes through an (n+4)-bit fixed-point intermediate with RTZ
    \\// (truncation!), then is scaled to n bits rounding half-DOWN:
    \\//     q = (k*(2^n-1) + 2^(n+3)-1) >> (n+4),   k = trunc(clamp(x,0,1) * 2^(n+4)).
    \\// The obvious convert_*_sat_rte(x*(2^n-1)) disagreed with placebo on ~4% of grain
    \\// pixels (one-sided -1: boundary shifted by up to 2^-(n+4) ≈ 1/16 code — legal under
    \\// the spec's 0.6-ULP float->unorm tolerance). The only exact tie is k = 2^(n+3)
    \\// (x==0.5 -> q*2 == 2^n-1, i.e. 127.5/32767.5), where hardware stores the LOW value
    \\// (127, not RTE's 128) — hence the -1 bias, which changes ONLY that tie. This form
    \\// leaves ~0.01% ulp-level upstream residual. pow2 scales are exact in fp32; the
    \\// (uint) float->int conversion is RTZ.
    \\inline uchar db_store8(float x) {
    \\    uint k = (uint)(clamp(x, 0.0f, 1.0f) * 4096.0f);          // 0.12 fixed, k <= 4096
    \\    return (uchar)((k * 255u + 2047u) >> 12);
    \\}
    \\inline ushort db_store16(float x) {
    \\    uint k = (uint)(clamp(x, 0.0f, 1.0f) * 1048576.0f);       // 0.20 fixed, k <= 2^20
    \\    return (ushort)(((ulong)k * 65535ul + 524287ul) >> 20);   // k*65535 overflows uint
    \\}
    \\#endif
    \\
    \\inline uint3 db_pcg3d(uint3 v) {
    \\    v = v * 1664525u + 1013904223u;
    \\    v.x += v.y * v.z;
    \\    v.y += v.z * v.x;
    \\    v.z += v.x * v.y;
    \\    v ^= v >> 16u;
    \\    v.x += v.y * v.z;
    \\    v.y += v.z * v.x;
    \\    v.z += v.x * v.y;
    \\    return v;
    \\}
    \\
    \\inline float db_get(global const io_t *src, int W, int H, int STRIDE,
    \\                    int px, int py, float ox, float oy) {
    \\    // Nearest texel with clamp-to-edge: texel = clamp(floor(pos*size), 0, size-1) where
    \\    // pos*size = px + 0.5 + off. Independent per call (do NOT reuse a negated offset).
    \\    // DELIBERATELY the exact formula: placebo's rasterizer-interpolated `pos` jitters
    \\    // ~U[0, 2^-21]*dim BELOW this, causing the ~0.03% iterations>=1 divergence — see
    \\    // the "Known divergence" header section; a mean-bias correction was tested (2x
    \\    // fewer flips) and REVERTED. Keep this exact; do not re-add a bias.
    \\    int cx = clamp((int)floor((float)px + 0.5f + ox), 0, W - 1);
    \\    int cy = clamp((int)floor((float)py + 0.5f + oy), 0, H - 1);
    \\    return LOADI(src, cy * STRIDE + cx);   // plane extent < 2^31 (gated at create()), int is safe
    \\}
    \\
    \\// The full pl_shader_deband body (shared by both kernels; expressions are VERBATIM the
    \\// verified float-path ones — for BITS==32 this is a pure code motion, output-identical).
    \\inline float db_core(global const io_t *src, int W, int H, int STRIDE,
    \\                     float threshold, float radius, float grain,
    \\                     uint zseed, int gx, int gy) {
    \\    uint3 state = (uint3)((uint)gx, (uint)gy, zseed);   // == uvec3(gl_FragCoord.xy, index)
    \\    float res = LOADI(src, gy * STRIDE + gx);            // center (nearest at pixel center)
    \\
    \\#if ITER > 0
    \\    for (int i = 1; i <= ITER; ++i) {
    \\        state = db_pcg3d(state);
    \\        float3 r = convert_float3(state) * INV_U32;       // uint->float RN, then *2^-32
    \\        float dist  = r.x * ((float)i * radius);
    \\        float theta = r.y * TWO_PI;
    \\        float dx = dist * native_cos(theta);
    \\        float dy = dist * native_sin(theta);
    \\        float avg = 0.0f;                                 // keep this exact 4-tap order + *0.25
    \\        avg += db_get(src, W, H, STRIDE, gx, gy,  dx,  dy);
    \\        avg += db_get(src, W, H, STRIDE, gx, gy, -dx,  dy);
    \\        avg += db_get(src, W, H, STRIDE, gx, gy, -dx, -dy);
    \\        avg += db_get(src, W, H, STRIDE, gx, gy,  dx, -dy);
    \\        avg *= 0.25f;
    \\        float diff  = fabs(res - avg);
    \\        float bound = threshold / (float)i;
    \\        res = (diff > bound) ? res : avg;                 // mix(avg, res, diff>bound)
    \\    }
    \\#endif
    \\
    \\#if GRAIN_ON
    \\    state = db_pcg3d(state);
    \\    float3 g = convert_float3(state) * INV_U32;
    \\    float strength = fmin(fabs(res), grain);              // grain_neutral == 0
    \\    res += strength * (g.x - 0.5f);                       // T(rand()) for 1 comp == .x
    \\#endif
    \\
    \\    return res;
    \\}
    \\
    \\kernel __attribute__((reqd_work_group_size(DB_BX, DB_BY, 1)))
    \\void deband(global io_t *dst, global const io_t *src, global const int *geom,
    \\            float threshold, float radius, float grain, uint zbase
    \\#if DITHERK && (DMODE == 0 || DMODE == 1)
    \\            , global const float *dlut
    \\#endif
    \\            ) {
    \\    const int z = get_global_id(2);
    \\    const int W = geom[z * 4], H = geom[z * 4 + 1];
    \\    const int STRIDE = geom[z * 4 + 2];
    \\    const int gx = get_global_id(0);
    \\    const int gy = get_global_id(1);
    \\    if (gx >= W || gy >= H) return;
    \\    long OFF = 0;
    \\    for (int r = 0; r < z; ++r) OFF += (long)geom[r * 4 + 1] * (long)geom[r * 4 + 2];
    \\    global const io_t *srcp = src + OFF;
    \\    global io_t *dstp = dst + OFF;
    \\    const uint zseed = (zbase + (uint)z) & 0xFFu;
    \\    float res = db_core(srcp, W, H, STRIDE, threshold, radius, grain, zseed, gx, gy);
    \\
    \\#if DITHERK
    \\    if (z == 0) {
    \\        float bias;
    \\#if DMODE == 0 || DMODE == 1
    \\        bias = dlut[((gy & 63) << 6) | (gx & 63)];
    \\#elif DMODE == 2
    \\        uint bx = ((uint)gx & 15u) ^ ((uint)gy & 15u);
    \\        uint by = (uint)gy & 15u;
    \\        bx = (bx | (bx << 2)) & 0x33333333u;
    \\        by = (by | (by << 2)) & 0x33333333u;
    \\        bx = (bx | (bx << 1)) & 0x55555555u;
    \\        by = (by | (by << 1)) & 0x55555555u;
    \\        uint b = bx + (by << 1);
    \\        b = (b * 0x0802u & 0x22110u) | (b * 0x8020u & 0x88440u);
    \\        b = 0x10101u * b;
    \\        b = (b >> 16) & 0xFFu;
    \\        bias = convert_float(b) * (1.0f / 256.0f);
    \\#else
    \\        uint3 ws = db_pcg3d((uint3)((uint)gx, (uint)gy, 0u));
    \\        bias = convert_float(ws.x) * INV_U32;
    \\#endif
    \\        res = (fabs(res) < 1e-5f) ? 0.0f : res;
    \\        float q = floor(255.0f * res + bias);
    \\        dstp[gy * STRIDE + gx] = convert_uchar_sat(q);
    \\        return;
    \\    }
    \\#endif
    \\
    \\    STOREI(dstp, gy * STRIDE + gx, res);
    \\}
;

const ndr = vszipcl.ndr;

const ZFrame = @typeInfo(@TypeOf(ZAPI.initZFrame)).@"fn".return_type.?;
const ZFrameW = @typeInfo(@TypeOf(ZFrame.newVideoFrame)).@"fn".return_type.?;

fn enqueueDeband(d: *Data, s: *Stream) !void {
    const lws: [3]usize = .{ d.blk_x, d.blk_y, 1 };
    if (d.fused) {
        const gws: [3]usize = .{ vszipcl.ceilTo(d.gws_w, d.blk_x), vszipcl.ceilTo(d.gws_h, d.blk_y), @intCast(d.n_proc) };
        try ndr(s, s.kerns[0], &gws, &lws);
    } else {
        for (0..@as(usize, @intCast(d.n_proc))) |r| {
            const p: usize = @intCast(d.rank_plane[r]);
            const off: [3]usize = .{ 0, 0, r };
            const gws: [3]usize = .{ vszipcl.ceilTo(@intCast(d.pw[p]), d.blk_x), vszipcl.ceilTo(@intCast(d.ph[p]), d.blk_y), 1 };
            if (cl.c.clEnqueueNDRangeKernel(s.queue.handle, s.kerns[r].handle, 3, &off, &gws, &lws, 0, null, null) != cl.c.CL_SUCCESS)
                return error.EnqueueKernel;
        }
    }
}

fn process(d: *Data, s: *Stream, src: ZFrame, dst: ZFrameW, n: c_int) !void {
    errdefer _ = cl.c.clFinish(s.queue.handle);

    if (d.n_proc == 0) return;

    const num_planes: u32 = @intCast(d.vi.format.numPlanes);
    const span: usize = d.buff_size * d.bytes;

    const zbase: c_uint = @intCast(@mod(@as(i64, n) * @as(i64, d.n_proc), 256));
    for (0..s.n_kern) |kk| try s.kerns[kk].setArg(c_uint, 6, zbase);

    if (s.stage != null) {
        d.memcpy_sem.waitUncancelable(vszipcl.io);
        var p: u32 = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) continue;
            const srcp = src.getReadSlice(p);
            if (srcp.len != @as(usize, @intCast(d.ph[p])) * @as(usize, @intCast(d.pstride[p])) * d.bytes) {
                d.memcpy_sem.post(vszipcl.io);
                return error.PlaneLayoutMismatch;
            }
            @memcpy(s.host[d.off_reg[p] * d.bytes ..][0..srcp.len], srcp);
        }
        d.memcpy_sem.post(vszipcl.io);
        try vszipcl.enqWrite(s.queue, s.src.handle, 0, s.host[0..span]);
        try enqueueDeband(d, s);
        try vszipcl.enqRead(s.queue, s.dst.handle, 0, s.host[span..][0..span]);
    } else {
        var p: u32 = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) continue;
            const srcp = src.getReadSlice(p);
            std.debug.assert(srcp.len == @as(usize, @intCast(d.ph[p])) * @as(usize, @intCast(d.pstride[p])) * d.bytes);
            try vszipcl.enqWrite(s.queue, s.src.handle, d.off_reg[p] * d.bytes, srcp);
        }
        try enqueueDeband(d, s);
        p = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) continue;
            try vszipcl.enqRead(s.queue, s.dst.handle, d.off_reg[p] * d.bytes, dst.getWriteSlice(p));
        }
    }

    if (cl.c.clFinish(s.queue.handle) != cl.c.CL_SUCCESS) return error.Finish;

    if (s.stage != null) {
        d.memcpy_sem.waitUncancelable(vszipcl.io);
        var p: u32 = 0;
        while (p < num_planes) : (p += 1) {
            if (!d.process[p]) continue;
            const dstp = dst.getWriteSlice(p);
            @memcpy(dstp, s.host[span + d.off_reg[p] * d.bytes ..][0..dstp.len]);
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

        process(d, s, src, dst, n) catch |err| {
            zapi.setFilterError("Deband: process frame failed.");
            std.log.err("Deband process frame failed: {}", .{err});
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
    if (d.dlut) |lut| allocator.free(lut);
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
        return map_out.setError("Deband: input bitdepth must be 8/16 (integer), 16 (half) or 32 (float), Gray/YUV/RGB.");
    d.bits = bits;
    d.half = fmt.sampleType == .Float and bits == 16;
    d.bytes = @intCast(fmt.bytesPerSample);

    var iterations: [3]i32 = undefined;
    var threshold: [3]f32 = undefined;
    var radius: [3]f32 = undefined;
    var grain: [3]f32 = undefined;
    for (0..3) |i| {
        iterations[i] = map_in.getValue2(i32, "iterations", i) orelse if (i > 0) iterations[i - 1] else 1;
        threshold[i] = map_in.getValue2(f32, "threshold", i) orelse if (i > 0) threshold[i - 1] else 3.0;
        radius[i] = map_in.getValue2(f32, "radius", i) orelse if (i > 0) radius[i - 1] else 16.0;
        grain[i] = map_in.getValue2(f32, "grain", i) orelse if (i > 0) grain[i - 1] else 4.0;

        if (iterations[i] < 0 or iterations[i] > max_iterations) return map_out.setError("Deband: iterations must be 0..32.");
        if (!math.isFinite(threshold[i]) or threshold[i] < 0.0) return map_out.setError("Deband: threshold must be a finite value >= 0.");
        if (!math.isFinite(radius[i]) or radius[i] < 0.0) return map_out.setError("Deband: radius must be a finite value >= 0.");
        if (!math.isFinite(grain[i]) or grain[i] < 0.0) return map_out.setError("Deband: grain must be a finite value >= 0.");
    }

    var sel = [3]bool{ true, true, true };
    if (map_in.numElements("planes")) |ne| {
        sel = .{ false, false, false };
        var e: u32 = 0;
        while (e < ne) : (e += 1) {
            const idx = map_in.getValue2(i32, "planes", e).?;
            if (idx < 0 or idx >= fmt.numPlanes) return map_out.setError("Deband: plane index out of range.");
            const ui: usize = @intCast(idx);
            if (sel[ui]) return map_out.setError("Deband: plane specified twice.");
            sel[ui] = true;
        }
    }
    const dither_req = map_in.getValue(i32, "dither");
    d.dither_on = (if (dither_req) |dv| dv != 0 else true) and bits == 8;
    const dither_algo = map_in.getValue(i32, "dither_algo") orelse 0;
    if (dither_algo < 0 or dither_algo > 3) return map_out.setError("Deband: dither_algo must be 0..3 (blue noise / bayer / ordered fixed / white noise).");
    d.dmode = dither_algo;
    const device_id = map_in.getValue(i32, "device_id") orelse 0;
    if (device_id < 0) return map_out.setError("Deband: invalid device ID.");
    const ns_req = map_in.getValue(i32, "num_streams");
    if (ns_req) |ns| if (ns < 1 or ns > 32) return map_out.setError("Deband: num_streams must be 1..32.");

    for (0..3) |i| {
        d.iterations[i] = iterations[i];
        d.threshold_s[i] = threshold[i] / 1000.0;
        d.radius[i] = radius[i];
        d.grain_s[i] = grain[i] / 1000.0;
        d.grain_on[i] = grain[i] > 0.0;
    }

    const strides = vszipcl.strideFromVi(d.vi);
    const num_planes: usize = @intCast(fmt.numPlanes);
    const subW: u5 = @intCast(fmt.subSamplingW);
    const subH: u5 = @intCast(fmt.subSamplingH);
    var np: i32 = 0;
    var pi: usize = 0;
    var region_sum: usize = 0;
    d.gws_w = 0;
    d.gws_h = 0;
    while (pi < 3) : (pi += 1) {
        d.process[pi] = false;
        d.rank[pi] = -1;
        d.rank_plane[pi] = -1;
        d.pw[pi] = 0;
        d.ph[pi] = 0;
        d.pstride[pi] = 0;
        d.off_reg[pi] = 0;
        if (pi >= num_planes) continue;
        d.pw[pi] = if (pi == 0) @intCast(d.vi.width) else @intCast(d.vi.width >> subW);
        d.ph[pi] = if (pi == 0) @intCast(d.vi.height) else @intCast(d.vi.height >> subH);
        d.pstride[pi] = @intCast(if (pi == 0) strides[0] else strides[1]);
        if (sel[pi]) {
            d.process[pi] = true;
            d.rank[pi] = np;
            d.rank_plane[@intCast(np)] = @intCast(pi);
            np += 1;
            d.off_reg[pi] = region_sum;
            region_sum += @as(usize, @intCast(d.ph[pi])) * @as(usize, @intCast(d.pstride[pi]));
            d.gws_w = @max(d.gws_w, @as(usize, @intCast(d.pw[pi])));
            d.gws_h = @max(d.gws_h, @as(usize, @intCast(d.ph[pi])));
        }
    }
    d.n_proc = np;
    d.buff_size = @max(region_sum, 1);

    d.fused = false;
    if (np > 0) {
        const p0: usize = @intCast(d.rank_plane[0]);
        d.fused = true;
        for (0..3) |i| {
            if (!d.process[i]) continue;
            if (iterations[i] != iterations[p0] or threshold[i] != threshold[p0] or
                radius[i] != radius[p0] or grain[i] != grain[p0]) d.fused = false;
        }
    }
    d.n_prog = 0;
    d.plane_prog = .{ 0, 0, 0 };
    for (0..3) |p| {
        if (!d.process[p]) continue;
        var hit: ?usize = null;
        for (0..d.n_prog) |m| {
            if (d.prog_iter[m] == d.iterations[p] and d.prog_grain[m] == d.grain_on[p]) hit = m;
        }
        if (hit) |m| {
            d.plane_prog[p] = m;
        } else {
            d.prog_iter[d.n_prog] = d.iterations[p];
            d.prog_grain[d.n_prog] = d.grain_on[p];
            d.plane_prog[p] = d.n_prog;
            d.n_prog += 1;
        }
    }

    {
        var gp: usize = 0;
        while (gp < 3) : (gp += 1) {
            if (d.process[gp] and @as(usize, @intCast(d.ph[gp])) * @as(usize, @intCast(d.pstride[gp])) >= (1 << 31))
                return map_out.setError("Deband: frame too large (a plane exceeds 2^31 samples).");
        }
    }

    var chk: usize = 0;
    while (chk < 3) : (chk += 1)
        if (d.process[chk] and (d.pw[chk] <= 0 or d.ph[chk] <= 0)) return map_out.setError("Deband: a processed plane has zero size.");

    d.dlut = null;
    if (d.dither_on and (d.dmode == 0 or d.dmode == 1)) {
        const lut = allocator.alloc(f32, dlut_len) catch return map_out.setError("Deband: out of memory.");
        if (d.dmode == 0) generateBlueNoise(lut) catch {
            allocator.free(lut);
            return map_out.setError("Deband: out of memory.");
        } else generateBayer(lut);
        d.dlut = lut;
    }

    vszipcl.initContext(&d, @intCast(device_id)) catch |err| {
        map_out.setError(if (err == error.InvalidDeviceID) "Deband: invalid device ID." else "Deband: OpenCL initialization failed.");
        std.log.err("Deband OpenCL init failed: {}", .{err});
        if (d.dlut) |lut| allocator.free(lut);
        return;
    };

    d.blk_x = 16;
    d.blk_y = 8;
    const streams: usize = if (ns_req) |ns| @intCast(ns) else 1;
    d.use_pinned = streams >= 2;
    {
        var terr: ?[:0]const u8 = null;
        if (map_in.numElements("tune")) |n| {
            if (n > tune_len) terr = "Deband: tune expects at most 3 entries [blk_x, blk_y, pinned].";
        }
        if (vszipcl.tuneEntry(map_in, 0)) |v| {
            if (v < 1 or v > 64) terr = "Deband: tune[0] (blk_x) must be 1..64." else d.blk_x = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 1)) |v| {
            if (v < 1 or v > 64) terr = "Deband: tune[1] (blk_y) must be 1..64." else d.blk_y = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 2)) |v| {
            if (v > 1) terr = "Deband: tune[2] (pinned) must be 0 or 1." else d.use_pinned = v != 0;
        }
        if (d.blk_x * d.blk_y > vszipcl.deviceMaxWG(d.device)) terr = "Deband: tune blk_x*blk_y exceeds the device max work-group size.";
        if (terr) |msg| {
            map_out.setError(msg);
            d.context.release();
            if (d.dlut) |lut| allocator.free(lut);
            return;
        }
    }

    d.pool = .{};
    d.memcpy_sem = .{ .permits = if (streams <= 2) streams else 1 };
    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;
    keep = true;

    data.pool.prime(data, streams);
    data.pool.prewarm(streams) catch |err| {
        map_out.setError("Deband: OpenCL stream init failed.");
        std.log.err("Deband stream init failed: {}", .{err});
        data.pool.deinit();
        data.context.release();
        if (data.dlut) |lut| allocator.free(lut);
        allocator.destroy(data);
        keep = false;
        return;
    };

    var dep = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };
    zapi.createVideoFilter(out, "Deband", d.vi, getFrame, free, .Parallel, &dep, data);
}
