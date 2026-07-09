const std = @import("std");
const vszipcl = @import("vszipcl.zig");
const clpool = @import("clpool.zig");

const cl = vszipcl.cl;
const vapoursynth = vszipcl.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
const REF_LUMA: u8 = 0;
const REF_CHROMA: u8 = 1;
const REF_YUV: u8 = 2;
const REF_RGB: u8 = 3;

const kernel_src =
    \\// Wire io (see file header): io_t is the type of u1/u1z; LOADU1/STOREU1 reference the
    \\// kernels' `u1`/`u1z` parameters BY NAME (every kernel names them exactly that), so
    \\// for BITS==32 each site expands token-identically to the old bare u1[i] / u1z[i]=x
    \\// (f32 md5 regression gate). HALF uses only pointer + vload/vstore_half — a half*
    \\// must never be dereferenced without cl_khr_fp16 (not exposed on NVIDIA).
    \\#if BITS == 32
    \\typedef float io_t;
    \\#define LOADU1(i)     (u1[i])
    \\#define STOREU1(i, x) (u1z[i] = (x))
    \\#elif BITS == 16 && HALF
    \\typedef half io_t;
    \\#define LOADU1(i)     vload_half((size_t)(i), u1)
    \\#define STOREU1(i, x) vstore_half_rte((x), (size_t)(i), u1z)
    \\#elif BITS == 16
    \\typedef ushort io_t;
    \\#define LOADU1(i)     (convert_float(u1[i]) / 65535.0f)
    \\#define STOREU1(i, x) (u1z[i] = convert_ushort_sat_rte((x) * 65535.0f))
    \\#else
    \\typedef uchar io_t;
    \\#define LOADU1(i)     (convert_float(u1[i]) / 255.0f)
    \\#define STOREU1(i, x) (u1z[i] = convert_uchar_sat_rte((x) * 255.0f))
    \\#endif
    \\
    \\#define NLM_NORM        (255.0f*255.0f)
    \\#define NLM_LEGACY      (3.0f)
    \\#define NLM_S_SIZE      ((2*NLM_S+1)*(2*NLM_S+1))
    \\#define NLM_H2_INV_NORM (NLM_NORM/(NLM_LEGACY*NLM_H*NLM_H*NLM_S_SIZE))
    \\// IDX (int or long, via -DIDX) is the flat plane/layer offset type. (2d+1)*PSTRIDE*PH
    \\// can exceed 2^31 for large d on large frames, which is why these were `long`; but on
    \\// typical configs 64-bit IMAD chains on every u1/u4a access are pure codegen tax
    \\// (fast-porting.md §8b — the documented exception to good-practice #5). Stream.init
    \\// PROVES the bound per instance: it passes -DIDX=int only when the max of the u1 /
    \\// u4a / u1z flat extents is < 2^31; otherwise -DIDX=long keeps the addressing
    \\// bit-for-bit identical to the old kernel.
    \\#define U1_LAYER        ((IDX)PSTRIDE*PH)             /* io_t ELEMENTS per zero-padded frame layer */
    \\#define U1_PLANE        ((IDX)(2*NLM_D+1)*U1_LAYER)   /* io_t elements per channel (all layers) */
    \\#define U4_LAYER        ((IDX)STRIDE*VI_DIM_Y)        /* floats per weight frame layer */
    \\#define NPIX            ((IDX)STRIDE*VI_DIM_Y)        /* io_t elements per output channel */
    \\// BLK_X, BLK_Y, VRT_RESULT arrive via -D (NVIDIA sweet spot 16x8, 3 output rows per
    \\// thread) so nlmWeight's work-group + tiling stay locked to the host launch geometry
    \\// in process(). (Model: vs-nlm-cuda kernel.cu / KNL NVIDIA path.)
    \\
    \\// Joint multi-channel patch distance for the center pixel (xx,y) vs the candidate
    \\// (xx+qx, y+qy) on layers t / t+qz. The +q read uses the zero margin of the
    \\// padded buffer (== KNLMeansCL's CLK_ADDRESS_CLAMP border 0). Constants/operand
    \\// order match NLMKernel.cpp's nlmDistance exactly.
    \\static float nlm_pix_dist(global const io_t *restrict u1, const int t, const int qz,
    \\                          const int xx, const int y, const int qx, const int qy) {
    \\    const IDX ac = (IDX)t*U1_LAYER + (IDX)(y+PAD)*PSTRIDE + (xx+PAD);
    \\    const IDX bc = (IDX)(t+qz)*U1_LAYER + (IDX)(y+qy+PAD)*PSTRIDE + (xx+qx+PAD);
    \\#if   NLM_REF == 0   /* LUMA */
    \\    const float d0 = LOADU1(ac) - LOADU1(bc);
    \\    return 3.0f * (d0 * d0);   // group the square first: bit-exact with KNL's 3*pown(d0,2)
    \\#elif NLM_REF == 1   /* CHROMA (U,V) */
    \\    const float du = LOADU1(ac) - LOADU1(bc);
    \\    const float dv = LOADU1(U1_PLANE + ac) - LOADU1(U1_PLANE + bc);
    \\    return 1.5f * (du*du + dv*dv);
    \\#elif NLM_REF == 2   /* YUV */
    \\    const float dy = LOADU1(ac) - LOADU1(bc);
    \\    const float du = LOADU1(U1_PLANE + ac) - LOADU1(U1_PLANE + bc);
    \\    const float dv = LOADU1(2*U1_PLANE + ac) - LOADU1(2*U1_PLANE + bc);
    \\    return dy*dy + du*du + dv*dv;
    \\#else                /* RGB */
    \\    const float ar = LOADU1(ac), br = LOADU1(bc);
    \\    const float dr = ar - br;
    \\    const float dg = LOADU1(U1_PLANE + ac) - LOADU1(U1_PLANE + bc);
    \\    const float db = LOADU1(2*U1_PLANE + ac) - LOADU1(2*U1_PLANE + bc);
    \\    const float m_red = (ar + br) / 6.0f;
    \\    return (2.0f/3.0f + m_red) * (dr*dr) + (4.0f/3.0f) * (dg*dg) + (1.0f - m_red) * (db*db);
    \\#endif
    \\}
    \\
    \\// (input upload is a strided H2D straight into the zero-padded d_u1 interior, host-side
    \\//  in writeU1 — no on-device copy kernel; the PAD margin keeps its one-time-init zero.)
    \\
    \\// Fused patch-distance + separable box-sum (horizontal then vertical) + weight
    \\// transform, all in local memory, producing u4a[t] directly (the old nlmDistHorz +
    \\// nlmVertical in ONE kernel; the u4b global scratch is gone). Halves the box-sum
    \\// launch count, which dominates the per-frame host overhead on small frames.
    \\// Each work-group owns a BLK_X-col x (VRT_RESULT*BLK_Y)-row output tile:
    \\//   1) compute per-pixel distance for the tile + a +-NLM_S halo in x AND y -> `dist`,
    \\//   2) horizontal box-sum (ascending x) over `dist` -> `hsum` (output cols x V-halo rows),
    \\//   3) vertical box-sum (ascending y) over `hsum` + weight transform -> u4a[t].
    \\// Tap order (ascending x then ascending y) and zero border (OOB dist cells -> 0) are
    \\// identical to the two-kernel version, so it stays bit-exact. Ref layer t, cand t+qz.
    \\// Q-BATCHED: one 3D launch computes the weight planes of SEVERAL passes — grid z picks
    \\// the pass, whose (t, qx, qy, qz, out slot) come from the create()-time sweep table
    \\// `wq` (stride-8 rows, uploaded once per Stream). Weight passes only read u1 (static
    \\// during the sweep), so they have NO cross-dependency and can share a launch freely;
    \\// this cut ~99 launches/frame (d=1 a=2) to ~26 — the measured ~4 ms/frame of WDDM
    \\// launch/submission overhead was the filter's dominant cost after the earlier rounds.
    \\// The weight math is untouched; only the parameter source and the u4a layer (a batch
    \\// SLOT instead of the temporal layer t) changed, neither of which can alter a float.
    \\kernel __attribute__((reqd_work_group_size(BLK_X, BLK_Y, 1)))
    \\void nlmWeight(global const io_t *restrict u1, global float *restrict u4a,
    \\               global const int *restrict wq, const int pass_base) {
    \\    const int prow = (pass_base + (int)get_group_id(2)) * 8;
    \\    const int t = wq[prow+0], qx = wq[prow+1], qy = wq[prow+2], qz = wq[prow+3];
    \\    const int slot = wq[prow+4];
    \\    local float dist[VRT_RESULT*BLK_Y + 2*NLM_S][BLK_X + 2*NLM_S];
    \\    local float hsum[VRT_RESULT*BLK_Y + 2*NLM_S][BLK_X];
    \\    const int lx = get_local_id(0), ly = get_local_id(1);
    \\    const int gx0 = get_group_id(0) * BLK_X;                // first output column
    \\    const int gy0 = get_group_id(1) * (VRT_RESULT*BLK_Y);   // first output row
    \\    // 1) per-pixel distance over the tile + halo; OOB (x or y) -> 0.0f (zero border).
    \\    for (int ry = ly; ry < VRT_RESULT*BLK_Y + 2*NLM_S; ry += BLK_Y) {
    \\        const int yy = gy0 - NLM_S + ry;
    \\        for (int cx = lx; cx < BLK_X + 2*NLM_S; cx += BLK_X) {
    \\            const int xx = gx0 - NLM_S + cx;
    \\            dist[ry][cx] = (xx >= 0 && xx < VI_DIM_X && yy >= 0 && yy < VI_DIM_Y)
    \\                           ? nlm_pix_dist(u1, t, qz, xx, yy, qx, qy) : 0.0f;
    \\        }
    \\    }
    \\    barrier(CLK_LOCAL_MEM_FENCE);
    \\    // 2) horizontal box-sum: hsum[ry][lx] = sum dist[ry][lx..lx+2*NLM_S] (out col gx0+lx).
    \\    for (int ry = ly; ry < VRT_RESULT*BLK_Y + 2*NLM_S; ry += BLK_Y) {
    \\        float sh = 0.0f;
    \\        for (int i = 0; i <= 2*NLM_S; ++i) sh += dist[ry][lx + i];   // xx = x-NLM_S .. x+NLM_S
    \\        hsum[ry][lx] = sh;
    \\    }
    \\    barrier(CLK_LOCAL_MEM_FENCE);
    \\    // 3) vertical box-sum + weight transform, VRT_RESULT output rows per thread.
    \\    const int x = gx0 + lx;
    \\    if (x >= VI_DIM_X) return;
    \\    int y = gy0 + ly, lc = ly;                  // lc = hsum row of tap yy = y-NLM_S
    \\    for (int r = 0; r < VRT_RESULT; ++r, y += BLK_Y, lc += BLK_Y) {
    \\        if (y >= VI_DIM_Y) return;              // strided y is increasing -> safe to bail
    \\        float sum = 0.0f;
    \\        for (int i = 0; i <= 2*NLM_S; ++i) sum += hsum[lc + i][lx];   // yy = y-NLM_S .. y+NLM_S
    \\        const float arg = sum * NLM_H2_INV_NORM;
    \\        float w;
    \\#if   WMODE == 0
    \\        w = exp(-arg);
    \\#elif WMODE == 1
    \\        w = fdim(1.0f, arg);
    \\#elif WMODE == 2
    \\        { float c = fdim(1.0f, arg); w = c*c; }
    \\#else
    \\        { float c = fdim(1.0f, arg); c = c*c; c = c*c; w = c*c; }
    \\#endif
    \\        u4a[(IDX)slot*U4_LAYER + y*STRIDE + x] = w;
    \\    }
    \\}
    \\
    \\// Accumulate both +q and -q at the center layer NLM_D, exploiting
    \\// weight(p,p+q)==weight(p,p-q). u2 is (C+1) interleaved per pixel: [c]=weighted-sum
    \\// of channel c, [C]=weight-sum.
    \\// Q-BATCHED: one launch folds a whole batch of displacements. Per pixel the loop
    \\// walks the batch's q's IN SWEEP ORDER (rows of the create()-time table `aq`:
    \\// qx, qy, qz, center slot, mirror slot — slot_m == slot_c when qz==0, reproducing
    \\// the old NLM_D-qz==NLM_D case), accumulating u2/u5 in REGISTERS and writing once.
    \\// BIT-EXACT: each q's update statements are the old per-launch kernel's statements
    \\// VERBATIM, applied in the same per-pixel order; the old kernel's global u2/u5
    \\// round-trip between q's returned exactly the register value being carried here.
    \\// Batched RMW is also why this can't race: one launch owns the whole batch.
    \\kernel __attribute__((reqd_work_group_size(BLK_X, BLK_Y, 1)))
    \\void nlmAccumulation(global const io_t *restrict u1, global float *restrict u2,
    \\                     global const float *restrict u4a, global float *restrict u5,
    \\                     global const int *restrict aq, const int q_base, const int nb) {
    \\    const int x = get_global_id(0), y = get_global_id(1);
    \\    if (x >= VI_DIM_X || y >= VI_DIM_Y) return;
    \\    const int g = y*STRIDE + x;
    \\    const int cx = x+PAD, cy = y+PAD;
    \\    float u5v = u5[g];
    \\    // Vector RMW of the (C+1)-interleaved u2: vloadN(g,u2) == u2[N*g .. N*g+N-1],
    \\    // so each lane maps 1:1 to the old scalar slot -> bit-exact, fewer transactions.
    \\#if   NLM_CHANNELS == 1
    \\    float2 acc = vload2(g, u2);
    \\#elif NLM_CHANNELS == 2
    \\    float3 acc = vload3(g, u2);
    \\#else
    \\    float4 acc = vload4(g, u2);
    \\#endif
    \\    for (int b = 0; b < nb; ++b) {
    \\        const int r = (q_base + b) * 8;   // params are warp-uniform (broadcast reads)
    \\        const int qx = aq[r+0], qy = aq[r+1], qz = aq[r+2];
    \\        const int sc = aq[r+3], sm = aq[r+4];
    \\        const float u4 = u4a[(IDX)sc*U4_LAYER + g];
    \\        const int xm = x-qx, ym = y-qy;
    \\        const float u4_mq = (xm < 0 || xm >= VI_DIM_X || ym < 0 || ym >= VI_DIM_Y)
    \\                            ? 0.0f : u4a[(IDX)sm*U4_LAYER + ym*STRIDE + xm];
    \\        u5v = fmax(u4, fmax(u4_mq, u5v));
    \\        const IDX pq = (IDX)(NLM_D+qz)*U1_LAYER + (IDX)(cy+qy)*PSTRIDE + (cx+qx);
    \\        const IDX mq = (IDX)(NLM_D-qz)*U1_LAYER + (IDX)(cy-qy)*PSTRIDE + (cx-qx);
    \\#if   NLM_CHANNELS == 1
    \\        const float pq0 = LOADU1(pq), mq0 = LOADU1(mq);
    \\        acc.x += (u4*pq0) + (u4_mq*mq0);
    \\        acc.y += (u4 + u4_mq);
    \\#elif NLM_CHANNELS == 2
    \\        const float pq0 = LOADU1(pq),            mq0 = LOADU1(mq);
    \\        const float pq1 = LOADU1(U1_PLANE + pq), mq1 = LOADU1(U1_PLANE + mq);
    \\        acc.x += (u4*pq0) + (u4_mq*mq0);
    \\        acc.y += (u4*pq1) + (u4_mq*mq1);
    \\        acc.z += (u4 + u4_mq);
    \\#else
    \\        const float pq0 = LOADU1(pq),              mq0 = LOADU1(mq);
    \\        const float pq1 = LOADU1(U1_PLANE + pq),   mq1 = LOADU1(U1_PLANE + mq);
    \\        const float pq2 = LOADU1(2*U1_PLANE + pq), mq2 = LOADU1(2*U1_PLANE + mq);
    \\        acc.x += (u4*pq0) + (u4_mq*mq0);
    \\        acc.y += (u4*pq1) + (u4_mq*mq1);
    \\        acc.z += (u4*pq2) + (u4_mq*mq2);
    \\        acc.w += (u4 + u4_mq);
    \\#endif
    \\    }
    \\#if   NLM_CHANNELS == 1
    \\    vstore2(acc, g, u2);
    \\#elif NLM_CHANNELS == 2
    \\    vstore3(acc, g, u2);
    \\#else
    \\    vstore4(acc, g, u2);
    \\#endif
    \\    u5[g] = u5v;
    \\}
    \\
    \\// out = (center*m + weighted_sum) / (m + weight_sum), m = wref*maxweight, per
    \\// channel. KNL does not guard den==0 (U5 seeded with FLT_EPS keeps den>0); match it.
    \\kernel __attribute__((reqd_work_group_size(BLK_X, BLK_Y, 1)))
    \\void nlmFinish(global const io_t *restrict u1, global io_t *restrict u1z,
    \\               global const float *restrict u2, global const float *restrict u5) {
    \\    const int x = get_global_id(0), y = get_global_id(1);
    \\    if (x >= VI_DIM_X || y >= VI_DIM_Y) return;
    \\    const int g = y*STRIDE + x;
    \\    const float m  = NLM_WREF * u5[g];
    \\    const IDX uc = (IDX)NLM_D*U1_LAYER + (IDX)(y+PAD)*PSTRIDE + (x+PAD);
    \\#if   NLM_CHANNELS == 1
    \\    float2 acc = vload2(g, u2);
    \\    const float den = m + acc.y;
    \\    STOREU1(g, (LOADU1(uc)*m + acc.x) / den);
    \\#elif NLM_CHANNELS == 2
    \\    float3 acc = vload3(g, u2);
    \\    const float den = m + acc.z;
    \\    STOREU1(g,        (LOADU1(uc)*m            + acc.x) / den);
    \\    STOREU1(NPIX + g, (LOADU1(U1_PLANE + uc)*m + acc.y) / den);
    \\#else
    \\    float4 acc = vload4(g, u2);
    \\    const float den = m + acc.w;
    \\    STOREU1(g,            (LOADU1(uc)*m              + acc.x) / den);
    \\    STOREU1(NPIX + g,     (LOADU1(U1_PLANE + uc)*m   + acc.y) / den);
    \\    STOREU1(2*NPIX + g,   (LOADU1(2*U1_PLANE + uc)*m + acc.z) / den);
    \\#endif
    \\}
;

const FLT_EPS: f32 = 1.1920929e-7;
const nlm_qb_small: u32 = 8;
const nlm_qb_large: u32 = 4;

const tune_len = 5;
const Variant = struct {
    w_base: u32,
    q_base: u32,
    q_cnt: u32,
    w_boff: []u32,
};

const Data = struct {
    node: ?*vs.Node = null,
    vi: vs.VideoInfo = undefined,
    ref_node: ?*vs.Node = null,
    has_ref: bool = false,

    d: u8 = 0,
    a: u8 = 0,
    s: u8 = 0,
    h: f32 = 0,
    wref: f32 = 0,
    wmode: u8 = 0,

    ref: u8 = REF_LUMA,
    chans: u8 = 1,
    plane0: u8 = 0,
    bits: i32 = 32,
    half: bool = false,
    wbytes: u32 = 4,

    w: u32 = 0,
    h_: u32 = 0,
    stride: u32 = 0,
    pad: u32 = 0,
    pstride: u32 = 0,
    ph: u32 = 0,

    platform: cl.Platform = undefined,
    device: cl.Device = undefined,
    context: cl.Context = undefined,
    use_pinned: bool = true,
    qb: u32 = nlm_qb_small,

    blk_x: usize = 16,
    blk_y: usize = 8,
    vrt: usize = 3,
    wq_host: []i32 = &.{},
    aq_host: []i32 = &.{},
    variants: []Variant = &.{},

    pool: clpool.Pool(Stream, Data) = .{},
};

fn freeTables(d: *Data) void {
    for (d.variants) |v| allocator.free(v.w_boff);
    allocator.free(d.variants);
    allocator.free(d.aq_host);
    allocator.free(d.wq_host);
}

const PinnedMirror = struct { stage: cl.Buffer(u8), host: []u8 };
fn mapPinnedMirror(queue: cl.CommandQueue, context: cl.Context, bytes: usize) ?PinnedMirror {
    const stage = cl.createBuffer(u8, context, .{ .alloc_host_ptr = true }, bytes) catch {
        std.log.warn("NLMeans: pinned staging alloc failed ({d} MB); falling back to pageable transfers.", .{bytes / (1 << 20)});
        return null;
    };
    var map_err: cl.c.cl_int = cl.c.CL_SUCCESS;
    const map_ptr = cl.c.clEnqueueMapBuffer(queue.handle, stage.handle, cl.c.CL_TRUE, cl.c.CL_MAP_READ | cl.c.CL_MAP_WRITE, 0, bytes, 0, null, null, &map_err);
    if (map_err != cl.c.CL_SUCCESS or map_ptr == null) {
        stage.release();
        std.log.warn("NLMeans: pinned staging map failed; falling back to pageable transfers.", .{});
        return null;
    }
    const host = @as([*]u8, @ptrCast(map_ptr.?))[0..bytes];
    @memset(host, 0);
    return .{ .stage = stage, .host = host };
}

const Stream = struct {
    program: cl.Program,
    queue: cl.CommandQueue,
    d_u1: cl.Buffer(u8),
    d_u1z: cl.Buffer(u8),
    d_u2: cl.Buffer(f32),
    d_u4a: cl.Buffer(f32),
    d_u5: cl.Buffer(f32),
    d_wq: cl.Buffer(i32),
    d_aq: cl.Buffer(i32),
    k_weight: cl.Kernel,
    k_accumulation: cl.Kernel,
    k_finish: cl.Kernel,
    stage: ?cl.Buffer(u8),
    host: ?[]u8,

    d_u1r: ?cl.Buffer(u8),
    stage_r: ?cl.Buffer(u8),
    host_r: ?[]u8,

    pub fn init(self: *Stream, d: *Data) !void {
        const npix = d.stride * d.h_;
        const layers = 2 * @as(usize, d.d) + 1;
        const c = @as(usize, d.chans);
        self.program = try cl.createProgramWithSource(d.context, kernel_src);
        errdefer self.program.release();
        const slots: u64 = if (d.d == 0) d.qb else 2 * @as(u64, d.qb);
        const npix64 = @as(u64, d.stride) * @as(u64, d.h_);
        const idx_max = @max(
            @as(u64, d.pstride) * @as(u64, d.ph) * @as(u64, layers) * @as(u64, c),
            @max(npix64 * slots, npix64 * @as(u64, c)),
        );
        const idx_t: []const u8 = if (idx_max < (1 << 31)) "int" else "long";
        const opts = try std.fmt.allocPrintSentinel(allocator,
            \\-cl-std=CL1.2 -DVI_DIM_X={d} -DVI_DIM_Y={d} -DSTRIDE={d} -DPSTRIDE={d} -DPAD={d} -DPH={d} -DNLM_S={d} -DNLM_D={d} -DNLM_REF={d} -DNLM_CHANNELS={d} -DWMODE={d} -DBLK_X={d} -DBLK_Y={d} -DVRT_RESULT={d} -DNLM_H={e}f -DNLM_WREF={e}f -DIDX={s} -DBITS={d} -DHALF={d}
        , .{ d.w, d.h_, d.stride, d.pstride, d.pad, d.ph, d.s, d.d, d.ref, d.chans, d.wmode, d.blk_x, d.blk_y, d.vrt, d.h, d.wref, idx_t, d.bits, @intFromBool(d.half) }, 0);
        defer allocator.free(opts);
        self.program.build(&.{d.device}, opts) catch |err| {
            if (err == error.BuildProgramFailure) {
                const log = try self.program.getBuildLog(allocator, d.device);
                defer allocator.free(log);
                std.log.err("NLMeans OpenCL build failed: {s}", .{log});
            }
            return err;
        };
        self.queue = try cl.createCommandQueue(d.context, d.device, .{});
        errdefer self.queue.release();
        const wb: usize = d.wbytes;
        self.d_u1 = try cl.createBuffer(u8, d.context, .{ .read_write = true }, d.pstride * d.ph * layers * c * wb);
        errdefer self.d_u1.release();
        self.d_u1r = null;
        if (d.has_ref) self.d_u1r = try cl.createBuffer(u8, d.context, .{ .read_write = true }, d.pstride * d.ph * layers * c * wb);
        errdefer if (self.d_u1r) |b| b.release();
        self.d_u1z = try cl.createBuffer(u8, d.context, .{ .read_write = true }, npix * c * wb);
        errdefer self.d_u1z.release();
        self.d_u2 = try cl.createBuffer(f32, d.context, .{ .read_write = true }, npix * (c + 1));
        errdefer self.d_u2.release();
        self.d_u4a = try cl.createBuffer(f32, d.context, .{ .read_write = true }, npix * @as(usize, @intCast(slots)));
        errdefer self.d_u4a.release();
        self.d_u5 = try cl.createBuffer(f32, d.context, .{ .read_write = true }, npix);
        errdefer self.d_u5.release();
        self.d_wq = try cl.createBufferWithData(i32, d.context, .{ .read_only = true }, d.wq_host);
        errdefer self.d_wq.release();
        self.d_aq = try cl.createBufferWithData(i32, d.context, .{ .read_only = true }, d.aq_host);
        errdefer self.d_aq.release();
        self.stage = null;
        self.host = null;
        const buff_stage = d.pstride * d.ph * layers * c * wb;
        if (d.use_pinned) {
            if (mapPinnedMirror(self.queue, d.context, buff_stage)) |pm| {
                self.stage = pm.stage;
                self.host = pm.host;
            }
        }
        errdefer if (self.host) |h| {
            _ = cl.c.clEnqueueUnmapMemObject(self.queue.handle, self.stage.?.handle, h.ptr, 0, null, null);
            _ = cl.c.clFinish(self.queue.handle);
            self.stage.?.release();
        };
        self.stage_r = null;
        self.host_r = null;
        if (d.has_ref and d.use_pinned) {
            if (mapPinnedMirror(self.queue, d.context, buff_stage)) |pm| {
                self.stage_r = pm.stage;
                self.host_r = pm.host;
            }
        }
        errdefer if (self.host_r) |h| {
            _ = cl.c.clEnqueueUnmapMemObject(self.queue.handle, self.stage_r.?.handle, h.ptr, 0, null, null);
            _ = cl.c.clFinish(self.queue.handle);
            self.stage_r.?.release();
        };
        self.k_weight = try cl.createKernel(self.program, "nlmWeight");
        errdefer self.k_weight.release();
        self.k_accumulation = try cl.createKernel(self.program, "nlmAccumulation");
        errdefer self.k_accumulation.release();
        self.k_finish = try cl.createKernel(self.program, "nlmFinish");
        errdefer self.k_finish.release();
        try self.setStaticArgs();
        try fillZero(self.queue, self.d_u1, d.pstride * d.ph * layers * c * wb);
        if (self.d_u1r) |b| try fillZero(self.queue, b, d.pstride * d.ph * layers * c * wb);
        if (cl.c.clFinish(self.queue.handle) != cl.c.CL_SUCCESS) return error.Finish;
    }

    fn setStaticArgs(self: *Stream) !void {
        try self.k_weight.setArg(@TypeOf(self.d_u1), 0, self.d_u1r orelse self.d_u1);
        try self.k_weight.setArg(@TypeOf(self.d_u4a), 1, self.d_u4a);
        try self.k_weight.setArg(@TypeOf(self.d_wq), 2, self.d_wq);
        try self.k_accumulation.setArg(@TypeOf(self.d_u1), 0, self.d_u1);
        try self.k_accumulation.setArg(@TypeOf(self.d_u2), 1, self.d_u2);
        try self.k_accumulation.setArg(@TypeOf(self.d_u4a), 2, self.d_u4a);
        try self.k_accumulation.setArg(@TypeOf(self.d_u5), 3, self.d_u5);
        try self.k_accumulation.setArg(@TypeOf(self.d_aq), 4, self.d_aq);
        try self.k_finish.setArg(@TypeOf(self.d_u1), 0, self.d_u1);
        try self.k_finish.setArg(@TypeOf(self.d_u1z), 1, self.d_u1z);
        try self.k_finish.setArg(@TypeOf(self.d_u2), 2, self.d_u2);
        try self.k_finish.setArg(@TypeOf(self.d_u5), 3, self.d_u5);
    }

    pub fn deinit(self: *Stream) void {
        if (self.host) |h| {
            _ = cl.c.clEnqueueUnmapMemObject(self.queue.handle, self.stage.?.handle, h.ptr, 0, null, null);
            _ = cl.c.clFinish(self.queue.handle);
        }
        if (self.host_r) |h| {
            _ = cl.c.clEnqueueUnmapMemObject(self.queue.handle, self.stage_r.?.handle, h.ptr, 0, null, null);
            _ = cl.c.clFinish(self.queue.handle);
        }
        self.k_finish.release();
        self.k_accumulation.release();
        self.k_weight.release();
        if (self.stage) |st| st.release();
        if (self.stage_r) |st| st.release();
        if (self.d_u1r) |b| b.release();
        self.d_aq.release();
        self.d_wq.release();
        self.d_u5.release();
        self.d_u4a.release();
        self.d_u2.release();
        self.d_u1z.release();
        self.d_u1.release();
        self.queue.release();
        self.program.release();
    }
};

fn fillF32(queue: cl.CommandQueue, buf: cl.Buffer(f32), value: f32, count: usize) !void {
    var pat = value;
    if (cl.c.clEnqueueFillBuffer(queue.handle, buf.handle, &pat, @sizeOf(f32), 0, count * @sizeOf(f32), 0, null, null) != cl.c.CL_SUCCESS) {
        return error.EnqueueFillBuffer;
    }
}

fn fillZero(queue: cl.CommandQueue, buf: cl.Buffer(u8), bytes: usize) !void {
    var pat: u8 = 0;
    if (cl.c.clEnqueueFillBuffer(queue.handle, buf.handle, &pat, 1, 0, bytes, 0, null, null) != cl.c.CL_SUCCESS) {
        return error.EnqueueFillBuffer;
    }
}

const ndr = vszipcl.ndr;

fn writeU1(d: *Data, s: *Stream, buf: cl.Buffer(u8), c: usize, t_layer: usize, src: []const u8) !void {
    const wsz: usize = d.wbytes;
    const layers = 2 * @as(usize, d.d) + 1;
    const ph: usize = d.ph;
    const base_row = c * layers * ph + t_layer * ph + @as(usize, d.pad);
    const buffer_origin = [3]usize{ @as(usize, d.pad) * wsz, base_row, 0 };
    const host_origin = [3]usize{ 0, 0, 0 };
    const region = [3]usize{ @as(usize, d.w) * wsz, @as(usize, d.h_), 1 };
    if (cl.c.clEnqueueWriteBufferRect(
        s.queue.handle,
        buf.handle,
        cl.c.CL_FALSE,
        &buffer_origin,
        &host_origin,
        &region,
        @as(usize, d.pstride) * wsz,
        0,
        @as(usize, d.stride) * wsz,
        0,
        src.ptr,
        0,
        null,
        null,
    ) != cl.c.CL_SUCCESS) return error.EnqueueWrite;
}

fn writeBuf(s: *Stream, buf: cl.Buffer(u8), off: usize, src: []const u8) !void {
    return vszipcl.enqWrite(s.queue, buf.handle, off, src);
}
fn readBuf(s: *Stream, buf: cl.Buffer(u8), offset: usize, dst: []u8) !void {
    return vszipcl.enqRead(s.queue, buf.handle, offset, dst);
}

fn uploadWindow(d: *Data, s: *Stream, buf: cl.Buffer(u8), host_opt: ?[]u8, srcps: []const []const u8, k_start: i32, k_end: i32) !void {
    const C: usize = d.chans;
    const center: i32 = @intCast(d.d);
    const layers = 2 * @as(usize, d.d) + 1;
    const npix = d.stride * d.h_;
    const wb: usize = d.wbytes;
    const pp: usize = d.pstride;
    const lay: usize = pp * @as(usize, d.ph);
    const pad: usize = d.pad;
    const w: usize = d.w;
    var fi: usize = 0;
    var k: i32 = k_start;
    while (k <= k_end) : (k += 1) {
        const t_layer: usize = @intCast(center + k);
        var c: usize = 0;
        while (c < C) : (c += 1) {
            const src = srcps[fi * C + c];
            std.debug.assert(src.len == npix * wb);
            if (host_opt) |host| {
                const base = ((c * layers + t_layer) * lay + pad * pp + pad) * wb;
                var y: usize = 0;
                while (y < d.h_) : (y += 1) {
                    @memcpy(host[base + y * pp * wb ..][0 .. w * wb], src[y * d.stride * wb ..][0 .. w * wb]);
                }
            } else {
                try writeU1(d, s, buf, c, t_layer, src);
            }
        }
        fi += 1;
    }
    if (host_opt) |host| {
        const t0: usize = @intCast(center + k_start);
        const nlay: usize = @intCast(k_end - k_start + 1);
        var c: usize = 0;
        while (c < C) : (c += 1) {
            const off = (c * layers + t0) * lay * wb;
            try writeBuf(s, buf, off, host[off..][0 .. nlay * lay * wb]);
        }
    }
}

fn process(d: *Data, s: *Stream, dstps: []const []u8, srcps: []const []const u8, refps: ?[]const []const u8, k_start: i32, k_end: i32) !void {
    const npix = d.stride * d.h_;
    const C: usize = d.chans;
    const wb: usize = d.wbytes;

    const BX: usize = d.blk_x;
    const BY: usize = d.blk_y;
    const VR: usize = d.vrt;
    const lws: [2]usize = .{ BX, BY };
    const gws_pix: [2]usize = .{ vszipcl.ceilTo(@as(usize, d.w), BX), vszipcl.ceilTo(@as(usize, d.h_), BY) };
    const gws_w: [2]usize = .{
        vszipcl.ceilTo(@as(usize, d.w), BX),
        ((@as(usize, d.h_) + VR * BY - 1) / (VR * BY)) * BY,
    };

    errdefer _ = cl.c.clFinish(s.queue.handle);
    try uploadWindow(d, s, s.d_u1, s.host, srcps, k_start, k_end);
    if (refps) |rp| try uploadWindow(d, s, s.d_u1r.?, s.host_r, rp, k_start, k_end);

    try fillF32(s.queue, s.d_u2, 0.0, npix * (C + 1));
    try fillF32(s.queue, s.d_u5, FLT_EPS, npix);

    const v = &d.variants[@intCast(k_end)];
    const qb: usize = d.qb;
    var q0: usize = 0;
    var bi: usize = 0;
    while (q0 < v.q_cnt) : ({
        q0 += qb;
        bi += 1;
    }) {
        const nb: usize = @min(qb, @as(usize, v.q_cnt) - q0);
        const p0 = v.w_boff[bi];
        const p1 = v.w_boff[bi + 1];
        try s.k_weight.setArg(c_int, 3, @intCast(v.w_base + p0));
        const gws_w3: [3]usize = .{ gws_w[0], gws_w[1], p1 - p0 };
        const lws3: [3]usize = .{ BX, BY, 1 };
        try ndr(s, s.k_weight, &gws_w3, &lws3);
        try s.k_accumulation.setArg(c_int, 5, @intCast(@as(usize, v.q_base) + q0));
        try s.k_accumulation.setArg(c_int, 6, @intCast(nb));
        try ndr(s, s.k_accumulation, &gws_pix, &lws);
    }

    try ndr(s, s.k_finish, &gws_pix, &lws);
    var c: usize = 0;
    while (c < C) : (c += 1) try readBuf(s, s.d_u1z, c * npix * wb, dstps[c]);
    if (cl.c.clFinish(s.queue.handle) != cl.c.CL_SUCCESS) return error.Finish;
}

fn getFrame(n: c_int, ar: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core_ptr, frame_ctx);
    const dd: i32 = @intCast(d.d);
    const ni: i32 = @intCast(n);
    const nf: i32 = @intCast(d.vi.numFrames);
    const m: i32 = @min(dd, ni);
    const k_start: i32 = -m;
    const k_end: i32 = m;

    if (ar == .Initial) {
        var k: i32 = k_start;
        while (k <= k_end) : (k += 1) {
            const idx: c_int = @intCast(@min(@max(ni + k, 0), nf - 1));
            zapi.requestFrameFilter(idx, d.node);
            if (d.ref_node) |rn| zapi.requestFrameFilter(idx, rn);
        }
    } else if (ar == .AllFramesReady) {
        const C: usize = d.chans;
        const plane0: u32 = d.plane0;
        const numPlanes: u32 = @intCast(d.vi.format.numPlanes);
        const count: usize = @intCast(k_end - k_start + 1);
        const Frame = @TypeOf(zapi.initZFrame(d.node, n));

        const frames = allocator.alloc(Frame, count) catch {
            zapi.setFilterError("NLMeans: out of memory.");
            return null;
        };
        defer allocator.free(frames);
        const srcps = allocator.alloc([]const u8, count * C) catch {
            zapi.setFilterError("NLMeans: out of memory.");
            return null;
        };
        defer allocator.free(srcps);
        const dstps = allocator.alloc([]u8, C) catch {
            zapi.setFilterError("NLMeans: out of memory.");
            return null;
        };
        defer allocator.free(dstps);

        var rframes: []Frame = &.{};
        var refps: ?[]const []const u8 = null;
        if (d.ref_node) |_| {
            rframes = allocator.alloc(Frame, count) catch {
                zapi.setFilterError("NLMeans: out of memory.");
                return null;
            };
            const rslices = allocator.alloc([]const u8, count * C) catch {
                allocator.free(rframes);
                zapi.setFilterError("NLMeans: out of memory.");
                return null;
            };
            refps = rslices;
        }
        defer if (refps) |rp| allocator.free(rp);
        defer allocator.free(rframes);

        var fi: usize = 0;
        var k: i32 = k_start;
        while (k <= k_end) : (k += 1) {
            const idx: c_int = @intCast(@min(@max(ni + k, 0), nf - 1));
            frames[fi] = zapi.initZFrame(d.node, idx);
            var c: usize = 0;
            while (c < C) : (c += 1) srcps[fi * C + c] = frames[fi].getReadSlice(plane0 + @as(u32, @intCast(c)));
            if (d.ref_node) |rn| {
                rframes[fi] = zapi.initZFrame(rn, idx);
                c = 0;
                while (c < C) : (c += 1) @constCast(refps.?)[fi * C + c] = rframes[fi].getReadSlice(plane0 + @as(u32, @intCast(c)));
            }
            fi += 1;
        }
        defer for (frames) |f| f.deinit();
        defer if (d.ref_node) |_| for (rframes) |f| f.deinit();

        const center_frame = frames[@as(usize, @intCast(-k_start))];
        const dst = center_frame.newVideoFrame();
        var c: usize = 0;
        while (c < C) : (c += 1) dstps[c] = dst.getWriteSlice(plane0 + @as(u32, @intCast(c)));

        var p: u32 = 0;
        while (p < numPlanes) : (p += 1) {
            if (p < plane0 or p >= plane0 + @as(u32, @intCast(C))) {
                @memcpy(dst.getWriteSlice(p), center_frame.getReadSlice(p));
            }
        }

        const s = d.pool.acquire();
        defer d.pool.release(s);

        process(d, s, dstps, srcps, refps, k_start, k_end) catch |err| {
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
    d.context.release();
    freeTables(d);
    vsapi.?.freeNode.?(d.node);
    if (d.ref_node) |rn| vsapi.?.freeNode.?(rn);
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
    defer if (!keep) {
        zapi.freeNode(d.node);
        if (d.ref_node) |rn| zapi.freeNode(rn);
    };

    const fmt = d.vi.format;
    const bits: i32 = fmt.bitsPerSample;
    const depth_ok = (fmt.sampleType == .Float and (bits == 32 or bits == 16)) or
        (fmt.sampleType == .Integer and (bits == 8 or bits == 16));
    if (!depth_ok) {
        return map_out.setError("NLMeans: input bitdepth must be 8/16 (integer), 16 (half) or 32 (float).");
    }
    d.bits = bits;
    d.half = fmt.sampleType == .Float and bits == 16;
    d.wbytes = @intCast(fmt.bytesPerSample);
    if (d.vi.width <= 0 or d.vi.height <= 0) {
        return map_out.setError("NLMeans: clip must have constant dimensions.");
    }
    if (d.vi.width > 8192 or d.vi.height > 8192) {
        return map_out.setError("NLMeans: 8192x8192 is the highest supported resolution.");
    }

    const dd = map_in.getValue(i32, "d") orelse 1;
    const a = map_in.getValue(i32, "a") orelse 2;
    const ss = map_in.getValue(i32, "s") orelse 4;
    d.h = map_in.getValue(f32, "h") orelse 1.2;
    const wmode = map_in.getValue(i32, "wmode") orelse 0;
    d.wref = map_in.getValue(f32, "wref") orelse 1.0;
    const chstr = map_in.getData("channels", 0) orelse "auto";
    const ns_req = map_in.getValue(i32, "num_streams");
    if (dd < 0 or dd > 16) return map_out.setError("NLMeans: d must be 0..16.");
    if (map_in.getNodeVi("rclip")) |rv| {
        d.ref_node = rv[0];
        d.has_ref = true;
        const rvi = rv[1];
        const rfmt = rvi.format;
        const same = rfmt.colorFamily == fmt.colorFamily and rfmt.sampleType == fmt.sampleType and
            rfmt.bitsPerSample == fmt.bitsPerSample and rfmt.subSamplingW == fmt.subSamplingW and
            rfmt.subSamplingH == fmt.subSamplingH and rvi.width == d.vi.width and
            rvi.height == d.vi.height and rvi.numFrames == d.vi.numFrames;
        if (!same) return map_out.setError("NLMeans: 'rclip' must match the source clip's format, dimensions and frame count.");
    }
    if (a < 1 or a > 64) return map_out.setError("NLMeans: a must be 1..64.");
    if (ss < 0 or ss > 8) return map_out.setError("NLMeans: s must be 0..8.");
    if (d.h <= 0) return map_out.setError("NLMeans: h must be > 0.");
    if (wmode < 0 or wmode > 3) return map_out.setError("NLMeans: wmode must be 0..3.");
    if (d.wref < 0) return map_out.setError("NLMeans: wref must be >= 0.");
    if (ns_req) |ns| {
        if (ns < 1 or ns > 32) return map_out.setError("NLMeans: num_streams must be 1..32.");
    }
    const device_id = map_in.getValue(i32, "device_id") orelse 0;
    if (device_id < 0) return map_out.setError("NLMeans: invalid device ID.");

    const eq = std.ascii.eqlIgnoreCase;
    switch (fmt.colorFamily) {
        .Gray => {
            if (!(eq(chstr, "Y") or eq(chstr, "auto"))) {
                return map_out.setError("NLMeans: 'channels' must be 'Y' with Gray.");
            }
            d.ref = REF_LUMA;
            d.chans = 1;
            d.plane0 = 0;
        },
        .YUV => {
            if (eq(chstr, "YUV")) {
                if (fmt.subSamplingW != 0 or fmt.subSamplingH != 0) {
                    return map_out.setError("NLMeans: 'channels'='YUV' requires 4:4:4.");
                }
                d.ref = REF_YUV;
                d.chans = 3;
                d.plane0 = 0;
            } else if (eq(chstr, "Y") or eq(chstr, "auto")) {
                d.ref = REF_LUMA;
                d.chans = 1;
                d.plane0 = 0;
            } else if (eq(chstr, "UV")) {
                d.ref = REF_CHROMA;
                d.chans = 2;
                d.plane0 = 1;
            } else {
                return map_out.setError("NLMeans: 'channels' must be 'YUV', 'Y' or 'UV' with YUV.");
            }
        },
        .RGB => {
            if (!(eq(chstr, "RGB") or eq(chstr, "auto"))) {
                return map_out.setError("NLMeans: 'channels' must be 'RGB' with RGB.");
            }
            d.ref = REF_RGB;
            d.chans = 3;
            d.plane0 = 0;
        },
        else => return map_out.setError("NLMeans: unsupported color family."),
    }

    const sw: u5 = @intCast(fmt.subSamplingW);
    const sh: u5 = @intCast(fmt.subSamplingH);
    if (d.ref == REF_CHROMA) {
        d.w = @as(u32, @intCast(d.vi.width)) >> sw;
        d.h_ = @as(u32, @intCast(d.vi.height)) >> sh;
    } else {
        d.w = @intCast(d.vi.width);
        d.h_ = @intCast(d.vi.height);
    }

    if (2 * a + 1 > @as(i32, @intCast(d.w)) or 2 * a + 1 > @as(i32, @intCast(d.h_))) {
        return map_out.setError("NLMeans: research window (2*a+1) larger than the frame.");
    }

    d.d = @intCast(dd);
    d.a = @intCast(a);
    d.s = @intCast(ss);
    d.wmode = @intCast(wmode);
    const strides = vszipcl.strideFromVi(&d.vi);
    d.stride = if (d.ref == REF_CHROMA) strides[1] else strides[0];
    d.pad = @intCast(a);
    d.pstride = @intCast(vsh.ceilN(@as(usize, d.w) + 2 * @as(usize, d.pad), 8));
    d.ph = d.h_ + 2 * d.pad;
    vszipcl.initContext(&d, @intCast(device_id)) catch |err| {
        map_out.setError(if (err == error.InvalidDeviceID) "NLMeans: invalid device ID." else "NLMeans: OpenCL init failed.");
        std.log.err("NLMeans OpenCL init failed: {}", .{err});
        return;
    };
    d.qb = if (@as(u64, d.stride) * @as(u64, d.h_) <= 1920 * 1152) nlm_qb_small else nlm_qb_large;
    var pin_min: usize = 2;
    {
        var terr: ?[:0]const u8 = null;
        if (map_in.numElements("tune")) |n| {
            if (n > tune_len) terr = "NLMeans: tune expects at most 5 entries [blk_x, blk_y, vrt, qb, pin_min_streams].";
        }
        if (vszipcl.tuneEntry(map_in, 0)) |v| {
            if (v < 1 or v > 64) terr = "NLMeans: tune[0] (blk_x) must be 1..64." else d.blk_x = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 1)) |v| {
            if (v < 1 or v > 64) terr = "NLMeans: tune[1] (blk_y) must be 1..64." else d.blk_y = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 2)) |v| {
            if (v < 1 or v > 8) terr = "NLMeans: tune[2] (vrt) must be 1..8." else d.vrt = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 3)) |v| {
            if (v < 1 or v > 32) terr = "NLMeans: tune[3] (qb) must be 1..32." else d.qb = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 4)) |v| {
            if (v < 1 or v > 33) terr = "NLMeans: tune[4] (pin_min_streams) must be 1..33." else pin_min = @intCast(v);
        }
        if (d.blk_x * d.blk_y > vszipcl.deviceMaxWG(d.device)) terr = "NLMeans: tune blk_x*blk_y exceeds the device max work-group size.";
        if (terr == null) {
            const s2: usize = 2 * @as(usize, d.s);
            const tile_h = d.vrt * d.blk_y + s2;
            const lds_bytes = (tile_h * (d.blk_x + s2) + tile_h * d.blk_x) * @sizeOf(f32);
            if (lds_bytes > vszipcl.deviceLocalMemSize(d.device))
                terr = "NLMeans: tune blk/vrt with this `s` needs more local memory than the device has (lower blk, vrt or s).";
        }
        if (terr) |msg| {
            map_out.setError(msg);
            d.context.release();
            return;
        }
    }

    {
        const spt_side: i32 = 2 * a + 1;
        const spt_area: i32 = spt_side * spt_side;
        const center: i32 = dd;
        var wq_list: std.ArrayListUnmanaged(i32) = .empty;
        var aq_list: std.ArrayListUnmanaged(i32) = .empty;
        const variants = allocator.alloc(Variant, @intCast(dd + 1)) catch unreachable;
        var m: i32 = 0;
        while (m <= dd) : (m += 1) {
            const v = &variants[@intCast(m)];
            v.w_base = @intCast(wq_list.items.len / 8);
            v.q_base = @intCast(aq_list.items.len / 8);
            var boff: std.ArrayListUnmanaged(u32) = .empty;
            var q_idx: u32 = 0;
            var kk: i32 = -m;
            while (kk <= 0) : (kk += 1) {
                var j: i32 = -a;
                while (j <= a) : (j += 1) {
                    var i: i32 = -a;
                    while (i <= a) : (i += 1) {
                        if (kk * spt_area + j * spt_side + i < 0) {
                            const b_local: u32 = q_idx % d.qb;
                            if (b_local == 0) boff.append(allocator, @intCast(wq_list.items.len / 8 - v.w_base)) catch unreachable;
                            const slot_c: i32 = if (dd == 0) @intCast(b_local) else 2 * @as(i32, @intCast(b_local));
                            const slot_m: i32 = if (kk != 0) slot_c + 1 else slot_c;
                            wq_list.appendSlice(allocator, &.{ center, i, j, kk, slot_c, 0, 0, 0 }) catch unreachable;
                            if (kk != 0) wq_list.appendSlice(allocator, &.{ center - kk, i, j, kk, slot_m, 0, 0, 0 }) catch unreachable;
                            aq_list.appendSlice(allocator, &.{ i, j, kk, slot_c, slot_m, 0, 0, 0 }) catch unreachable;
                            q_idx += 1;
                        }
                    }
                }
            }
            boff.append(allocator, @intCast(wq_list.items.len / 8 - v.w_base)) catch unreachable;
            v.q_cnt = q_idx;
            v.w_boff = boff.toOwnedSlice(allocator) catch unreachable;
        }
        d.wq_host = wq_list.toOwnedSlice(allocator) catch unreachable;
        d.aq_host = aq_list.toOwnedSlice(allocator) catch unreachable;
        d.variants = variants;
    }

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;
    keep = true;

    const streams: usize = if (ns_req) |ns| @intCast(ns) else 1;
    data.use_pinned = streams >= pin_min;
    data.pool.prime(data, streams);
    data.pool.prewarm(streams) catch |err| {
        map_out.setError("NLMeans: OpenCL stream init failed.");
        std.log.err("NLMeans stream init failed: {}", .{err});
        data.pool.deinit();
        data.context.release();
        freeTables(data);
        allocator.destroy(data);
        keep = false;
        return;
    };

    const rp: vs.RequestPattern = if (d.d > 0) .General else .StrictSpatial;
    var dep = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = rp },
        .{ .source = d.ref_node, .requestPattern = rp },
    };
    const deps = if (d.has_ref) dep[0..2] else dep[0..1];
    zapi.createVideoFilter(out, "NLMeans", &d.vi, getFrame, free, .Parallel, deps, data);
}
