const std = @import("std");
const vszipcl = @import("vszipcl.zig");
const clpool = @import("clpool.zig");

const cl = vszipcl.cl;
const vapoursynth = vszipcl.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;
const zon = @import("zon");

const allocator = std.heap.c_allocator;

const mdis_max = 40;
const tpitch_max = 2 * mdis_max + 1;
const tune_len = 4;

const kernel_src =
    \\#define TPMAX 85  /* tpitch_max + 4 sentinels (2 per side; K=2 fused DP reads tid..tid+4) */
    \\#define FLTMAX9 3.0e38f
    \\#ifndef CN
    \\#define CN 2      /* = nrad; overridden by -DCN=<nrad> at build time so the
    \\                     hot cost loop has a compile-time bound and fully unrolls */
    \\#endif
    \\#ifndef MDIS
    \\#define MDIS 20   /* overridden by -DMDIS=<mdis>: bakes tpitch so the strip fill's
    \\                     i/TP remap is a mul-shift and cst[] indexing is compile-time */
    \\#endif
    \\#ifndef BX
    \\#define BX 8      /* strip width: x-positions whose costs are batch-computed per
    \\                     cooperative fill (one barrier), then BX cheap DP steps */
    \\#endif
    \\#define TP  (2*MDIS + 1)   /* directions, non-hp */
    \\#define TPH (4*MDIS + 1)   /* directions, hp     */
    \\
    \\/* ---- io layer (-DBITS={8,16,32} -DHALF={0,1}; io type is clip-wide) ----
    \\ * io_t   — element type of the RAW-domain buffers (src/dst/dst2/in-/outframe/scp).
    \\ *          half is pointer-only without cl_khr_fp16 (not exposed on NVIDIA): NEVER
    \\ *          dereference an io_t* directly on the HALF path — only LOAD_IO/STORE_IO.
    \\ * raw_t  — bit-pattern type for VERBATIM element moves (copy_kept, transpose): kept
    \\ *          rows and transposes must stay bit-lossless for every io type. float for
    \\ *          BITS=32 keeps the f32 build token-identical to the pre-io kernel.
    \\ * LOAD_IO(p,i)    — io -> f32: identity (f32) / exact widen (f16) / UNORM decode
    \\ *                   v/PEAK, PEAK=(1<<BITS)-1 (full-range convention, like Deband).
    \\ * STORE_IO(p,i,x) — f32 -> io: identity / vstore_half_rte / convert_sat_rte(x*PEAK)
    \\ *                   (_sat is REQUIRED: the cubic interp overshoots [0,1] on edges). */
    \\#ifndef BITS
    \\#define BITS 32
    \\#endif
    \\#ifndef HALF
    \\#define HALF 0
    \\#endif
    \\#if BITS == 32
    \\typedef float io_t;
    \\typedef float raw_t;
    \\#define LOAD_IO(p, i) ((p)[i])
    \\#define STORE_IO(p, i, x) ((p)[i] = (x))
    \\#elif BITS == 16 && HALF
    \\typedef half io_t;
    \\typedef ushort raw_t;
    \\#define LOAD_IO(p, i) vload_half((size_t)(i), p)
    \\#define STORE_IO(p, i, x) vstore_half_rte((x), (size_t)(i), p)
    \\#elif BITS == 16
    \\typedef ushort io_t;
    \\typedef ushort raw_t;
    \\#define LOAD_IO(p, i) (convert_float((p)[i]) / 65535.0f)
    \\#define STORE_IO(p, i, x) ((p)[i] = convert_ushort_sat_rte((x) * 65535.0f))
    \\#else
    \\typedef uchar io_t;
    \\typedef uchar raw_t;
    \\#define LOAD_IO(p, i) (convert_float((p)[i]) / 255.0f)
    \\#define STORE_IO(p, i, x) ((p)[i] = convert_uchar_sat_rte((x) * 255.0f))
    \\#endif
    \\
    \\// mirror-reflect a horizontal index into [0,w) (fractional axis, no edge dup)
    \\static int refl(int i, const int w) {
    \\    if (w == 1) return 0;
    \\    while (i < 0 || i >= w) { if (i < 0) i = -i; if (i >= w) i = 2*(w-1) - i; }
    \\    return i;
    \\}
    \\
    \\// Build a horizontally mirror-padded copy of `src`: every source row of width
    \\// `w` (stride `stride`) becomes a row of width pad+w+pad (stride `pstride`) in
    \\// `out`, so out[y*pstride + (j+pad)] == src[y*stride + refl(j,w)] for any j in
    \\// [-pad, w-1+pad]. The full multi-bounce refl is computed here ONCE per padded
    \\// column, so the hot interp loops can index out[... + (idx+pad)] with no refl.
    \\// global = (pad+w+pad, src_h).
    \\// The io->f32 conversion happens HERE (LOAD_IO), once per padded column — the entire
    \\// downstream pipeline (pad_hp, costs, DP, interp reads) then works on f32 srcpad rows.
    \\// soff: the plane's element offset into the src REGION buffer (0 unless vcheck>0
    \\// multi-plane, where d_src holds every plane so the fused vcheck can read them all).
    \\kernel void pad_src(global float *out, global const io_t *src,
    \\                    const int w, const int stride, const int pstride,
    \\                    const int pad, const int src_h, const int soff) {
    \\    const int px = get_global_id(0);     // padded column [0, pad+w+pad)
    \\    const int y  = get_global_id(1);
    \\    if (px >= pad + w + pad || y >= src_h) return;
    \\    out[y*pstride + px] = LOAD_IO(src, soff + y*stride + refl(px - pad, w));
    \\}
    \\
    \\// Half-pel companion of pad_src (only used when hp=1): hpout[y][px] is the cubic
    \\// half-pel sample between full-pel columns px and px+1 of the mirror-padded srcpad
    \\// row. It depends ONLY on the source row (not on direction u or DP position x), so
    \\// precomputing it once per frame removes ALL inline HP() recompute from the
    \\// barrier-bound hp DP loop (interp_hp reads hpX[j] instead of recomputing). Same
    \\// 0.5625/0.0625 coefficients + tap grouping as the old inline HP()/CPU computeHpRow,
    \\// over the same padded data, so it stays bit-identical. global = (pad+w+pad, src_h).
    \\kernel void pad_hp(global float *hpout, global const float *srcpad,
    \\                   const int pstride, const int src_h, const int pad, const int w) {
    \\    const int px = get_global_id(0);
    \\    const int y  = get_global_id(1);
    \\    const int pw = pad + w + pad;
    \\    if (px >= pw || y >= src_h) return;
    \\    global const float *r = srcpad + (long)y*pstride;
    \\    float v = (px >= 1 && px + 2 <= pw - 1)
    \\        ? 0.5625f*(r[px]+r[px+1]) - 0.0625f*(r[px-1]+r[px+2])
    \\        : r[px];
    \\    hpout[(long)y*pstride + px] = v;
    \\}
    \\
    \\// connection cost for output position x, direction u (matches interpLine).
    \\// Row pointers are PADDED rows; index [idx+pad] with no per-access refl().
    \\static float conn_cost(global const float * restrict r3p, global const float * restrict r1p,
    \\                       global const float * restrict r1n, global const float * restrict r3n,
    \\                       const int x, const int u, const int w, const int nrad,
    \\                       const int pad,
    \\                       const float alpha, const float beta, const float one_minus_ab) {
    \\    const int two_u = 2*u;
    \\    const int xp = x + pad;            // x mapped into the padded row
    \\    float sw = 0.0f;
    \\    #pragma unroll
    \\    for (int k = -CN; k <= CN; k++) {
    \\        int a0 = xp + u + k,    b0 = a0 - two_u;
    \\        int a1 = xp + k,        b1 = a1 - two_u;
    \\        int a2 = xp + two_u + k, b2 = a2 - two_u;
    \\        sw += fabs(r3p[a0]-r1p[b0]) + fabs(r1p[a0]-r1n[b0]) + fabs(r1n[a0]-r3n[b0]);
    \\        sw += fabs(r3p[a1]-r1p[b1]) + fabs(r1p[a1]-r1n[b1]) + fabs(r1n[a1]-r3n[b1]);
    \\        sw += fabs(r3p[a2]-r1p[b2]) + fabs(r1p[a2]-r1n[b2]) + fabs(r1n[a2]-r3n[b2]);
    \\    }
    \\    float ip = (r1p[xp+u] + r1n[xp-u]) * 0.5f;
    \\    float v = fabs(r1p[xp] - ip) + fabs(r1n[xp] - ip);
    \\    return alpha * sw + beta * (float)abs(u) + one_minus_ab * v;
    \\}
    \\
    \\// ---- h-precompute cost fill (round 4) ----
    \\// The cost algebra: with h_{+u}(c) = fabs(r3p[c]-r1p[c-2u]) + fabs(r1p[c]-r1n[c-2u]) +
    \\// fabs(r1n[c]-r3n[c-2u]) (ONE statement, 3 fabs L-to-R — VERBATIM one accumulation
    \\// statement of conn_cost) and h_{-u} its c+2u mirror, conn_cost's loop is EXACTLY
    \\//   for k: sw += h_u(xp+u+k); sw += h_u(xp+k); sw += h_u(xp+2u+k);
    \\// i.e. three sliding windows of one per-(row,u) column function. A run of RUN
    \\// consecutive x at fixed u therefore needs only the 6 right-edge h values per new x
    \\// (each window slides by exactly +1), cutting the fill's loads/cost ~73 -> ~20-32
    \\// (the fill was measured 64% of the frame). BIT-EXACT: every h value is the original
    \\// expression tree on the same addresses (no muls in h or the window sums -> no
    \\// FMA-contraction freedom), each accumulator adds the same value sequence in the
    \\// same per-k interleaved order, and the tails are conn_cost_pair's trees verbatim.
    \\#define CW (2*CN + 1)          /* h-window length: 2*nrad+1 */
    \\#ifndef RUN
    \\#define RUN 8                  /* consecutive x per fill unit; BX % RUN == 0 */
    \\#endif
    \\#ifndef RUN_HP
    \\#define RUN_HP 4               /* hp run length; BX % RUN_HP == 0. The original "4 beat
    \\                                  8 by ~4%" measurement PREDATES the round-4 fill rework
    \\                                  + reg cap: re-measured 2026-07 on the shipped kernel,
    \\                                  tune run_hp=8 now WINS +4.6% at heavy hp (see tune.md
    \\                                  RTX 30 table) — default left at 4 pending a full
    \\                                  multi-config gate; 16 collapsed hp ~-17%. */
    \\#endif
    \\
    \\static float h_pos(global const float * restrict r3p, global const float * restrict r1p,
    \\                   global const float * restrict r1n, global const float * restrict r3n,
    \\                   const int c, const int two_u) {
    \\    const int b = c - two_u;
    \\    return fabs(r3p[c]-r1p[b]) + fabs(r1p[c]-r1n[b]) + fabs(r1n[c]-r3n[b]);
    \\}
    \\static float h_neg(global const float * restrict r3p, global const float * restrict r1p,
    \\                   global const float * restrict r1n, global const float * restrict r3n,
    \\                   const int c, const int two_u) {
    \\    const int b = c + two_u;
    \\    return fabs(r3p[c]-r1p[b]) + fabs(r1p[c]-r1n[b]) + fabs(r1n[c]-r3n[b]);
    \\}
    \\// both signs at the SAME c: the 3 center loads are shared (9 loads for 2 h values)
    \\static float2 h_pair(global const float * restrict r3p, global const float * restrict r1p,
    \\                     global const float * restrict r1n, global const float * restrict r3n,
    \\                     const int c, const int two_u) {
    \\    const float A = r3p[c], B = r1p[c], C = r1n[c];
    \\    const float Bm = r1p[c-two_u], Cm = r1n[c-two_u], Dm = r3n[c-two_u];
    \\    const float Bp = r1p[c+two_u], Cp = r1n[c+two_u], Dp = r3n[c+two_u];
    \\    return (float2)(fabs(A-Bm) + fabs(B-Cm) + fabs(C-Dm),     /* h_{+u}(c) */
    \\                    fabs(A-Bp) + fabs(B-Cp) + fabs(C-Dp));    /* h_{-u}(c) */
    \\}
    \\
    \\// RUN-batched pair fill unit: cost(x,+u) and cost(x,-u) for x = xs .. xs+rl-1 into
    \\// co (= cst + dx0*TP). Six register windows, index t <-> column (x + delta - CN + t):
    \\//   wp1: h_{+u} @ x+u+k | wp0: h_{+u} @ x+k | wp2: h_{+u} @ x+2u+k
    \\//   wm1: h_{-u} @ x-u+k | wm0: h_{-u} @ x+k | wm2: h_{-u} @ x-2u+k
    \\static void cost_pair_run(global const float * restrict r3p, global const float * restrict r1p,
    \\                          global const float * restrict r1n, global const float * restrict r3n,
    \\                          local float *co, const int xs, const int rl, const int u,
    \\                          const int pad, const float alpha, const float beta,
    \\                          const float one_minus_ab)
    \\{
    \\    const int two_u = 2*u;
    \\    const int x0p = xs + pad;
    \\    float wp1[CW], wp0[CW], wp2[CW], wm1[CW], wm0[CW], wm2[CW];
    \\    #pragma unroll
    \\    for (int t = 0; t < CW; t++) {              /* prime the six windows at x = xs */
    \\        const int c = x0p - CN + t;
    \\        const float2 hc = h_pair(r3p,r1p,r1n,r3n, c, two_u);
    \\        wp0[t] = hc.x;  wm0[t] = hc.y;
    \\        wp1[t] = h_pos(r3p,r1p,r1n,r3n, c + u,     two_u);
    \\        wp2[t] = h_pos(r3p,r1p,r1n,r3n, c + two_u, two_u);
    \\        wm1[t] = h_neg(r3p,r1p,r1n,r3n, c - u,     two_u);
    \\        wm2[t] = h_neg(r3p,r1p,r1n,r3n, c - two_u, two_u);
    \\    }
    \\    #pragma unroll
    \\    for (int t = 0; t < RUN; t++) {
    \\        if (t >= rl) break;
    \\        const int xp = x0p + t;
    \\        float swp = 0.0f, swm = 0.0f;
    \\        #pragma unroll
    \\        for (int k = 0; k < CW; k++) {          /* ORIGINAL per-k interleave: a0,a1,a2 */
    \\            swp += wp1[k]; swp += wp0[k]; swp += wp2[k];
    \\            swm += wm1[k]; swm += wm0[k]; swm += wm2[k];
    \\        }
    \\        // Tails: expression trees kept VERBATIM per sign (no shared temps in the final
    \\        // sum — a hoisted beta*|u| could change the compiler's FMA contraction shape).
    \\        const float b1pc = r1p[xp], b1nc = r1n[xp];
    \\        float ipp = (r1p[xp+u] + r1n[xp-u]) * 0.5f;
    \\        float vp = fabs(b1pc - ipp) + fabs(b1nc - ipp);
    \\        float ipm = (r1p[xp-u] + r1n[xp+u]) * 0.5f;
    \\        float vm = fabs(b1pc - ipm) + fabs(b1nc - ipm);
    \\        co[t*TP + (MDIS + u)] = alpha * swp + beta * (float)abs(u) + one_minus_ab * vp;
    \\        co[t*TP + (MDIS - u)] = alpha * swm + beta * (float)abs(u) + one_minus_ab * vm;
    \\        if (t + 1 < rl) {                       /* slide all six windows by +1 */
    \\            #pragma unroll
    \\            for (int k = 0; k < CW-1; k++) {
    \\                wp1[k]=wp1[k+1]; wp0[k]=wp0[k+1]; wp2[k]=wp2[k+1];
    \\                wm1[k]=wm1[k+1]; wm0[k]=wm0[k+1]; wm2[k]=wm2[k+1];
    \\            }
    \\            const int c = xp + 1 + CN;          /* new right edge per window */
    \\            const float2 hc = h_pair(r3p,r1p,r1n,r3n, c, two_u);
    \\            wp0[CW-1] = hc.x;  wm0[CW-1] = hc.y;
    \\            wp1[CW-1] = h_pos(r3p,r1p,r1n,r3n, c + u,     two_u);
    \\            wp2[CW-1] = h_pos(r3p,r1p,r1n,r3n, c + two_u, two_u);
    \\            wm1[CW-1] = h_neg(r3p,r1p,r1n,r3n, c - u,     two_u);
    \\            wm2[CW-1] = h_neg(r3p,r1p,r1n,r3n, c - two_u, two_u);
    \\        }
    \\    }
    \\}
    \\
    \\// Copy the kept field rows from src into dst, and (when vcheck off) acts as a
    \\// no-op for interp rows. global = (w, dst_h). raw_t: a VERBATIM bit-pattern element
    \\// copy so kept rows are BIT-LOSSLESS for every io type (u8/u16/f16/f32 alike) — do
    \\// NOT route this through LOAD_IO/STORE_IO (a decode+re-encode round-trip).
    \\// dual=1 (vcheck on): ALSO write dst2, seeding the vcheck output buffer's kept rows
    \\// directly — replaces the old whole-frame clEnqueueCopyBuffer(dst->dst2) (same bytes,
    \\// written once here instead of copied device-side; the interp kernels seed the interp
    \\// rows the same way). dual=0 binds a 1-byte dummy dst2 that is never touched.
    \\// soff/doff: plane element offsets into the src/dst(+dst2) region buffers (see pad_src).
    \\kernel void copy_kept(global raw_t *dst, global const raw_t *src,
    \\                      const int w, const int stride, const int dh,
    \\                      const int field, const int src_h, const int dst_h,
    \\                      global raw_t *dst2, const int dual,
    \\                      const int soff, const int doff) {
    \\    const int x = get_global_id(0);
    \\    const int y = get_global_id(1);
    \\    if (x >= w || y >= dst_h) return;
    \\    int is_interp, sy = -1;
    \\    if (dh) {
    \\        // kept dst rows: 2*ky + (1-field); interp rows: field, field+2, ...
    \\        is_interp = ((y & 1) == field);
    \\        if (!is_interp) sy = (y - (1 - field)) >> 1;
    \\    } else {
    \\        is_interp = ((y & 1) == field);
    \\        if (!is_interp) sy = y;
    \\    }
    \\    if (!is_interp) {
    \\        const raw_t v = src[soff + sy*stride + x];
    \\        dst[doff + y*stride + x] = v;
    \\        if (dual) dst2[doff + y*stride + x] = v;
    \\    }
    \\}
    \\
    \\// One workgroup per interpolated row. local_size = padded tpitch.
    \\// rowidx[off*4 + {0,1,2,3}] = src y-index for r3p,r1p,r1n,r3n.
    \\// dst_y[off] = destination row.
    \\//
    \\// STRIP-BATCHED DP: the conn_cost work is hoisted OUT of the barrier-locked DP loop.
    \\// For each strip of BX x-positions, ALL lanes (not just the tpitch active ones)
    \\// cooperatively compute the BX*TP costs into local `cst` — each lane gets several
    \\// INDEPENDENT cost evaluations (ILP, no barrier between them) — then BX DP steps run
    \\// with a cheap barrier each (~15 instr instead of ~230). Bit-exact: every cost(x,u) is
    \\// still evaluated ONCE by one thread with the identical conn_cost expression (thread
    \\// REASSIGNMENT cannot change a float); the DP combine order/pbackt bytes are untouched.
    \\// The path array lives in the dmap row (global) instead of dynamic local — it freed 92%
    \\// of the local budget (w ints), roughly 2.4x-ing resident workgroups per SM.
    \\kernel void interp(global io_t * restrict dst, global const float * restrict srcpad,
    \\                   global const int * restrict rowidx, global const int * restrict dst_y,
    \\                   global char * restrict pbackt, global int * restrict dmap,
    \\                   const int w, const int stride, const int pstride, const int pad,
    \\                   const int mdis, const int nrad,
    \\                   const float alpha, const float beta, const float gamma,
    \\                   const float one_minus_ab,
    \\                   global io_t * restrict dst2, const int dual,
    \\                   const int doff, const int moff) {
    \\    const int off = get_global_id(1);
    \\    const int tid = get_local_id(0);
    \\    const int lsz = get_local_size(0);
    \\
    \\    // PADDED rows: index [j+pad] for any j in [-pad, w-1+pad], no refl().
    \\    global const float * restrict r3p = srcpad + rowidx[off*4+0]*pstride;
    \\    global const float * restrict r1p = srcpad + rowidx[off*4+1]*pstride;
    \\    global const float * restrict r1n = srcpad + rowidx[off*4+2]*pstride;
    \\    global const float * restrict r3n = srcpad + rowidx[off*4+3]*pstride;
    \\    global char *pb = pbackt + (long)off * w * TP;
    \\
    \\    local float pc[2][TPMAX];
    \\    local float cst[BX * TP];          // strip costs (TP odd => bank-conflict-free)
    \\    const int u = tid - MDIS;          // direction for this thread (valid if tid<TP)
    \\    const int active = (tid < TP);
    \\
    \\    // sentinels: TWO per side (positions {0,1} and {TP+2,TP+3}; lane t's slot is t+2)
    \\    // for the K=2 fused DP step, which reads pc[tid..tid+4]. Sentinels are never
    \\    // relaxed — they stay FLTMAX9 forever, exactly like the old single sentinels.
    \\    if (tid == 0) {
    \\        pc[0][0]=FLTMAX9; pc[0][1]=FLTMAX9; pc[1][0]=FLTMAX9; pc[1][1]=FLTMAX9;
    \\        pc[0][TP+2]=FLTMAX9; pc[0][TP+3]=FLTMAX9; pc[1][TP+2]=FLTMAX9; pc[1][TP+3]=FLTMAX9;
    \\    }
    \\    barrier(CLK_LOCAL_MEM_FENCE);
    \\
    \\    int ping = 0;
    \\    if (active) pc[ping][tid+2] = conn_cost(r3p,r1p,r1n,r3n, 0, u, w, nrad, pad, alpha, beta, one_minus_ab);
    \\    barrier(CLK_LOCAL_MEM_FENCE);
    \\
    \\    #define NUNIT (MDIS + 1)   /* fill units per x-run: u=0 single + MDIS (+u,-u) pairs */
    \\    for (int x0 = 1; x0 < w; x0 += BX) {
    \\        const int bn = min(BX, w - x0);
    \\        // cooperative fill in (pair, RUN-of-x) units: each lane slides six register
    \\        // h-windows across RUN consecutive x at its pair's u (see cost_pair_run) —
    \\        // the LSU-bound bulk of this kernel drops ~2.3-3x in loads. i/NUNIT stays
    \\        // the same baked mul-shift; r (run index) is the slow axis so consecutive
    \\        // lanes keep consecutive u (coalescing unchanged).
    \\        const int nrun = (bn + RUN - 1) / RUN;
    \\        for (int i = tid; i < nrun*NUNIT; i += lsz) {
    \\            const int r  = i / NUNIT;
    \\            const int j  = i - r*NUNIT;
    \\            const int xs = x0 + r*RUN;
    \\            const int rl = min(RUN, bn - r*RUN);
    \\            local float *co = cst + (r*RUN)*TP;
    \\            if (j == 0) {
    \\                for (int t = 0; t < rl; t++)    /* u=0: original conn_cost, per x */
    \\                    co[t*TP + MDIS] = conn_cost(r3p,r1p,r1n,r3n, xs+t, 0, w, nrad, pad, alpha, beta, one_minus_ab);
    \\            } else {
    \\                cost_pair_run(r3p,r1p,r1n,r3n, co, xs, rl, j, pad, alpha, beta, one_minus_ab);
    \\            }
    \\        }
    \\        barrier(CLK_LOCAL_MEM_FENCE);
    \\        // K=2 FUSED DP: one barrier per TWO columns. Each lane redundantly recomputes its
    \\        // neighbours' column-x relaxes (qL/qR, VALUE only) plus its own (qC, emits the pb
    \\        // byte), then relaxes column x+1 from those registers — the intermediate pc state
    \\        // never touches local. BIT-EXACT: the relax contains NO multiplies (only fadd,
    \\        // strict-less-than select and fmin, all exactly rounded; no fast-math flags), so
    \\        // a verbatim clone fed the same inputs produces the same bits under any schedule;
    \\        // the tie-break chain (cent, then left, then right, strict <) and the
    \\        // fmin(bval+cost, FLTMAX9) clamp placement are cloned per sub-step; sentinel
    \\        // positions are never relaxed (the tid==0 / tid==TP-1 overrides reproduce exactly
    \\        // the FLTMAX9 a real sentinel read returns; their clamped cst reads are discarded).
    \\        int dx = 0;
    \\        for (; dx + 1 < bn; dx += 2) {
    \\            const int pong = ping ^ 1;
    \\            if (active) {
    \\                const float p0 = pc[ping][tid];
    \\                const float p1 = pc[ping][tid+1];
    \\                const float p2 = pc[ping][tid+2];   // own slot
    \\                const float p3 = pc[ping][tid+3];
    \\                const float p4 = pc[ping][tid+4];
    \\                const float cL = cst[dx*TP + max(tid-1, 0)];      // value discarded on edge lane
    \\                const float cC = cst[dx*TP + tid];
    \\                const float cR = cst[dx*TP + min(tid+1, TP-1)];   // value discarded on edge lane
    \\                const float cN = cst[(dx+1)*TP + tid];
    \\                // relax_x at slot tid+1 (dir u-1): VALUE only
    \\                float lft = p0 + gamma, cnt = p1, rgt = p2 + gamma;
    \\                float bL = cnt; if (lft < bL) bL = lft; if (rgt < bL) bL = rgt;
    \\                const float qL = (tid == 0) ? FLTMAX9 : fmin(bL + cL, FLTMAX9);
    \\                // relax_x at own slot tid+2: VALUE + column-x pb byte (delta stored at x-1)
    \\                lft = p1 + gamma; cnt = p2; rgt = p3 + gamma;
    \\                float bC = cnt; char bdC = 0;
    \\                if (lft < bC) { bC = lft; bdC = -1; }
    \\                if (rgt < bC) { bC = rgt; bdC =  1; }
    \\                const float qC = fmin(bC + cC, FLTMAX9);
    \\                pb[(x0+dx-1)*TP + tid] = bdC;
    \\                // relax_x at slot tid+3 (dir u+1): VALUE only
    \\                lft = p2 + gamma; cnt = p3; rgt = p4 + gamma;
    \\                float bR = cnt; if (lft < bR) bR = lft; if (rgt < bR) bR = rgt;
    \\                const float qR = (tid == TP-1) ? FLTMAX9 : fmin(bR + cR, FLTMAX9);
    \\                // relax_{x+1} at own slot from the register intermediates
    \\                lft = qL + gamma; cnt = qC; rgt = qR + gamma;
    \\                float bN = cnt; char bdN = 0;
    \\                if (lft < bN) { bN = lft; bdN = -1; }
    \\                if (rgt < bN) { bN = rgt; bdN =  1; }
    \\                pc[pong][tid+2] = fmin(bN + cN, FLTMAX9);
    \\                pb[(x0+dx)*TP + tid] = bdN;
    \\            }
    \\            barrier(CLK_LOCAL_MEM_FENCE);
    \\            ping = pong;                            // ONE flip per 2 columns
    \\        }
    \\        if (dx < bn) {                              // odd-bn tail: original single step
    \\            const int pong = ping ^ 1;
    \\            if (active) {
    \\                const float cost = cst[dx*TP + tid];
    \\                float left  = pc[ping][tid+1] + gamma;  // dir u-1
    \\                float cent  = pc[ping][tid+2];          // dir u
    \\                float right = pc[ping][tid+3] + gamma;  // dir u+1
    \\                float bval = cent; char bd = 0;
    \\                if (left  < bval) { bval = left;  bd = -1; }
    \\                if (right < bval) { bval = right; bd =  1; }
    \\                pc[pong][tid+2] = fmin(bval + cost, FLTMAX9);
    \\                pb[(x0+dx-1)*TP + tid] = bd;   // CPU stores backtrack delta at x-1
    \\            }
    \\            barrier(CLK_LOCAL_MEM_FENCE);
    \\            ping = pong;
    \\        }
    \\    }
    \\
    \\    // serial backtrack: all pbackt global writes must be visible to thread 0. The path
    \\    // goes straight into this row's dmap slice (same values the interpolate stage used
    \\    // to store) — vcheck reads the identical dirs.
    \\    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    \\    global int *dm = dmap + moff + (long)off*stride;
    \\    // LOCAL-STAGED serial backtrack: the walk is w-1 DEPENDENT loads (f feeds the next
    \\    // address), so its makespan is per-load latency x w. cst[] is DEAD after the last DP
    \\    // strip, and pb's x-major layout makes a BTC-column band a CONTIGUOUS byte block —
    \\    // so all lanes cooperatively stage each band into cst (coalesced), and thread 0 walks
    \\    // LOCAL (~30 cy) instead of L2/DRAM (~400+ cy). BTC*TP bytes == sizeof(cst) exactly
    \\    // (4*BX*TP == BX*TP*4). Identical bytes, identical f recurrence -> identical dm.
    \\    {
    \\        local char *stg = (local char *)cst;   // char, NOT uchar: bd deltas are SIGNED
    \\        int f = 0;
    \\        if (tid == 0) dm[w-1] = 0;
    \\        for (int xhi = w-2; xhi >= 0; xhi -= 4*BX) {
    \\            const int xlo = max(xhi - 4*BX + 1, 0);
    \\            const int nb = (xhi - xlo + 1) * TP;
    \\            for (int i = tid; i < nb; i += lsz)
    \\                stg[i] = pb[xlo*TP + i];
    \\            barrier(CLK_LOCAL_MEM_FENCE);
    \\            if (tid == 0) {
    \\                for (int bx = xhi; bx >= xlo; bx--) {
    \\                    f += (int)stg[(bx - xlo)*TP + (MDIS + f)];
    \\                    dm[bx] = f;
    \\                }
    \\            }
    \\            barrier(CLK_LOCAL_MEM_FENCE);   // walk done before stg is overwritten
    \\        }
    \\    }
    \\    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);  // dm[] must be visible to all
    \\
    \\    // parallel interpolate (STORE_IO: the f32 result quantizes to the io type here).
    \\    // dual=1 (vcheck on): also seed the vcheck output buffer's interp rows — together
    \\    // with copy_kept's dual store this fully replaces the old dst->dst2 device copy.
    \\    global io_t *drow = dst + doff + dst_y[off]*stride;
    \\    global io_t *drow2 = dst2 + doff + dst_y[off]*stride;
    \\    for (int x = tid; x < w; x += lsz) {
    \\        const int dir = dm[x];
    \\        int ad = abs(dir);
    \\        const int xp = x + pad;
    \\        float val;
    \\        if (x >= ad*3 && x + ad*3 <= w-1) {
    \\            val = 0.5625f * (r1p[xp+dir] + r1n[xp-dir])
    \\                - 0.0625f * (r3p[xp+dir*3] + r3n[xp-dir*3]);
    \\        } else {
    \\            val = (r1p[xp+dir] + r1n[xp-dir]) * 0.5f;
    \\        }
    \\        STORE_IO(drow, x, val);
    \\        if (dual) STORE_IO(drow2, x, val);
    \\    }
    \\}
    \\
    \\// ---- half-pel (hp=1) path ----
    \\#define TPMAX_HP 165   /* 4*mdis_max+1 + 4 sentinels */
    \\// Rows passed to the hp helpers are PADDED rows already advanced by `pad`, so
    \\// R(row,j) is a plain row[j] (j may be negative / >= w within the pad margin).
    \\static float R(global const float * restrict row, int j) { return row[j]; }
    \\// The hp* rows are the PRECOMPUTED half-pel padded rows from pad_hp, pre-advanced by
    \\// `pad` just like the R() rows, so hpX[j] == the old inline HP(rX,j) cubic column for
    \\// column. The odd-u cost now just LOADS them (no per-k/per-x cubic recompute) — the
    \\// summation order + operand grouping are unchanged, so it stays bit-identical.
    \\static float conn_cost_hp(global const float * restrict r3p, global const float * restrict r1p,
    \\                          global const float * restrict r1n, global const float * restrict r3n,
    \\                          global const float * restrict hp3p, global const float * restrict hp1p,
    \\                          global const float * restrict hp1n, global const float * restrict hp3n,
    \\                          const int x, const int u, const int w, const int nrad,
    \\                          const float alpha3, const float beta255, const float one_minus_ab) {
    \\    const int uh = u >> 1;
    \\    const int odd = u & 1;
    \\    const int lo0 = odd ? (-uh - 1) : (-uh);
    \\    float s0=0.0f, s1=0.0f, s2=0.0f;
    \\    #pragma unroll
    \\    for (int k=-CN; k<=CN; k++) {
    \\        int xk = x+k, hi = x+uh+k, lo = x+lo0+k, xu = x+u+k, xmu = x-u+k;
    \\        s1 += fabs(R(r3p,xk)-R(r1p,xmu)) + fabs(R(r1p,xk)-R(r1n,xmu)) + fabs(R(r1n,xk)-R(r3n,xmu));
    \\        s2 += fabs(R(r3p,xu)-R(r1p,xk)) + fabs(R(r1p,xu)-R(r1n,xk)) + fabs(R(r1n,xu)-R(r3n,xk));
    \\        if (odd)
    \\            s0 += fabs(hp3p[hi]-hp1p[lo]) + fabs(hp1p[hi]-hp1n[lo]) + fabs(hp1n[hi]-hp3n[lo]);
    \\        else
    \\            s0 += fabs(R(r3p,hi)-R(r1p,lo)) + fabs(R(r1p,hi)-R(r1n,lo)) + fabs(R(r1n,hi)-R(r3n,lo));
    \\    }
    \\    float Bxuh = odd ? hp1p[x+uh] : R(r1p,x+uh);
    \\    float Cxlo = odd ? hp1n[x+lo0] : R(r1n,x+lo0);
    \\    float ip = (Bxuh + Cxlo) * 0.5f;
    \\    float v = fabs(R(r1p,x)-ip) + fabs(R(r1n,x)-ip);
    \\    return alpha3*(s0+s1+s2) + beta255*(float)abs(u)*0.5f + one_minus_ab*v;
    \\}
    \\
    \\// hp h-precompute fill (round 4, see the non-hp block): conn_cost_hp is window sums
    \\// of the SAME h shape with partner distance u (not 2u) — s1 = SUM_k h1_{+u}(x+k),
    \\// s2 = SUM_k h1_{+u}(x+u+k) over the full rows, and s0 = SUM_k g_{+u}(x+uh+k) over
    \\// the parity-selected S rows (hp rows when odd, full rows when even; hi-lo == u for
    \\// both parities under arithmetic >>, the shipped invariant). The h_pos/h_neg/h_pair
    \\// helpers are reused with two_u := u. s0/s1/s2 stay SEPARATE accumulators (summed
    \\// only in the tail, like the original), so each window sum needs only ITS OWN
    \\// ascending-k order — bit-exact by the same argument as cost_pair_run.
    \\// Six register windows per (pair, run) unit:
    \\//   V0p: h1_{+u} @ x+k   | V1p: h1_{+u} @ x+u+k  | G0p: g_{+u} @ x+uh+k
    \\//   V0m: h1_{-u} @ x+k   | V1m: h1_{-u} @ x-u+k  | G0m: g_{-u} @ x+lo0+k
    \\static void cost_hp_pair_run(
    \\        global const float * restrict r3p, global const float * restrict r1p,
    \\        global const float * restrict r1n, global const float * restrict r3n,
    \\        global const float * restrict S3p, global const float * restrict S1p,
    \\        global const float * restrict S1n, global const float * restrict S3n,
    \\        global const float * restrict Sel1p, global const float * restrict Sel1n,
    \\        local float *co, const int xs, const int rl, const int u,
    \\        const float alpha3, const float beta255, const float one_minus_ab)
    \\{
    \\    const int uh = u >> 1;
    \\    const int lo0 = (u & 1) ? (-uh - 1) : (-uh);   // == uh of -u (both parities)
    \\    float V0p[CW], V0m[CW], V1p[CW], V1m[CW], G0p[CW], G0m[CW];
    \\    #pragma unroll
    \\    for (int t = 0; t < CW; t++) {                 /* prime at x = xs (rows pre-+pad) */
    \\        const int c = xs - CN + t;
    \\        const float2 hc = h_pair(r3p,r1p,r1n,r3n, c, u);
    \\        V0p[t] = hc.x;  V0m[t] = hc.y;
    \\        V1p[t] = h_pos(r3p,r1p,r1n,r3n, c + u, u);
    \\        V1m[t] = h_neg(r3p,r1p,r1n,r3n, c - u, u);
    \\        G0p[t] = h_pos(S3p,S1p,S1n,S3n, c + uh,  u);
    \\        G0m[t] = h_neg(S3p,S1p,S1n,S3n, c + lo0, u);
    \\    }
    \\    #pragma unroll
    \\    for (int t = 0; t < RUN_HP; t++) {
    \\        if (t >= rl) break;
    \\        const int x = xs + t;
    \\        float s0p=0.0f, s1p=0.0f, s2p=0.0f;
    \\        float s0m=0.0f, s1m=0.0f, s2m=0.0f;
    \\        #pragma unroll
    \\        for (int k = 0; k < CW; k++) {             /* per-accumulator ascending k */
    \\            s1p += V0p[k]; s2p += V1p[k];
    \\            s1m += V0m[k]; s2m += V1m[k];
    \\            s0p += G0p[k]; s0m += G0m[k];
    \\        }
    \\        // Tails per sign, expression trees verbatim; Sel* == the parity-selected rows
    \\        // (odd ? hp : full), so Sel1p[x+uh] == the original odd? ternary value.
    \\        const float b1pc = R(r1p,x), b1nc = R(r1n,x);
    \\        float Bxuh_p = Sel1p[x+uh];
    \\        float Cxlo_p = Sel1n[x+lo0];
    \\        float ip_p = (Bxuh_p + Cxlo_p) * 0.5f;
    \\        float v_p = fabs(b1pc-ip_p) + fabs(b1nc-ip_p);
    \\        float Bxuh_m = Sel1p[x+lo0];
    \\        float Cxlo_m = Sel1n[x+uh];
    \\        float ip_m = (Bxuh_m + Cxlo_m) * 0.5f;
    \\        float v_m = fabs(b1pc-ip_m) + fabs(b1nc-ip_m);
    \\        co[t*TPH + (2*MDIS + u)] = alpha3*(s0p+s1p+s2p) + beta255*(float)abs(u)*0.5f + one_minus_ab*v_p;
    \\        co[t*TPH + (2*MDIS - u)] = alpha3*(s0m+s1m+s2m) + beta255*(float)abs(u)*0.5f + one_minus_ab*v_m;
    \\        if (t + 1 < rl) {                          /* slide all six windows by +1 */
    \\            #pragma unroll
    \\            for (int k = 0; k < CW-1; k++) {
    \\                V0p[k]=V0p[k+1]; V0m[k]=V0m[k+1]; V1p[k]=V1p[k+1];
    \\                V1m[k]=V1m[k+1]; G0p[k]=G0p[k+1]; G0m[k]=G0m[k+1];
    \\            }
    \\            const int c = x + 1 + CN;
    \\            const float2 hc = h_pair(r3p,r1p,r1n,r3n, c, u);
    \\            V0p[CW-1] = hc.x;  V0m[CW-1] = hc.y;
    \\            V1p[CW-1] = h_pos(r3p,r1p,r1n,r3n, c + u, u);
    \\            V1m[CW-1] = h_neg(r3p,r1p,r1n,r3n, c - u, u);
    \\            G0p[CW-1] = h_pos(S3p,S1p,S1n,S3n, c + uh,  u);
    \\            G0m[CW-1] = h_neg(S3p,S1p,S1n,S3n, c + lo0, u);
    \\        }
    \\    }
    \\}
    \\
    \\// Strip-batched exactly like `interp` (see its header comment); TPH directions, 5-way
    \\// min, path array in the dmap row.
    \\kernel void interp_hp(global io_t * restrict dst, global const float * restrict srcpad,
    \\                      global const int * restrict rowidx, global const int * restrict dst_y,
    \\                      global char * restrict pbackt, global int * restrict dmap,
    \\                      const int w, const int stride, const int pstride, const int pad,
    \\                      const int mdis, const int nrad,
    \\                      const float alpha3, const float beta255, const float gamma255,
    \\                      const float one_minus_ab,
    \\                      global const float * restrict hpsrcpad,
    \\                      global io_t * restrict dst2, const int dual,
    \\                      const int doff, const int moff) {
    \\    const int off = get_global_id(1);
    \\    const int tid = get_local_id(0);
    \\    const int lsz = get_local_size(0);
    \\    const int cen = 2*MDIS;
    \\    // PADDED rows, pre-advanced by `pad` so R(row,j)=row[j] needs no refl().
    \\    global const float * restrict r3p = srcpad + rowidx[off*4+0]*pstride + pad;
    \\    global const float * restrict r1p = srcpad + rowidx[off*4+1]*pstride + pad;
    \\    global const float * restrict r1n = srcpad + rowidx[off*4+2]*pstride + pad;
    \\    global const float * restrict r3n = srcpad + rowidx[off*4+3]*pstride + pad;
    \\    // Precomputed half-pel rows (pad_hp), pre-advanced by `pad` exactly like the R() rows.
    \\    global const float * restrict hp3p = hpsrcpad + rowidx[off*4+0]*pstride + pad;
    \\    global const float * restrict hp1p = hpsrcpad + rowidx[off*4+1]*pstride + pad;
    \\    global const float * restrict hp1n = hpsrcpad + rowidx[off*4+2]*pstride + pad;
    \\    global const float * restrict hp3n = hpsrcpad + rowidx[off*4+3]*pstride + pad;
    \\    global char *pb = pbackt + (long)off * w * TPH;
    \\    local float pc[2][TPMAX_HP];
    \\    local float cst[BX * TPH];
    \\    const int u = tid - cen;
    \\    const int active = (tid < TPH);
    \\    if (tid == 0) { for (int b=0;b<2;b++){ pc[b][0]=FLTMAX9; pc[b][1]=FLTMAX9; pc[b][TPH+2]=FLTMAX9; pc[b][TPH+3]=FLTMAX9; } }
    \\    barrier(CLK_LOCAL_MEM_FENCE);
    \\    int ping = 0;
    \\    if (active) pc[ping][tid+2] = conn_cost_hp(r3p,r1p,r1n,r3n, hp3p,hp1p,hp1n,hp3n, 0, u, w, nrad, alpha3, beta255, one_minus_ab);
    \\    barrier(CLK_LOCAL_MEM_FENCE);
    \\    const float g1 = gamma255*0.5f, g2 = gamma255;
    \\    for (int x0 = 1; x0 < w; x0 += BX) {
    \\        const int bn = min(BX, w - x0);
    \\        // PARITY-SPLIT fill in (pair, RUN-of-x) units: pairs form within a parity so
    \\        // the parity-dependent row selection stays warp-uniform per pass (S*/Sel*
    \\        // pointers picked at the call site), and each lane slides six register
    \\        // h-windows across RUN consecutive x (cost_hp_pair_run — the LSU-bound bulk
    \\        // of this kernel drops ~2.3-3x in loads, like the non-hp fill).
    \\        // Every (x,u) is still computed exactly once with the identical expressions.
    \\        #define NEVU (MDIS + 1)   /* even units: u=0 single + MDIS pairs (2j,-2j) */
    \\        #define NODU (MDIS)       /* odd units: MDIS pairs (2j+1,-(2j+1)) */
    \\        const int nrun = (bn + RUN_HP - 1) / RUN_HP;
    \\        for (int i = tid; i < nrun*NEVU; i += lsz) {       // even directions (S = full rows)
    \\            const int r = i / NEVU;
    \\            const int j = i - r*NEVU;
    \\            const int xs = x0 + r*RUN_HP;
    \\            const int rl = min(RUN_HP, bn - r*RUN_HP);
    \\            local float *co = cst + (r*RUN_HP)*TPH;
    \\            if (j == 0) {
    \\                for (int t = 0; t < rl; t++)    /* u=0: original conn_cost_hp, per x */
    \\                    co[t*TPH + cen] = conn_cost_hp(r3p,r1p,r1n,r3n, hp3p,hp1p,hp1n,hp3n, xs+t, 0, w, nrad, alpha3, beta255, one_minus_ab);
    \\            } else {
    \\                cost_hp_pair_run(r3p,r1p,r1n,r3n, r3p,r1p,r1n,r3n, r1p,r1n,
    \\                                 co, xs, rl, 2*j, alpha3, beta255, one_minus_ab);
    \\            }
    \\        }
    \\        for (int i = tid; i < nrun*NODU; i += lsz) {       // odd directions (S = hp rows)
    \\            const int r = i / NODU;
    \\            const int j = i - r*NODU;
    \\            const int xs = x0 + r*RUN_HP;
    \\            const int rl = min(RUN_HP, bn - r*RUN_HP);
    \\            cost_hp_pair_run(r3p,r1p,r1n,r3n, hp3p,hp1p,hp1n,hp3n, hp1p,hp1n,
    \\                             cst + (r*RUN_HP)*TPH, xs, rl, 2*j + 1, alpha3, beta255, one_minus_ab);
    \\        }
    \\        barrier(CLK_LOCAL_MEM_FENCE);
    \\        for (int dx = 0; dx < bn; dx++) {
    \\            const int pong = ping ^ 1;
    \\            if (active) {
    \\                const float cost = cst[dx*TPH + tid];
    \\                float c_m2 = fmin(pc[ping][tid+0]+g2, FLTMAX9);
    \\                float c_m1 = fmin(pc[ping][tid+1]+g1, FLTMAX9);
    \\                float c_0  = fmin(pc[ping][tid+2],    FLTMAX9);
    \\                float c_p1 = fmin(pc[ping][tid+3]+g1, FLTMAX9);
    \\                float c_p2 = fmin(pc[ping][tid+4]+g2, FLTMAX9);
    \\                float bval = c_m2; char bd = -2;
    \\                if (c_m1 < bval) { bval = c_m1; bd = -1; }
    \\                if (c_0  < bval) { bval = c_0;  bd =  0; }
    \\                if (c_p1 < bval) { bval = c_p1; bd =  1; }
    \\                if (c_p2 < bval) { bval = c_p2; bd =  2; }
    \\                pc[pong][tid+2] = fmin(bval + cost, FLTMAX9);
    \\                pb[(x0+dx-1)*TPH + tid] = bd;
    \\            }
    \\            barrier(CLK_LOCAL_MEM_FENCE);
    \\            ping = pong;
    \\        }
    \\    }
    \\    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    \\    global int *dm = dmap + moff + (long)off*stride;
    \\    // LOCAL-STAGED serial backtrack (see interp): 4*BX*TPH bytes == sizeof(cst) exactly.
    \\    {
    \\        local char *stg = (local char *)cst;   // char, NOT uchar: bd deltas are SIGNED
    \\        int f = 0;
    \\        if (tid == 0) dm[w-1] = 0;
    \\        for (int xhi = w-2; xhi >= 0; xhi -= 4*BX) {
    \\            const int xlo = max(xhi - 4*BX + 1, 0);
    \\            const int nb = (xhi - xlo + 1) * TPH;
    \\            for (int i = tid; i < nb; i += lsz)
    \\                stg[i] = pb[xlo*TPH + i];
    \\            barrier(CLK_LOCAL_MEM_FENCE);
    \\            if (tid == 0) {
    \\                for (int bx = xhi; bx >= xlo; bx--) {
    \\                    f += (int)stg[(bx - xlo)*TPH + (cen + f)];
    \\                    dm[bx] = f;
    \\                }
    \\            }
    \\            barrier(CLK_LOCAL_MEM_FENCE);   // walk done before stg is overwritten
    \\        }
    \\    }
    \\    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    \\    global io_t *drow = dst + doff + dst_y[off]*stride;
    \\    global io_t *drow2 = dst2 + doff + dst_y[off]*stride;   // dual=1: seed vcheck's out (see interp)
    \\    for (int x = tid; x < w; x += lsz) {
    \\        const int dir = dm[x];
    \\        float val;
    \\        if ((dir & 1) == 0) {
    \\            int d2 = dir >> 1; int ad = abs(d2);
    \\            if (x >= ad*3 && x + ad*3 <= w-1)
    \\                val = 0.5625f*(R(r1p,x+d2)+R(r1n,x-d2)) - 0.0625f*(R(r3p,x+3*d2)+R(r3n,x-3*d2));
    \\            else
    \\                val = (R(r1p,x+d2)+R(r1n,x-d2))*0.5f;
    \\        } else {
    \\            int d20 = dir>>1, d21 = (dir+1)>>1, d30 = (dir*3)>>1, d31 = (dir*3+1)>>1;
    \\            int ad = max(abs(d30), abs(d31));
    \\            if (x >= ad && x + ad <= w-1) {
    \\                float c0 = R(r3p,x+d30)+R(r3p,x+d31);
    \\                float c1 = R(r1p,x+d20)+R(r1p,x+d21);
    \\                float c2 = R(r1n,x-d20)+R(r1n,x-d21);
    \\                float c3 = R(r3n,x-d30)+R(r3n,x-d31);
    \\                val = 0.28125f*(c1+c2) - 0.03125f*(c0+c3);
    \\            } else {
    \\                val = (R(r1p,x+d20)+R(r1p,x+d21)+R(r1n,x-d20)+R(r1n,x-d21))*0.25f;
    \\            }
    \\        }
    \\        STORE_IO(drow, x, val);
    \\        if (dual) STORE_IO(drow2, x, val);
    \\    }
    \\}
    \\
    \\// Sequential vCheck in ONE launch: a single workgroup loops over the
    \\// interpolated rows (`off`) in increasing order, barrier()-ing between rows.
    \\// d2p (row 2 above = the previous interpolated row) is read from the OUTPUT
    \\// buffer `out`, which by the barrier already holds that row's vchecked
    \\// result — reproducing the CPU's in-place ±2 feedback exactly, and bit-exact
    \\// despite the horizontal ±dir coupling (d2p is read at x±offh) because the
    \\// whole prior row is finished before the barrier. The just-written prior row
    \\// stays L2-hot, so reading d2p from global is as fast as a local copy would
    \\// be (shared-mem staging of it was benchmarked and was flat-to-worse). Every
    \\// other neighbour comes from the interp snapshot `dst`. Launch global = local
    \\// = one workgroup; work-items grid-stride over x within each row. The host
    \\// seeds `out` (= copy of `dst`) so kept + non-vcheckable rows pass through,
    \\// and so the first row's d2p reads a valid (off=0 interp) row.
    \\// io note: vcheck runs in the IO domain end-to-end — every row it touches (the interp
    \\// snapshot `dst`, its own output `out`, the raw source rows d3p/d3n and the sclip) is
    \\// io_t, so all reads go through LOAD_IO and both stores through STORE_IO. The ±2-row
    \\// feedback therefore sees the QUANTIZED previously-stored row for int/f16 io (like a
    \\// CPU int-domain implementation); for f32 io the macros are the identity (unchanged).
    \\// VC_WG (baked): the exact single-workgroup launch size. reqd_work_group_size is
    \\// LOAD-BEARING under -cl-nv-maxrregcount=96: without it ptxas budgets registers for a
    \\// possible 1024-thread launch ONLY up to the 64-reg default — raising the cap to 96 let
    \\// this kernel exceed 64 regs (f16 io hit 70) and the 1024-thread enqueue then failed
    \\// with CL_INVALID_WORK_GROUP_SIZE. The hard reqd restores the exact-size budget.
    \\// FUSED PLANE BATCH (round 4): the per-plane vcheck chains are mutually independent —
    \\// the only reason they serialized was the shared in-order queue. With every io buffer
    \\// holding per-plane REGIONS (element offsets in `geom`), ONE launch of nplanes
    \\// workgroups runs the chains CONCURRENTLY (group_id = plane; each WG executes the
    \\// original single-plane body verbatim over its own disjoint regions, barrier() is
    \\// WG-local). YUV ns1 makespan drops from sum(0.75+0.4+0.4) to max(~0.78) ms; a Gray
    \\// launch (nplanes=1) degenerates to exactly the old single-WG schedule.
    \\// geom[p*6+..] = {w, stride, dst_h, off_io, off_src, off_dmap}; n_interp is derived
    \\// in-kernel with process()'s exact formula (dst_h - field + 1)/2.
    \\__attribute__((reqd_work_group_size(VC_WG, 1, 1)))
    \\kernel void vcheck(global io_t *out_b, global const io_t *dst_b, global const io_t *src_b,
    \\                   global const int *rowidx0, global const int *rowidx1, global const int *rowidx2,
    \\                   global const int *dmap_b, global const io_t *scp_b,
    \\                   global const int *geom, const int field,
    \\                   const int vmode, const int use_scp, const int hp,
    \\                   const float rcp0, const float rcp1, const float rcp2, const float vthresh2,
    \\                   const int nplanes) {
    \\    const int p = get_group_id(0);
    \\    if (p >= nplanes) return;
    \\    const int w = geom[p*6+0], stride = geom[p*6+1], dst_h = geom[p*6+2];
    \\    global io_t *out = out_b + geom[p*6+3];
    \\    global const io_t *dst = dst_b + geom[p*6+3];
    \\    global const io_t *src = src_b + geom[p*6+4];
    \\    global const int *dmap = dmap_b + geom[p*6+5];
    \\    global const io_t *scp = scp_b + geom[p*6+3];
    \\    global const int *rowidx = (p == 0) ? rowidx0 : ((p == 1) ? rowidx1 : rowidx2);
    \\    const int n_interp = (dst_h - field + 1) / 2;
    \\    const int lid = get_local_id(0);
    \\    const int lsz = get_local_size(0);
    \\    for (int off = 1; off + 1 < n_interp; ++off) {
    \\        const int y = field + 2*off;
    \\        if (y >= 2 && y + 2 < dst_h) {
    \\            global const io_t *drow = dst + y*stride;
    \\            global const io_t *d1p = dst + (y-1)*stride;
    \\            global const io_t *d2p = out + (y-2)*stride;
    \\            global const io_t *d1n = dst + (y+1)*stride;
    \\            global const io_t *d2n = dst + (y+2)*stride;
    \\            global const io_t *d3p = src + rowidx[off*4+0]*stride;
    \\            global const io_t *d3n = src + rowidx[off*4+3]*stride;
    \\            for (int x = lid; x < w; x += lsz) {
    \\                const int dirc = dmap[(long)off*stride + x];
    \\                float cint = use_scp ? LOAD_IO(scp, y*stride + x) : (0.5625f*(LOAD_IO(d1p,x)+LOAD_IO(d1n,x)) - 0.0625f*(LOAD_IO(d3p,x)+LOAD_IO(d3n,x)));
    \\                int dirt = dmap[(long)(off-1)*stride + x];
    \\                int dirb = dmap[(long)(off+1)*stride + x];
    \\                int maxoff = hp ? ((dirc & 1) == 0 ? abs(dirc>>1) : max(abs(dirc>>1), abs((dirc+1)>>1))) : abs(dirc);
    \\                if (dirc == 0 || max(dirc*dirt, dirc*dirb) < 0 || (dirt==dirb && dirt==0)
    \\                    || x + maxoff >= w || x - maxoff < 0) {
    \\                    STORE_IO(out, y*stride + x, cint);
    \\                    continue;
    \\                }
    \\                float it, ib, vt, vb;
    \\                int dabs;
    \\                if (hp && (dirc & 1) != 0) {
    \\                    int d20 = dirc>>1, d21 = (dirc+1)>>1;
    \\                    int xp0 = x+d20, xp1 = x+d21, xm0 = x-d20, xm1 = x-d21;
    \\                    float s2psum = LOAD_IO(d2p,xp0)+LOAD_IO(d2p,xp1), s1psum = LOAD_IO(d1p,xp0)+LOAD_IO(d1p,xp1);
    \\                    float pa0 = LOAD_IO(drow,xp0)+LOAD_IO(drow,xp1), ps0 = LOAD_IO(drow,xm0)+LOAD_IO(drow,xm1);
    \\                    float s1nsum = LOAD_IO(d1n,xm0)+LOAD_IO(d1n,xm1), s2nsum = LOAD_IO(d2n,xm0)+LOAD_IO(d2n,xm1);
    \\                    it = (s2psum + ps0)*0.25f;
    \\                    vt = (fabs(s2psum-s1psum) + fabs(pa0-s1psum))*0.5f;
    \\                    ib = (pa0 + s2nsum)*0.25f;
    \\                    vb = (fabs(s2nsum-s1nsum) + fabs(ps0-s1nsum))*0.5f;
    \\                    dabs = abs(dirc) >> 1;
    \\                } else {
    \\                    int offh = hp ? (dirc>>1) : dirc;
    \\                    int xpd = x+offh, xmd = x-offh;
    \\                    it = (LOAD_IO(d2p,xpd) + LOAD_IO(drow,xmd)) * 0.5f;
    \\                    ib = (LOAD_IO(drow,xpd) + LOAD_IO(d2n,xmd)) * 0.5f;
    \\                    vt = fabs(LOAD_IO(d2p,xpd)-LOAD_IO(d1p,xpd)) + fabs(LOAD_IO(drow,xpd)-LOAD_IO(d1p,xpd));
    \\                    vb = fabs(LOAD_IO(d2n,xmd)-LOAD_IO(d1n,xmd)) + fabs(LOAD_IO(drow,xmd)-LOAD_IO(d1n,xmd));
    \\                    dabs = hp ? (abs(dirc)>>1) : abs(dirc);
    \\                }
    \\                float vc = fabs(LOAD_IO(drow,x)-LOAD_IO(d1p,x)) + fabs(LOAD_IO(drow,x)-LOAD_IO(d1n,x));
    \\                float d0 = fabs(it-LOAD_IO(d1p,x)), d1 = fabs(ib-LOAD_IO(d1n,x)), d2 = fabs(vt-vc), d3 = fabs(vb-vc);
    \\                float mdiff0 = (vmode==1)?fmin(d0,d1):(vmode==2)?(d0+d1)*0.5f:fmax(d0,d1);
    \\                float mdiff1 = (vmode==1)?fmin(d2,d3):(vmode==2)?(d2+d3)*0.5f:fmax(d2,d3);
    \\                float a0 = mdiff0*rcp0, a1 = mdiff1*rcp1;
    \\                float a2 = fmax((vthresh2 - (float)dabs)*rcp2, 0.0f);
    \\                float a = fmin(fmax(a0, fmax(a1,a2)), 1.0f);
    \\                STORE_IO(out, y*stride + x, (1.0f-a)*LOAD_IO(drow,x) + a*cint);
    \\            }
    \\        }
    \\        barrier(CLK_GLOBAL_MEM_FENCE);
    \\    }
    \\}
    \\
    \\// out[c*out_stride + r] = in[r*in_stride + c], for r<in_h, c<in_w. global=(in_w,in_h).
    \\// Local-tiled (16x16, +1 col to break bank conflicts): reads AND writes are coalesced
    \\// (the naive version wrote out[c*out_stride + r] with c varying per lane — one 32B
    \\// transaction per lane). Pure permutation of the same values — bit-exact trivially.
    \\// raw_t: transpose sits on the RAW ingress/egress/sclip paths (EEDI3H), so it moves io
    \\// elements as bit patterns (avoids declaring `half` objects, which core CL forbids).
    \\kernel __attribute__((reqd_work_group_size(16, 16, 1)))
    \\void transpose(global raw_t *out, global const raw_t *in,
    \\               const int in_w, const int in_h, const int in_stride, const int out_stride,
    \\               const int soff, const int doff) {
    \\    local raw_t tile[16][17];
    \\    const int lx = get_local_id(0), ly = get_local_id(1);
    \\    const int c0 = get_group_id(0) * 16, r0 = get_group_id(1) * 16;
    \\    const int c = c0 + lx, r = r0 + ly;
    \\    if (c < in_w && r < in_h) tile[ly][lx] = in[soff + r*in_stride + c];
    \\    barrier(CLK_LOCAL_MEM_FENCE);
    \\    // write transposed: lane lx walks r (coalesced along out rows)
    \\    const int oc = c0 + ly, orr = r0 + lx;
    \\    if (oc < in_w && orr < in_h) out[doff + oc*out_stride + orr] = tile[lx][ly];
    \\}
    \\
;

const max_cfg = 2;
const Config = struct {
    w: u32,
    src_h: u32,
    dst_h: u32,
    stride: u32,
    pstride: u32,
    n_interp_max: u32,
    in_w: u32 = 0,
    in_h: u32 = 0,
    out_w: u32 = 0,
    in_stride: u32 = 0,
    out_stride: u32 = 0,
};

const Data = struct {
    node: ?*vs.Node = null,
    sclip: ?*vs.Node = null,
    vi: vs.VideoInfo = undefined,

    horizontal: bool = false,
    hp: bool = false,
    bits: i32 = 32,
    half: bool = false,
    bytes: u32 = 4,
    field: u8 = 0,
    dh: bool = false,
    mdis: u8 = 0,
    nrad: u8 = 0,
    alpha: f32 = 0,
    beta: f32 = 0,
    gamma: f32 = 0,
    one_minus_ab: f32 = 0,
    vcheck: u8 = 0,
    rcp0: f32 = 0,
    rcp1: f32 = 0,
    rcp2: f32 = 0,
    vthresh2: f32 = 0,

    configs: [max_cfg]Config = undefined,
    n_cfg: usize = 0,
    plane_cfg: [3]usize = .{ 0, 0, 0 },
    off_io: [3]u32 = .{ 0, 0, 0 },
    off_src: [3]u32 = .{ 0, 0, 0 },
    off_dmap: [3]u32 = .{ 0, 0, 0 },
    sum_io: usize = 0,
    sum_src: usize = 0,
    sum_dmap: usize = 0,
    vc_geom: [18]i32 = [_]i32{0} ** 18,
    rowidx_host: [max_cfg][2][]i32 = undefined,
    dsty_host: [max_cfg][2][]i32 = undefined,

    tpitch: u32 = 0,
    lws: usize = 0,
    max_wg: usize = 256,
    is_nv: bool = false,
    local_mem: usize = 32 * 1024,
    bx: u32 = 48,
    run: u32 = 8,
    run_hp: u32 = 4,
    pad: u32 = 0,
    platform: cl.Platform = undefined,
    device: cl.Device = undefined,
    context: cl.Context = undefined,
    pool: clpool.Pool(Stream, Data) = .{},
};

const Stream = struct {
    program: cl.Program,
    queue: cl.CommandQueue,
    d_src: cl.Buffer(u8),
    d_srcpad: cl.Buffer(f32),
    d_hpsrcpad: cl.Buffer(f32),
    d_dst: cl.Buffer(u8),
    d_rowidx: [max_cfg][2]cl.Buffer(i32),
    d_dsty: [max_cfg][2]cl.Buffer(i32),
    d_vcgeom: cl.Buffer(i32),
    d_pbackt: cl.Buffer(i8),
    d_dmap: cl.Buffer(i32),
    d_dst2: cl.Buffer(u8),
    d_inframe: cl.Buffer(u8),
    d_outframe: cl.Buffer(u8),
    d_scp: cl.Buffer(u8),
    d_scpframe: cl.Buffer(u8),
    k_copy: [max_cfg]cl.Kernel,
    k_pad: [max_cfg]cl.Kernel,
    k_pad_hp: [max_cfg]cl.Kernel,
    k_interp: [max_cfg]cl.Kernel,
    k_interp_hp: [max_cfg]cl.Kernel,
    k_vcheck: [max_cfg]cl.Kernel,
    k_transpose: cl.Kernel,
    n_kern_cfg: usize,
    n_tbl_sets: usize,

    pub fn init(self: *Stream, d: *Data) !void {
        self.n_kern_cfg = 0;
        self.n_tbl_sets = 0;
        self.program = try cl.createProgramWithSource(d.context, kernel_src);
        errdefer self.program.release();
        const bx: u32 = d.bx;
        var run_buf: [64]u8 = undefined;
        const run_opt: []const u8 = if (d.run != 8 or d.run_hp != 4)
            std.fmt.bufPrint(&run_buf, " -DRUN={d} -DRUN_HP={d}", .{ d.run, d.run_hp }) catch unreachable
        else
            "";
        const nv_opt: []const u8 = if (d.is_nv) " -cl-nv-maxrregcount=96" else "";
        const vc_wg: usize = @min(@as(usize, 1024), d.max_wg);
        const build_opts = try std.fmt.allocPrintSentinel(allocator, "-cl-std=CL1.2 -DCN={d} -DMDIS={d} -DBX={d} -DBITS={d} -DHALF={d} -DVC_WG={d}{s}{s}", .{ d.nrad, d.mdis, bx, d.bits, @intFromBool(d.half), vc_wg, run_opt, nv_opt }, 0);
        defer allocator.free(build_opts);
        self.program.build(&.{d.device}, build_opts) catch |err| {
            if (err == error.BuildProgramFailure) {
                const log = try self.program.getBuildLog(allocator, d.device);
                defer allocator.free(log);
                std.log.err("EEDI3 OpenCL build failed: {s}", .{log});
            }
            return err;
        };
        self.queue = try cl.createCommandQueue(d.context, d.device, .{});
        errdefer self.queue.release();
        errdefer _ = cl.c.clFinish(self.queue.handle);
        const c0 = &d.configs[0];
        const bytes: usize = d.bytes;
        const use_reg = d.vcheck > 0;
        const src_elems: usize = if (use_reg) d.sum_src else @as(usize, c0.stride) * c0.src_h;
        const io_elems: usize = if (use_reg) d.sum_io else @as(usize, c0.stride) * c0.dst_h;
        const dmap_elems: usize = if (use_reg) d.sum_dmap else @as(usize, c0.n_interp_max) * c0.stride;
        self.d_src = try cl.createBuffer(u8, d.context, .{ .read_only = true }, src_elems * bytes);
        errdefer self.d_src.release();
        self.d_srcpad = try cl.createBuffer(f32, d.context, .{ .read_write = true }, c0.pstride * c0.src_h);
        errdefer self.d_srcpad.release();
        self.d_hpsrcpad = try cl.createBuffer(f32, d.context, .{ .read_write = true }, if (d.hp) c0.pstride * c0.src_h else 1);
        errdefer self.d_hpsrcpad.release();
        self.d_dst = try cl.createBuffer(u8, d.context, .{ .read_write = true }, io_elems * bytes);
        errdefer self.d_dst.release();
        self.d_pbackt = try cl.createBuffer(i8, d.context, .{ .read_write = true }, @as(usize, c0.n_interp_max) * c0.w * d.tpitch);
        errdefer self.d_pbackt.release();
        self.d_dmap = try cl.createBuffer(i32, d.context, .{ .read_write = true }, dmap_elems);
        errdefer self.d_dmap.release();
        self.d_dst2 = try cl.createBuffer(u8, d.context, .{ .read_write = true }, if (d.vcheck > 0) io_elems * bytes else 1);
        errdefer self.d_dst2.release();
        self.d_inframe = try cl.createBuffer(u8, d.context, .{ .read_write = true }, if (d.horizontal) @as(usize, c0.in_stride) * c0.in_h * bytes else 1);
        errdefer self.d_inframe.release();
        self.d_outframe = try cl.createBuffer(u8, d.context, .{ .read_write = true }, if (d.horizontal) @as(usize, c0.out_stride) * c0.in_h * bytes else 1);
        errdefer self.d_outframe.release();
        const have_scp = d.sclip != null and d.vcheck > 0;
        self.d_scp = try cl.createBuffer(u8, d.context, .{ .read_write = true }, if (have_scp) io_elems * bytes else 1);
        errdefer self.d_scp.release();
        self.d_scpframe = try cl.createBuffer(u8, d.context, .{ .read_write = true }, if (have_scp and d.horizontal) @as(usize, c0.out_stride) * c0.in_h * bytes else 1);
        errdefer self.d_scpframe.release();
        self.d_vcgeom = try cl.createBuffer(i32, d.context, .{ .read_only = true }, if (d.vcheck > 0) d.vc_geom.len else 1);
        errdefer self.d_vcgeom.release();
        if (d.vcheck > 0) try writeBuf(i32, self, self.d_vcgeom, &d.vc_geom);

        errdefer self.releaseKernels();
        errdefer self.releaseTables();
        for (0..d.n_cfg) |ci| {
            for (0..2) |fb| {
                self.d_rowidx[ci][fb] = try cl.createBuffer(i32, d.context, .{ .read_only = true }, d.rowidx_host[ci][fb].len);
                errdefer self.d_rowidx[ci][fb].release();
                try writeBuf(i32, self, self.d_rowidx[ci][fb], d.rowidx_host[ci][fb]);
                self.d_dsty[ci][fb] = try cl.createBuffer(i32, d.context, .{ .read_only = true }, d.dsty_host[ci][fb].len);
                errdefer self.d_dsty[ci][fb].release();
                try writeBuf(i32, self, self.d_dsty[ci][fb], d.dsty_host[ci][fb]);
                self.n_tbl_sets = ci * 2 + fb + 1;
            }
            self.k_copy[ci] = try cl.createKernel(self.program, "copy_kept");
            errdefer self.k_copy[ci].release();
            self.k_pad[ci] = try cl.createKernel(self.program, "pad_src");
            errdefer self.k_pad[ci].release();
            self.k_pad_hp[ci] = try cl.createKernel(self.program, "pad_hp");
            errdefer self.k_pad_hp[ci].release();
            self.k_interp[ci] = try cl.createKernel(self.program, "interp");
            errdefer self.k_interp[ci].release();
            self.k_interp_hp[ci] = try cl.createKernel(self.program, "interp_hp");
            errdefer self.k_interp_hp[ci].release();
            self.k_vcheck[ci] = try cl.createKernel(self.program, "vcheck");
            errdefer self.k_vcheck[ci].release();
            try self.setStaticArgs(d, ci);
            self.n_kern_cfg = ci + 1;
        }
        self.k_transpose = try cl.createKernel(self.program, "transpose");
    }

    fn releaseKernels(self: *Stream) void {
        var i: usize = self.n_kern_cfg;
        while (i > 0) {
            i -= 1;
            self.k_vcheck[i].release();
            self.k_interp_hp[i].release();
            self.k_interp[i].release();
            self.k_pad_hp[i].release();
            self.k_pad[i].release();
            self.k_copy[i].release();
        }
    }

    fn releaseTables(self: *Stream) void {
        var i: usize = self.n_tbl_sets;
        while (i > 0) {
            i -= 1;
            self.d_dsty[i / 2][i % 2].release();
            self.d_rowidx[i / 2][i % 2].release();
        }
    }

    fn setStaticArgs(self: *Stream, d: *Data, ci: usize) !void {
        const cfg = &d.configs[ci];
        const w: c_int = @intCast(cfg.w);
        const stride: c_int = @intCast(cfg.stride);
        const src_h_i: c_int = @intCast(cfg.src_h);

        try self.k_pad[ci].setArg(@TypeOf(self.d_srcpad), 0, self.d_srcpad);
        try self.k_pad[ci].setArg(@TypeOf(self.d_src), 1, self.d_src);
        try self.k_pad[ci].setArg(c_int, 2, w);
        try self.k_pad[ci].setArg(c_int, 3, stride);
        try self.k_pad[ci].setArg(c_int, 4, @intCast(cfg.pstride));
        try self.k_pad[ci].setArg(c_int, 5, @intCast(d.pad));
        try self.k_pad[ci].setArg(c_int, 6, src_h_i);
        try self.k_pad[ci].setArg(c_int, 7, 0);

        if (d.hp) {
            try self.k_pad_hp[ci].setArg(@TypeOf(self.d_hpsrcpad), 0, self.d_hpsrcpad);
            try self.k_pad_hp[ci].setArg(@TypeOf(self.d_srcpad), 1, self.d_srcpad);
            try self.k_pad_hp[ci].setArg(c_int, 2, @intCast(cfg.pstride));
            try self.k_pad_hp[ci].setArg(c_int, 3, src_h_i);
            try self.k_pad_hp[ci].setArg(c_int, 4, @intCast(d.pad));
            try self.k_pad_hp[ci].setArg(c_int, 5, w);
        }

        try self.k_copy[ci].setArg(@TypeOf(self.d_dst), 0, self.d_dst);
        try self.k_copy[ci].setArg(@TypeOf(self.d_src), 1, self.d_src);
        try self.k_copy[ci].setArg(c_int, 2, w);
        try self.k_copy[ci].setArg(c_int, 3, stride);
        try self.k_copy[ci].setArg(c_int, 4, @intFromBool(d.dh));
        try self.k_copy[ci].setArg(c_int, 6, src_h_i);
        try self.k_copy[ci].setArg(c_int, 7, @intCast(cfg.dst_h));
        const dual: c_int = @intFromBool(d.vcheck > 0);
        try self.k_copy[ci].setArg(@TypeOf(self.d_dst2), 8, self.d_dst2);
        try self.k_copy[ci].setArg(c_int, 9, dual);
        try self.k_copy[ci].setArg(c_int, 10, 0);
        try self.k_copy[ci].setArg(c_int, 11, 0);

        const ik = if (d.hp) self.k_interp_hp[ci] else self.k_interp[ci];
        try ik.setArg(@TypeOf(self.d_dst), 0, self.d_dst);
        try ik.setArg(@TypeOf(self.d_srcpad), 1, self.d_srcpad);
        try ik.setArg(@TypeOf(self.d_rowidx[ci][0]), 2, self.d_rowidx[ci][0]);
        try ik.setArg(@TypeOf(self.d_dsty[ci][0]), 3, self.d_dsty[ci][0]);
        try ik.setArg(@TypeOf(self.d_pbackt), 4, self.d_pbackt);
        try ik.setArg(@TypeOf(self.d_dmap), 5, self.d_dmap);
        try ik.setArg(c_int, 6, w);
        try ik.setArg(c_int, 7, stride);
        try ik.setArg(c_int, 8, @intCast(cfg.pstride));
        try ik.setArg(c_int, 9, @intCast(d.pad));
        try ik.setArg(c_int, 10, @intCast(d.mdis));
        try ik.setArg(c_int, 11, @intCast(d.nrad));
        try ik.setArg(f32, 12, d.alpha);
        try ik.setArg(f32, 13, d.beta);
        try ik.setArg(f32, 14, d.gamma);
        try ik.setArg(f32, 15, d.one_minus_ab);
        if (d.hp) {
            try ik.setArg(@TypeOf(self.d_hpsrcpad), 16, self.d_hpsrcpad);
            try ik.setArg(@TypeOf(self.d_dst2), 17, self.d_dst2);
            try ik.setArg(c_int, 18, dual);
            try ik.setArg(c_int, 19, 0);
            try ik.setArg(c_int, 20, 0);
        } else {
            try ik.setArg(@TypeOf(self.d_dst2), 16, self.d_dst2);
            try ik.setArg(c_int, 17, dual);
            try ik.setArg(c_int, 18, 0);
            try ik.setArg(c_int, 19, 0);
        }

        if (d.vcheck > 0 and ci == 0) {
            const use_scp: c_int = if (d.sclip != null) 1 else 0;
            const kv = self.k_vcheck[0];
            try kv.setArg(@TypeOf(self.d_dst2), 0, self.d_dst2);
            try kv.setArg(@TypeOf(self.d_dst), 1, self.d_dst);
            try kv.setArg(@TypeOf(self.d_src), 2, self.d_src);
            try kv.setArg(@TypeOf(self.d_rowidx[0][0]), 3, self.d_rowidx[0][0]);
            try kv.setArg(@TypeOf(self.d_rowidx[0][0]), 4, self.d_rowidx[0][0]);
            try kv.setArg(@TypeOf(self.d_rowidx[0][0]), 5, self.d_rowidx[0][0]);
            try kv.setArg(@TypeOf(self.d_dmap), 6, self.d_dmap);
            try kv.setArg(@TypeOf(self.d_scp), 7, self.d_scp);
            try kv.setArg(@TypeOf(self.d_vcgeom), 8, self.d_vcgeom);
            try kv.setArg(c_int, 10, @intCast(d.vcheck));
            try kv.setArg(c_int, 11, use_scp);
            try kv.setArg(c_int, 12, @intFromBool(d.hp));
            try kv.setArg(f32, 13, d.rcp0);
            try kv.setArg(f32, 14, d.rcp1);
            try kv.setArg(f32, 15, d.rcp2);
            try kv.setArg(f32, 16, d.vthresh2);
            try kv.setArg(c_int, 17, @intCast(d.vi.format.numPlanes));
        }
    }

    pub fn deinit(self: *Stream) void {
        _ = cl.c.clFinish(self.queue.handle);
        self.k_transpose.release();
        self.releaseKernels();
        self.releaseTables();
        self.d_vcgeom.release();
        self.d_scpframe.release();
        self.d_scp.release();
        self.d_outframe.release();
        self.d_inframe.release();
        self.d_dst2.release();
        self.d_dmap.release();
        self.d_pbackt.release();
        self.d_dst.release();
        self.d_hpsrcpad.release();
        self.d_srcpad.release();
        self.d_src.release();
        self.queue.release();
        self.program.release();
    }
};

fn reflectRow(y: i32, h: i32) u32 {
    if (h == 1) return 0;
    var r = y;
    while (r < 0 or r >= h) {
        if (r < 0) r = -r;
        if (r >= h) r = 2 * (h - 1) - r;
    }
    return @intCast(r);
}

fn stencilRow(yy: i32, sh: i32, dh: bool) i32 {
    return @intCast(if (dh) reflectRow(yy, 2 * sh) / 2 else reflectRow(yy, sh));
}

fn initOpenCL(d: *Data, device_id: usize) !void {
    try vszipcl.initContext(d, device_id);
    var dev_max_wg: usize = 0;
    if (cl.c.clGetDeviceInfo(d.device.id, cl.c.CL_DEVICE_MAX_WORK_GROUP_SIZE, @sizeOf(usize), &dev_max_wg, null) == cl.c.CL_SUCCESS and dev_max_wg > 0) {
        d.max_wg = dev_max_wg;
    }
    var vend: [256]u8 = undefined;
    var vlen: usize = 0;
    if (cl.c.clGetDeviceInfo(d.device.id, cl.c.CL_DEVICE_VENDOR, vend.len, &vend, &vlen) == cl.c.CL_SUCCESS and vlen > 0) {
        d.is_nv = std.mem.indexOf(u8, vend[0..vlen], "NVIDIA") != null;
    }
    d.local_mem = vszipcl.deviceLocalMemSize(d.device);
}

const ndr = vszipcl.ndr;
fn writeBuf(comptime T: type, s: *Stream, buf: cl.Buffer(T), src: []const T) !void {
    return vszipcl.enqWrite(s.queue, buf.handle, 0, std.mem.sliceAsBytes(src));
}
fn readBuf(comptime T: type, s: *Stream, buf: cl.Buffer(T), dst: []T) !void {
    return vszipcl.enqRead(s.queue, buf.handle, 0, std.mem.sliceAsBytes(dst));
}

fn processPlane(d: *Data, s: *Stream, ci: usize, plane: usize, srcp: []const u8, scpp: ?[]const u8, field: u8) !void {
    const cfg = &d.configs[ci];
    const n_interp: u32 = (cfg.dst_h - field + 1) / 2;
    errdefer _ = cl.c.clFinish(s.queue.handle);

    const use_reg = d.vcheck > 0;
    const off_io: c_int = if (use_reg) @intCast(d.off_io[plane]) else 0;
    const off_src: c_int = if (use_reg) @intCast(d.off_src[plane]) else 0;
    const off_dm: c_int = if (use_reg) @intCast(d.off_dmap[plane]) else 0;

    const fb: usize = field & 1;
    const ik = if (d.hp) s.k_interp_hp[ci] else s.k_interp[ci];
    try ik.setArg(@TypeOf(s.d_rowidx[ci][fb]), 2, s.d_rowidx[ci][fb]);
    try ik.setArg(@TypeOf(s.d_dsty[ci][fb]), 3, s.d_dsty[ci][fb]);
    if (d.hp) {
        try ik.setArg(c_int, 19, off_io);
        try ik.setArg(c_int, 20, off_dm);
    } else {
        try ik.setArg(c_int, 18, off_io);
        try ik.setArg(c_int, 19, off_dm);
    }
    try s.k_pad[ci].setArg(c_int, 7, off_src);
    try s.k_copy[ci].setArg(c_int, 10, off_src);
    try s.k_copy[ci].setArg(c_int, 11, off_io);

    if (d.horizontal) {
        try writeBuf(u8, s, s.d_inframe, srcp);
        try runTranspose(s, s.d_src, s.d_inframe, cfg.in_w, cfg.in_h, cfg.in_stride, cfg.stride, 0, off_src);
    } else {
        try vszipcl.enqWrite(s.queue, s.d_src.handle, @as(usize, @intCast(off_src)) * d.bytes, srcp);
    }

    const pad_gws: [2]usize = .{ vsh.ceilN(@as(usize, d.pad) * 2 + cfg.w, 16), vsh.ceilN(@as(usize, cfg.src_h), 8) };
    const pad_lws: [2]usize = .{ 16, 8 };
    try ndr(s, s.k_pad[ci], &pad_gws, &pad_lws);
    if (d.hp) try ndr(s, s.k_pad_hp[ci], &pad_gws, &pad_lws);

    try s.k_copy[ci].setArg(c_int, 5, field);
    const copy_gws: [2]usize = .{ vsh.ceilN(@as(usize, cfg.w), 16), vsh.ceilN(@as(usize, cfg.dst_h), 8) };
    const copy_lws: [2]usize = .{ 16, 8 };
    try ndr(s, s.k_copy[ci], &copy_gws, &copy_lws);

    const interp_gws: [2]usize = .{ d.lws, n_interp };
    const interp_lws: [2]usize = .{ d.lws, 1 };
    try ndr(s, ik, &interp_gws, &interp_lws);

    if (d.vcheck > 0) {
        if (scpp) |sp| {
            if (d.horizontal) {
                try writeBuf(u8, s, s.d_scpframe, sp);
                try runTranspose(s, s.d_scp, s.d_scpframe, cfg.out_w, cfg.in_h, cfg.out_stride, cfg.stride, 0, off_io);
            } else {
                try vszipcl.enqWrite(s.queue, s.d_scp.handle, @as(usize, @intCast(off_io)) * d.bytes, sp);
            }
        }
    }
}

fn vcheckFused(d: *Data, s: *Stream, field: u8) !void {
    errdefer _ = cl.c.clFinish(s.queue.handle);
    const fb: usize = field & 1;
    const np: usize = @intCast(d.vi.format.numPlanes);
    const kv = s.k_vcheck[0];
    for (0..3) |p| {
        const ci = d.plane_cfg[if (p < np) p else 0];
        try kv.setArg(@TypeOf(s.d_rowidx[ci][fb]), @intCast(3 + p), s.d_rowidx[ci][fb]);
    }
    try kv.setArg(c_int, 9, field);
    const vc_wg: usize = @min(@as(usize, 1024), d.max_wg);
    const vc_gws: [1]usize = .{vc_wg * np};
    const vc_lws: [1]usize = .{vc_wg};
    try ndr(s, kv, &vc_gws, &vc_lws);
}

fn downloadPlane(d: *Data, s: *Stream, ci: usize, plane: usize, dstp: []u8, sync: bool) !void {
    const cfg = &d.configs[ci];
    errdefer _ = cl.c.clFinish(s.queue.handle);
    const use_reg = d.vcheck > 0;
    const off_io: usize = if (use_reg) d.off_io[plane] else 0;
    const result = if (d.vcheck > 0) s.d_dst2 else s.d_dst;
    if (d.horizontal) {
        try runTranspose(s, s.d_outframe, result, cfg.w, cfg.dst_h, cfg.stride, cfg.out_stride, @intCast(off_io), 0);
        try readBuf(u8, s, s.d_outframe, dstp);
    } else {
        try vszipcl.enqRead(s.queue, result.handle, off_io * d.bytes, dstp);
    }
    if (sync) {
        if (cl.c.clFinish(s.queue.handle) != cl.c.CL_SUCCESS) return error.Finish;
    }
}

fn runTranspose(s: *Stream, dst: cl.Buffer(u8), src: cl.Buffer(u8), in_w: u32, in_h: u32, in_stride: u32, out_stride: u32, soff: c_int, doff: c_int) !void {
    try s.k_transpose.setArg(@TypeOf(dst), 0, dst);
    try s.k_transpose.setArg(@TypeOf(src), 1, src);
    try s.k_transpose.setArg(c_int, 2, @intCast(in_w));
    try s.k_transpose.setArg(c_int, 3, @intCast(in_h));
    try s.k_transpose.setArg(c_int, 4, @intCast(in_stride));
    try s.k_transpose.setArg(c_int, 5, @intCast(out_stride));
    try s.k_transpose.setArg(c_int, 6, soff);
    try s.k_transpose.setArg(c_int, 7, doff);
    const gws: [2]usize = .{ vsh.ceilN(@as(usize, in_w), 16), vsh.ceilN(@as(usize, in_h), 16) };
    const lws: [2]usize = .{ 16, 16 };
    try ndr(s, s.k_transpose, &gws, &lws);
}

fn getFrame(n: c_int, ar: vs.ActivationReason, instance_data: ?*anyopaque, _: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core_ptr, frame_ctx);
    const src_n: c_int = if (d.field > 1) @divTrunc(n, 2) else n;

    if (ar == .Initial) {
        zapi.requestFrameFilter(src_n, d.node);
        if (d.vcheck > 0 and d.sclip != null) zapi.requestFrameFilter(n, d.sclip);
    } else if (ar == .AllFramesReady) {
        const src = zapi.initZFrame(d.node, src_n);
        defer src.deinit();
        const scp = if (d.vcheck > 0 and d.sclip != null) zapi.initZFrame(d.sclip, n) else null;
        defer if (scp) |sc| sc.deinit();
        const dst = if (d.horizontal) src.newVideoFrame3(.{ .width = d.vi.width }) else src.newVideoFrame3(.{ .height = d.vi.height });
        const dst_props = dst.getPropertiesRW();

        var field: u8 = d.field & 1;
        switch (dst_props.getFieldBased() orelse .PROGRESSIVE) {
            .BOTTOM => field = 0,
            .TOP => field = 1,
            else => {},
        }
        if (d.field > 1) field = @as(u8, @intCast(n & 1)) ^ field;

        const s = d.pool.acquire();
        defer d.pool.release(s);

        const nplanes: u32 = @intCast(d.vi.format.numPlanes);
        var failed = false;
        var plane: u32 = 0;
        while (plane < nplanes and !failed) : (plane += 1) {
            const ci = d.plane_cfg[plane];
            const srcp = src.getReadSlice(plane);
            const scpp: ?[]const u8 = if (scp) |sc| sc.getReadSlice(plane) else null;
            std.debug.assert(src.getStride2(u8, plane) ==
                (if (d.horizontal) d.configs[ci].in_stride else d.configs[ci].stride) * d.bytes);
            processPlane(d, s, ci, plane, srcp, scpp, field) catch |err| {
                std.log.err("EEDI3 process failed: {}", .{err});
                failed = true;
                break;
            };
            if (d.vcheck == 0) {
                downloadPlane(d, s, ci, plane, dst.getWriteSlice(plane), plane == nplanes - 1) catch |err| {
                    std.log.err("EEDI3 process failed: {}", .{err});
                    failed = true;
                };
            }
        }
        if (!failed and d.vcheck > 0) {
            blk: {
                vcheckFused(d, s, field) catch |err| {
                    std.log.err("EEDI3 process failed: {}", .{err});
                    failed = true;
                    break :blk;
                };
                plane = 0;
                while (plane < nplanes) : (plane += 1) {
                    downloadPlane(d, s, d.plane_cfg[plane], plane, dst.getWriteSlice(plane), plane == nplanes - 1) catch |err| {
                        std.log.err("EEDI3 process failed: {}", .{err});
                        failed = true;
                        break;
                    };
                }
            }
        }
        if (failed) {
            zapi.setFilterError("EEDI3: process failed.");
            dst.deinit();
            return null;
        }

        dst_props.setFieldBased(.PROGRESSIVE);
        if (d.field > 1) {
            var dn = dst_props.getDurationNum();
            var dd = dst_props.getDurationDen();
            if (dn != null and dd != null) {
                vsh.muldivRational(&dn.?, &dd.?, 1, 2);
                dst_props.setDurationNum(dn.?);
                dst_props.setDurationDen(dd.?);
            }
        }
        return dst.frame;
    }
    return null;
}

fn free(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    d.pool.deinit();
    d.context.release();
    freeStencilTables(d);
    vsapi.?.freeNode.?(d.node);
    vsapi.?.freeNode.?(d.sclip);
    allocator.destroy(d);
}

fn freeStencilTables(d: *Data) void {
    for (0..d.n_cfg) |ci| {
        for (0..2) |f| {
            allocator.free(d.rowidx_host[ci][f]);
            allocator.free(d.dsty_host[ci][f]);
        }
    }
}

pub fn createEEDI3(in: ?*const vs.Map, out: ?*vs.Map, ud: ?*anyopaque, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    createImpl(in, out, ud, core_ptr, vsapi, false);
}
pub fn createEEDI3H(in: ?*const vs.Map, out: ?*vs.Map, ud: ?*anyopaque, core_ptr: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    createImpl(in, out, ud, core_ptr, vsapi, true);
}

fn createImpl(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core_ptr: ?*vs.Core, vsapi: ?*const vs.API, horizontal: bool) void {
    var d: Data = .{};
    d.horizontal = horizontal;
    const zapi = ZAPI.init(vsapi, core_ptr, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, const vi_in = map_in.getNodeVi("clip").?;
    d.vi = vi_in.*;

    const vcheck = map_in.getValue(i32, "vcheck") orelse 2;
    d.sclip = if (vcheck > 0) map_in.getNode("sclip") else null;

    var keep = false;
    defer if (!keep) {
        zapi.freeNode(d.node);
        zapi.freeNode(d.sclip);
    };

    const cf = d.vi.format.colorFamily;
    const fmt = d.vi.format;
    const io_bits: i32 = fmt.bitsPerSample;
    const depth_ok = (fmt.sampleType == .Integer and (io_bits == 8 or io_bits == 16)) or
        (fmt.sampleType == .Float and (io_bits == 16 or io_bits == 32));
    if (!depth_ok or d.vi.width <= 0 or d.vi.height <= 0 or
        (cf != .Gray and cf != .YUV and cf != .RGB))
    {
        map_out.setError("EEDI3: input bitdepth must be 8/16 (integer), 16 (half) or 32 (float), Gray/YUV/RGB.");
        return;
    }
    d.bits = io_bits;
    d.half = fmt.sampleType == .Float and io_bits == 16;
    d.bytes = @intCast(fmt.bytesPerSample);

    const field = map_in.getValue(i32, "field") orelse 0;
    const mdis = map_in.getValue(i32, "mdis") orelse 20;
    const nrad = map_in.getValue(i32, "nrad") orelse 2;
    d.alpha = map_in.getValue(f32, "alpha") orelse 0.2;
    d.beta = map_in.getValue(f32, "beta") orelse 0.25;
    d.gamma = map_in.getValue(f32, "gamma") orelse 20.0;
    d.dh = map_in.getBool("dh") orelse false;
    d.hp = map_in.getBool("hp") orelse false;
    const ns_req = map_in.getValue(i32, "num_streams");
    const interp_axis: i32 = if (horizontal) d.vi.width else d.vi.height;

    if (field < 0 or field > 3) return map_out.setError("EEDI3: field must be 0..3.");
    if (d.dh and field > 1) return map_out.setError("EEDI3: field must be 0 or 1 when dh=True.");
    if (!d.dh and (interp_axis & 1) != 0) return map_out.setError("EEDI3: interpolated axis must be mod 2 when dh=False.");
    if (d.alpha < 0 or d.alpha > 1) return map_out.setError("EEDI3: alpha 0..1.");
    if (d.beta < 0 or d.beta > 1) return map_out.setError("EEDI3: beta 0..1.");
    if (d.alpha + d.beta > 1) return map_out.setError("EEDI3: alpha+beta must be <= 1.");
    if (d.gamma < 0) return map_out.setError("EEDI3: gamma >= 0.");
    if (nrad < 0 or nrad > 3) return map_out.setError("EEDI3: nrad 0..3.");
    if (mdis < 1 or mdis > 40) return map_out.setError("EEDI3: mdis 1..40.");
    if (vcheck < 0 or vcheck > 3) return map_out.setError("EEDI3: vcheck 0..3.");
    if (ns_req) |ns| {
        if (ns < 1 or ns > 32) return map_out.setError("EEDI3: num_streams must be 1..32.");
    }
    const device_id = map_in.getValue(i32, "device_id") orelse 0;
    if (device_id < 0) return map_out.setError("EEDI3: invalid device ID.");

    if (field > 1) {
        if (d.vi.numFrames > std.math.maxInt(i32) / 2) return map_out.setError("EEDI3: clip too long.");
        d.vi.numFrames *= 2;
        vsh.muldivRational(&d.vi.fpsNum, &d.vi.fpsDen, 2, 1);
    }
    if (d.dh) {
        if (horizontal) d.vi.width *= 2 else d.vi.height *= 2;
    }

    if (d.sclip) |sc| {
        const svi = zapi.getVideoInfo(sc);
        if (!vsh.isSameVideoInfo(svi, &d.vi) or svi.numFrames != d.vi.numFrames) {
            return map_out.setError("EEDI3: sclip's format/dimensions/length must match the output.");
        }
    }

    d.field = @intCast(field);
    d.mdis = @intCast(mdis);
    d.nrad = @intCast(nrad);
    d.vcheck = @intCast(vcheck);
    d.one_minus_ab = 1.0 - d.alpha - d.beta;
    d.alpha /= 3.0;
    d.beta /= 255.0;
    d.gamma /= 255.0;

    const vthresh0 = (map_in.getValue(f32, "vthresh0") orelse 32.0) / 255.0;
    const vthresh1 = (map_in.getValue(f32, "vthresh1") orelse 64.0) / 255.0;
    d.vthresh2 = map_in.getValue(f32, "vthresh2") orelse 4.0;
    if (vcheck > 0 and (vthresh0 <= 0 or vthresh1 <= 0 or d.vthresh2 <= 0)) return map_out.setError("EEDI3: vthresh* must be > 0.");
    d.rcp0 = 1.0 / vthresh0;
    d.rcp1 = 1.0 / vthresh1;
    d.rcp2 = 1.0 / d.vthresh2;

    d.tpitch = @intCast(if (d.hp) 4 * mdis + 1 else 2 * mdis + 1);
    d.pad = @intCast(3 * mdis + nrad + 2);

    {
        const strides_out = vszipcl.strideFromVi(&d.vi);
        const strides_in = vszipcl.strideFromVi(vi_in);
        const nplanes: usize = @intCast(d.vi.format.numPlanes);
        var p: usize = 0;
        while (p < nplanes) : (p += 1) {
            const cat: usize = if (p == 0) 0 else 1;
            const sw: u5 = if (p == 0) 0 else @intCast(d.vi.format.subSamplingW);
            const sh: u5 = if (p == 0) 0 else @intCast(d.vi.format.subSamplingH);
            var cfg: Config = .{ .w = 0, .src_h = 0, .dst_h = 0, .stride = 0, .pstride = 0, .n_interp_max = 0 };
            if (horizontal) {
                cfg.in_w = @as(u32, @intCast(vi_in.width)) >> sw;
                cfg.in_h = @as(u32, @intCast(vi_in.height)) >> sh;
                cfg.out_w = @as(u32, @intCast(d.vi.width)) >> sw;
                cfg.w = cfg.in_h;
                cfg.src_h = cfg.in_w;
                cfg.dst_h = cfg.out_w;
                cfg.in_stride = strides_in[cat];
                cfg.out_stride = strides_out[cat];
                const n_align: u32 = @divExact(vszipcl.vsFrameAlignment(), @as(u32, @intCast(d.vi.format.bytesPerSample)));
                cfg.stride = @max(strides_out[cat], @as(u32, @intCast(vsh.ceilN(@as(usize, cfg.w), n_align))));
            } else {
                cfg.w = @as(u32, @intCast(vi_in.width)) >> sw;
                cfg.src_h = @as(u32, @intCast(vi_in.height)) >> sh;
                cfg.dst_h = @as(u32, @intCast(d.vi.height)) >> sh;
                cfg.stride = strides_out[cat];
            }
            cfg.pstride = @intCast(vsh.ceilN(@as(usize, d.pad) * 2 + cfg.w, 8));
            cfg.n_interp_max = (cfg.dst_h + 1) / 2;
            var ci: usize = 0;
            while (ci < d.n_cfg) : (ci += 1) {
                if (std.meta.eql(cfg, d.configs[ci])) break;
            }
            if (ci == d.n_cfg) {
                d.configs[ci] = cfg;
                d.n_cfg += 1;
            }
            d.plane_cfg[p] = ci;
        }
    }

    {
        var acc_io: u64 = 0;
        var acc_src: u64 = 0;
        var acc_dm: u64 = 0;
        const nplanes: usize = @intCast(d.vi.format.numPlanes);
        for (0..nplanes) |p| {
            const cfg = &d.configs[d.plane_cfg[p]];
            acc_io += @as(u64, cfg.stride) * cfg.dst_h;
            acc_src += @as(u64, cfg.stride) * cfg.src_h;
            acc_dm += @as(u64, cfg.n_interp_max) * cfg.stride;
        }
        const plane0: u64 = @as(u64, d.configs[0].stride) * d.configs[0].dst_h;
        const max_extent = @max(plane0, if (vcheck > 0) @max(acc_io, @max(acc_src, acc_dm)) else 0);
        if (max_extent >= (1 << 31)) {
            map_out.setError("EEDI3: frame too large (a plane exceeds 2^31 samples).");
            return;
        }
        if (vcheck > 0) {
            var io: u64 = 0;
            var sr: u64 = 0;
            var dm: u64 = 0;
            for (0..nplanes) |p| {
                const cfg = &d.configs[d.plane_cfg[p]];
                d.off_io[p] = @intCast(io);
                d.off_src[p] = @intCast(sr);
                d.off_dmap[p] = @intCast(dm);
                d.vc_geom[p * 6 + 0] = @intCast(cfg.w);
                d.vc_geom[p * 6 + 1] = @intCast(cfg.stride);
                d.vc_geom[p * 6 + 2] = @intCast(cfg.dst_h);
                d.vc_geom[p * 6 + 3] = @intCast(io);
                d.vc_geom[p * 6 + 4] = @intCast(sr);
                d.vc_geom[p * 6 + 5] = @intCast(dm);
                io += @as(u64, cfg.stride) * cfg.dst_h;
                sr += @as(u64, cfg.stride) * cfg.src_h;
                dm += @as(u64, cfg.n_interp_max) * cfg.stride;
            }
        }
        d.sum_io = @intCast(acc_io);
        d.sum_src = @intCast(acc_src);
        d.sum_dmap = @intCast(acc_dm);
    }

    for (0..d.n_cfg) |ci| {
        const cfg = &d.configs[ci];
        const sh_i: i32 = @intCast(cfg.src_h);
        for (0..2) |f| {
            const n: u32 = (cfg.dst_h - @as(u32, @intCast(f)) + 1) / 2;
            const ri = allocator.alloc(i32, @as(usize, n) * 4) catch unreachable;
            const dy = allocator.alloc(i32, n) catch unreachable;
            var off: u32 = 0;
            var iy: i32 = @intCast(f);
            while (off < n) : (off += 1) {
                ri[off * 4 + 0] = stencilRow(iy - 3, sh_i, d.dh);
                ri[off * 4 + 1] = stencilRow(iy - 1, sh_i, d.dh);
                ri[off * 4 + 2] = stencilRow(iy + 1, sh_i, d.dh);
                ri[off * 4 + 3] = stencilRow(iy + 3, sh_i, d.dh);
                dy[off] = iy;
                iy += 2;
            }
            d.rowidx_host[ci][f] = ri;
            d.dsty_host[ci][f] = dy;
        }
    }

    initOpenCL(&d, @intCast(device_id)) catch |err| {
        map_out.setError(if (err == error.InvalidDeviceID) "EEDI3: invalid device ID." else "EEDI3: OpenCL init failed.");
        std.log.err("EEDI3 OpenCL init failed: {}", .{err});
        freeStencilTables(&d);
        return;
    };

    d.bx = 48;
    d.run = 8;
    d.run_hp = 4;
    var lws_floor: usize = 128;
    {
        var terr: ?[:0]const u8 = null;
        if (map_in.numElements("tune")) |n| {
            if (n > tune_len) terr = "EEDI3: tune expects at most 4 entries [bx, run, run_hp, lws_floor].";
        }
        if (vszipcl.tuneEntry(map_in, 0)) |v| {
            if (v < 8 or v > 256) terr = "EEDI3: tune[0] (bx) must be 8..256." else d.bx = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 1)) |v| {
            if (v < 1 or v > 64) terr = "EEDI3: tune[1] (run) must be 1..64." else d.run = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 2)) |v| {
            if (v < 1 or v > 64) terr = "EEDI3: tune[2] (run_hp) must be 1..64." else d.run_hp = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 3)) |v| {
            if (v < 32 or v > 1024) terr = "EEDI3: tune[3] (lws_floor) must be 32..1024." else lws_floor = @intCast(v);
        }
        if (terr == null and (d.bx % d.run != 0 or d.bx % d.run_hp != 0))
            terr = "EEDI3: tune bx must be a multiple of both run and run_hp.";
        if (terr) |msg| {
            map_out.setError(msg);
            d.context.release();
            freeStencilTables(&d);
            return;
        }
    }
    {
        const step: u32 = (d.run * d.run_hp) / std.math.gcd(d.run, d.run_hp);
        const want: u32 = d.bx;
        const tph: usize = 4 * @as(usize, d.mdis) + 1;
        const budget = d.local_mem -| 1024;
        var bx = want;
        while (bx > step and (@as(usize, bx) * tph + 2 * 165) * 4 > budget) bx -= step;
        if ((@as(usize, bx) * tph + 2 * 165) * 4 > budget) {
            map_out.setError("EEDI3: no strip width fits the device local memory at this mdis (lower mdis, run or run_hp).");
            d.context.release();
            freeStencilTables(&d);
            return;
        }
        if (bx != want) std.log.warn("EEDI3: tune BX {d} over the local-mem budget ({d} B = device {d} B - 1 KiB reserve) — clamped to {d}.", .{ want, budget, d.local_mem, bx });
        d.bx = bx;
    }
    d.lws = @max(vsh.ceilN(@as(usize, d.tpitch), 32), lws_floor);
    if (d.lws > d.max_wg) d.lws = vsh.ceilN(@as(usize, d.tpitch), 32);
    if (d.lws > d.max_wg) {
        map_out.setError("EEDI3: device max work-group size is too small for this mdis.");
        d.context.release();
        freeStencilTables(&d);
        return;
    }
    if (horizontal and d.max_wg < 256) {
        map_out.setError("EEDI3H: device max work-group size < 256 (required by the transpose kernel).");
        d.context.release();
        freeStencilTables(&d);
        return;
    }
    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;
    keep = true;

    const streams: usize = if (ns_req) |ns| @intCast(ns) else 1;
    data.pool.prime(data, streams);
    data.pool.prewarm(streams) catch |err| {
        map_out.setError("EEDI3: OpenCL stream init failed.");
        std.log.err("EEDI3 stream init failed: {}", .{err});
        data.pool.deinit();
        data.context.release();
        freeStencilTables(data);
        allocator.destroy(data);
        keep = false;
        return;
    };

    var dep_buf: [2]vs.FilterDependency = undefined;
    dep_buf[0] = .{ .source = d.node, .requestPattern = if (d.field > 1) .General else .StrictSpatial };
    var ndeps: usize = 1;
    if (d.sclip) |sc| {
        dep_buf[ndeps] = .{ .source = sc, .requestPattern = .StrictSpatial };
        ndeps += 1;
    }
    zapi.createVideoFilter(out, if (horizontal) "EEDI3H" else "EEDI3", &d.vi, getFrame, free, .Parallel, dep_buf[0..ndeps], data);
}
