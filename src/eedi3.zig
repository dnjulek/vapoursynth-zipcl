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
const tpitch_max = 2 * mdis_max + 1; // 81

const kernel_src =
    \\#define TPMAX 83  /* tpitch_max + 2 sentinels */
    \\#define FLTMAX9 3.0e38f
    \\#ifndef CN
    \\#define CN 2      /* = nrad; overridden by -DCN=<nrad> at build time so the
    \\                     hot cost loop has a compile-time bound and fully unrolls */
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
    \\kernel void pad_src(global float *out, global const float *src,
    \\                    const int w, const int stride, const int pstride,
    \\                    const int pad, const int src_h) {
    \\    const int px = get_global_id(0);     // padded column [0, pad+w+pad)
    \\    const int y  = get_global_id(1);
    \\    if (px >= pad + w + pad || y >= src_h) return;
    \\    out[y*pstride + px] = src[y*stride + refl(px - pad, w)];
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
    \\// Copy the kept field rows from src into dst, and (when vcheck off) acts as a
    \\// no-op for interp rows. global = (w, dst_h).
    \\kernel void copy_kept(global float *dst, global const float *src,
    \\                      const int w, const int stride, const int dh,
    \\                      const int field, const int src_h) {
    \\    const int x = get_global_id(0);
    \\    const int y = get_global_id(1);
    \\    if (x >= w) return;
    \\    int is_interp, sy = -1;
    \\    if (dh) {
    \\        // kept dst rows: 2*ky + (1-field); interp rows: field, field+2, ...
    \\        is_interp = ((y & 1) == field);
    \\        if (!is_interp) sy = (y - (1 - field)) >> 1;
    \\    } else {
    \\        is_interp = ((y & 1) == field);
    \\        if (!is_interp) sy = y;
    \\    }
    \\    if (!is_interp) dst[y*stride + x] = src[sy*stride + x];
    \\}
    \\
    \\// One workgroup per interpolated row. local_size = padded tpitch.
    \\// rowidx[off*4 + {0,1,2,3}] = src y-index for r3p,r1p,r1n,r3n.
    \\// dst_y[off] = destination row.  fpath is dynamic local (w ints).
    \\kernel void interp(global float * restrict dst, global const float * restrict srcpad,
    \\                   global const int * restrict rowidx, global const int * restrict dst_y,
    \\                   global char * restrict pbackt, global int * restrict dmap,
    \\                   const int w, const int stride, const int pstride, const int pad,
    \\                   const int mdis, const int nrad,
    \\                   const float alpha, const float beta, const float gamma,
    \\                   const float one_minus_ab,
    \\                   local int *fpath) {
    \\    const int off = get_global_id(1);
    \\    const int tid = get_local_id(0);
    \\    const int lsz = get_local_size(0);
    \\    const int tpitch = 2*mdis + 1;
    \\
    \\    // PADDED rows: index [j+pad] for any j in [-pad, w-1+pad], no refl().
    \\    global const float * restrict r3p = srcpad + rowidx[off*4+0]*pstride;
    \\    global const float * restrict r1p = srcpad + rowidx[off*4+1]*pstride;
    \\    global const float * restrict r1n = srcpad + rowidx[off*4+2]*pstride;
    \\    global const float * restrict r3n = srcpad + rowidx[off*4+3]*pstride;
    \\    global char *pb = pbackt + (long)off * w * tpitch;
    \\
    \\    local float pc[2][TPMAX];
    \\    const int u = tid - mdis;          // direction for this thread (valid if tid<tpitch)
    \\    const int active = (tid < tpitch);
    \\
    \\    // sentinels
    \\    if (tid == 0) { pc[0][0]=FLTMAX9; pc[1][0]=FLTMAX9; pc[0][tpitch+1]=FLTMAX9; pc[1][tpitch+1]=FLTMAX9; }
    \\    barrier(CLK_LOCAL_MEM_FENCE);
    \\
    \\    int ping = 0;
    \\    if (active) pc[ping][tid+1] = conn_cost(r3p,r1p,r1n,r3n, 0, u, w, nrad, pad, alpha, beta, one_minus_ab);
    \\    barrier(CLK_LOCAL_MEM_FENCE);
    \\
    \\    for (int x = 1; x < w; x++) {
    \\        const int pong = ping ^ 1;
    \\        if (active) {
    \\            float cost = conn_cost(r3p,r1p,r1n,r3n, x, u, w, nrad, pad, alpha, beta, one_minus_ab);
    \\            float left  = pc[ping][tid]   + gamma;  // dir u-1
    \\            float cent  = pc[ping][tid+1];          // dir u
    \\            float right = pc[ping][tid+2] + gamma;  // dir u+1
    \\            float bval = cent; char bd = 0;
    \\            if (left  < bval) { bval = left;  bd = -1; }
    \\            if (right < bval) { bval = right; bd =  1; }
    \\            pc[pong][tid+1] = fmin(bval + cost, FLTMAX9);
    \\            pb[(x-1)*tpitch + tid] = bd;   // CPU stores backtrack delta at x-1
    \\        }
    \\        barrier(CLK_LOCAL_MEM_FENCE);
    \\        ping = pong;
    \\    }
    \\
    \\    // serial backtrack: all pbackt global writes must be visible to thread 0
    \\    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    \\    if (tid == 0) {
    \\        fpath[w-1] = 0;
    \\        for (int bx = w-2; bx >= 0; bx--) {
    \\            int ui = mdis + fpath[bx+1];
    \\            fpath[bx] = fpath[bx+1] + (int)pb[bx*tpitch + ui];
    \\        }
    \\    }
    \\    barrier(CLK_LOCAL_MEM_FENCE);
    \\
    \\    // parallel interpolate
    \\    global float *drow = dst + dst_y[off]*stride;
    \\    global int *dm = dmap + (long)off*stride;
    \\    for (int x = tid; x < w; x += lsz) {
    \\        int dir = fpath[x];
    \\        dm[x] = dir;
    \\        int ad = abs(dir);
    \\        const int xp = x + pad;
    \\        float val;
    \\        if (x >= ad*3 && x + ad*3 <= w-1) {
    \\            val = 0.5625f * (r1p[xp+dir] + r1n[xp-dir])
    \\                - 0.0625f * (r3p[xp+dir*3] + r3n[xp-dir*3]);
    \\        } else {
    \\            val = (r1p[xp+dir] + r1n[xp-dir]) * 0.5f;
    \\        }
    \\        drow[x] = val;
    \\    }
    \\}
    \\
    \\// ---- half-pel (hp=1) path ----
    \\#define TPMAX_HP 165   /* 4*mdis_max+1 + 4 sentinels */
    \\// Rows passed to the hp helpers are PADDED rows already advanced by `pad`, so
    \\// R(row,j) is a plain row[j] (j may be negative / >= w within the pad margin).
    \\static float R(global const float * restrict row, int j) { return row[j]; }
    \\// cubic half-pel sample between full-pel j and j+1 (matches computeHpRow)
    \\static float HP(global const float * restrict row, int j) {
    \\    return 0.5625f*(row[j]+row[j+1]) - 0.0625f*(row[j-1]+row[j+2]);
    \\}
    \\static float conn_cost_hp(global const float * restrict r3p, global const float * restrict r1p,
    \\                          global const float * restrict r1n, global const float * restrict r3n,
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
    \\            s0 += fabs(HP(r3p,hi)-HP(r1p,lo)) + fabs(HP(r1p,hi)-HP(r1n,lo)) + fabs(HP(r1n,hi)-HP(r3n,lo));
    \\        else
    \\            s0 += fabs(R(r3p,hi)-R(r1p,lo)) + fabs(R(r1p,hi)-R(r1n,lo)) + fabs(R(r1n,hi)-R(r3n,lo));
    \\    }
    \\    float Bxuh = odd ? HP(r1p,x+uh) : R(r1p,x+uh);
    \\    float Cxlo = odd ? HP(r1n,x+lo0) : R(r1n,x+lo0);
    \\    float ip = (Bxuh + Cxlo) * 0.5f;
    \\    float v = fabs(R(r1p,x)-ip) + fabs(R(r1n,x)-ip);
    \\    return alpha3*(s0+s1+s2) + beta255*(float)abs(u)*0.5f + one_minus_ab*v;
    \\}
    \\
    \\kernel void interp_hp(global float * restrict dst, global const float * restrict srcpad,
    \\                      global const int * restrict rowidx, global const int * restrict dst_y,
    \\                      global char * restrict pbackt, global int * restrict dmap,
    \\                      const int w, const int stride, const int pstride, const int pad,
    \\                      const int mdis, const int nrad,
    \\                      const float alpha3, const float beta255, const float gamma255,
    \\                      const float one_minus_ab, local int *fpath) {
    \\    const int off = get_global_id(1);
    \\    const int tid = get_local_id(0);
    \\    const int lsz = get_local_size(0);
    \\    const int cen = 2*mdis;
    \\    const int tpitch = 4*mdis + 1;
    \\    // PADDED rows, pre-advanced by `pad` so R(row,j)=row[j] needs no refl().
    \\    global const float * restrict r3p = srcpad + rowidx[off*4+0]*pstride + pad;
    \\    global const float * restrict r1p = srcpad + rowidx[off*4+1]*pstride + pad;
    \\    global const float * restrict r1n = srcpad + rowidx[off*4+2]*pstride + pad;
    \\    global const float * restrict r3n = srcpad + rowidx[off*4+3]*pstride + pad;
    \\    global char *pb = pbackt + (long)off * w * tpitch;
    \\    local float pc[2][TPMAX_HP];
    \\    const int u = tid - cen;
    \\    const int active = (tid < tpitch);
    \\    if (tid == 0) { for (int b=0;b<2;b++){ pc[b][0]=FLTMAX9; pc[b][1]=FLTMAX9; pc[b][tpitch+2]=FLTMAX9; pc[b][tpitch+3]=FLTMAX9; } }
    \\    barrier(CLK_LOCAL_MEM_FENCE);
    \\    int ping = 0;
    \\    if (active) pc[ping][tid+2] = conn_cost_hp(r3p,r1p,r1n,r3n, 0, u, w, nrad, alpha3, beta255, one_minus_ab);
    \\    barrier(CLK_LOCAL_MEM_FENCE);
    \\    const float g1 = gamma255*0.5f, g2 = gamma255;
    \\    for (int x = 1; x < w; x++) {
    \\        const int pong = ping ^ 1;
    \\        if (active) {
    \\            float cost = conn_cost_hp(r3p,r1p,r1n,r3n, x, u, w, nrad, alpha3, beta255, one_minus_ab);
    \\            float c_m2 = fmin(pc[ping][tid+0]+g2, FLTMAX9);
    \\            float c_m1 = fmin(pc[ping][tid+1]+g1, FLTMAX9);
    \\            float c_0  = fmin(pc[ping][tid+2],    FLTMAX9);
    \\            float c_p1 = fmin(pc[ping][tid+3]+g1, FLTMAX9);
    \\            float c_p2 = fmin(pc[ping][tid+4]+g2, FLTMAX9);
    \\            float bval = c_m2; char bd = -2;
    \\            if (c_m1 < bval) { bval = c_m1; bd = -1; }
    \\            if (c_0  < bval) { bval = c_0;  bd =  0; }
    \\            if (c_p1 < bval) { bval = c_p1; bd =  1; }
    \\            if (c_p2 < bval) { bval = c_p2; bd =  2; }
    \\            pc[pong][tid+2] = fmin(bval + cost, FLTMAX9);
    \\            pb[(x-1)*tpitch + tid] = bd;
    \\        }
    \\        barrier(CLK_LOCAL_MEM_FENCE);
    \\        ping = pong;
    \\    }
    \\    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    \\    if (tid == 0) {
    \\        fpath[w-1] = 0;
    \\        for (int bx = w-2; bx >= 0; bx--) {
    \\            int ui = cen + fpath[bx+1];
    \\            fpath[bx] = fpath[bx+1] + (int)pb[bx*tpitch + ui];
    \\        }
    \\    }
    \\    barrier(CLK_LOCAL_MEM_FENCE);
    \\    global float *drow = dst + dst_y[off]*stride;
    \\    global int *dm = dmap + (long)off*stride;
    \\    for (int x = tid; x < w; x += lsz) {
    \\        int dir = fpath[x];
    \\        dm[x] = dir;
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
    \\        drow[x] = val;
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
    \\kernel void vcheck(global float *out, global const float *dst, global const float *src,
    \\                   global const int *rowidx, global const int *dmap, global const float *scp,
    \\                   const int w, const int stride, const int dst_h, const int field,
    \\                   const int n_interp, const int vmode, const int use_scp, const int hp,
    \\                   const float rcp0, const float rcp1, const float rcp2, const float vthresh2) {
    \\    const int lid = get_local_id(0);
    \\    const int lsz = get_local_size(0);
    \\    for (int off = 1; off + 1 < n_interp; ++off) {
    \\        const int y = field + 2*off;
    \\        if (y >= 2 && y + 2 < dst_h) {
    \\            global const float *drow = dst + y*stride;
    \\            global const float *d1p = dst + (y-1)*stride;
    \\            global const float *d2p = out + (y-2)*stride;
    \\            global const float *d1n = dst + (y+1)*stride;
    \\            global const float *d2n = dst + (y+2)*stride;
    \\            global const float *d3p = src + rowidx[off*4+0]*stride;
    \\            global const float *d3n = src + rowidx[off*4+3]*stride;
    \\            for (int x = lid; x < w; x += lsz) {
    \\                const int dirc = dmap[(long)off*stride + x];
    \\                float cint = use_scp ? scp[y*stride + x] : (0.5625f*(d1p[x]+d1n[x]) - 0.0625f*(d3p[x]+d3n[x]));
    \\                int dirt = dmap[(long)(off-1)*stride + x];
    \\                int dirb = dmap[(long)(off+1)*stride + x];
    \\                int maxoff = hp ? ((dirc & 1) == 0 ? abs(dirc>>1) : max(abs(dirc>>1), abs((dirc+1)>>1))) : abs(dirc);
    \\                if (dirc == 0 || max(dirc*dirt, dirc*dirb) < 0 || (dirt==dirb && dirt==0)
    \\                    || x + maxoff >= w || x - maxoff < 0) {
    \\                    out[y*stride + x] = cint;
    \\                    continue;
    \\                }
    \\                float it, ib, vt, vb;
    \\                int dabs;
    \\                if (hp && (dirc & 1) != 0) {
    \\                    int d20 = dirc>>1, d21 = (dirc+1)>>1;
    \\                    int xp0 = x+d20, xp1 = x+d21, xm0 = x-d20, xm1 = x-d21;
    \\                    float s2psum = d2p[xp0]+d2p[xp1], s1psum = d1p[xp0]+d1p[xp1];
    \\                    float pa0 = drow[xp0]+drow[xp1], ps0 = drow[xm0]+drow[xm1];
    \\                    float s1nsum = d1n[xm0]+d1n[xm1], s2nsum = d2n[xm0]+d2n[xm1];
    \\                    it = (s2psum + ps0)*0.25f;
    \\                    vt = (fabs(s2psum-s1psum) + fabs(pa0-s1psum))*0.5f;
    \\                    ib = (pa0 + s2nsum)*0.25f;
    \\                    vb = (fabs(s2nsum-s1nsum) + fabs(ps0-s1nsum))*0.5f;
    \\                    dabs = abs(dirc) >> 1;
    \\                } else {
    \\                    int offh = hp ? (dirc>>1) : dirc;
    \\                    int xpd = x+offh, xmd = x-offh;
    \\                    it = (d2p[xpd] + drow[xmd]) * 0.5f;
    \\                    ib = (drow[xpd] + d2n[xmd]) * 0.5f;
    \\                    vt = fabs(d2p[xpd]-d1p[xpd]) + fabs(drow[xpd]-d1p[xpd]);
    \\                    vb = fabs(d2n[xmd]-d1n[xmd]) + fabs(drow[xmd]-d1n[xmd]);
    \\                    dabs = hp ? (abs(dirc)>>1) : abs(dirc);
    \\                }
    \\                float vc = fabs(drow[x]-d1p[x]) + fabs(drow[x]-d1n[x]);
    \\                float d0 = fabs(it-d1p[x]), d1 = fabs(ib-d1n[x]), d2 = fabs(vt-vc), d3 = fabs(vb-vc);
    \\                float mdiff0 = (vmode==1)?fmin(d0,d1):(vmode==2)?(d0+d1)*0.5f:fmax(d0,d1);
    \\                float mdiff1 = (vmode==1)?fmin(d2,d3):(vmode==2)?(d2+d3)*0.5f:fmax(d2,d3);
    \\                float a0 = mdiff0*rcp0, a1 = mdiff1*rcp1;
    \\                float a2 = fmax((vthresh2 - (float)dabs)*rcp2, 0.0f);
    \\                float a = fmin(fmax(a0, fmax(a1,a2)), 1.0f);
    \\                out[y*stride + x] = (1.0f-a)*drow[x] + a*cint;
    \\            }
    \\        }
    \\        barrier(CLK_GLOBAL_MEM_FENCE);
    \\    }
    \\}
    \\
    \\// out[c*out_stride + r] = in[r*in_stride + c], for r<in_h, c<in_w. global=(in_w,in_h).
    \\kernel void transpose(global float *out, global const float *in,
    \\                      const int in_w, const int in_h, const int in_stride, const int out_stride) {
    \\    const int c = get_global_id(0);
    \\    const int r = get_global_id(1);
    \\    if (c >= in_w || r >= in_h) return;
    \\    out[c*out_stride + r] = in[r*in_stride + c];
    \\}
    \\
;

const Data = struct {
    node: ?*vs.Node = null,
    sclip: ?*vs.Node = null,
    vi: vs.VideoInfo = undefined,

    horizontal: bool = false,
    hp: bool = false,
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

    // dimensions of the plane-0 src/dst (vertical orientation)
    w: u32 = 0,
    src_h: u32 = 0,
    dst_h: u32 = 0,
    stride: u32 = 0,
    n_interp: u32 = 0,
    tpitch: u32 = 0,
    lws: usize = 0,
    max_wg: usize = 256, // device CL_DEVICE_MAX_WORK_GROUP_SIZE (caps the vcheck workgroup)

    // horizontal mirror-pad margin (covers max interp/cost reach) + padded stride
    pad: u32 = 0,
    pstride: u32 = 0,

    // EEDI3H (transpose) only: input/output frame dims (untransposed)
    in_w: u32 = 0, // = src_h
    in_h: u32 = 0, // = w
    out_w: u32 = 0, // = dst_h
    in_stride: u32 = 0,
    out_stride: u32 = 0,

    platform: cl.Platform = undefined,
    device: cl.Device = undefined,
    pool: clpool.Pool(Stream, Data) = .{},
};

const Stream = struct {
    context: cl.Context,
    program: cl.Program,
    queue: cl.CommandQueue,
    d_src: cl.Buffer(f32),
    d_srcpad: cl.Buffer(f32),
    d_dst: cl.Buffer(f32),
    d_rowidx: cl.Buffer(i32),
    d_dsty: cl.Buffer(i32),
    d_pbackt: cl.Buffer(i8),
    d_dmap: cl.Buffer(i32),
    d_dst2: cl.Buffer(f32),
    d_inframe: cl.Buffer(f32),
    d_outframe: cl.Buffer(f32),
    d_scp: cl.Buffer(f32),
    d_scpframe: cl.Buffer(f32),
    k_copy: cl.Kernel,
    k_pad: cl.Kernel,
    k_interp: cl.Kernel,
    k_interp_hp: cl.Kernel,
    k_vcheck: cl.Kernel,
    k_transpose: cl.Kernel,

    pub fn init(self: *Stream, d: *Data) !void {
        self.context = try cl.createContext(&.{d.device}, .{ .platform = d.platform });
        errdefer self.context.release();
        self.program = try cl.createProgramWithSource(self.context, kernel_src);
        errdefer self.program.release();
        const build_opts = try std.fmt.allocPrintSentinel(allocator, "-cl-std=CL3.0 -DCN={d}", .{d.nrad}, 0);
        defer allocator.free(build_opts);
        self.program.build(&.{d.device}, build_opts) catch |err| {
            if (err == error.BuildProgramFailure) {
                const log = try self.program.getBuildLog(allocator, d.device);
                defer allocator.free(log);
                std.log.err("EEDI3 OpenCL build failed: {s}", .{log});
            }
            return err;
        };
        self.queue = try cl.createCommandQueue(self.context, d.device, .{});
        errdefer self.queue.release();
        self.d_src = try cl.createBuffer(f32, self.context, .{ .read_only = true }, d.stride * d.src_h);
        errdefer self.d_src.release();
        self.d_srcpad = try cl.createBuffer(f32, self.context, .{ .read_write = true }, d.pstride * d.src_h);
        errdefer self.d_srcpad.release();
        self.d_dst = try cl.createBuffer(f32, self.context, .{ .read_write = true }, d.stride * d.dst_h);
        errdefer self.d_dst.release();
        self.d_rowidx = try cl.createBuffer(i32, self.context, .{ .read_only = true }, d.n_interp * 4);
        errdefer self.d_rowidx.release();
        self.d_dsty = try cl.createBuffer(i32, self.context, .{ .read_only = true }, d.n_interp);
        errdefer self.d_dsty.release();
        self.d_pbackt = try cl.createBuffer(i8, self.context, .{ .read_write = true }, @as(usize, d.n_interp) * d.w * d.tpitch);
        errdefer self.d_pbackt.release();
        self.d_dmap = try cl.createBuffer(i32, self.context, .{ .read_write = true }, @as(usize, d.n_interp) * d.stride);
        errdefer self.d_dmap.release();
        self.d_dst2 = try cl.createBuffer(f32, self.context, .{ .read_write = true }, if (d.vcheck > 0) d.stride * d.dst_h else 1);
        errdefer self.d_dst2.release();
        self.d_inframe = try cl.createBuffer(f32, self.context, .{ .read_write = true }, if (d.horizontal) d.in_stride * d.in_h else 1);
        errdefer self.d_inframe.release();
        self.d_outframe = try cl.createBuffer(f32, self.context, .{ .read_write = true }, if (d.horizontal) d.out_stride * d.in_h else 1);
        errdefer self.d_outframe.release();
        const have_scp = d.sclip != null and d.vcheck > 0;
        self.d_scp = try cl.createBuffer(f32, self.context, .{ .read_write = true }, if (have_scp) d.stride * d.dst_h else 1);
        errdefer self.d_scp.release();
        self.d_scpframe = try cl.createBuffer(f32, self.context, .{ .read_write = true }, if (have_scp and d.horizontal) d.out_stride * d.in_h else 1);
        errdefer self.d_scpframe.release();
        self.k_copy = try cl.createKernel(self.program, "copy_kept");
        errdefer self.k_copy.release();
        self.k_pad = try cl.createKernel(self.program, "pad_src");
        errdefer self.k_pad.release();
        self.k_interp = try cl.createKernel(self.program, "interp");
        errdefer self.k_interp.release();
        self.k_interp_hp = try cl.createKernel(self.program, "interp_hp");
        errdefer self.k_interp_hp.release();
        self.k_vcheck = try cl.createKernel(self.program, "vcheck");
        errdefer self.k_vcheck.release();
        self.k_transpose = try cl.createKernel(self.program, "transpose");
        try self.setStaticArgs(d);
    }

    // Set all immutable kernel args ONCE (this Stream owns its kernels for life and
    // is used by one frame at a time, so OpenCL's sticky arg state carries them
    // frame-to-frame). Only `field` (copy_kept arg 5, vcheck arg 9) varies per frame
    // and is re-set in process(); the transpose kernel is reused per-call with
    // different buffers so its args stay in runTranspose. Buffer CONTENTS change per
    // frame (re-uploaded), but the buffer-handle args are fixed per Stream.
    fn setStaticArgs(self: *Stream, d: *Data) !void {
        const w: c_int = @intCast(d.w);
        const stride: c_int = @intCast(d.stride);
        const src_h_i: c_int = @intCast(d.src_h);

        // pad_src
        try self.k_pad.setArg(@TypeOf(self.d_srcpad), 0, self.d_srcpad);
        try self.k_pad.setArg(@TypeOf(self.d_src), 1, self.d_src);
        try self.k_pad.setArg(c_int, 2, w);
        try self.k_pad.setArg(c_int, 3, stride);
        try self.k_pad.setArg(c_int, 4, @intCast(d.pstride));
        try self.k_pad.setArg(c_int, 5, @intCast(d.pad));
        try self.k_pad.setArg(c_int, 6, src_h_i);

        // copy_kept (arg 5 = field is per-frame)
        try self.k_copy.setArg(@TypeOf(self.d_dst), 0, self.d_dst);
        try self.k_copy.setArg(@TypeOf(self.d_src), 1, self.d_src);
        try self.k_copy.setArg(c_int, 2, w);
        try self.k_copy.setArg(c_int, 3, stride);
        try self.k_copy.setArg(c_int, 4, @intFromBool(d.dh));
        try self.k_copy.setArg(c_int, 6, src_h_i);

        // interp / interp_hp (no field arg; all immutable, incl. the dynamic __local)
        const ik = if (d.hp) self.k_interp_hp else self.k_interp;
        try ik.setArg(@TypeOf(self.d_dst), 0, self.d_dst);
        try ik.setArg(@TypeOf(self.d_srcpad), 1, self.d_srcpad);
        try ik.setArg(@TypeOf(self.d_rowidx), 2, self.d_rowidx);
        try ik.setArg(@TypeOf(self.d_dsty), 3, self.d_dsty);
        try ik.setArg(@TypeOf(self.d_pbackt), 4, self.d_pbackt);
        try ik.setArg(@TypeOf(self.d_dmap), 5, self.d_dmap);
        try ik.setArg(c_int, 6, w);
        try ik.setArg(c_int, 7, stride);
        try ik.setArg(c_int, 8, @intCast(d.pstride));
        try ik.setArg(c_int, 9, @intCast(d.pad));
        try ik.setArg(c_int, 10, @intCast(d.mdis));
        try ik.setArg(c_int, 11, @intCast(d.nrad));
        try ik.setArg(f32, 12, d.alpha);
        try ik.setArg(f32, 13, d.beta);
        try ik.setArg(f32, 14, d.gamma);
        try ik.setArg(f32, 15, d.one_minus_ab);
        if (cl.c.clSetKernelArg(ik.handle, 16, d.w * @sizeOf(i32), null) != cl.c.CL_SUCCESS) return error.SetKernelArg;

        // vcheck (arg 9 = field is per-frame; use_scp is constant). Only used when on.
        if (d.vcheck > 0) {
            const use_scp: c_int = if (d.sclip != null) 1 else 0;
            try self.k_vcheck.setArg(@TypeOf(self.d_dst2), 0, self.d_dst2);
            try self.k_vcheck.setArg(@TypeOf(self.d_dst), 1, self.d_dst);
            try self.k_vcheck.setArg(@TypeOf(self.d_src), 2, self.d_src);
            try self.k_vcheck.setArg(@TypeOf(self.d_rowidx), 3, self.d_rowidx);
            try self.k_vcheck.setArg(@TypeOf(self.d_dmap), 4, self.d_dmap);
            try self.k_vcheck.setArg(@TypeOf(self.d_scp), 5, self.d_scp);
            try self.k_vcheck.setArg(c_int, 6, w);
            try self.k_vcheck.setArg(c_int, 7, stride);
            try self.k_vcheck.setArg(c_int, 8, @intCast(d.dst_h));
            try self.k_vcheck.setArg(c_int, 10, @intCast(d.n_interp));
            try self.k_vcheck.setArg(c_int, 11, @intCast(d.vcheck));
            try self.k_vcheck.setArg(c_int, 12, use_scp);
            try self.k_vcheck.setArg(c_int, 13, @intFromBool(d.hp));
            try self.k_vcheck.setArg(f32, 14, d.rcp0);
            try self.k_vcheck.setArg(f32, 15, d.rcp1);
            try self.k_vcheck.setArg(f32, 16, d.rcp2);
            try self.k_vcheck.setArg(f32, 17, d.vthresh2);
        }
    }

    pub fn deinit(self: *Stream) void {
        self.k_transpose.release();
        self.k_vcheck.release();
        self.k_interp_hp.release();
        self.k_interp.release();
        self.k_pad.release();
        self.k_copy.release();
        self.d_scpframe.release();
        self.d_scp.release();
        self.d_outframe.release();
        self.d_inframe.release();
        self.d_dst2.release();
        self.d_dmap.release();
        self.d_pbackt.release();
        self.d_dsty.release();
        self.d_rowidx.release();
        self.d_dst.release();
        self.d_srcpad.release();
        self.d_src.release();
        self.queue.release();
        self.program.release();
        self.context.release();
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

fn initOpenCL(d: *Data) !void {
    const platforms = try cl.getPlatforms(allocator);
    defer allocator.free(platforms);
    if (platforms.len == 0) return error.NoPlatforms;
    const platform = platforms[0];
    const devices = try platform.getDevices(allocator, cl.DeviceType.all);
    defer allocator.free(devices);
    if (devices.len == 0) return error.NoDevices;
    d.device = devices[0];
    // Cap the single-workgroup vcheck launch at the device limit — a hardcoded
    // 1024 would error (CL_INVALID_WORK_GROUP_SIZE) on devices whose max WG < 1024
    // (CPU runtimes, some iGPUs). vcheck grid-strides over x, so a smaller WG is
    // still correct, just fewer warps to hide the inter-row latency.
    var dev_max_wg: usize = 0;
    if (cl.c.clGetDeviceInfo(d.device.id, cl.c.CL_DEVICE_MAX_WORK_GROUP_SIZE, @sizeOf(usize), &dev_max_wg, null) == cl.c.CL_SUCCESS and dev_max_wg > 0) {
        d.max_wg = dev_max_wg;
    }
    d.platform = platform;
}

fn process(d: *Data, s: *Stream, dstp: []f32, srcp: []const f32, scpp: ?[]const f32, field: u8) !void {
    const w: i32 = @intCast(d.w);
    const src_h_i: i32 = @intCast(d.src_h);
    const none: []const cl.Event = &.{};

    // Per-frame: source row indices for the 4 stencil rows of each interp row.
    var rowidx = try allocator.alloc(i32, d.n_interp * 4);
    defer allocator.free(rowidx);
    var dsty = try allocator.alloc(i32, d.n_interp);
    defer allocator.free(dsty);

    var off: u32 = 0;
    var iy: i32 = field;
    while (off < d.n_interp) : (off += 1) {
        const r = struct {
            fn f(yy: i32, sh: i32, dh: bool) i32 {
                return @intCast(if (dh) reflectRow(yy, 2 * sh) / 2 else reflectRow(yy, sh));
            }
        }.f;
        rowidx[off * 4 + 0] = r(iy - 3, src_h_i, d.dh);
        rowidx[off * 4 + 1] = r(iy - 1, src_h_i, d.dh);
        rowidx[off * 4 + 2] = r(iy + 1, src_h_i, d.dh);
        rowidx[off * 4 + 3] = r(iy + 3, src_h_i, d.dh);
        dsty[off] = iy;
        iy += 2;
    }

    // Commands run on an in-order queue, so each implicitly waits for the prior.
    (try s.queue.enqueueWriteBuffer(i32, s.d_rowidx, false, 0, rowidx, none)).release();
    (try s.queue.enqueueWriteBuffer(i32, s.d_dsty, false, 0, dsty, none)).release();

    // Get src into d_src (transpose the frame into column-major for EEDI3H).
    if (d.horizontal) {
        (try s.queue.enqueueWriteBuffer(f32, s.d_inframe, false, 0, srcp, none)).release();
        try runTranspose(s, s.d_src, s.d_inframe, d.in_w, d.in_h, d.in_stride, d.stride);
    } else {
        (try s.queue.enqueueWriteBuffer(f32, s.d_src, false, 0, srcp, none)).release();
    }

    // Build the horizontally mirror-padded source once, so interp/interp_hp can
    // read contiguous padded memory with no per-access refl() in the hot loop.
    // (immutable kernel args set once in Stream.setStaticArgs)
    const pad_gws: [2]usize = .{ vsh.ceilN(@as(usize, d.pad) * 2 + d.w, 16), d.src_h };
    const pad_lws: [2]usize = .{ 16, 1 };
    (try s.queue.enqueueNDRangeKernel(s.k_pad, null, &pad_gws, &pad_lws, none)).release();

    // copy kept field rows (only `field` is per-frame; other args set in init)
    try s.k_copy.setArg(c_int, 5, field);
    const copy_gws: [2]usize = .{ vsh.ceilN(@intCast(w), 16), d.dst_h };
    const copy_lws: [2]usize = .{ 16, 1 };
    (try s.queue.enqueueNDRangeKernel(s.k_copy, null, &copy_gws, &copy_lws, none)).release();

    // interp rows (half-pel uses a distinct kernel; all args set in init)
    const ik = if (d.hp) s.k_interp_hp else s.k_interp;
    const interp_gws: [2]usize = .{ d.lws, d.n_interp };
    const interp_lws: [2]usize = .{ d.lws, 1 };
    (try s.queue.enqueueNDRangeKernel(ik, null, &interp_gws, &interp_lws, none)).release();

    // pick the buffer holding the finished (vertical-orientation) result
    var result = s.d_dst;
    if (d.vcheck > 0) {
        // sclip (if any) is uploaded per frame; its buffer arg + use_scp are static.
        if (scpp) |sp| {
            // sclip is at output resolution; transpose it into column-major for EEDI3H.
            if (d.horizontal) {
                (try s.queue.enqueueWriteBuffer(f32, s.d_scpframe, false, 0, sp, none)).release();
                try runTranspose(s, s.d_scp, s.d_scpframe, d.out_w, d.in_h, d.out_stride, d.stride);
            } else {
                (try s.queue.enqueueWriteBuffer(f32, s.d_scp, false, 0, sp, none)).release();
            }
        }

        // Seed the output buffer with the interp snapshot so kept rows and
        // non-vcheckable interp rows pass through, and so each row's d2p (the
        // previous interp row) reads a valid value before it is overwritten.
        if (cl.c.clEnqueueCopyBuffer(s.queue.handle, s.d_dst.handle, s.d_dst2.handle, 0, 0, d.stride * d.dst_h * @sizeOf(f32), 0, null, null) != cl.c.CL_SUCCESS) {
            return error.EnqueueCopyBuffer;
        }

        // `field` is the only per-frame vcheck arg; the rest are set in init.
        try s.k_vcheck.setArg(c_int, 9, field);

        // Single launch: one workgroup walks the interpolated rows in order,
        // barrier()-ing between them so each row's d2p (the previous interp row,
        // read at x±dir) sees the finished, already-vchecked prior row. Bit-exact
        // with the CPU's in-place feedback, with no per-row launch overhead. A
        // large workgroup keeps enough warps resident to hide the serial inter-row
        // dependency latency on the single SM this (necessarily) runs on.
        const vc_wg: usize = @min(@as(usize, 1024), d.max_wg);
        const vc_gws: [1]usize = .{vc_wg};
        const vc_lws: [1]usize = .{vc_wg};
        (try s.queue.enqueueNDRangeKernel(s.k_vcheck, null, &vc_gws, &vc_lws, none)).release();
        result = s.d_dst2;
    }

    // download (transpose back to row-major for EEDI3H)
    if (d.horizontal) {
        try runTranspose(s, s.d_outframe, result, d.w, d.dst_h, d.stride, d.out_stride);
        const rd = try s.queue.enqueueReadBuffer(f32, s.d_outframe, false, 0, dstp, none);
        try cl.waitForEvents(&.{rd});
        rd.release();
    } else {
        const rd = try s.queue.enqueueReadBuffer(f32, result, false, 0, dstp, none);
        try cl.waitForEvents(&.{rd});
        rd.release();
    }
}

fn runTranspose(s: *Stream, dst: cl.Buffer(f32), src: cl.Buffer(f32), in_w: u32, in_h: u32, in_stride: u32, out_stride: u32) !void {
    try s.k_transpose.setArg(@TypeOf(dst), 0, dst);
    try s.k_transpose.setArg(@TypeOf(src), 1, src);
    try s.k_transpose.setArg(c_int, 2, @intCast(in_w));
    try s.k_transpose.setArg(c_int, 3, @intCast(in_h));
    try s.k_transpose.setArg(c_int, 4, @intCast(in_stride));
    try s.k_transpose.setArg(c_int, 5, @intCast(out_stride));
    const gws: [2]usize = .{ vsh.ceilN(@as(usize, in_w), 16), vsh.ceilN(@as(usize, in_h), 16) };
    const lws: [2]usize = .{ 16, 16 };
    (try s.queue.enqueueNDRangeKernel(s.k_transpose, null, &gws, &lws, &.{})).release();
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

        var plane: u32 = 0;
        while (plane < d.vi.format.numPlanes) : (plane += 1) {
            const srcp = src.getReadSlice2(f32, plane);
            const dstp = dst.getWriteSlice2(f32, plane);
            const scpp: ?[]const f32 = if (scp) |sc| sc.getReadSlice2(f32, plane) else null;
            process(d, s, dstp, srcp, scpp, field) catch |err| {
                zapi.setFilterError("EEDI3: process failed.");
                std.log.err("EEDI3 process failed: {}", .{err});
                dst.deinit();
                return null;
            };
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
    vsapi.?.freeNode.?(d.node);
    vsapi.?.freeNode.?(d.sclip);
    allocator.destroy(d);
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

    if (d.vi.format.sampleType != .Float or d.vi.format.bitsPerSample != 32) {
        map_out.setError("EEDI3: only 32-bit float (GrayS) is supported.");
        return;
    }

    const field = map_in.getValue(i32, "field") orelse 0;
    const mdis = map_in.getValue(i32, "mdis") orelse 20;
    const nrad = map_in.getValue(i32, "nrad") orelse 2;
    d.alpha = map_in.getValue(f32, "alpha") orelse 0.2;
    d.beta = map_in.getValue(f32, "beta") orelse 0.25;
    d.gamma = map_in.getValue(f32, "gamma") orelse 20.0;
    d.dh = map_in.getBool("dh") orelse false;
    d.hp = map_in.getBool("hp") orelse false;

    // the interpolated axis (height for EEDI3, width for EEDI3H)
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

    // Internal pipeline is always vertical. For EEDI3H the buffers are the
    // transposed frame: w = input height, src_h = input width, dst_h = output width.
    if (horizontal) {
        d.in_w = @intCast(vi_in.width);
        d.in_h = @intCast(vi_in.height);
        d.out_w = @intCast(d.vi.width);
        d.w = @intCast(vi_in.height);
        d.src_h = @intCast(vi_in.width);
        d.dst_h = @intCast(d.vi.width);
        d.in_stride = @intCast(vsh.ceilN(@as(usize, d.in_w), 8));
        d.out_stride = @intCast(vsh.ceilN(@as(usize, d.out_w), 8));
    } else {
        d.w = @intCast(vi_in.width);
        d.src_h = @intCast(vi_in.height);
        d.dst_h = @intCast(d.vi.height);
    }

    d.stride = vszipcl.strideFromVi(&d.vi)[0];
    d.n_interp = if (d.dh) d.src_h else d.src_h / 2;
    d.tpitch = @intCast(if (d.hp) 4 * mdis + 1 else 2 * mdis + 1);
    d.lws = vsh.ceilN(@as(usize, d.tpitch), 32);
    // Horizontal mirror-pad margin: covers the max reach of the cost/interp
    // loops (interp tail x±3*mdis, conn cost x±(2*mdis+nrad), hp ±(mdis+nrad+2)).
    // pad = 3*mdis + nrad + 2 is a safe upper bound for all of them.
    d.pad = @intCast(3 * mdis + nrad + 2);
    d.pstride = @intCast(vsh.ceilN(@as(usize, d.pad) * 2 + d.w, 8));

    initOpenCL(&d) catch |err| {
        map_out.setError("EEDI3: OpenCL init failed.");
        std.log.err("EEDI3 OpenCL init failed: {}", .{err});
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
        map_out.setError("EEDI3: OpenCL stream init failed.");
        std.log.err("EEDI3 stream init failed: {}", .{err});
        data.pool.deinit();
        allocator.destroy(data);
        keep = false;
        return;
    };

    var dep_buf: [2]vs.FilterDependency = undefined;
    dep_buf[0] = .{ .source = d.node, .requestPattern = .StrictSpatial };
    var ndeps: usize = 1;
    if (d.sclip) |sc| {
        dep_buf[ndeps] = .{ .source = sc, .requestPattern = .StrictSpatial };
        ndeps += 1;
    }
    zapi.createVideoFilter(out, if (horizontal) "EEDI3H" else "EEDI3", &d.vi, getFrame, free, .Unordered, dep_buf[0..ndeps], data);
}
