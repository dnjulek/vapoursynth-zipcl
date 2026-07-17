#define FMA(a, b, c) ({ float _fm = (a) * (b); _fm + (c); })
#define FMS(a, b, c) ({ float _fm = (a) * (b); _fm - (c); })
#define FNMS(a, b, c) ({ float _fm = (a) * (b); (c) - _fm; })

#define FLT_MAX_ 3.402823466e+38f
#define FLT_EPS_ 1.192092896e-07f
#define INT_MAX_ 0x7fffffff

#define smem_stride 33

#define CONCAT_(a, b) a##b
#define CONCAT(a, b) CONCAT_(a, b)

#if TEMPORAL
#define KRADIUS RADIUS
#else
#define KRADIUS 0
#endif

#define TEMPORAL_WIDTH (2 * KRADIUS + 1)
#define TEMPORAL_STRIDE (HEIGHT * STRIDE)
#define PLANE_STRIDE (TEMPORAL_WIDTH * TEMPORAL_STRIDE)
#define NUM_PLANES (CHROMA ? 3 : 1)
#define CLIP_STRIDE (NUM_PLANES * TEMPORAL_WIDTH * TEMPORAL_STRIDE)

#define THREADS (32 * WARPS)
__attribute__((always_inline)) static void dct_fwd(float *v) {
    const float KP414213562 = +0.414213562373095048801688724209698078569671875;
    const float KP1_847759065 = +1.847759065022573512256366378793576573644833252;
    const float KP198912367 = +0.198912367379658006911597622644676228597850501;
    const float KP1_961570560 = +1.961570560806460898252364472268478073947867462;
    const float KP1_414213562 = +1.414213562373095048801688724209698078569671875;
    const float KP668178637 = +0.668178637919298919997757686523080761552472251;
    const float KP1_662939224 = +1.662939224605090474157576755235811513477121624;
    const float KP707106781 = +0.707106781186547524400844362104849039284835938;

    float T1 = v[0];
    float T2 = v[7];
    float T3 = T1 - T2;
    float Tj = T1 + T2;
    float Tc = v[4];
    float Td = v[3];
    float Te = Tc - Td;
    float Tk = Tc + Td;
    float T4 = v[2];
    float T5 = v[5];
    float T6 = T4 - T5;
    float T7 = v[1];
    float T8 = v[6];
    float T9 = T7 - T8;
    float Ta = T6 + T9;
    float Tn = T7 + T8;
    float Tf = T6 - T9;
    float Tm = T4 + T5;
    float Tb = FNMS(KP707106781, Ta, T3);
    float Tg = FNMS(KP707106781, Tf, Te);
    v[3] = KP1_662939224 * (FMA(KP668178637, Tg, Tb));
    v[5] = -(KP1_662939224 * (FNMS(KP668178637, Tb, Tg)));
    float Tp = Tj + Tk;
    float Tq = Tm + Tn;
    v[4] = KP1_414213562 * (Tp - Tq);
    v[0] = KP1_414213562 * (Tp + Tq);
    float Th = FMA(KP707106781, Ta, T3);
    float Ti = FMA(KP707106781, Tf, Te);
    v[1] = KP1_961570560 * (FNMS(KP198912367, Ti, Th));
    v[7] = KP1_961570560 * (FMA(KP198912367, Th, Ti));
    float Tl = Tj - Tk;
    float To = Tm - Tn;
    v[2] = KP1_847759065 * (FNMS(KP414213562, To, Tl));
    v[6] = KP1_847759065 * (FMA(KP414213562, Tl, To));
}

__attribute__((always_inline)) static void dct_inv(float *v) {
    const float KP1_662939224 = +1.662939224605090474157576755235811513477121624;
    const float KP668178637 = +0.668178637919298919997757686523080761552472251;
    const float KP1_961570560 = +1.961570560806460898252364472268478073947867462;
    const float KP198912367 = +0.198912367379658006911597622644676228597850501;
    const float KP1_847759065 = +1.847759065022573512256366378793576573644833252;
    const float KP707106781 = +0.707106781186547524400844362104849039284835938;
    const float KP414213562 = +0.414213562373095048801688724209698078569671875;
    const float KP1_414213562 = +1.414213562373095048801688724209698078569671875;

    float T1 = v[0] * KP1_414213562;
    float T2 = v[4];
    float T3 = FMA(KP1_414213562, T2, T1);
    float Tj = FNMS(KP1_414213562, T2, T1);
    float T4 = v[2];
    float T5 = v[6];
    float T6 = FMA(KP414213562, T5, T4);
    float Tk = FMS(KP414213562, T4, T5);
    float T8 = v[1];
    float Td = v[7];
    float T9 = v[5];
    float Ta = v[3];
    float Tb = T9 + Ta;
    float Te = Ta - T9;
    float Tc = FMA(KP707106781, Tb, T8);
    float Tn = FNMS(KP707106781, Te, Td);
    float Tf = FMA(KP707106781, Te, Td);
    float Tm = FNMS(KP707106781, Tb, T8);
    float T7 = FMA(KP1_847759065, T6, T3);
    float Tg = FMA(KP198912367, Tf, Tc);
    v[7] = FNMS(KP1_961570560, Tg, T7);
    v[0] = FMA(KP1_961570560, Tg, T7);
    float Tp = FNMS(KP1_847759065, Tk, Tj);
    float Tq = FMA(KP668178637, Tm, Tn);
    v[5] = FNMS(KP1_662939224, Tq, Tp);
    v[2] = FMA(KP1_662939224, Tq, Tp);
    float Th = FNMS(KP1_847759065, T6, T3);
    float Ti = FNMS(KP198912367, Tc, Tf);
    v[3] = FNMS(KP1_961570560, Ti, Th);
    v[4] = FMA(KP1_961570560, Ti, Th);
    float Tl = FMA(KP1_847759065, Tk, Tj);
    float To = FNMS(KP668178637, Tn, Tm);
    v[6] = FNMS(KP1_662939224, To, Tl);
    v[1] = FMA(KP1_662939224, To, Tl);
}

__attribute__((always_inline)) static void haar_fwd(float *v) {
    const float KP1_414213562 = +1.414213562373095048801688724209698078569671875;
    const float KP2_000000000 = +2.000000000000000000000000000000000000000000000;

    float T1 = v[0] + v[1];
    float T2 = v[0] - v[1];
    float T3 = v[2] + v[3];
    float T4 = v[2] - v[3];
    float T5 = v[4] + v[5];
    float T6 = v[4] - v[5];
    float T7 = v[6] + v[7];
    float T8 = v[6] - v[7];

    float T9 = T1 + T3;
    float T10 = KP1_414213562 * (T1 - T3);
    float T11 = T5 + T7;
    float T12 = KP1_414213562 * (T5 - T7);

    float scale = KP1_414213562;
    v[0] = scale * (T9 + T11);
    v[1] = scale * (T9 - T11);
    v[2] = scale * T10;
    v[3] = scale * T12;
    v[4] = scale * KP2_000000000 * T2;
    v[5] = scale * KP2_000000000 * T4;
    v[6] = scale * KP2_000000000 * T6;
    v[7] = scale * KP2_000000000 * T8;
}

__attribute__((always_inline)) static void haar_inv(float *v) {
    const float KP1_414213562 = +1.414213562373095048801688724209698078569671875;
    const float KP2_000000000 = +2.000000000000000000000000000000000000000000000;

    float T1 = v[0] + v[1];
    float T2 = v[0] - v[1];
    float T3 = KP1_414213562 * v[2] + KP2_000000000 * v[4];
    float T4 = KP1_414213562 * v[2] - KP2_000000000 * v[4];
    float T5 = -KP1_414213562 * v[2] + KP2_000000000 * v[4];
    float T6 = -KP1_414213562 * v[2] - KP2_000000000 * v[4];
    float T7 = KP1_414213562 * v[2] + KP2_000000000 * v[4];
    float T8 = KP1_414213562 * v[2] - KP2_000000000 * v[4];
    float T9 = -KP1_414213562 * v[2] + KP2_000000000 * v[4];
    float T10 = -KP1_414213562 * v[2] - KP2_000000000 * v[4];

    float scale = KP1_414213562;
    v[0] = scale * (T1 + T3);
    v[1] = scale * (T1 + T4);
    v[2] = scale * (T1 + T5);
    v[3] = scale * (T1 + T6);
    v[4] = scale * (T2 + T7);
    v[5] = scale * (T2 + T8);
    v[6] = scale * (T2 + T9);
    v[7] = scale * (T2 + T10);
}

__attribute__((always_inline)) static void wht_fwd(float *v) {
    const float KP1_414213562 = +1.414213562373095048801688724209698078569671875;

    float T1 = v[0] + v[1];
    float T2 = v[0] - v[1];
    float T3 = v[2] + v[3];
    float T4 = v[2] - v[3];
    float T5 = v[4] + v[5];
    float T6 = v[4] - v[5];
    float T7 = v[6] + v[7];
    float T8 = v[6] - v[7];

    float T9 = T1 + T3;
    float T10 = T1 - T3;
    float T11 = T2 + T4;
    float T12 = T2 - T4;
    float T13 = T5 + T7;
    float T14 = T5 - T7;
    float T15 = T6 + T8;
    float T16 = T6 - T8;

    float scale = KP1_414213562;
    v[0] = scale * (T9 + T13);
    v[1] = scale * (T9 - T13);
    v[2] = scale * (T10 - T14);
    v[3] = scale * (T10 + T14);
    v[4] = scale * (T12 + T16);
    v[5] = scale * (T12 - T16);
    v[6] = scale * (T11 - T15);
    v[7] = scale * (T11 + T15);
}

__attribute__((always_inline)) static void wht_inv(float *v) {
    const float KP1_414213562 = +1.414213562373095048801688724209698078569671875;

    float T1 = v[0] + v[1];
    float T2 = v[0] - v[1];
    float T3 = v[2] + v[3];
    float T4 = v[2] - v[3];
    float T5 = v[4] + v[5];
    float T6 = v[4] - v[5];
    float T7 = v[6] + v[7];
    float T8 = v[6] - v[7];

    float T9 = T1 + T3;
    float T10 = T1 - T3;
    float T11 = T2 + T4;
    float T12 = T2 - T4;
    float T13 = T5 + T7;
    float T14 = T5 - T7;
    float T15 = T6 + T8;
    float T16 = T6 - T8;

    float scale = KP1_414213562;
    v[0] = scale * (T9 + T13);
    v[1] = scale * (T9 - T13);
    v[2] = scale * (T10 - T14);
    v[3] = scale * (T10 + T14);
    v[4] = scale * (T12 + T16);
    v[5] = scale * (T12 - T16);
    v[6] = scale * (T11 - T15);
    v[7] = scale * (T11 + T15);
}

__attribute__((always_inline)) static void bior1_5_fwd(float *v) {
    const float KP1_414213562 = +1.414213562373095048801688724209698078569671875;
    const float KP877670597 = +0.877670597010003062405456290501163901200537360;
    const float KP1_797135031 = +1.797135031972863413496886690073811797696338403;
    const float KP2_277437593 = +2.277437593371384746027611934034934695963515886;
    const float KP1_609389232 = +1.609389232649111887192845766718020518480884560;
    const float KP334024180 = +0.334024180361136429417383083658457088741315663;
    const float KP2_828427124 = +2.828427124746190097603377448419396157139343751;

    float T1 = v[0] + v[3];
    float T2 = v[0] - v[3];
    float T3 = v[1] + v[2];
    float T4 = v[1] - v[2];
    float T5 = v[4] + v[7];
    float T6 = v[4] - v[7];
    float T7 = v[5] + v[6];
    float T8 = v[5] - v[6];
    float T9 = v[0] - v[1];
    float T10 = v[2] - v[3];
    float T11 = v[4] - v[5];
    float T12 = v[6] - v[7];

    v[0] = KP1_414213562 * (T1 + T5 + T3 + T7);
    v[1] = KP877670597 * (T1 - T5) + KP1_797135031 * (T3 - T7);
    v[2] = KP2_277437593 * T2 + KP1_609389232 * T4 + KP334024180 * (T8 - T6);
    v[3] = KP2_277437593 * T6 + KP1_609389232 * T8 + KP334024180 * (T4 - T2);
    v[4] = KP2_828427124 * T9;
    v[5] = KP2_828427124 * T10;
    v[6] = KP2_828427124 * T11;
    v[7] = KP2_828427124 * T12;
}

__attribute__((always_inline)) static void bior1_5_inv(float *v) {
    const float KP1_414213562 = +1.414213562373095048801688724209698078569671875;
    const float KP1_495435764 = +1.495435764250674860795011090214036706658653686;
    const float KP2_058234225 = +2.058234225009388964222454285384072231477027482;
    const float KP486135912 = +0.486135912065751423025580498947083714508324707;
    const float KP2_828427124 = +2.828427124746190097603377448419396157139343751;

    float T1 = KP1_414213562 * v[0];
    float T2 = KP1_495435764 * v[1];
    float T3 = KP2_058234225 * v[2];
    float T4 = KP2_058234225 * v[3];
    float T5 = KP2_828427124 * v[4];
    float T6 = KP486135912 * v[4];
    float T7 = KP2_828427124 * v[5];
    float T8 = KP486135912 * v[5];
    float T9 = KP2_828427124 * v[6];
    float T10 = KP486135912 * v[6];
    float T11 = KP2_828427124 * v[7];
    float T12 = KP486135912 * v[7];

    float T13 = T1 + T2;
    float T14 = T1 - T2;
    float T15 = T8 - T12;
    float T16 = T6 - T10;

    v[0] = (T13 + T3) + (T5 - T15);
    v[1] = (T13 + T3) - (T5 + T15);
    v[2] = (T13 - T3) + (T16 + T7);
    v[3] = (T13 - T3) + (T16 - T7);
    v[4] = (T14 + T4) + (T15 + T9);
    v[5] = (T14 + T4) + (T15 - T9);
    v[6] = (T14 - T4) - (T16 - T11);
    v[7] = (T14 - T4) - (T16 + T11);
}

#define T2D_FWD CONCAT(TRANSFORM_2D, _fwd)
#define T2D_INV CONCAT(TRANSFORM_2D, _inv)
#define T1D_FWD CONCAT(TRANSFORM_1D, _fwd)
#define T1D_INV CONCAT(TRANSFORM_1D, _inv)

#define TRANSFORM_PACK8(TFN, DATA, STRIDE_, HMSTRIDE_)                 \
    do {                                                               \
        float *_dp = (DATA);                                           \
        _Pragma("unroll")    \
        for (int _it = 0; _it < 8; ++_it, _dp += (HMSTRIDE_)) {        \
            float _v[8];                                               \
            _Pragma("unroll")    \
            for (int _i = 0; _i < 8; ++_i) _v[_i] = _dp[_i * (STRIDE_)]; \
            TFN(_v);                                                   \
            _Pragma("unroll")    \
            for (int _i = 0; _i < 8; ++_i) _dp[_i * (STRIDE_)] = _v[_i]; \
        }                                                              \
    } while (0)

#define TRANSPOSE_PACK8(DATA, BUF, LANE)                                              \
    do {                                                                              \
        float *_dp = (DATA);                                                          \
        _Pragma("unroll")    \
        for (int _it = 0; _it < 8; ++_it, _dp += 8) {                                 \
            barrier(CLK_LOCAL_MEM_FENCE);                                             \
            _Pragma("unroll")    \
            for (int _i = 0; _i < 8; ++_i) BUF[_i * smem_stride + (LANE)] = _dp[_i];  \
            barrier(CLK_LOCAL_MEM_FENCE);                                             \
            _Pragma("unroll")    \
            for (int _i = 0; _i < 8; ++_i)                                            \
                _dp[_i] = BUF[((LANE) % 8) * smem_stride + ((LANE) & -8) + _i];       \
        }                                                                             \
    } while (0)

__attribute__((always_inline)) static float reduce_group8(float x, local float *red, int lane_id) {
    barrier(CLK_LOCAL_MEM_FENCE);
    red[lane_id] = x;
    barrier(CLK_LOCAL_MEM_FENCE);
    const int g = lane_id & -8;
    return ((red[g + 0] + red[g + 1]) + (red[g + 2] + red[g + 3])) +
           ((red[g + 4] + red[g + 5]) + (red[g + 6] + red[g + 7]));
}

__attribute__((always_inline)) static float ssd_pre(const float *cur) {
    return 0.0f;
}

__attribute__((always_inline)) static float ssd_err(const float *cur, float pre, const global float *base, int cx, int cy) {
    float col[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        float errors[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float val = cur[c * 8 + i] - base[(cy + i) * STRIDE + (cx + c)];
            errors[i % 2] += val * val;
        }
        col[c] = errors[0] + errors[1];
    }
    return ((col[0] + col[1]) + (col[2] + col[3])) + ((col[4] + col[5]) + (col[6] + col[7]));
}

__attribute__((always_inline)) static float sad_pre(const float *cur) {
    return 0.0f;
}

__attribute__((always_inline)) static float sad_err(const float *cur, float pre, const global float *base, int cx, int cy) {
    float col[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        float errors[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float val = cur[c * 8 + i] - base[(cy + i) * STRIDE + (cx + c)];
            errors[i % 2] += fabs(val);
        }
        col[c] = errors[0] + errors[1];
    }
    return ((col[0] + col[1]) + (col[2] + col[3])) + ((col[4] + col[5]) + (col[6] + col[7]));
}

__attribute__((always_inline)) static float zmean_of(const float *p) {
    float cs[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        cs[c] = (((p[c * 8 + 0] + p[c * 8 + 1]) + (p[c * 8 + 2] + p[c * 8 + 3])) +
                 ((p[c * 8 + 4] + p[c * 8 + 5]) + (p[c * 8 + 6] + p[c * 8 + 7])));
    }
    return (((cs[0] + cs[1]) + (cs[2] + cs[3])) + ((cs[4] + cs[5]) + (cs[6] + cs[7]))) * (1.0f / 64.f);
}

__attribute__((always_inline)) static float zssd_pre(const float *cur) {
    return zmean_of(cur);
}

__attribute__((always_inline)) static float zssd_err(const float *cur, float center_mean, const global float *base, int cx, int cy) {
    float nb[64];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) nb[c * 8 + i] = base[(cy + i) * STRIDE + (cx + c)];
    }
    float neighbor_mean = zmean_of(nb);
    float dm = center_mean - neighbor_mean;
    float col[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        float errors[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float val = cur[c * 8 + i] - nb[c * 8 + i] - dm;
            errors[i % 2] += val * val;
        }
        col[c] = errors[0] + errors[1];
    }
    return ((col[0] + col[1]) + (col[2] + col[3])) + ((col[4] + col[5]) + (col[6] + col[7]));
}

__attribute__((always_inline)) static float zsad_pre(const float *cur) {
    return zmean_of(cur);
}

__attribute__((always_inline)) static float zsad_err(const float *cur, float center_mean, const global float *base, int cx, int cy) {
    float nb[64];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) nb[c * 8 + i] = base[(cy + i) * STRIDE + (cx + c)];
    }
    float neighbor_mean = zmean_of(nb);
    float dm = center_mean - neighbor_mean;
    float col[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        float errors[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float val = cur[c * 8 + i] - nb[c * 8 + i] - dm;
            errors[i % 2] += fabs(val);
        }
        col[c] = errors[0] + errors[1];
    }
    return ((col[0] + col[1]) + (col[2] + col[3])) + ((col[4] + col[5]) + (col[6] + col[7]));
}

__attribute__((always_inline)) static float znorm_ssd(const float *p) {
    float cd[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        float ssds[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            ssds[i % 2] += p[c * 8 + i] * p[c * 8 + i];
        }
        cd[c] = ssds[0] + ssds[1];
    }
    return ((cd[0] + cd[1]) + (cd[2] + cd[3])) + ((cd[4] + cd[5]) + (cd[6] + cd[7]));
}

__attribute__((always_inline)) static float ssd_norm_pre(const float *cur) {
    return native_sqrt(znorm_ssd(cur));
}

__attribute__((always_inline)) static float ssd_norm_err(const float *cur, float center_norm, const global float *base, int cx, int cy) {
    float nb[64];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) nb[c * 8 + i] = base[(cy + i) * STRIDE + (cx + c)];
    }
    float neighbor_norm = native_sqrt(znorm_ssd(nb));
    float col[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        float errors[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float val = cur[c * 8 + i] * nb[c * 8 + i];
            errors[i % 2] += val;
        }
        col[c] = errors[0] + errors[1];
    }
    float err = ((col[0] + col[1]) + (col[2] + col[3])) + ((col[4] + col[5]) + (col[6] + col[7]));
    float q = native_recip(center_norm * neighbor_norm + FLT_EPS_);
    return FMA(-(2.0f * err), q, 2.0f);
}

#define BM_PRE CONCAT(BM_ERROR, _pre)
#define BM_ERR CONCAT(BM_ERROR, _err)

#define INSERT8(e, xs, ys, sq, err_, cx_, cy_, cs_)                         do {                                                                        if ((err_) < e[7]) {                                                    /* BRANCHLESS (all selects): conditional stores compile as branches around             st.local and pin the arrays in stack memory; unconditional ternary                  assignments promote to registers. Values identical: pos = first slot                with err < e[k] (strict <), shift down, place — the reference network. */         int _pos = 8;                                                           _Pragma("unroll")                                                       for (int _k = 7; _k >= 0; --_k) {                                           _pos = ((err_) < e[_k]) ? _k : _pos;                                }                                                                       _Pragma("unroll")                                                       for (int _k = 7; _k >= 1; --_k) {                                           e[_k] = (_k == _pos) ? (err_) : ((_k > _pos) ? e[_k - 1] : e[_k]);                   xs[_k] = (_k == _pos) ? (cx_) : ((_k > _pos) ? xs[_k - 1] : xs[_k]);                 ys[_k] = (_k == _pos) ? (cy_) : ((_k > _pos) ? ys[_k - 1] : ys[_k]);                 sq[_k] = (_k == _pos) ? (cs_) : ((_k > _pos) ? sq[_k - 1] : sq[_k]);         }                                                                       e[0] = (_pos == 0) ? (err_) : e[0];                                     xs[0] = (_pos == 0) ? (cx_) : xs[0];                                    ys[0] = (_pos == 0) ? (cy_) : ys[0];                                    sq[0] = (_pos == 0) ? (cs_) : sq[0];                            }                                                                   } while (0)

#define GINSERT8(e, xs, ys, zs, err_, cx_, cy_, cz_)                         do {                                                                        if ((err_) < e[7]) {                                                    /* BRANCHLESS (all selects): conditional stores compile as branches around             st.local and pin the arrays in stack memory; unconditional ternary                  assignments promote to registers. Values identical: pos = first slot                with err < e[k] (strict <), shift down, place — the reference network. */         int _pos = 8;                                                           _Pragma("unroll")                                                       for (int _k = 7; _k >= 0; --_k) {                                           _pos = ((err_) < e[_k]) ? _k : _pos;                                }                                                                       _Pragma("unroll")                                                       for (int _k = 7; _k >= 1; --_k) {                                           e[_k] = (_k == _pos) ? (err_) : ((_k > _pos) ? e[_k - 1] : e[_k]);                   xs[_k] = (_k == _pos) ? (cx_) : ((_k > _pos) ? xs[_k - 1] : xs[_k]);                 ys[_k] = (_k == _pos) ? (cy_) : ((_k > _pos) ? ys[_k - 1] : ys[_k]);                 zs[_k] = (_k == _pos) ? (cz_) : ((_k > _pos) ? zs[_k - 1] : zs[_k]);         }                                                                       e[0] = (_pos == 0) ? (err_) : e[0];                                     xs[0] = (_pos == 0) ? (cx_) : xs[0];                                    ys[0] = (_pos == 0) ? (cy_) : ys[0];                                    zs[0] = (_pos == 0) ? (cz_) : zs[0];                            }                                                                   } while (0)

__attribute__((always_inline)) static void merge_group64(local const float *le, local const int *lx, local const int *ly,
                          local const int *ls, int base, float *me, int *mx, int *my, int *ms) {
    int head[8];
    #pragma unroll
    for (int l = 0; l < 8; ++l) head[l] = 0;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        int bl = 0;
        int bh = 0;
        float be = FLT_MAX_;
        int bs = INT_MAX_;
        int found = 0;
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            int hv = head[l];
            int valid = hv < 8;
            int idx = base + l * 8 + (valid ? hv : 7);
            float e = valid ? le[idx] : FLT_MAX_;
            int sq = valid ? ls[idx] : INT_MAX_;
            int better = valid && (!found || e < be || (e == be && sq < bs));
            bl = better ? l : bl;
            bh = better ? hv : bh;
            be = better ? e : be;
            bs = better ? sq : bs;
            found = found || valid;
        }
        me[k] = be;
        mx[k] = lx[base + bl * 8 + bh];
        my[k] = ly[base + bl * 8 + bh];
        ms[k] = bs;
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            head[l] += (l == bl);
        }
    }
}

__attribute__((always_inline)) static float hard_thresholding_s(float *data, float sigma, local float *red, int lane_id) {
    float ks[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        float val = data[i];

        float thr;
        if (i == 0) {
            thr = (lane_id % 8) ? sigma : 0.0f;
        } else {
            thr = sigma;
        }

        float flag = fabs(val) >= thr;

        ks[i % 4] += flag;
        data[i] = (flag != 0.0f) ? (val * (1.0f / 4096.0f)) : 0.0f;
    }

    float k = (ks[0] + ks[1]) + (ks[2] + ks[3]);
    k = reduce_group8(k, red, lane_id);

    return native_recip(k);
}

__attribute__((always_inline)) static float collaborative_hard(float *denoising_patch, float sigma, local float *buffer,
                                local float *red, int lane_id) {
    for (int ndim = 0; ndim < 2; ++ndim) {
        TRANSFORM_PACK8(T2D_FWD, denoising_patch, 1, 8);
        TRANSPOSE_PACK8(denoising_patch, buffer, lane_id);
    }
    TRANSFORM_PACK8(T1D_FWD, denoising_patch, 8, 1);

    float adaptive_weight = hard_thresholding_s(denoising_patch, sigma, red, lane_id);

    for (int ndim = 0; ndim < 2; ++ndim) {
        TRANSFORM_PACK8(T2D_INV, denoising_patch, 1, 8);
        TRANSPOSE_PACK8(denoising_patch, buffer, lane_id);
    }
    TRANSFORM_PACK8(T1D_INV, denoising_patch, 8, 1);

    return adaptive_weight;
}

#pragma OPENCL FP_CONTRACT OFF
__attribute__((always_inline)) static float wiener_coeff(float ref_val, float sigma) {
    return native_divide(ref_val * ref_val, ref_val * ref_val + sigma * sigma);
}

#pragma OPENCL FP_CONTRACT ON

__attribute__((always_inline)) static float wiener_filtering_s(float *data, float *ref, float sigma, local float *red, int lane_id) {
    float ks[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        float val = data[i];
        float ref_val = ref[i];
        float coeff = wiener_coeff(ref_val, sigma);
        if (i == 0) {
            coeff = (lane_id % 8) ? coeff : 1.0f;
        }
        val *= coeff;
        ks[i % 4] += coeff * coeff;
        data[i] = val * (1.0f / 4096.0f);
    }

    float k = (ks[0] + ks[1]) + (ks[2] + ks[3]);
    k = reduce_group8(k, red, lane_id);

    return native_recip(k);
}

__attribute__((always_inline)) static float collaborative_wiener(float *denoising_patch, float *ref_patch, float sigma,
                                  local float *buffer, local float *red, int lane_id) {
    for (int ndim = 0; ndim < 2; ++ndim) {
        TRANSFORM_PACK8(T2D_FWD, denoising_patch, 1, 8);
        TRANSPOSE_PACK8(denoising_patch, buffer, lane_id);
    }
    TRANSFORM_PACK8(T1D_FWD, denoising_patch, 8, 1);

    for (int ndim = 0; ndim < 2; ++ndim) {
        TRANSFORM_PACK8(T2D_FWD, ref_patch, 1, 8);
        TRANSPOSE_PACK8(ref_patch, buffer, lane_id);
    }
    TRANSFORM_PACK8(T1D_FWD, ref_patch, 8, 1);

    float adaptive_weight = wiener_filtering_s(denoising_patch, ref_patch, sigma, red, lane_id);

    for (int ndim = 0; ndim < 2; ++ndim) {
        TRANSFORM_PACK8(T2D_INV, denoising_patch, 1, 8);
        TRANSPOSE_PACK8(denoising_patch, buffer, lane_id);
    }
    TRANSFORM_PACK8(T1D_INV, denoising_patch, 8, 1);

    return adaptive_weight;
}

__attribute__((always_inline)) static void atom_add_f(volatile global float *p, float val) {
    volatile global uint *up = (volatile global uint *)p;
    uint old = *up;
    uint assumed;
    do {
        assumed = old;
        old = atomic_cmpxchg(up, assumed, as_uint(as_float(assumed) + val));
    } while (old != assumed);
}

__attribute__((reqd_work_group_size(THREADS, 1, 1)))
kernel void bm3d(
    /* shape: [NUM_PLANES, TEMPORAL_WIDTH, 2, HEIGHT, STRIDE] at res_off */
    global float * restrict res0,
    /* shape: [(FINAL ? 2 : 1), NUM_PLANES, TEMPORAL_WIDTH, HEIGHT, STRIDE] at src_off */
    global const float * restrict src0,
    const int res_off,
    const int src_off
) {
    global float * restrict res = res0 + res_off;
    global const float * restrict src = src0 + src_off;

    local float buffer_all[WARPS][8 * smem_stride];
    local float red_all[WARPS][32];
    local float l_e[WARPS][4][64];
    local int l_x[WARPS][4][64];
    local int l_y[WARPS][4][64];
    local int l_s[WARPS][4][64];

    const int lid = (int)get_local_id(0);
    const int warp_id = lid >> 5;
    const int lane_id = lid & 31;
    const int group = lane_id >> 3;
    const int sub_lane_id = lane_id & 7;
    local float *const buffer = buffer_all[warp_id];
    local float *const red = red_all[warp_id];
    local float *const ge_l = l_e[warp_id][group];
    local int *const gx_l = l_x[warp_id][group];
    local int *const gy_l = l_y[warp_id][group];
    local int *const gs_l = l_s[warp_id][group];

    const int gid = (int)get_group_id(0) * WARPS + warp_id;

    int x = (4 * gid + group) * BLOCK_STEP;
    int y = BLOCK_STEP * (int)get_group_id(1);
    if (y >= HEIGHT - 8 + BLOCK_STEP) {
        return;
    }
    const int active = x < WIDTH - 8 + BLOCK_STEP;
    x = min(active ? x : 0, WIDTH - 8);
    y = min(y, HEIGHT - 8);

    const global float *const srcpc = &src[KRADIUS * TEMPORAL_STRIDE];
    float cur[64];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            cur[c * 8 + i] = srcpc[(y + i) * STRIDE + (x + c)];
        }
    }
    const float center_pre = BM_PRE(cur);

    float ge[8];
    int gx[8], gy[8], gz[8];
    {
        float pe[8];
        int px[8], py[8], ps[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            pe[k] = FLT_MAX_;
            px[k] = 0;
            py[k] = 0;
            ps[k] = INT_MAX_;
        }
        const int left = max(x - BM_RANGE, 0);
        const int right = min(x + BM_RANGE, WIDTH - 8);
        const int top = max(y - BM_RANGE, 0);
        const int bottom = min(y + BM_RANGE, HEIGHT - 8);
        const int rw = right - left + 1;
        const int total = (bottom - top + 1) * rw;
        for (int s = sub_lane_id; s < total; s += 8) {
            const int row_i = top + s / rw;
            const int col_i = left + s % rw;
            const float error = BM_ERR(cur, center_pre, srcpc, col_i, row_i);
            INSERT8(pe, px, py, ps, error, col_i, row_i, s);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            ge_l[sub_lane_id * 8 + k] = pe[k];
            gx_l[sub_lane_id * 8 + k] = px[k];
            gy_l[sub_lane_id * 8 + k] = py[k];
            gs_l[sub_lane_id * 8 + k] = ps[k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int gseq[8];
        merge_group64(ge_l, gx_l, gy_l, gs_l, 0, ge, gx, gy, gseq);
        #pragma unroll
        for (int k = 0; k < 8; ++k) gz[k] = KRADIUS;
    }

#if TEMPORAL
    {
        int sx[8], sy[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            sx[k] = gx[k];
            sy[k] = gy[k];
        }

        #pragma unroll
        for (int direction = -1; direction <= 1; direction += 2) {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (sub_lane_id == 0) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    gx_l[k] = sx[k];
                    gy_l[k] = sy[k];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int t = 1; t <= KRADIUS; ++t) {
                const int temporal_index = KRADIUS + direction * t;
                const global float *const temporal_srcpc = &src[temporal_index * TEMPORAL_STRIDE];

                int cxs[PS_NUM], cys[PS_NUM];
                #pragma unroll
                for (int i = 0; i < PS_NUM; ++i) {
                    cxs[i] = gx_l[i];
                    cys[i] = gy_l[i];
                }

                float pe[8];
                int px[8], py[8], ps[8];
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    pe[k] = FLT_MAX_;
                    px[k] = 0;
                    py[k] = 0;
                    ps[k] = INT_MAX_;
                }
                #pragma unroll
                for (int i = 0; i < PS_NUM; ++i) {
                    const int left = max(cxs[i] - PS_RANGE, 0);
                    const int right = min(cxs[i] + PS_RANGE, WIDTH - 8);
                    const int top = max(cys[i] - PS_RANGE, 0);
                    const int bottom = min(cys[i] + PS_RANGE, HEIGHT - 8);
                    const int rw = right - left + 1;
                    const int total = (bottom - top + 1) * rw;
                    const int seq_base = i * ((2 * PS_RANGE + 1) * (2 * PS_RANGE + 1));
                    for (int s = sub_lane_id; s < total; s += 8) {
                        const int row_i = top + s / rw;
                        const int col_i = left + s % rw;
                        const float error = BM_ERR(cur, center_pre, temporal_srcpc, col_i, row_i);
                        INSERT8(pe, px, py, ps, error, col_i, row_i, seq_base + s);
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE);
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    ge_l[sub_lane_id * 8 + k] = pe[k];
                    gx_l[sub_lane_id * 8 + k] = px[k];
                    gy_l[sub_lane_id * 8 + k] = py[k];
                    gs_l[sub_lane_id * 8 + k] = ps[k];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                if (sub_lane_id == 0) {
                    float fe[8];
                    int fx[8], fy[8], fs[8];
                    merge_group64(ge_l, gx_l, gy_l, gs_l, 0, fe, fx, fy, fs);
                    #pragma unroll
                    for (int i = 0; i < PS_NUM; ++i) {
                        GINSERT8(ge, gx, gy, gz, fe[i], fx[i], fy[i], temporal_index);
                    }
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        gx_l[k] = fx[k];
                        gy_l[k] = fy[k];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
    }
#endif

    barrier(CLK_LOCAL_MEM_FENCE);
    if (sub_lane_id == 0) {
        int match = 0;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int m = (gx[k] == x) && (gy[k] == y);
#if TEMPORAL
            m = m && (gz[k] == KRADIUS);
#endif
            match += m;
        }
        if (!match) {
            #pragma unroll
            for (int k = 7; k >= 1; --k) {
                gx[k] = gx[k - 1];
                gy[k] = gy[k - 1];
                gz[k] = gz[k - 1];
            }
            gx[0] = x;
            gy[0] = y;
            gz[0] = KRADIUS;
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            gx_l[k] = gx[k];
            gy_l[k] = gy[k];
            gs_l[k] = gz[k];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float denoising_patch[64];
#if FINAL
    float ref_patch[64];
#endif

    #pragma unroll
    for (int plane = 0; plane < NUM_PLANES; ++plane) {
        float sigma;
        if (plane == 0) {
            sigma = SIGMA_Y;
        } else if (plane == 1) {
            sigma = SIGMA_U;
        } else {
            sigma = SIGMA_V;
        }

#if CHROMA
        if (sigma < FLT_EPS_) {
            continue;
        }
#endif

        const global float *const splane = &src[plane * PLANE_STRIDE];
        global float *const rplane = &res[plane * 2 * TEMPORAL_STRIDE * TEMPORAL_WIDTH];

        float adaptive_weight;
#if FINAL
        {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int tmp_x = gx_l[i];
                const int tmp_y = gy_l[i];
                const int tmp_z = gs_l[i];
                const global float *refp = &splane[tmp_z * TEMPORAL_STRIDE + tmp_y * STRIDE + tmp_x + sub_lane_id];
                const global float *srcp = &refp[CLIP_STRIDE];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    ref_patch[i * 8 + j] = refp[j * STRIDE];
                    denoising_patch[i * 8 + j] = srcp[j * STRIDE];
                }
            }
            adaptive_weight = collaborative_wiener(denoising_patch, ref_patch, sigma, buffer, red, lane_id);
        }
#else
        {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int tmp_x = gx_l[i];
                const int tmp_y = gy_l[i];
                const int tmp_z = gs_l[i];
                const global float *srcp = &splane[tmp_z * TEMPORAL_STRIDE + tmp_y * STRIDE + tmp_x + sub_lane_id];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    denoising_patch[i * 8 + j] = srcp[j * STRIDE];
                }
            }
            adaptive_weight = collaborative_hard(denoising_patch, sigma, buffer, red, lane_id);
        }
#endif

        if (active) {
            volatile global float *const wdstpc = (volatile global float *)&rplane[sub_lane_id];
            volatile global float *const weightpc = (volatile global float *)&rplane[TEMPORAL_STRIDE + sub_lane_id];

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int tmp_x = gx_l[i];
                const int tmp_y = gy_l[i];
                const int tmp_z = gs_l[i];
                const int offset = tmp_z * 2 * TEMPORAL_STRIDE + tmp_y * STRIDE + tmp_x;

                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float wdst_val = (adaptive_weight * denoising_patch[i * 8 + j] + EXTRACTOR) - EXTRACTOR;
                    float weight_val = (adaptive_weight + EXTRACTOR) - EXTRACTOR;

                    atom_add_f(&wdstpc[offset + j * STRIDE], wdst_val);
                    atom_add_f(&weightpc[offset + j * STRIDE], weight_val);
                }
            }
        }
    }
}

kernel void aggregate(
    /* [NUM_PLANES, HEIGHT, STRIDE] at dst_off */
    global float * restrict dst0,
    /* [NUM_PLANES, 2, HEIGHT, STRIDE] at res_off */
    global const float * restrict res0,
    const int dst_off,
    const int res_off
) {
    global float * restrict dst = dst0 + dst_off;
    global const float * restrict res = res0 + res_off;

    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= WIDTH || y >= HEIGHT) {
        return;
    }

    const int plane = (int)get_global_id(2);
    if (!((PROC_MASK >> plane) & 1)) {
        return;
    }

    const global float *wdst = &res[plane * 2 * TEMPORAL_STRIDE];
    const global float *weight = &wdst[TEMPORAL_STRIDE];
    global float *dstp = &dst[plane * TEMPORAL_STRIDE];

    const int i = y * STRIDE + x;
    dstp[i] = wdst[i] / weight[i];
}
