const std = @import("std");
const builtin = @import("builtin");
const vszipcl = @import("vszipcl.zig");
const clpool = @import("clpool.zig");

const cl = vszipcl.cl;
const math = std.math;
const vapoursynth = vszipcl.vapoursynth;
const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;

const libm = struct {
    extern fn powf(x: f32, y: f32) f32;
    extern fn expf(x: f32) f32;
    extern fn log2f(x: f32) f32;
};

const besselJ1 = if (builtin.os.tag == .windows)
    struct {
        extern fn _j1(x: f64) f64;
    }._j1
else
    struct {
        extern fn j1(x: f64) f64;
    }.j1;

const SCALER_LUT_SIZE = 256;
const POLAR_CUTOFF: f32 = 1e-3;

const FilterCtx = struct {
    radius: f32,
    params: [2]f32,
};

const WeightFn = *const fn (f: *const FilterCtx, x: f64) f64;

const FilterFunction = struct {
    name: []const u8,
    weight: WeightFn,
    radius: f32,
    resizable: bool = false,
    params: [2]f32 = .{ 0, 0 },
    tunable: [2]bool = .{ false, false },
};

fn wBox(_: *const FilterCtx, _: f64) f64 {
    return 1.0;
}

fn wTriangle(f: *const FilterCtx, x: f64) f64 {
    return 1.0 - x / @as(f64, f.radius);
}

fn wCosine(_: *const FilterCtx, x: f64) f64 {
    return @cos(x);
}

fn wHann(_: *const FilterCtx, x: f64) f64 {
    return 0.5 + 0.5 * @cos(math.pi * x);
}

fn wHamming(_: *const FilterCtx, x: f64) f64 {
    return 0.54 + 0.46 * @cos(math.pi * x);
}

fn wWelch(_: *const FilterCtx, x: f64) f64 {
    return 1.0 - x * x;
}

fn besselI0(x: f64) f64 {
    var s: f64 = 1.0;
    const y = x * x / 4.0;
    var t: f64 = y;
    var i: i32 = 2;
    while (t > 1e-12) {
        s += t;
        t *= y / @as(f64, @floatFromInt(i * i));
        i += 1;
    }
    return s;
}

fn wKaiser(f: *const FilterCtx, x: f64) f64 {
    const alpha = @max(@as(f64, f.params[0]), 0.0);
    const scale = besselI0(alpha);
    return besselI0(alpha * @sqrt(1.0 - x * x)) / scale;
}

fn wBlackman(f: *const FilterCtx, x: f64) f64 {
    const a: f64 = f.params[0];
    const a0 = (1 - a) / 2.0;
    const a1 = 1 / 2.0;
    const a2 = a / 2.0;
    const px = x * math.pi;
    return a0 + a1 * @cos(px) + a2 * @cos(2 * px);
}

fn wBohman(_: *const FilterCtx, x: f64) f64 {
    const pix = math.pi * x;
    return (1.0 - x) * @cos(pix) + @sin(pix) / math.pi;
}

fn wGaussian(f: *const FilterCtx, x: f64) f64 {
    return @exp(-2.0 * x * x / @as(f64, f.params[0]));
}

fn wQuadratic(_: *const FilterCtx, x: f64) f64 {
    if (x < 0.5) {
        return 1.0 - 4.0 / 3.0 * (x * x);
    } else {
        return 2.0 / 3.0 * (x - 1.5) * (x - 1.5);
    }
}

fn wSinc(_: *const FilterCtx, x_in: f64) f64 {
    if (x_in < 1e-8) return 1.0;
    const x = x_in * math.pi;
    return @sin(x) / x;
}

fn wJinc(_: *const FilterCtx, x_in: f64) f64 {
    if (x_in < 1e-8) return 1.0;
    const x = x_in * math.pi;
    return 2.0 * besselJ1(x) / x;
}

fn wSphinx(_: *const FilterCtx, x_in: f64) f64 {
    if (x_in < 1e-8) return 1.0;
    const x = x_in * math.pi;
    return 3.0 * (@sin(x) - x * @cos(x)) / (x * x * x);
}

fn wCubic(f: *const FilterCtx, x: f64) f64 {
    const b: f64 = f.params[0];
    const c: f64 = f.params[1];
    const p0 = 6.0 - 2.0 * b;
    const p2 = -18.0 + 12.0 * b + 6.0 * c;
    const p3 = 12.0 - 9.0 * b - 6.0 * c;
    const q0 = 8.0 * b + 24.0 * c;
    const q1 = -12.0 * b - 48.0 * c;
    const q2 = 6.0 * b + 30.0 * c;
    const q3 = -b - 6.0 * c;
    if (x < 1.0) {
        return (p0 + x * x * (p2 + x * p3)) / p0;
    } else {
        return (q0 + x * (q1 + x * (q2 + x * q3))) / p0;
    }
}

fn wSpline16(_: *const FilterCtx, x: f64) f64 {
    if (x < 1.0) {
        return ((x - 9.0 / 5.0) * x - 1.0 / 5.0) * x + 1.0;
    } else {
        return ((-1.0 / 3.0 * (x - 1) + 4.0 / 5.0) * (x - 1) - 7.0 / 15.0) * (x - 1);
    }
}

fn wSpline36(_: *const FilterCtx, x: f64) f64 {
    if (x < 1.0) {
        return ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0;
    } else if (x < 2.0) {
        return ((-6.0 / 11.0 * (x - 1) + 270.0 / 209.0) * (x - 1) - 156.0 / 209.0) * (x - 1);
    } else {
        return ((1.0 / 11.0 * (x - 2) - 45.0 / 209.0) * (x - 2) + 26.0 / 209.0) * (x - 2);
    }
}

fn wSpline64(_: *const FilterCtx, x: f64) f64 {
    if (x < 1.0) {
        return ((49.0 / 41.0 * x - 6387.0 / 2911.0) * x - 3.0 / 2911.0) * x + 1.0;
    } else if (x < 2.0) {
        return ((-24.0 / 41.0 * (x - 1) + 4032.0 / 2911.0) * (x - 1) - 2328.0 / 2911.0) * (x - 1);
    } else if (x < 3.0) {
        return ((6.0 / 41.0 * (x - 2) - 1008.0 / 2911.0) * (x - 2) + 582.0 / 2911.0) * (x - 2);
    } else {
        return ((-1.0 / 41.0 * (x - 3) + 168.0 / 2911.0) * (x - 3) - 97.0 / 2911.0) * (x - 3);
    }
}

const ff_box = FilterFunction{ .name = "box", .weight = wBox, .radius = 1.0, .resizable = true };
const ff_triangle = FilterFunction{ .name = "triangle", .weight = wTriangle, .radius = 1.0, .resizable = true };
const ff_cosine = FilterFunction{ .name = "cosine", .weight = wCosine, .radius = math.pi / 2.0 };
const ff_hann = FilterFunction{ .name = "hann", .weight = wHann, .radius = 1.0 };
const ff_hamming = FilterFunction{ .name = "hamming", .weight = wHamming, .radius = 1.0 };
const ff_welch = FilterFunction{ .name = "welch", .weight = wWelch, .radius = 1.0 };
const ff_kaiser = FilterFunction{ .name = "kaiser", .weight = wKaiser, .radius = 1.0, .params = .{ 2.0, 0 }, .tunable = .{ true, false } };
const ff_blackman = FilterFunction{ .name = "blackman", .weight = wBlackman, .radius = 1.0, .params = .{ 0.16, 0 }, .tunable = .{ true, false } };
const ff_bohman = FilterFunction{ .name = "bohman", .weight = wBohman, .radius = 1.0 };
const ff_gaussian = FilterFunction{ .name = "gaussian", .weight = wGaussian, .radius = 2.0, .resizable = true, .params = .{ 1.0, 0 }, .tunable = .{ true, false } };
const ff_quadratic = FilterFunction{ .name = "quadratic", .weight = wQuadratic, .radius = 1.5 };
const ff_sinc = FilterFunction{ .name = "sinc", .weight = wSinc, .radius = 1.0, .resizable = true };
const ff_jinc = FilterFunction{ .name = "jinc", .weight = wJinc, .radius = 1.2196698912665045, .resizable = true };
const ff_sphinx = FilterFunction{ .name = "sphinx", .weight = wSphinx, .radius = 1.4302966531242027, .resizable = true };
const ff_cubic = FilterFunction{ .name = "cubic", .weight = wCubic, .radius = 2.0, .params = .{ 1.0, 0.0 }, .tunable = .{ true, true } };
const ff_hermite = FilterFunction{ .name = "hermite", .weight = wCubic, .radius = 1.0, .params = .{ 0.0, 0.0 } };
const ff_spline16 = FilterFunction{ .name = "spline16", .weight = wSpline16, .radius = 2.0 };
const ff_spline36 = FilterFunction{ .name = "spline36", .weight = wSpline36, .radius = 3.0 };
const ff_spline64 = FilterFunction{ .name = "spline64", .weight = wSpline64, .radius = 4.0 };

const USAGE_UP: u8 = 1 << 0;
const USAGE_DOWN: u8 = 1 << 1;
const USAGE_MIXING: u8 = 1 << 2;
const USAGE_SCALING: u8 = USAGE_UP | USAGE_DOWN;
const USAGE_ALL: u8 = USAGE_SCALING | USAGE_MIXING;

const M_SQRT2: f64 = 1.41421356237309504880;
const JINC_ZERO3: f32 = 3.2383154841662362076499;
const JINC_ZERO4: f32 = 4.2410628637960698819573;

const FilterConfig = struct {
    name: []const u8,
    kernel: *const FilterFunction,
    window: ?*const FilterFunction = null,
    radius: f32 = 0,
    params: [2]f32 = .{ 0, 0 },
    wparams: [2]f32 = .{ 0, 0 },
    clamp: f32 = 0,
    blur: f32 = 0,
    taper: f32 = 0,
    antiring: f32 = 0,
    polar: bool = false,
    allowed: u8,
};

const robidoux_b: f32 = @floatCast(12.0 / (19.0 + 9.0 * M_SQRT2));
const robidoux_c: f32 = @floatCast(113.0 / (58.0 + 216.0 * M_SQRT2));
const robidouxsharp_b: f32 = @floatCast(6.0 / (13.0 + 7.0 * M_SQRT2));
const robidouxsharp_c: f32 = @floatCast(7.0 / (2.0 + 12.0 * M_SQRT2));

const filter_configs = [_]FilterConfig{
    .{ .name = "bilinear", .kernel = &ff_triangle, .allowed = USAGE_ALL },
    .{ .name = "triangle", .kernel = &ff_triangle, .allowed = USAGE_SCALING },
    .{ .name = "linear", .kernel = &ff_triangle, .allowed = USAGE_MIXING },
    .{ .name = "nearest", .kernel = &ff_box, .radius = 0.5, .allowed = USAGE_UP },
    .{ .name = "spline16", .kernel = &ff_spline16, .allowed = USAGE_ALL },
    .{ .name = "spline36", .kernel = &ff_spline36, .allowed = USAGE_ALL },
    .{ .name = "spline64", .kernel = &ff_spline64, .allowed = USAGE_ALL },
    .{ .name = "lanczos", .kernel = &ff_sinc, .window = &ff_sinc, .radius = 3.0, .allowed = USAGE_ALL },
    .{ .name = "ewa_lanczos", .kernel = &ff_jinc, .window = &ff_jinc, .radius = JINC_ZERO3, .polar = true, .allowed = USAGE_SCALING },
    .{ .name = "ewa_lanczossharp", .kernel = &ff_jinc, .window = &ff_jinc, .radius = JINC_ZERO3, .blur = 0.98125058372237073562493, .polar = true, .allowed = USAGE_SCALING },
    .{ .name = "ewa_lanczos4sharpest", .kernel = &ff_jinc, .window = &ff_jinc, .radius = JINC_ZERO4, .blur = 0.88451209326050047745788, .antiring = 0.8, .polar = true, .allowed = USAGE_SCALING },
    .{ .name = "bicubic", .kernel = &ff_cubic, .params = .{ 1.0, 0.0 }, .allowed = USAGE_SCALING },
    .{ .name = "cubic", .kernel = &ff_cubic, .params = .{ 1.0, 0.0 }, .allowed = USAGE_MIXING },
    .{ .name = "hermite", .kernel = &ff_hermite, .allowed = USAGE_ALL },
    .{ .name = "gaussian", .kernel = &ff_gaussian, .params = .{ 1.0, 0 }, .allowed = USAGE_ALL },
    .{ .name = "mitchell", .kernel = &ff_cubic, .params = .{ 1.0 / 3.0, 1.0 / 3.0 }, .allowed = USAGE_ALL },
    .{ .name = "mitchell_clamp", .kernel = &ff_cubic, .params = .{ 1.0 / 3.0, 1.0 / 3.0 }, .clamp = 1.0, .allowed = USAGE_ALL },
    .{ .name = "sinc", .kernel = &ff_sinc, .radius = 2.0, .allowed = USAGE_ALL },
    .{ .name = "ginseng", .kernel = &ff_sinc, .window = &ff_jinc, .radius = 3.0, .allowed = USAGE_ALL },
    .{ .name = "ewa_jinc", .kernel = &ff_jinc, .radius = JINC_ZERO3, .polar = true, .allowed = USAGE_SCALING },
    .{ .name = "ewa_ginseng", .kernel = &ff_jinc, .window = &ff_sinc, .radius = JINC_ZERO3, .polar = true, .allowed = USAGE_SCALING },
    .{ .name = "ewa_hann", .kernel = &ff_jinc, .window = &ff_hann, .radius = JINC_ZERO3, .polar = true, .allowed = USAGE_SCALING },
    .{ .name = "ewa_hanning", .kernel = &ff_jinc, .window = &ff_hann, .radius = JINC_ZERO3, .polar = true, .allowed = USAGE_SCALING },
    .{ .name = "catmull_rom", .kernel = &ff_cubic, .params = .{ 0.0, 0.5 }, .allowed = USAGE_ALL },
    .{ .name = "robidoux", .kernel = &ff_cubic, .params = .{ robidoux_b, robidoux_c }, .allowed = USAGE_ALL },
    .{ .name = "robidouxsharp", .kernel = &ff_cubic, .params = .{ robidouxsharp_b, robidouxsharp_c }, .allowed = USAGE_ALL },
    .{ .name = "ewa_robidoux", .kernel = &ff_cubic, .params = .{ robidoux_b, robidoux_c }, .polar = true, .allowed = USAGE_SCALING },
    .{ .name = "ewa_robidouxsharp", .kernel = &ff_cubic, .params = .{ robidouxsharp_b, robidouxsharp_c }, .polar = true, .allowed = USAGE_SCALING },
};

fn findFilterConfig(name: []const u8) ?*const FilterConfig {
    for (&filter_configs) |*c| {
        if ((c.allowed & USAGE_SCALING) != USAGE_SCALING) continue;
        if (std.mem.eql(u8, name, c.name)) return c;
    }
    return null;
}

const ResolvedConfig = struct {
    kernel: FilterFunction,
    window: ?*const FilterFunction,
    radius: f32,
    params: [2]f32,
    wparams: [2]f32,
    clamp: f32,
    blur: f32,
    taper: f32,
    polar: bool,
};

fn radiusBound(c: *const ResolvedConfig) f32 {
    const r: f32 = if (c.radius != 0 and c.kernel.resizable) c.radius else c.kernel.radius;
    return if (c.blur > 0.0) r * c.blur else r;
}

fn plFilterSample(c: *const ResolvedConfig, x_in: f64) f64 {
    const radius: f32 = radiusBound(c);
    const x = @abs(x_in);
    if (x > @as(f64, radius)) return 0.0;

    var kx: f64 = if (x <= @as(f64, c.taper))
        0.0
    else
        (x - @as(f64, c.taper)) / (1.0 - @as(f64, c.taper / radius));
    if (c.blur > 0.0) kx /= @as(f64, c.blur);

    const kctx = FilterCtx{
        .radius = radius,
        .params = .{
            if (c.kernel.tunable[0]) c.params[0] else c.kernel.params[0],
            if (c.kernel.tunable[1]) c.params[1] else c.kernel.params[1],
        },
    };
    var k = c.kernel.weight(&kctx, kx);

    if (c.window) |w| {
        const wx: f64 = x / @as(f64, radius) * @as(f64, w.radius);
        const wctx = FilterCtx{
            .radius = w.radius,
            .params = .{
                if (w.tunable[0]) c.wparams[0] else w.params[0],
                if (w.tunable[1]) c.wparams[1] else w.params[1],
            },
        };
        k *= w.weight(&wctx, wx);
    }

    return if (k < 0) (1.0 - @as(f64, c.clamp)) * k else k;
}

fn filterCutoffs(c: *const ResolvedConfig, cutoff: f32, out_radius: *f32, out_radius_zero: *f32) void {
    const bound: f32 = radiusBound(c);
    var prev: f32 = 0.0;
    var fprev: f32 = @floatCast(plFilterSample(c, prev));
    var found_root = false;

    const step: f32 = 1e-2;
    var x: f32 = 0.0;
    while (x < bound + step) : (x += step) {
        const fx: f32 = @floatCast(plFilterSample(c, x));
        if ((fprev > cutoff and fx <= cutoff) or (fprev < -cutoff and fx >= -cutoff)) {
            var root: f32 = x - fx * (x - prev) / (fx - fprev);
            root = @min(root, bound);
            out_radius.* = root;
            if (!found_root) out_radius_zero.* = root;
            found_root = true;
        }
        prev = x;
        fprev = fx;
    }

    if (!found_root) {
        out_radius.* = bound;
        out_radius_zero.* = bound;
    }
}

const GeneratedFilter = struct {
    radius: f32,
    radius_zero: f32,
    row_size: i32 = 0,
    row_stride: i32 = 0,
    weights: []f32,

    fn deinit(self: *GeneratedFilter) void {
        allocator.free(self.weights);
    }
};

fn generateFilter(c: *const ResolvedConfig, cutoff: f32, max_row_size: i32) !GeneratedFilter {
    var f: GeneratedFilter = undefined;
    filterCutoffs(c, cutoff, &f.radius, &f.radius_zero);

    if (c.polar) {
        const w = try allocator.alloc(f32, SCALER_LUT_SIZE);
        for (0..SCALER_LUT_SIZE) |i| {
            const x: f32 = f.radius * @as(f32, @floatFromInt(i)) / @as(f32, SCALER_LUT_SIZE - 1);
            w[i] = @floatCast(plFilterSample(c, x));
        }
        f.weights = w;
        return f;
    }

    var row_size: i32 = @intFromFloat(@ceil(f.radius) * 2.0);
    if (max_row_size > 0 and row_size > max_row_size) {
        std.log.warn("Resample: required filter size {d} exceeds the maximum {d}; expect aliasing.", .{ row_size, max_row_size });
        row_size = max_row_size;
    }
    const row_stride: i32 = (row_size + 3) & ~@as(i32, 3);
    f.row_size = row_size;
    f.row_stride = row_stride;

    const n: usize = @intCast(row_stride);
    const w = try allocator.alloc(f32, SCALER_LUT_SIZE * n);
    @memset(w, 0);

    for (0..SCALER_LUT_SIZE) |i| {
        const offset: f64 = @as(f64, @floatFromInt(i)) / @as(f64, SCALER_LUT_SIZE - 1);
        const row = w[i * n ..][0..n];
        var wsum: f64 = 0.0;
        const base: i32 = @divTrunc(row_size, 2) - 1;
        const center: f64 = @as(f64, @floatFromInt(base)) + offset;
        for (0..@intCast(row_size)) |j| {
            const ww = plFilterSample(c, @as(f64, @floatFromInt(j)) - center);
            row[j] = @floatCast(ww);
            wsum += ww;
        }
        for (0..@intCast(row_size)) |j|
            row[j] = @floatCast(@as(f64, row[j]) / wsum);
    }

    if (f.radius == f.radius_zero) {
        for (0..SCALER_LUT_SIZE) |i| {
            const row = w[i * n ..][0..n];
            var j: usize = 0;
            while (j < @as(usize, @intCast(row_size))) : (j += 2) {
                const w0 = row[j];
                const w1 = row[j + 1];
                row[j] = w0 + w1;
                row[j + 1] = w1 / (w0 + w1);
            }
            while (j < n) : (j += 1)
                row[j] = if (j >= 4) row[j - 4] else 0;
        }
    }

    f.weights = w;
    return f;
}

const PL_COLOR_SDR_WHITE: f32 = 203.0;
const PL_COLOR_HDR_BLACK: f32 = 1e-6;
const PL_COLOR_HLG_PEAK: f32 = 1000.0;
const PL_COLOR_SDR_CONTRAST: f32 = 1000.0;

const TRC_UNKNOWN = 0;
const TRC_BT_1886 = 1;
const TRC_SRGB = 2;
const TRC_LINEAR = 3;
const TRC_GAMMA18 = 4;
const TRC_GAMMA20 = 5;
const TRC_GAMMA22 = 6;
const TRC_GAMMA24 = 7;
const TRC_GAMMA26 = 8;
const TRC_GAMMA28 = 9;
const TRC_PRO_PHOTO = 10;
const TRC_ST428 = 11;
const TRC_PQ = 12;
const TRC_HLG = 13;
const TRC_V_LOG = 14;
const TRC_S_LOG1 = 15;
const TRC_S_LOG2 = 16;
const TRC_SCRGB = 17;
const TRC_COUNT = 18;

const PQ_M1: f32 = 2610.0 / 4096.0 * 1.0 / 4.0;
const PQ_M2: f32 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f32 = 3424.0 / 4096.0;
const PQ_C2: f32 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f32 = 2392.0 / 4096.0 * 32.0;
const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 0.28466892;
const HLG_C: f32 = 0.55991073;
const HLG_REF: f32 = 1000.0 / PL_COLOR_SDR_WHITE;
const HLG_75: f32 = 3.17955;
const VLOG_B: f32 = 0.00873;
const VLOG_C: f32 = 0.241514;
const VLOG_D: f32 = 0.598206;
const SLOG_A: f32 = 0.432699;
const SLOG_B: f32 = 0.037584;
const SLOG_C: f32 = 0.616596 + 0.03;
const SLOG_P: f32 = 3.538813;
const SLOG_Q: f32 = 0.030001;
const SLOG_K2: f32 = 155.0 / 219.0;

fn nominalPeak(trc: i32) f32 {
    return switch (trc) {
        TRC_SCRGB, TRC_PQ => 10000.0 / PL_COLOR_SDR_WHITE,
        TRC_HLG => 12.0 / HLG_75,
        TRC_V_LOG => 46.0855,
        TRC_S_LOG1 => 6.52,
        TRC_S_LOG2 => 9.212,
        else => 1.0,
    };
}

fn isHdr(trc: i32) bool {
    return nominalPeak(trc) > 1.0;
}

fn hdrNitsToNorm(x_in: f32) f32 {
    if (x_in == 0) return 0;
    return @max(x_in, 0.0) / PL_COLOR_SDR_WHITE;
}

fn hdrPqToNorm(x_in: f32) f32 {
    if (x_in == 0) return 0;
    var x = @max(x_in, 0.0);
    x = libm.powf(x, 1.0 / PQ_M2);
    x = @max(x - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * x);
    x = libm.powf(x, 1.0 / PQ_M1);
    x *= 10000.0;
    return x / PL_COLOR_SDR_WHITE;
}

fn nominalLuma(trc: i32, user_min_luma: f32, out_min: *f32, out_max: *f32) void {
    var min_luma: f32 = hdrNitsToNorm(user_min_luma);
    var max_luma: f32 = 0;

    const hdr_min: f32 = hdrNitsToNorm(PL_COLOR_HDR_BLACK);
    const hdr_max: f32 = hdrPqToNorm(1.0);
    min_luma = if (min_luma != 0) math.clamp(min_luma, hdr_min, hdr_max) else 0;
    if (min_luma >= hdr_max) {
        min_luma = 0;
        max_luma = 0;
    }

    if (max_luma == 0) {
        if (trc == TRC_HLG) {
            max_luma = hdrNitsToNorm(PL_COLOR_HLG_PEAK);
        } else {
            max_luma = nominalPeak(trc);
        }
    }
    if (min_luma == 0) {
        if (isHdr(trc)) {
            min_luma = hdr_min;
        } else {
            const peak: f32 = max_luma * PL_COLOR_SDR_WHITE;
            min_luma = hdrNitsToNorm(peak / PL_COLOR_SDR_CONTRAST);
        }
    }

    out_min.* = min_luma;
    out_max.* = max_luma;
}

fn signalBlack(min: f32) f32 {
    return if (min <= hdrNitsToNorm(PL_COLOR_HDR_BLACK)) 0 else min;
}

fn isBlackScaled(trc: i32) bool {
    return switch (trc) {
        TRC_UNKNOWN, TRC_SRGB, TRC_LINEAR, TRC_GAMMA18, TRC_GAMMA20, TRC_GAMMA22, TRC_GAMMA24, TRC_GAMMA26, TRC_GAMMA28, TRC_PRO_PHOTO, TRC_ST428, TRC_HLG => true,
        else => false,
    };
}

fn bt709LumaCoeffs() [3]f32 {
    const rx: f32 = 0.640;
    const ry: f32 = 0.330;
    const gx: f32 = 0.300;
    const gy: f32 = 0.600;
    const bx: f32 = 0.150;
    const by: f32 = 0.060;
    const wx: f32 = 0.3127;
    const wy: f32 = 0.3290;
    const X = [4]f32{ rx / ry, gx / gy, bx / by, wx / wy };
    const Z = [4]f32{ (1 - rx - ry) / ry, (1 - gx - gy) / gy, (1 - bx - by) / by, (1 - wx - wy) / wy };

    var m: [3][3]f32 = .{ .{ X[0], X[1], X[2] }, .{ 1, 1, 1 }, .{ Z[0], Z[1], Z[2] } };
    {
        const m00: f64 = m[0][0];
        const m01: f64 = m[0][1];
        const m02: f64 = m[0][2];
        const m10: f64 = m[1][0];
        const m11: f64 = m[1][1];
        const m12: f64 = m[1][2];
        const m20: f64 = m[2][0];
        const m21: f64 = m[2][1];
        const m22: f64 = m[2][2];
        const a00 = m11 * m22 - m21 * m12;
        const a01 = -(m01 * m22 - m21 * m02);
        const a02 = m01 * m12 - m11 * m02;
        const a10 = -(m10 * m22 - m20 * m12);
        const a11 = m00 * m22 - m20 * m02;
        const a12 = -(m00 * m12 - m10 * m02);
        const a20 = m10 * m21 - m20 * m11;
        const a21 = -(m00 * m21 - m20 * m01);
        const a22 = m00 * m11 - m10 * m01;
        var det = m00 * a00 + m10 * a01 + m20 * a02;
        det = 1.0 / det;
        m = .{
            .{ @floatCast(det * a00), @floatCast(det * a01), @floatCast(det * a02) },
            .{ @floatCast(det * a10), @floatCast(det * a11), @floatCast(det * a12) },
            .{ @floatCast(det * a20), @floatCast(det * a21), @floatCast(det * a22) },
        };
    }
    var S: [3]f32 = undefined;
    for (0..3) |i|
        S[i] = m[i][0] * X[3] + m[i][1] * 1 + m[i][2] * Z[3];
    return .{ S[0], S[1], S[2] };
}

fn f6(x: f64) f32 {
    var buf: [64]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "{d:.6}", .{x}) catch unreachable;
    return std.fmt.parseFloat(f32, s) catch unreachable;
}

fn foldDiv(a: f64, b: f64) f32 {
    return @floatCast(a / b);
}

const resample_src =
    \\#if BITS == 32
    \\typedef float io_t;
    \\#define LOADI(p, i) ((p)[i])
    \\#define STOREI(p, i, x) ((p)[i] = (x))
    \\#elif BITS == 16 && HALF
    \\typedef half io_t;
    \\#define LOADI(p, i) vload_half((size_t)(i), p)
    \\#define STOREI(p, i, x) vstore_half_rtz((x), (size_t)(i), p)
    \\#elif BITS == 16
    \\typedef ushort io_t;
    \\#define LOADI(p, i) (convert_float((p)[i]) / 65535.0f)
    \\#define STOREI(p, i, x) ((p)[i] = rs_store16(x))
    \\#else
    \\typedef uchar io_t;
    \\#define LOADI(p, i) (convert_float((p)[i]) / 255.0f)
    \\#define STOREI(p, i, x) ((p)[i] = rs_store8(x))
    \\#endif
    \\
    \\#if BITS != 32 && !HALF
    \\inline uchar rs_store8(float x) {
    \\    uint k = (uint)(clamp(x, 0.0f, 1.0f) * 4096.0f);
    \\    return (uchar)((k * 255u + 2047u) >> 12);
    \\}
    \\inline ushort rs_store16(float x) {
    \\    uint k = (uint)(clamp(x, 0.0f, 1.0f) * 1048576.0f);
    \\    return (ushort)(((ulong)k * 65535ul + 524287ul) >> 20);
    \\}
    \\#endif
    \\
    \\#define FRACT(x) ((x) - floor(x))
    \\#define MIX(a, b, t) ((a) + ((b) - (a)) * (t))
    \\
    \\#pragma OPENCL FP_CONTRACT OFF
    \\inline float acc_madd(float acc, float w, float c) { float m = w * c; return acc + m; }
    \\inline float acc_add(float acc, float w) { return acc + w; }
    \\#pragma OPENCL FP_CONTRACT ON
    \\
    \\#ifndef FX8
    \\#define FX8 2
    \\#endif
    \\#if FX8 == 1
    \\#define QFRAC(a) (floor((a) * 256.0f) * (1.0f / 256.0f))
    \\#elif FX8 == 2
    \\#define QFRAC(a) (floor((a) * 256.0f + 0.5f) * (1.0f / 256.0f))
    \\#else
    \\#define QFRAC(a) (a)
    \\#endif
    \\
    \\inline float tex_linear(global const io_t *t, float u, float v, int tw, int th, int tstride) {
    \\    float px = u * (float)tw - 0.5f;
    \\    float py = v * (float)th - 0.5f;
    \\    float fx = floor(px), fy = floor(py);
    \\    float ax = QFRAC(px - fx), ay = QFRAC(py - fy);
    \\    int x0 = clamp((int)fx, 0, tw - 1);
    \\    int x1 = clamp((int)fx + 1, 0, tw - 1);
    \\    int y0 = clamp((int)fy, 0, th - 1);
    \\    int y1 = clamp((int)fy + 1, 0, th - 1);
    \\    float c00 = LOADI(t, y0 * tstride + x0), c10 = LOADI(t, y0 * tstride + x1);
    \\    float c01 = LOADI(t, y1 * tstride + x0), c11 = LOADI(t, y1 * tstride + x1);
    \\    float c0 = MIX(c00, c10, ax);
    \\    float c1 = MIX(c01, c11, ax);
    \\    return MIX(c0, c1, ay);
    \\}
    \\inline float tex_nearest(global const io_t *t, float u, float v, int tw, int th, int tstride) {
    \\    int x = clamp((int)floor(u * (float)tw), 0, tw - 1);
    \\    int y = clamp((int)floor(v * (float)th), 0, th - 1);
    \\    return LOADI(t, y * tstride + x);
    \\}
    \\
    \\inline float4 lut2d(global const float4 *lut, int col, float fcoord, int lutw) {
    \\    float fpos = clamp(fcoord, 0.0f, 1.0f) * 255.0f;
    \\    float fbase = floor(fpos);
    \\    float fr = fpos - fbase;
    \\    int r0 = (int)fbase;
    \\    int r1 = min(r0 + 1, 255);
    \\    float4 a = lut[r0 * lutw + col];
    \\    float4 b = lut[r1 * lutw + col];
    \\    return a + (b - a) * fr;
    \\}
    \\inline float lut1d(global const float *lut, float x) {
    \\    float fpos = clamp(x, 0.0f, 1.0f) * 255.0f;
    \\    float fbase = floor(fpos);
    \\    float fr = fpos - fbase;
    \\    int i0 = (int)fbase;
    \\    int i1 = min(i0 + 1, 255);
    \\    return lut[i0] + (lut[i1] - lut[i0]) * fr;
    \\}
    \\
    \\#define WS_AT(ws, i) ((i) == 0 ? (ws).x : ((i) == 1 ? (ws).y : ((i) == 2 ? (ws).z : (ws).w)))
    \\
    \\#if LINEARIZE
    \\inline float pl_linearize(float v) {
    \\#if TRC != 17
    \\    v = fmax(v, 0.0f);
    \\#endif
    \\#if TRC == 2
    \\    v = (v > 0.04045f) ? native_powr((v + 0.055f) / 1.055f, 2.4f) : v * SRGB_LO;
    \\#elif TRC == 1
    \\    v = BT1886_A * native_powr(v + BT1886_B, 2.4f);
    \\#elif TRC == 0 || TRC == 4 || TRC == 5 || TRC == 6 || TRC == 7 || TRC == 8 || TRC == 9
    \\    v = native_powr(v, GAMMA_EXP);
    \\#elif TRC == 10
    \\    v = (v > 0.03125f) ? native_powr(v, 1.8f) : v * PROPHOTO_LO;
    \\#elif TRC == 11
    \\    v = ST428_S * native_powr(v, 2.6f);
    \\#elif TRC == 12
    \\    v = native_powr(v, PQ_IM2R);
    \\    v = fmax(v - PQ_C1R, 0.0f) / (PQ_C2R - PQ_C3R * v);
    \\    v = native_powr(v, PQ_IM1R);
    \\    v *= PQ_LSCALE;
    \\#elif TRC == 13
    \\    v = HLG_1MB * v + HLG_BB;
    \\    v = (v > 0.5f) ? native_exp((v - HLG_CR) * HLG_IAR) + HLG_BR : 4.0f * v * v;
    \\    v *= 1.0f / 12.0f;
    \\    v *= HLG_CSPMAX * native_powr(fmax(HLG_LUMA_R * v, 0.0f), HLG_YM1);
    \\#elif TRC == 14
    \\    v = (v >= 0.181f) ? native_powr(10.0f, (v - VLOG_DR) * VLOG_ICR) - VLOG_BR : (v - 0.125f) * VLOG_LO;
    \\#elif TRC == 15
    \\    v = native_powr(10.0f, (v - SLOG_CR) * SLOG_IAR) - SLOG_BR;
    \\#elif TRC == 16
    \\    v = (v >= SLOG_QR) ? (native_powr(10.0f, (v - SLOG_CR) * SLOG_IAR) - SLOG_BR) * SLOG_IK2R : (v - SLOG_QR) * SLOG_IPR;
    \\#elif TRC == 17
    \\    v *= SCRGB_L;
    \\#endif
    \\#if LIN_SCALE_OUT
    \\    v = LIN_SA * v + LIN_SB;
    \\#endif
    \\    return v;
    \\}
    \\
    \\inline float pl_delinearize(float v) {
    \\#if DELIN_SCALE
    \\    v = DELIN_SA * v + DELIN_SB;
    \\#endif
    \\#if TRC != 17
    \\    v = fmax(v, 0.0f);
    \\#endif
    \\#if TRC == 2
    \\    v = (v >= 0.0031308f) ? 1.055f * native_powr(v, DELIN_G24) - 0.055f : v * 12.92f;
    \\#elif TRC == 1
    \\    v = native_powr(BT1886_IA * v, DELIN_G24) - BT1886_B;
    \\#elif TRC == 0 || TRC == 4 || TRC == 5 || TRC == 6 || TRC == 7 || TRC == 8 || TRC == 9
    \\    v = native_powr(v, GAMMA_IEXP);
    \\#elif TRC == 10
    \\    v = (v >= 0.001953f) ? native_powr(v, DELIN_G18) : v * 16.0f;
    \\#elif TRC == 11
    \\    v = native_powr(v * ST428_IS, DELIN_G26);
    \\#elif TRC == 12
    \\    v *= PQ_ILSCALE;
    \\    v = native_powr(v, PQ_M1R);
    \\    v = (PQ_C1R + PQ_C2R * v) / (1.0f + PQ_C3R * v);
    \\    v = native_powr(v, PQ_M2R);
    \\#elif TRC == 13
    \\    v *= HLG_ICSPMAX;
    \\    v *= 12.0f * native_powr(fmax(1e-6f, HLG_LUMA_R * v), HLG_IYM1);
    \\    v = (v > 1.0f) ? HLG_AR * native_log(v - HLG_BR) + HLG_CR : 0.5f * sqrt(v);
    \\    v = HLG_OSC * v + HLG_OOF;
    \\#elif TRC == 14
    \\    v = (v >= 0.01f) ? VLOG_CLN * native_log(v + VLOG_BR) + VLOG_DR : 5.6f * v + 0.125f;
    \\#elif TRC == 15
    \\    v = SLOG_ALN * native_log(v + SLOG_BR) + SLOG_CR;
    \\#elif TRC == 16
    \\    v = (v >= 0.0f) ? SLOG_ALN * native_log(SLOG_K2R * v + SLOG_BR) + SLOG_CR : SLOG_PR * v + SLOG_QR;
    \\#elif TRC == 17
    \\    v *= SCRGB_IL;
    \\#endif
    \\    return v;
    \\}
    \\#endif
    \\
    \\#if SIGMOID
    \\inline float pl_sigmoidize(float v) {
    \\    v = clamp(v, 0.0f, 1.0f);
    \\    float sig = v * SIG_SCALE + SIG_OFF;
    \\    return (native_log(sig / (1.0f - sig)) - native_log(SIG_OFF / (1.0f - SIG_OFF))) * SIG_ISLOPE;
    \\}
    \\inline float pl_unsigmoidize(float v) {
    \\    v = clamp(v, 0.0f, 1.0f);
    \\    float bias = native_log(SIG_OFF / (1.0f - SIG_OFF));
    \\    return (1.0f / (1.0f + native_exp(-(v * SIG_SLOPE + bias))) - 1.0f / (1.0f + native_exp(-bias))) * SIG_ISCALE;
    \\}
    \\#endif
    \\
    \\inline float rs_tail(float v) {
    \\#if SIGMOID
    \\    v = pl_unsigmoidize(v);
    \\#endif
    \\#if LINEARIZE
    \\    v = pl_delinearize(v);
    \\#endif
    \\    return v;
    \\}
    \\
    \\#if PREP
    \\kernel __attribute__((reqd_work_group_size(BX, BY, 1)))
    \\void prep(global io_t *dst, global const io_t *src) {
    \\    const int gx = get_global_id(0), gy = get_global_id(1);
    \\    if (gx >= SW || gy >= SH) return;
    \\    float v = LOADI(src, gy * SSTRIDE + gx);
    \\#if LINEARIZE
    \\    v = pl_linearize(v);
    \\#endif
    \\#if SIGMOID
    \\    v = pl_sigmoidize(v);
    \\#endif
    \\    STOREI(dst, gy * SSTRIDE + gx, v);
    \\}
    \\#endif
    \\
    \\#if !POLAR
    \\kernel __attribute__((reqd_work_group_size(BX, BY, 1)))
    \\void pass_v(global io_t *dst, global const io_t *src, global const float4 *lut) {
    \\    const int gx = get_global_id(0), gy = get_global_id(1);
    \\    if (gx >= SW || gy >= DH) return;
    \\    const float posx = MIX(0.0f, VNX1, ((float)gx + 0.5f) * VOSX);
    \\    const float posy = MIX(VNY0, VNY1, ((float)gy + 0.5f) * VOSY);
    \\    const float fcoord = FRACT(posy * (float)SH - 0.5f);
    \\    const float basey = posy - fcoord * VPT - VPT * (float)(NV / 2 - 1);
    \\    float4 ws;
    \\    float wsum = 0.0f;
    \\    float ca = 0.0f;
    \\#if ARV
    \\    float lo = 1e9f, hi = 0.0f;
    \\#endif
    \\    __attribute__((opencl_unroll_hint))
    \\    for (uint n = 0u; n < (uint)NV; n += LINV ? 2u : 1u) {
    \\        if (n % 4u == 0u) ws = lut2d(lut, (int)(n / 4u), fcoord, LUTWV);
    \\        float off = (float)n;
    \\#if LINV
    \\        off += WS_AT(ws, n % 4u + 1u);
    \\#endif
    \\        float c = tex_linear(src, posx, basey + VPT * off, SW, SH, SSTRIDE);
    \\#if ARV
    \\        if (n == (uint)NV / 2u - 1u || n == (uint)NV / 2u) {
    \\            lo = fmin(lo, c);
    \\            hi = fmax(hi, c);
    \\        }
    \\#endif
    \\        ca = acc_madd(ca, WS_AT(ws, n % 4u), c);
    \\        wsum = acc_add(wsum, WS_AT(ws, n % 4u));
    \\    }
    \\    ca /= wsum;
    \\#if ARV
    \\    ca = MIX(ca, clamp(ca, lo, hi), ANTIRING);
    \\#endif
    \\    STOREI(dst, gy * TSTRIDE + gx, ca);
    \\}
    \\
    \\kernel __attribute__((reqd_work_group_size(BX, BY, 1)))
    \\void pass_h(global io_t *dst, global const io_t *src, global const float4 *lut) {
    \\    const int gx = get_global_id(0), gy = get_global_id(1);
    \\    if (gx >= DW || gy >= DH) return;
    \\    const float posx = MIX(HNX0, HNX1, ((float)gx + 0.5f) * HOSX);
    \\    const float posy = MIX(0.0f, HNY1, ((float)gy + 0.5f) * HOSY);
    \\    const float fcoord = FRACT(posx * (float)SW - 0.5f);
    \\    const float basex = posx - fcoord * HPT - HPT * (float)(NH / 2 - 1);
    \\    float4 ws;
    \\    float wsum = 0.0f;
    \\    float ca = 0.0f;
    \\#if ARH
    \\    float lo = 1e9f, hi = 0.0f;
    \\#endif
    \\    __attribute__((opencl_unroll_hint))
    \\    for (uint n = 0u; n < (uint)NH; n += LINH ? 2u : 1u) {
    \\        if (n % 4u == 0u) ws = lut2d(lut, (int)(n / 4u), fcoord, LUTWH);
    \\        float off = (float)n;
    \\#if LINH
    \\        off += WS_AT(ws, n % 4u + 1u);
    \\#endif
    \\        float c = tex_linear(src, basex + HPT * off, posy, SW, DH, TSTRIDE);
    \\#if ARH
    \\        if (n == (uint)NH / 2u - 1u || n == (uint)NH / 2u) {
    \\            lo = fmin(lo, c);
    \\            hi = fmax(hi, c);
    \\        }
    \\#endif
    \\        ca = acc_madd(ca, WS_AT(ws, n % 4u), c);
    \\        wsum = acc_add(wsum, WS_AT(ws, n % 4u));
    \\    }
    \\    ca /= wsum;
    \\#if ARH
    \\    ca = MIX(ca, clamp(ca, lo, hi), ANTIRING);
    \\#endif
    \\    ca = rs_tail(ca);
    \\    STOREI(dst, gy * DSTRIDE + gx, ca);
    \\}
    \\#endif
    \\
    \\#if POLAR
    \\kernel __attribute__((reqd_work_group_size(BX, BY, 1)))
    \\void polar_k(global io_t *dst, global const io_t *src, global const float *lut) {
    \\    const int gx = get_global_id(0), gy = get_global_id(1);
    \\    if (gx >= DW || gy >= DH) return;
    \\    const float posx = MIX(PNX0, PNX1, ((float)gx + 0.5f) * POSX_);
    \\    const float posy = MIX(PNY0, PNY1, ((float)gy + 0.5f) * POSY_);
    \\    const float fcx = FRACT(posx * (float)SW - 0.5f);
    \\    const float fcy = FRACT(posy * (float)SH - 0.5f);
    \\    const float basex = posx - PPTX * fcx;
    \\    const float basey = posy - PPTY * fcy;
    \\    float color = 0.0f, wsum = 0.0f;
    \\#if ARP
    \\    float2 ar = (float2)(0.0f), wwsum = (float2)(0.0f);
    \\#endif
    \\#if TAP_UNROLL
    \\    __attribute__((opencl_unroll_hint))
    \\#endif
    \\    for (int yy = 1 - BOUND; yy <= BOUND; ++yy) {
    \\#if TAP_UNROLL
    \\        __attribute__((opencl_unroll_hint))
    \\#endif
    \\        for (int xx = 1 - BOUND; xx <= BOUND; ++xx) {
    \\            const int y2 = yy > 0 ? yy - 1 : yy;
    \\            const int x2 = xx > 0 ? xx - 1 : xx;
    \\            const float dmin = sqrt((float)(x2 * x2 + y2 * y2));
    \\            if (dmin >= PRADIUS) continue;
    \\            const float dx = (float)xx - fcx, dy = (float)yy - fcy;
    \\            const float d = sqrt(dx * dx + dy * dy);
    \\            if (dmin >= PRADIUS - 1.41421356237309504880f && !(d < PRADIUS)) continue;
    \\            const float w = lut1d(lut, d / PRADIUS);
    \\            wsum += w;
    \\            const float c = tex_nearest(src, basex + PPTX * (float)xx, basey + PPTY * (float)yy, SW, SH, SSTRIDE);
    \\            color += w * c;
    \\#if ARP
    \\            if (dmin < PARRADIUS) {
    \\                if (d <= PARRADIUS) {
    \\                    float2 cc = (float2)(1.0f - c, c);
    \\                    float2 ww = cc + 0.10f;
    \\                    ww = ww * ww;
    \\                    ww = ww * ww;
    \\                    ww = ww * ww;
    \\                    ww = ww * ww;
    \\                    ww = ww * ww;
    \\                    ww = w * ww;
    \\                    ar += ww * cc;
    \\                    wwsum += ww;
    \\                }
    \\            }
    \\#endif
    \\        }
    \\    }
    \\    color = 1.0f / wsum * color;
    \\#if ARP
    \\    float2 wwv = ar / wwsum;
    \\    wwv.x = 1.0f - wwv.x;
    \\    float wcl = clamp(color, wwv.x, wwv.y);
    \\    wcl = (wwv.x > wwv.y) ? (0.5f * wwv.x + 0.5f * wwv.y) : wcl;
    \\    color = MIX(color, wcl, ANTIRING);
    \\#endif
    \\    color = rs_tail(color);
    \\    STOREI(dst, gy * DSTRIDE + gx, color);
    \\}
    \\#endif
;

const OrthoPass = struct {
    n: i32,
    lutw: i32,
    use_linear: bool,
    use_ar: bool,
    lut: []f32,
};

const Config = struct {
    sw: i32,
    sh: i32,
    dw: i32,
    dh: i32,
    sstride: i32,
    dstride: i32,
    tstride: i32,
    sxf: f32,
    syf: f32,
    srcw_eff: f32,
    srch_eff: f32,
    rx: f32,
    ry: f32,
    p_radius: f32 = 0,
    p_radius_zero: f32 = 0,
    p_bound: i32 = 0,
    p_use_ar: bool = false,
    p_lut: []f32 = &.{},
    vert: OrthoPass = undefined,
    horiz: OrthoPass = undefined,

    opts: [:0]u8 = undefined,
};

const Data = struct {
    node: ?*vs.Node,
    vi: *const vs.VideoInfo,
    vi_out: vs.VideoInfo,

    platform: cl.Platform,
    device: cl.Device,
    context: cl.Context,

    bits: i32,
    half: bool,
    bytes: u32,

    polar: bool,
    prep: bool,
    n_cfg: usize,
    cfgs: [2]Config,
    plane_cfg: [3]usize,
    n_planes: usize,

    src_bytes: usize,
    tmp2_bytes: usize,
    dst_bytes: usize,

    width: i32,
    height: i32,
    src_width: f32,
    src_height: f32,

    blk_x: usize,
    blk_y: usize,

    use_pinned: bool,
    stage_src_off: [3]usize,
    stage_dst_off: [3]usize,
    stage_src_sum: usize,
    stage_bytes: usize,
    memcpy_sem: std.Io.Semaphore,

    pool: clpool.Pool(Stream, Data),
};

const Stream = struct {
    programs: [2]cl.Program,
    n_prog: usize,
    queue: cl.CommandQueue,
    d_src: cl.Buffer(u8),
    d_tmp1: ?cl.Buffer(u8),
    d_tmp2: ?cl.Buffer(u8),
    d_dst: cl.Buffer(u8),
    stage: ?cl.Buffer(u8),
    host: []u8,
    luts: [4]cl.Buffer(f32),
    n_lut: usize,
    kerns: [8]cl.Kernel,
    n_kern: usize,
    launches: [8]Launch,
    n_launch: [2]usize,
    launch_off: [2]usize,

    const Launch = struct {
        kern: cl.Kernel,
        gw: usize,
        gh: usize,
    };

    pub fn init(self: *Stream, d: *Data) !void {
        self.n_prog = 0;
        self.n_lut = 0;
        self.n_kern = 0;
        self.d_tmp1 = null;
        self.d_tmp2 = null;
        self.stage = null;

        errdefer {
            var m: usize = 0;
            while (m < self.n_prog) : (m += 1) self.programs[m].release();
        }
        for (0..d.n_cfg) |m| {
            self.programs[m] = try cl.createProgramWithSource(d.context, resample_src);
            self.n_prog = m + 1;
            self.programs[m].build(&.{d.device}, d.cfgs[m].opts) catch |err| {
                if (err == error.BuildProgramFailure) {
                    const log = try self.programs[m].getBuildLog(allocator, d.device);
                    defer allocator.free(log);
                    std.log.err("Resample OpenCL build failed: {s}", .{log});
                }
                return err;
            };
        }

        self.queue = try cl.createCommandQueue(d.context, d.device, .{});
        errdefer self.queue.release();
        self.d_src = try cl.createBuffer(u8, d.context, .{ .read_only = true }, d.src_bytes);
        errdefer self.d_src.release();
        self.d_dst = try cl.createBuffer(u8, d.context, .{ .read_write = true }, d.dst_bytes);
        errdefer self.d_dst.release();

        if (d.use_pinned) blk: {
            const st = cl.createBuffer(u8, d.context, .{ .alloc_host_ptr = true }, d.stage_bytes) catch break :blk;
            var map_err: cl.c.cl_int = cl.c.CL_SUCCESS;
            const map_ptr = cl.c.clEnqueueMapBuffer(self.queue.handle, st.handle, cl.c.CL_TRUE, cl.c.CL_MAP_READ | cl.c.CL_MAP_WRITE, 0, d.stage_bytes, 0, null, null, &map_err);
            if (map_err != cl.c.CL_SUCCESS or map_ptr == null) {
                st.release();
                std.log.warn("Resample: pinned staging unavailable; this stream runs pageable transfers.", .{});
                break :blk;
            }
            self.stage = st;
            self.host = @as([*]u8, @ptrCast(map_ptr.?))[0..d.stage_bytes];
        }
        errdefer if (self.stage) |st| {
            _ = cl.c.clEnqueueUnmapMemObject(self.queue.handle, st.handle, self.host.ptr, 0, null, null);
            _ = cl.c.clFinish(self.queue.handle);
            st.release();
        };

        if (d.prep) self.d_tmp1 = try cl.createBuffer(u8, d.context, .{ .read_write = true }, d.src_bytes);
        errdefer if (self.d_tmp1) |b| b.release();
        if (!d.polar) self.d_tmp2 = try cl.createBuffer(u8, d.context, .{ .read_write = true }, d.tmp2_bytes);
        errdefer if (self.d_tmp2) |b| b.release();

        errdefer {
            var m: usize = 0;
            while (m < self.n_lut) : (m += 1) self.luts[m].release();
        }
        errdefer {
            var m: usize = 0;
            while (m < self.n_kern) : (m += 1) self.kerns[m].release();
        }

        var launch_i: usize = 0;
        for (0..d.n_cfg) |m| {
            const cfg = &d.cfgs[m];
            self.launch_off[m] = launch_i;
            var nl: usize = 0;

            const sampler_src: cl.Buffer(u8) = if (d.prep) self.d_tmp1.? else self.d_src;

            if (d.prep) {
                const k = try cl.createKernel(self.programs[m], "prep");
                self.kerns[self.n_kern] = k;
                self.n_kern += 1;
                try k.setArg(@TypeOf(self.d_tmp1.?), 0, self.d_tmp1.?);
                try k.setArg(@TypeOf(self.d_src), 1, self.d_src);
                self.launches[launch_i + nl] = .{ .kern = k, .gw = @intCast(cfg.sw), .gh = @intCast(cfg.sh) };
                nl += 1;
            }

            if (d.polar) {
                const lut = try self.uploadLut(d, cfg.p_lut);
                const k = try cl.createKernel(self.programs[m], "polar_k");
                self.kerns[self.n_kern] = k;
                self.n_kern += 1;
                try k.setArg(@TypeOf(self.d_dst), 0, self.d_dst);
                try k.setArg(@TypeOf(sampler_src), 1, sampler_src);
                try k.setArg(@TypeOf(lut), 2, lut);
                self.launches[launch_i + nl] = .{ .kern = k, .gw = @intCast(cfg.dw), .gh = @intCast(cfg.dh) };
                nl += 1;
            } else {
                const lut_v = try self.uploadLut(d, cfg.vert.lut);
                const lut_h = try self.uploadLut(d, cfg.horiz.lut);
                const kv = try cl.createKernel(self.programs[m], "pass_v");
                self.kerns[self.n_kern] = kv;
                self.n_kern += 1;
                try kv.setArg(@TypeOf(self.d_tmp2.?), 0, self.d_tmp2.?);
                try kv.setArg(@TypeOf(sampler_src), 1, sampler_src);
                try kv.setArg(@TypeOf(lut_v), 2, lut_v);
                self.launches[launch_i + nl] = .{ .kern = kv, .gw = @intCast(cfg.sw), .gh = @intCast(cfg.dh) };
                nl += 1;
                const kh = try cl.createKernel(self.programs[m], "pass_h");
                self.kerns[self.n_kern] = kh;
                self.n_kern += 1;
                try kh.setArg(@TypeOf(self.d_dst), 0, self.d_dst);
                try kh.setArg(@TypeOf(self.d_tmp2.?), 1, self.d_tmp2.?);
                try kh.setArg(@TypeOf(lut_h), 2, lut_h);
                self.launches[launch_i + nl] = .{ .kern = kh, .gw = @intCast(cfg.dw), .gh = @intCast(cfg.dh) };
                nl += 1;
            }
            self.n_launch[m] = nl;
            launch_i += nl;
        }
    }

    fn uploadLut(self: *Stream, d: *Data, host: []const f32) !cl.Buffer(f32) {
        const buf = try cl.createBuffer(f32, d.context, .{ .read_only = true }, host.len);
        self.luts[self.n_lut] = buf;
        self.n_lut += 1;
        if (cl.c.clEnqueueWriteBuffer(self.queue.handle, buf.handle, cl.c.CL_TRUE, 0, host.len * @sizeOf(f32), host.ptr, 0, null, null) != cl.c.CL_SUCCESS)
            return error.EnqueueWrite;
        return buf;
    }

    pub fn deinit(self: *Stream) void {
        if (self.stage) |st| _ = cl.c.clEnqueueUnmapMemObject(self.queue.handle, st.handle, self.host.ptr, 0, null, null);
        _ = cl.c.clFinish(self.queue.handle);
        var i: usize = 0;
        while (i < self.n_kern) : (i += 1) self.kerns[i].release();
        i = 0;
        while (i < self.n_lut) : (i += 1) self.luts[i].release();
        if (self.stage) |st| st.release();
        if (self.d_tmp2) |b| b.release();
        if (self.d_tmp1) |b| b.release();
        self.d_dst.release();
        self.d_src.release();
        self.queue.release();
        var m: usize = 0;
        while (m < self.n_prog) : (m += 1) self.programs[m].release();
    }
};

const ndr = vszipcl.ndr;

const ZFrame = @typeInfo(@TypeOf(ZAPI.initZFrame)).@"fn".return_type.?;
const ZFrameW = @typeInfo(@TypeOf(ZFrame.newVideoFrame)).@"fn".return_type.?;

fn process(d: *Data, s: *Stream, src: ZFrame, dst: ZFrameW) !void {
    errdefer _ = cl.c.clFinish(s.queue.handle);

    const lws: [2]usize = .{ d.blk_x, d.blk_y };
    for (0..d.n_planes) |p| {
        const cfgi = d.plane_cfg[p];
        const c = &d.cfgs[cfgi];
        const srcp = src.getReadSlice(@intCast(p));
        const dstp = dst.getWriteSlice(@intCast(p));

        if (s.stage != null) {
            const src_region = @as(usize, @intCast(c.sh)) * @as(usize, @intCast(c.sstride)) * d.bytes;
            const dst_region = @as(usize, @intCast(c.dh)) * @as(usize, @intCast(c.dstride)) * d.bytes;
            if (srcp.len != src_region or dstp.len != dst_region) return error.PlaneLayoutMismatch;
            d.memcpy_sem.waitUncancelable(vszipcl.io);
            @memcpy(s.host[d.stage_src_off[p]..][0..srcp.len], srcp);
            d.memcpy_sem.post(vszipcl.io);
            try vszipcl.enqWrite(s.queue, s.d_src.handle, 0, s.host[d.stage_src_off[p]..][0..srcp.len]);
        } else {
            try vszipcl.enqWrite(s.queue, s.d_src.handle, 0, srcp);
        }

        const off = s.launch_off[cfgi];
        for (0..s.n_launch[cfgi]) |k| {
            const L = s.launches[off + k];
            const gws: [2]usize = .{ vszipcl.ceilTo(L.gw, d.blk_x), vszipcl.ceilTo(L.gh, d.blk_y) };
            try ndr(s, L.kern, &gws, &lws);
        }

        if (s.stage != null) {
            try vszipcl.enqRead(s.queue, s.d_dst.handle, 0, s.host[d.stage_src_sum + d.stage_dst_off[p] ..][0..dstp.len]);
        } else {
            try vszipcl.enqRead(s.queue, s.d_dst.handle, 0, dstp);
        }
    }
    if (cl.c.clFinish(s.queue.handle) != cl.c.CL_SUCCESS) return error.Finish;

    if (s.stage != null) {
        d.memcpy_sem.waitUncancelable(vszipcl.io);
        for (0..d.n_planes) |q| {
            const dstp = dst.getWriteSlice(@intCast(q));
            @memcpy(dstp, s.host[d.stage_src_sum + d.stage_dst_off[q] ..][0..dstp.len]);
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
        const dst = src.newVideoFrame3(.{ .width = d.vi_out.width, .height = d.vi_out.height });

        const s = d.pool.acquire();
        defer d.pool.release(s);

        process(d, s, src, dst) catch |err| {
            zapi.setFilterError("Resample: process frame failed.");
            std.log.err("Resample process frame failed: {}", .{err});
            dst.deinit();
            return null;
        };

        propagateSar(d, src, dst);
        return dst.frame;
    }
    return null;
}

fn propagateSar(d: *Data, src: ZFrame, dst: ZFrameW) void {
    const sprops = src.getPropertiesRO();
    const dprops = dst.getPropertiesRW();
    var sar_num: i64 = sprops.getInt(i64, "_SARNum") orelse 0;
    var sar_den: i64 = sprops.getInt(i64, "_SARDen") orelse 0;

    if (sar_num <= 0 or sar_den <= 0) {
        dprops.deleteKey("_SARNum");
        dprops.deleteKey("_SARDen");
        return;
    }

    const width: f32 = @floatFromInt(d.vi.width);
    const height: f32 = @floatFromInt(d.vi.height);
    const dst_width: f32 = @floatFromInt(d.width);
    const dst_height: f32 = @floatFromInt(d.height);

    if (!math.isNan(d.src_width) and d.src_width != width) {
        vsh.muldivRational(&sar_num, &sar_den, @intFromFloat(@round(@as(f64, d.src_width) * 16.0)), @intFromFloat(dst_width * 16.0));
    } else {
        vsh.muldivRational(&sar_num, &sar_den, d.vi.width, @intFromFloat(dst_width));
    }
    if (!math.isNan(d.src_height) and d.src_height != height) {
        vsh.muldivRational(&sar_num, &sar_den, @intFromFloat(dst_height * 16.0), @intFromFloat(@round(@as(f64, d.src_height) * 16.0)));
    } else {
        vsh.muldivRational(&sar_num, &sar_den, @intFromFloat(dst_height), d.vi.height);
    }

    dprops.setInt("_SARNum", sar_num, .Replace);
    dprops.setInt("_SARDen", sar_den, .Replace);
}

fn freeCfgAllocs(d: *Data) void {
    for (0..d.n_cfg) |m| {
        const cfg = &d.cfgs[m];
        if (d.polar) {
            if (cfg.p_lut.len > 0) allocator.free(cfg.p_lut);
        } else {
            allocator.free(cfg.vert.lut);
            allocator.free(cfg.horiz.lut);
        }
        allocator.free(cfg.opts);
    }
}

fn free(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));
    d.pool.deinit();
    d.context.release();
    freeCfgAllocs(d);
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

fn deviceHasCrDivSqrt(device: cl.Device) bool {
    var cfg: cl.c.cl_device_fp_config = 0;
    if (cl.c.clGetDeviceInfo(device.id, cl.c.CL_DEVICE_SINGLE_FP_CONFIG, @sizeOf(cl.c.cl_device_fp_config), &cfg, null) != cl.c.CL_SUCCESS) return false;
    return (cfg & cl.c.CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) != 0;
}

fn orthoMaxRowSize(device: cl.Device) i32 {
    var w: usize = 0;
    var h: usize = 0;
    if (cl.c.clGetDeviceInfo(device.id, cl.c.CL_DEVICE_IMAGE2D_MAX_WIDTH, @sizeOf(usize), &w, null) != cl.c.CL_SUCCESS) w = 0;
    if (cl.c.clGetDeviceInfo(device.id, cl.c.CL_DEVICE_IMAGE2D_MAX_HEIGHT, @sizeOf(usize), &h, null) != cl.c.CL_SUCCESS) h = 0;
    const dim = @min(w, h);
    if (dim == 0 or dim > math.maxInt(i32)) return 8192;
    return @intCast(dim / 4);
}

const OptsW = std.ArrayList(u8);

fn defI(w: *OptsW, name: []const u8, v: i64) void {
    w.print(allocator, " -D{s}={d}", .{ name, v }) catch unreachable;
}

fn defF(w: *OptsW, name: []const u8, v: f32) void {
    w.print(allocator, " -D{s}={x}f", .{ name, v }) catch unreachable;
}

fn buildOpts(d: *Data, cfg: *Config, trc: i32, linearize: bool, sigm: bool, antiring: f32, csp_min: f32, csp_max: f32, sig_offset: f32, sig_scale: f32, sig_slope: f32) ![:0]u8 {
    var w: OptsW = .empty;
    defer w.deinit(allocator);
    try w.appendSlice(allocator, "-cl-std=CL1.2");
    if (deviceHasCrDivSqrt(d.device)) try w.appendSlice(allocator, " -cl-fp32-correctly-rounded-divide-sqrt");

    defI(&w, "BITS", d.bits);
    defI(&w, "HALF", @intFromBool(d.half));
    defI(&w, "POLAR", @intFromBool(d.polar));
    defI(&w, "PREP", @intFromBool(d.prep));
    defI(&w, "LINEARIZE", @intFromBool(linearize));
    defI(&w, "SIGMOID", @intFromBool(sigm));
    defI(&w, "TRC", trc);
    defI(&w, "FX8", 2);
    defF(&w, "ANTIRING", antiring);

    defI(&w, "SW", cfg.sw);
    defI(&w, "SH", cfg.sh);
    defI(&w, "DW", cfg.dw);
    defI(&w, "DH", cfg.dh);
    defI(&w, "SSTRIDE", cfg.sstride);
    defI(&w, "TSTRIDE", cfg.tstride);
    defI(&w, "DSTRIDE", cfg.dstride);

    const inv_sw: f32 = 1.0 / @as(f32, @floatFromInt(cfg.sw));
    const inv_sh: f32 = 1.0 / @as(f32, @floatFromInt(cfg.sh));
    const inv_dw: f32 = 1.0 / @as(f32, @floatFromInt(cfg.dw));
    const inv_dh: f32 = 1.0 / @as(f32, @floatFromInt(cfg.dh));

    if (d.polar) {
        defI(&w, "BOUND", cfg.p_bound);
        defF(&w, "PRADIUS", cfg.p_radius);
        defF(&w, "PARRADIUS", cfg.p_radius_zero);
        defI(&w, "ARP", @intFromBool(cfg.p_use_ar));
        defI(&w, "TAP_UNROLL", @intFromBool(cfg.p_bound <= 8));
        defF(&w, "PNX0", inv_sw * cfg.sxf);
        defF(&w, "PNX1", inv_sw * (cfg.sxf + cfg.srcw_eff));
        defF(&w, "PNY0", inv_sh * cfg.syf);
        defF(&w, "PNY1", inv_sh * (cfg.syf + cfg.srch_eff));
        defF(&w, "POSX_", inv_dw);
        defF(&w, "POSY_", inv_dh);
        defF(&w, "PPTX", inv_sw);
        defF(&w, "PPTY", inv_sh);
    } else {
        defI(&w, "NV", cfg.vert.n);
        defI(&w, "LUTWV", cfg.vert.lutw);
        defI(&w, "LINV", @intFromBool(cfg.vert.use_linear));
        defI(&w, "ARV", @intFromBool(cfg.vert.use_ar));
        defF(&w, "VNX1", inv_sw * @as(f32, @floatFromInt(cfg.sw)));
        defF(&w, "VNY0", inv_sh * cfg.syf);
        defF(&w, "VNY1", inv_sh * (cfg.syf + cfg.srch_eff));
        defF(&w, "VOSX", inv_sw);
        defF(&w, "VOSY", inv_dh);
        defF(&w, "VPT", inv_sh);
        defI(&w, "NH", cfg.horiz.n);
        defI(&w, "LUTWH", cfg.horiz.lutw);
        defI(&w, "LINH", @intFromBool(cfg.horiz.use_linear));
        defI(&w, "ARH", @intFromBool(cfg.horiz.use_ar));
        defF(&w, "HNX0", inv_sw * cfg.sxf);
        defF(&w, "HNX1", inv_sw * (cfg.sxf + cfg.srcw_eff));
        defF(&w, "HNY1", inv_dh * @as(f32, @floatFromInt(cfg.dh)));
        defF(&w, "HOSX", inv_dw);
        defF(&w, "HOSY", inv_dh);
        defF(&w, "HPT", inv_sw);
    }

    if (linearize) {
        const cmin: f64 = csp_min;
        const cmax: f64 = csp_max;
        switch (trc) {
            TRC_SRGB => defF(&w, "SRGB_LO", foldDiv(1.0, 12.92)),
            TRC_BT_1886 => {
                const lb = libm.powf(csp_min, 1.0 / 2.4);
                const lw = libm.powf(csp_max, 1.0 / 2.4);
                const a = libm.powf(lw - lb, 2.4);
                const b = lb / (lw - lb);
                defF(&w, "BT1886_A", a);
                defF(&w, "BT1886_B", b);
                defF(&w, "BT1886_IA", @floatCast(1.0 / @as(f64, a)));
            },
            TRC_UNKNOWN, TRC_GAMMA22 => {
                defF(&w, "GAMMA_EXP", 2.2);
                defF(&w, "GAMMA_IEXP", foldDiv(1.0, 2.2));
            },
            TRC_GAMMA18 => {
                defF(&w, "GAMMA_EXP", 1.8);
                defF(&w, "GAMMA_IEXP", foldDiv(1.0, 1.8));
            },
            TRC_GAMMA20 => {
                defF(&w, "GAMMA_EXP", 2.0);
                defF(&w, "GAMMA_IEXP", foldDiv(1.0, 2.0));
            },
            TRC_GAMMA24 => {
                defF(&w, "GAMMA_EXP", 2.4);
                defF(&w, "GAMMA_IEXP", foldDiv(1.0, 2.4));
            },
            TRC_GAMMA26 => {
                defF(&w, "GAMMA_EXP", 2.6);
                defF(&w, "GAMMA_IEXP", foldDiv(1.0, 2.6));
            },
            TRC_GAMMA28 => {
                defF(&w, "GAMMA_EXP", 2.8);
                defF(&w, "GAMMA_IEXP", foldDiv(1.0, 2.8));
            },
            TRC_PRO_PHOTO => {
                defF(&w, "PROPHOTO_LO", foldDiv(1.0, 16.0));
                defF(&w, "DELIN_G18", foldDiv(1.0, 1.8));
            },
            TRC_ST428 => {
                defF(&w, "ST428_S", foldDiv(52.37, 48.0));
                defF(&w, "ST428_IS", foldDiv(48.0, 52.37));
                defF(&w, "DELIN_G26", foldDiv(1.0, 2.6));
            },
            TRC_PQ => {
                defF(&w, "PQ_IM2R", @floatCast(1.0 / @as(f64, f6(PQ_M2))));
                defF(&w, "PQ_C1R", f6(PQ_C1));
                defF(&w, "PQ_C2R", f6(PQ_C2));
                defF(&w, "PQ_C3R", f6(PQ_C3));
                defF(&w, "PQ_IM1R", @floatCast(1.0 / @as(f64, f6(PQ_M1))));
                defF(&w, "PQ_LSCALE", f6(10000.0 / @as(f64, PL_COLOR_SDR_WHITE)));
                defF(&w, "PQ_ILSCALE", @floatCast(1.0 / @as(f64, f6(10000.0 / @as(f64, PL_COLOR_SDR_WHITE)))));
                defF(&w, "PQ_M1R", f6(PQ_M1));
                defF(&w, "PQ_M2R", f6(PQ_M2));
            },
            TRC_HLG => {
                const y: f32 = 1.2 * libm.powf(1.111, libm.log2f(csp_max / HLG_REF));
                const b: f32 = @sqrt(3.0 * libm.powf(csp_min / csp_max, 1.0 / y));
                defF(&w, "HLG_1MB", @floatCast(1.0 - @as(f64, b)));
                defF(&w, "HLG_BB", b);
                defF(&w, "HLG_CR", f6(HLG_C));
                defF(&w, "HLG_IAR", @floatCast(1.0 / @as(f64, f6(HLG_A))));
                defF(&w, "HLG_BR", f6(HLG_B));
                defF(&w, "HLG_CSPMAX", csp_max);
                defF(&w, "HLG_YM1", @floatCast(@as(f64, y) - 1.0));
                defF(&w, "HLG_LUMA_R", bt709LumaCoeffs()[0]);
                defF(&w, "HLG_ICSPMAX", @floatCast(1.0 / cmax));
                defF(&w, "HLG_IYM1", @floatCast((1.0 - @as(f64, y)) / @as(f64, y)));
                defF(&w, "HLG_AR", f6(HLG_A));
                defF(&w, "HLG_OSC", @floatCast(1.0 / (1.0 - @as(f64, b))));
                defF(&w, "HLG_OOF", @floatCast(-@as(f64, b) / (1.0 - @as(f64, b))));
            },
            TRC_V_LOG => {
                defF(&w, "VLOG_DR", f6(VLOG_D));
                defF(&w, "VLOG_ICR", @floatCast(1.0 / @as(f64, f6(VLOG_C))));
                defF(&w, "VLOG_BR", f6(VLOG_B));
                defF(&w, "VLOG_LO", foldDiv(1.0, 5.6));
                defF(&w, "VLOG_CLN", f6(@as(f64, VLOG_C) / @log(@as(f64, 10.0))));
            },
            TRC_S_LOG1, TRC_S_LOG2 => {
                defF(&w, "SLOG_CR", f6(SLOG_C));
                defF(&w, "SLOG_IAR", @floatCast(1.0 / @as(f64, f6(SLOG_A))));
                defF(&w, "SLOG_BR", f6(SLOG_B));
                defF(&w, "SLOG_ALN", f6(@as(f64, SLOG_A) / @log(@as(f64, 10.0))));
                if (trc == TRC_S_LOG2) {
                    defF(&w, "SLOG_IK2R", @floatCast(1.0 / @as(f64, f6(SLOG_K2))));
                    defF(&w, "SLOG_QR", f6(SLOG_Q));
                    defF(&w, "SLOG_IPR", @floatCast(1.0 / @as(f64, f6(SLOG_P))));
                    defF(&w, "SLOG_K2R", f6(SLOG_K2));
                    defF(&w, "SLOG_PR", f6(SLOG_P));
                }
            },
            TRC_SCRGB => {
                defF(&w, "SCRGB_L", f6(80.0 / @as(f64, PL_COLOR_SDR_WHITE)));
                defF(&w, "SCRGB_IL", f6(@as(f64, PL_COLOR_SDR_WHITE) / 80.0));
            },
            else => {},
        }
        const scale_out_trc = switch (trc) {
            TRC_SRGB, TRC_UNKNOWN, TRC_GAMMA18, TRC_GAMMA20, TRC_GAMMA22, TRC_GAMMA24, TRC_GAMMA26, TRC_GAMMA28, TRC_PRO_PHOTO, TRC_ST428 => true,
            else => false,
        };
        const scaled = csp_max != 1.0 or csp_min != 0.0;
        defI(&w, "LIN_SCALE_OUT", @intFromBool(scale_out_trc and scaled));
        if (scale_out_trc and scaled) {
            defF(&w, "LIN_SA", @floatCast(cmax - cmin));
            defF(&w, "LIN_SB", csp_min);
        }
        const delin_scale = isBlackScaled(trc) and trc != TRC_HLG and scaled;
        defI(&w, "DELIN_SCALE", @intFromBool(delin_scale));
        if (delin_scale) {
            defF(&w, "DELIN_SA", @floatCast(1.0 / (cmax - cmin)));
            defF(&w, "DELIN_SB", @floatCast(-cmin / (cmax - cmin)));
        }
        defF(&w, "DELIN_G24", foldDiv(1.0, 2.4));
    }

    if (sigm) {
        defF(&w, "SIG_OFF", sig_offset);
        defF(&w, "SIG_SCALE", sig_scale);
        defF(&w, "SIG_SLOPE", sig_slope);
        defF(&w, "SIG_ISLOPE", @floatCast(1.0 / @as(f64, sig_slope)));
        defF(&w, "SIG_ISCALE", @floatCast(1.0 / @as(f64, sig_scale)));
    }

    return std.fmt.allocPrintSentinel(allocator, "{s}", .{w.items}, 0);
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = undefined;
    d.n_cfg = 0;
    d.polar = false;

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
        return map_out.setError("Resample: input bitdepth must be 8/16 (integer), 16 (half) or 32 (float), Gray/YUV/RGB.");
    d.bits = bits;
    d.half = fmt.sampleType == .Float and bits == 16;
    d.bytes = @intCast(fmt.bytesPerSample);

    d.width = map_in.getValue(i32, "width").?;
    d.height = map_in.getValue(i32, "height").?;
    if (d.width <= 0 or d.height <= 0) return map_out.setError("Resample: width/height must be positive.");
    const subW: u5 = @intCast(fmt.subSamplingW);
    const subH: u5 = @intCast(fmt.subSamplingH);
    if ((d.width >> subW) <= 0 or (d.height >> subH) <= 0)
        return map_out.setError("Resample: output too small for the chroma subsampling.");

    d.src_width = map_in.getValue(f32, "src_width") orelse @floatFromInt(d.vi.width);
    d.src_height = map_in.getValue(f32, "src_height") orelse @floatFromInt(d.vi.height);
    const src_x = map_in.getValue(f32, "sx") orelse 0.0;
    const src_y = map_in.getValue(f32, "sy") orelse 0.0;
    if (!math.isFinite(d.src_width) or d.src_width <= 0 or !math.isFinite(d.src_height) or d.src_height <= 0)
        return map_out.setError("Resample: src_width/src_height must be finite and positive.");
    if (!math.isFinite(src_x) or !math.isFinite(src_y))
        return map_out.setError("Resample: sx/sy must be finite.");

    const is_rgb_default: i32 = @intFromBool(fmt.colorFamily == .RGB);
    var linear = (map_in.getValue(i32, "linearize") orelse is_rgb_default) != 0;
    linear = linear and (fmt.colorFamily == .RGB or fmt.colorFamily == .Gray);
    var sigm = (map_in.getValue(i32, "sigmoidize") orelse is_rgb_default) != 0;
    sigm = sigm and (fmt.colorFamily == .RGB or fmt.colorFamily == .Gray);
    const trc = map_in.getValue(i32, "trc") orelse 1;
    if (trc < 0 or trc >= TRC_COUNT) return map_out.setError("Resample: trc must be 0..17.");
    const min_luma = map_in.getValue(f32, "min_luma") orelse PL_COLOR_HDR_BLACK;
    if (!math.isFinite(min_luma) or min_luma < 0) return map_out.setError("Resample: min_luma must be finite and >= 0.");
    const sig_center = map_in.getValue(f32, "sigmoid_center") orelse 0.75;
    const sig_slope = map_in.getValue(f32, "sigmoid_slope") orelse 6.50;
    if (!math.isFinite(sig_center) or sig_center <= 0.0 or sig_center >= 1.0)
        return map_out.setError("Resample: sigmoid_center must be in (0, 1).");
    if (!math.isFinite(sig_slope) or sig_slope <= 0.0)
        return map_out.setError("Resample: sigmoid_slope must be > 0.");

    const linearize = linear and trc != TRC_LINEAR;
    d.prep = linearize or sigm;

    const filter_name = map_in.getData("filter", 0) orelse blk: {
        std.log.warn("Resample: unspecified filter... selecting ewa_lanczos.", .{});
        break :blk "ewa_lanczos";
    };
    const fcfg: *const FilterConfig = findFilterConfig(filter_name) orelse blk: {
        std.log.warn("Resample: unknown filter... selecting ewa_lanczos.", .{});
        break :blk findFilterConfig("ewa_lanczos").?;
    };
    d.polar = fcfg.polar;

    const user_clamp = map_in.getValue(f32, "clamp") orelse 0.0;
    const user_blur = map_in.getValue(f32, "blur") orelse 0.0;
    const user_taper = map_in.getValue(f32, "taper") orelse 0.0;
    const user_antiring = map_in.getValue(f32, "antiring") orelse 0.0;
    if (!math.isFinite(user_clamp) or user_clamp < 0 or user_clamp > 1) return map_out.setError("Resample: clamp must be in [0, 1].");
    if (!math.isFinite(user_blur) or user_blur < 0) return map_out.setError("Resample: blur must be finite and >= 0.");
    if (!math.isFinite(user_taper) or user_taper < 0) return map_out.setError("Resample: taper must be finite and >= 0.");
    if (!math.isFinite(user_antiring) or user_antiring < 0 or user_antiring > 1) return map_out.setError("Resample: antiring must be in [0, 1].");

    var res: ResolvedConfig = .{
        .kernel = fcfg.kernel.*,
        .window = fcfg.window,
        .radius = fcfg.radius,
        .params = fcfg.params,
        .wparams = fcfg.wparams,
        .clamp = user_clamp,
        .blur = user_blur,
        .taper = user_taper,
        .polar = fcfg.polar,
    };
    if (res.kernel.resizable) {
        if (map_in.getValue(f32, "radius")) |r| {
            if (!math.isFinite(r) or r <= 0) return map_out.setError("Resample: radius must be finite and > 0.");
            res.kernel.radius = r;
        }
    }
    if (map_in.getValue(f32, "param1")) |p1| {
        if (res.kernel.tunable[0]) {
            if (!math.isFinite(p1)) return map_out.setError("Resample: param1 must be finite.");
            res.params[0] = p1;
        }
    }
    if (map_in.getValue(f32, "param2")) |p2| {
        if (res.kernel.tunable[1]) {
            if (!math.isFinite(p2)) return map_out.setError("Resample: param2 must be finite.");
            res.params[1] = p2;
        }
    }
    const antiring: f32 = if (fcfg.antiring != 0) fcfg.antiring else user_antiring;

    const device_id = map_in.getValue(i32, "device_id") orelse 0;
    if (device_id < 0) return map_out.setError("Resample: invalid device ID.");
    const ns_req = map_in.getValue(i32, "num_streams");
    if (ns_req) |ns| if (ns < 1 or ns > 32) return map_out.setError("Resample: num_streams must be 1..32.");

    d.vi_out = d.vi.*;
    d.vi_out.width = d.width;
    d.vi_out.height = d.height;
    const src_strides = vszipcl.strideFromVi(d.vi);
    const dst_strides = vszipcl.strideFromVi(&d.vi_out);
    d.n_planes = @intCast(fmt.numPlanes);

    var csp_min: f32 = 0;
    var csp_max: f32 = 1;
    if (linearize) {
        nominalLuma(trc, min_luma, &csp_min, &csp_max);
        csp_min = signalBlack(csp_min);
    }

    vszipcl.initContext(&d, @intCast(device_id)) catch |err| {
        map_out.setError(if (err == error.InvalidDeviceID) "Resample: invalid device ID." else "Resample: OpenCL initialization failed.");
        std.log.err("Resample OpenCL init failed: {}", .{err});
        return;
    };

    const sig_offset: f32 = 1.0 / (1.0 + libm.expf(sig_slope * sig_center));
    const sig_scale: f32 = 1.0 / (1.0 + libm.expf(sig_slope * (sig_center - 1.0))) - sig_offset;

    var build_err: ?[:0]const u8 = null;
    for (0..d.n_planes) |p| {
        const sw: i32 = if (p == 0) d.vi.width else d.vi.width >> subW;
        const sh_: i32 = if (p == 0) d.vi.height else d.vi.height >> subH;
        const dw: i32 = if (p == 0) d.width else d.width >> subW;
        const dh: i32 = if (p == 0) d.height else d.height >> subH;

        const ssw: f32 = @floatFromInt(@as(i32, 1) << subW);
        const ssh: f32 = @floatFromInt(@as(i32, 1) << subH);
        const shift_w: f32 = (0.5 * (1.0 - @as(f32, @floatFromInt(dw)) / @as(f32, @floatFromInt(d.vi.width)))) / ssw;
        const shifted = fmt.colorFamily == .YUV and p > 0;
        const sxp: f32 = if (shifted) shift_w + src_x / ssw else src_x;
        const syp: f32 = if (shifted) 0.0 + src_y / ssh else src_y;
        const srcwp: f32 = if (shifted) d.src_width / ssw else d.src_width;
        const srchp: f32 = if (shifted) d.src_height / ssh else d.src_height;

        const srcw_eff: f32 = (sxp + srcwp) - sxp;
        const srch_eff: f32 = (syp + srchp) - syp;

        var hit: ?usize = null;
        for (0..d.n_cfg) |m| {
            const c = &d.cfgs[m];
            if (c.sw == sw and c.sh == sh_ and c.dw == dw and c.dh == dh and
                c.sxf == sxp and c.syf == syp and c.srcw_eff == srcw_eff and c.srch_eff == srch_eff)
                hit = m;
        }
        if (hit) |m| {
            d.plane_cfg[p] = m;
            continue;
        }

        const m = d.n_cfg;
        var cfg = &d.cfgs[m];
        cfg.* = .{
            .sw = sw,
            .sh = sh_,
            .dw = dw,
            .dh = dh,
            .sstride = @intCast(if (p == 0) src_strides[0] else src_strides[1]),
            .dstride = @intCast(if (p == 0) dst_strides[0] else dst_strides[1]),
            .tstride = (sw + 63) & ~@as(i32, 63),
            .sxf = sxp,
            .syf = syp,
            .srcw_eff = srcw_eff,
            .srch_eff = srch_eff,
            .rx = undefined,
            .ry = undefined,
        };
        cfg.rx = @floatCast(@as(f64, @floatFromInt(dw)) / @abs(@as(f64, cfg.srcw_eff)));
        cfg.ry = @floatCast(@as(f64, @floatFromInt(dh)) / @abs(@as(f64, cfg.srch_eff)));

        const max_row_size = orthoMaxRowSize(d.device);
        if (d.polar) {
            var inv_scale: f32 = @floatCast(1.0 / @as(f64, @min(cfg.rx, cfg.ry)));
            inv_scale = @max(inv_scale, 1.0);
            var pres = res;
            pres.blur = (if (user_blur != 0) user_blur else 1.0) * inv_scale;
            const gf = generateFilter(&pres, POLAR_CUTOFF, 0) catch {
                build_err = "Resample: out of memory.";
                break;
            };
            cfg.p_radius = gf.radius;
            cfg.p_radius_zero = gf.radius_zero;
            cfg.p_bound = @intFromFloat(@ceil(gf.radius));
            cfg.p_use_ar = antiring > 0;
            cfg.p_lut = gf.weights;
        } else {
            var vres = res;
            vres.polar = false;
            var inv_v: f32 = @floatCast(1.0 / @as(f64, cfg.ry));
            inv_v = @max(inv_v, 1.0);
            vres.blur = (if (user_blur != 0) user_blur else 1.0) * inv_v;
            const gv = generateFilter(&vres, 0.0, max_row_size) catch {
                build_err = "Resample: out of memory.";
                break;
            };
            cfg.vert = .{
                .n = gv.row_size,
                .lutw = @divExact(gv.row_stride, 4),
                .use_linear = gv.radius == gv.radius_zero,
                .use_ar = antiring > 0 and cfg.ry > 1.0 and !(gv.radius == gv.radius_zero),
                .lut = gv.weights,
            };
            var hres = res;
            hres.polar = false;
            var inv_h: f32 = @floatCast(1.0 / @as(f64, cfg.rx));
            inv_h = @max(inv_h, 1.0);
            hres.blur = (if (user_blur != 0) user_blur else 1.0) * inv_h;
            const gh = generateFilter(&hres, 0.0, max_row_size) catch {
                allocator.free(cfg.vert.lut);
                build_err = "Resample: out of memory.";
                break;
            };
            cfg.horiz = .{
                .n = gh.row_size,
                .lutw = @divExact(gh.row_stride, 4),
                .use_linear = gh.radius == gh.radius_zero,
                .use_ar = antiring > 0 and cfg.rx > 1.0 and !(gh.radius == gh.radius_zero),
                .lut = gh.weights,
            };
        }

        cfg.opts = buildOpts(&d, cfg, trc, linearize, sigm, antiring, csp_min, csp_max, sig_offset, sig_scale, sig_slope) catch {
            build_err = "Resample: out of memory.";
            if (d.polar) allocator.free(cfg.p_lut) else {
                allocator.free(cfg.vert.lut);
                allocator.free(cfg.horiz.lut);
            }
            break;
        };
        d.n_cfg = m + 1;
        d.plane_cfg[p] = m;
    }
    if (build_err) |msg| {
        freeCfgAllocs(&d);
        d.context.release();
        return map_out.setError(msg);
    }

    for (0..d.n_cfg) |m| {
        const c = &d.cfgs[m];
        const exts = [_]u64{
            @as(u64, @intCast(c.sh)) * @as(u64, @intCast(c.sstride)),
            @as(u64, @intCast(c.dh)) * @as(u64, @intCast(c.tstride)),
            @as(u64, @intCast(c.dh)) * @as(u64, @intCast(c.dstride)),
        };
        for (exts) |e| if (e >= (1 << 31)) {
            freeCfgAllocs(&d);
            d.context.release();
            return map_out.setError("Resample: frame too large (a plane exceeds 2^31 samples).");
        };
    }

    d.src_bytes = 0;
    d.tmp2_bytes = 1;
    d.dst_bytes = 0;
    for (0..d.n_cfg) |m| {
        const c = &d.cfgs[m];
        d.src_bytes = @max(d.src_bytes, @as(usize, @intCast(c.sh)) * @as(usize, @intCast(c.sstride)) * d.bytes);
        d.dst_bytes = @max(d.dst_bytes, @as(usize, @intCast(c.dh)) * @as(usize, @intCast(c.dstride)) * d.bytes);
        if (!d.polar) d.tmp2_bytes = @max(d.tmp2_bytes, @as(usize, @intCast(c.dh)) * @as(usize, @intCast(c.tstride)) * d.bytes);
    }

    d.blk_x = 16;
    d.blk_y = 8;
    const streams: usize = if (ns_req) |ns| @intCast(ns) else 1;
    {
        var terr: ?[:0]const u8 = null;
        if (map_in.numElements("tune")) |ne| {
            if (ne > 2) terr = "Resample: tune expects at most 2 entries [blk_x, blk_y].";
        }
        if (vszipcl.tuneEntry(map_in, 0)) |v| {
            if (v < 1 or v > 64) terr = "Resample: tune[0] (blk_x) must be 1..64." else d.blk_x = @intCast(v);
        }
        if (vszipcl.tuneEntry(map_in, 1)) |v| {
            if (v < 1 or v > 64) terr = "Resample: tune[1] (blk_y) must be 1..64." else d.blk_y = @intCast(v);
        }
        if (d.blk_x * d.blk_y > vszipcl.deviceMaxWG(d.device)) terr = "Resample: tune blk_x*blk_y exceeds the device max work-group size.";
        if (terr) |msg| {
            map_out.setError(msg);
            d.context.release();
            freeCfgAllocs(&d);
            return;
        }
    }
    for (0..d.n_cfg) |m| {
        const cfg = &d.cfgs[m];
        const with_blk = std.fmt.allocPrintSentinel(allocator, "{s} -DBX={d} -DBY={d}", .{ cfg.opts, d.blk_x, d.blk_y }, 0) catch {
            map_out.setError("Resample: out of memory.");
            d.context.release();
            freeCfgAllocs(&d);
            return;
        };
        allocator.free(cfg.opts);
        cfg.opts = with_blk;
    }

    d.stage_src_sum = 0;
    for (0..d.n_planes) |p| {
        d.stage_src_off[p] = d.stage_src_sum;
        const c = &d.cfgs[d.plane_cfg[p]];
        d.stage_src_sum += @as(usize, @intCast(c.sh)) * @as(usize, @intCast(c.sstride)) * d.bytes;
    }
    var dst_sum: usize = 0;
    for (0..d.n_planes) |p| {
        d.stage_dst_off[p] = dst_sum;
        const c = &d.cfgs[d.plane_cfg[p]];
        dst_sum += @as(usize, @intCast(c.dh)) * @as(usize, @intCast(c.dstride)) * d.bytes;
    }
    d.stage_bytes = @max(d.stage_src_sum + dst_sum, 1);
    d.use_pinned = streams >= 2 and dst_sum <= d.stage_src_sum;
    d.memcpy_sem = .{ .permits = if (streams <= 2 or d.n_planes > 1) streams else 1 };

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;
    keep = true;

    data.pool = .{};
    data.pool.prime(data, streams);
    data.pool.prewarm(streams) catch |err| {
        map_out.setError("Resample: OpenCL stream init failed.");
        std.log.err("Resample stream init failed: {}", .{err});
        data.pool.deinit();
        data.context.release();
        freeCfgAllocs(data);
        allocator.destroy(data);
        keep = false;
        return;
    };

    var dep = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };
    zapi.createVideoFilter(out, "Resample", &data.vi_out, getFrame, free, .Parallel, &dep, data);
}
