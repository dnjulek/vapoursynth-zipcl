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

const Data = struct {
    node: ?*vs.Node,
    vi: *const vs.VideoInfo,

    sigma_spatial: f32,
    sigma_color: f32,
    radius: i32,

    platform: cl.Platform,
    device: cl.Device,
    buff_size: usize,

    pool: clpool.Pool(Stream, Data),
};

/// Per-concurrent-frame OpenCL resources. One is acquired per getFrame call so
/// frames never share a queue, buffers or kernel.
const Stream = struct {
    context: cl.Context,
    program: cl.Program,
    queue: cl.CommandQueue,
    src: cl.Buffer(f32),
    dst: cl.Buffer(f32),
    kernel: cl.Kernel,

    pub fn init(self: *Stream, d: *Data) !void {
        self.context = try cl.createContext(&.{d.device}, .{ .platform = d.platform });
        errdefer self.context.release();
        self.program = try cl.createProgramWithSource(self.context, bilateral);
        errdefer self.program.release();
        self.program.build(&.{d.device}, "-cl-std=CL3.0") catch |err| {
            if (err == error.BuildProgramFailure) {
                const log = try self.program.getBuildLog(allocator, d.device);
                defer allocator.free(log);
                std.log.err("OpenCL kernel build failed: {s}", .{log});
            }
            return err;
        };
        self.queue = try cl.createCommandQueue(self.context, d.device, .{});
        errdefer self.queue.release();
        self.src = try cl.createBuffer(f32, self.context, .{ .read_only = true }, d.buff_size);
        errdefer self.src.release();
        self.dst = try cl.createBuffer(f32, self.context, .{ .read_write = true }, d.buff_size);
        errdefer self.dst.release();
        self.kernel = try cl.createKernel(self.program, "bilateral");
    }

    pub fn deinit(self: *Stream) void {
        self.kernel.release();
        self.dst.release();
        self.src.release();
        self.queue.release();
        self.program.release();
        self.context.release();
    }
};

const bilateral =
    \\kernel void bilateral(global float *dst, global const float *src,
    \\                      const int w, const int h, const int stride,
    \\                      const float sigma_spatial, const float sigma_color, const int radius) {
    \\
    \\    const int x = get_global_id(0);
    \\    const int y = get_global_id(1);
    \\    if (x >= w || y >= h) {
    \\        return;
    \\    }
    \\
    \\    float center = src[y * stride + x];
    \\
    \\    float num = 0.0f;
    \\    float den = 0.0f;
    \\
    \\    for (int cy = max(y - radius, 0); cy <= min(y + radius, h - 1); ++cy) {
    \\        for (int cx = max(x - radius, 0); cx <= min(x + radius, w - 1); ++cx) {
    \\            const float value = src[cy * stride + cx];
    \\
    \\            float space = (x - cx) * (x - cx) + (y - cy) * (y - cy);
    \\            float range = (value - center) * (value - center);
    \\
    \\            float weight = exp(space * sigma_spatial + range * sigma_color);
    \\
    \\            num += weight * value;
    \\            den += weight;
    \\        }
    \\    }
    \\
    \\    dst[y * stride + x] = num / den;
    \\}
    \\
;

fn initOpenCL(d: *Data) !void {
    const platforms = try cl.getPlatforms(allocator);
    defer allocator.free(platforms);

    if (platforms.len == 0) {
        return error.NoPlatforms;
    }

    const platform = platforms[0];
    const devices = try platform.getDevices(allocator, cl.DeviceType.all);
    defer allocator.free(devices);

    if (devices.len == 0) {
        return error.NoDevices;
    }

    d.device = devices[0];

    d.platform = platform;
}

fn process(d: *Data, s: *Stream, dstp: []f32, srcp: []const f32, w: i32, h: i32, stride: i32) !void {
    const local_work_size: [2]usize = .{ 16, 16 };
    const global_work_size: [2]usize = .{
        vsh.ceilN(@intCast(w), local_work_size[0]),
        vsh.ceilN(@intCast(h), local_work_size[1]),
    };

    const write_complete = try s.queue.enqueueWriteBuffer(f32, s.src, false, 0, srcp, &.{});
    defer write_complete.release();

    try s.kernel.setArg(@TypeOf(s.dst), 0, s.dst);
    try s.kernel.setArg(@TypeOf(s.src), 1, s.src);
    try s.kernel.setArg(c_int, 2, w);
    try s.kernel.setArg(c_int, 3, h);
    try s.kernel.setArg(c_int, 4, @intCast(stride));
    try s.kernel.setArg(f32, 5, d.sigma_spatial);
    try s.kernel.setArg(f32, 6, d.sigma_color);
    try s.kernel.setArg(c_int, 7, d.radius);

    const kernel_complete = try s.queue.enqueueNDRangeKernel(
        s.kernel,
        null,
        &global_work_size,
        &local_work_size,
        &.{write_complete},
    );
    defer kernel_complete.release();

    // The bilateral kernel keeps the D2H read on PoCL-CUDA's fast path, so we
    // read straight into the VS frame (no pinned staging buffer / extra memcpy).
    const read_complete = try s.queue.enqueueReadBuffer(f32, s.dst, false, 0, dstp, &.{kernel_complete});
    defer read_complete.release();

    try cl.waitForEvents(&.{read_complete});
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

        const s = d.pool.acquire();
        defer d.pool.release(s);

        var plane: u32 = 0;
        while (plane < d.vi.format.numPlanes) : (plane += 1) {
            const srcp = src.getReadSlice2(f32, plane);
            const dstp = dst.getWriteSlice2(f32, plane);

            const w = src.getWidthSigned(plane);
            const h = src.getHeightSigned(plane);
            const stride = src.getStride2(f32, plane);

            process(d, s, dstp, srcp, w, h, @intCast(stride)) catch |err| {
                zapi.setFilterError("Bilateral: process frame failed.");
                std.log.err("OpenCL process frame failed: {}", .{err});
                dst.deinit();
                return null;
            };
        }

        return dst.frame;
    }

    return null;
}

fn free(instance_data: ?*anyopaque, _: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *Data = @ptrCast(@alignCast(instance_data));

    d.pool.deinit();

    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub fn create(in: ?*const vs.Map, out: ?*vs.Map, _: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    var d: Data = undefined;

    const zapi = ZAPI.init(vsapi, core, null);
    const map_in = zapi.initZMap(in);
    const map_out = zapi.initZMap(out);

    d.node, d.vi = map_in.getNodeVi("clip").?;
    if (zapi.getVideoFormatID(d.vi) != .GrayS) {
        map_out.setError("Bilateral: test build, GRAYS format only.");
        zapi.freeNode(d.node);
        return;
    }

    const sigma_spatial: f32 = map_in.getValue(f32, "sigma_spatial") orelse 3;
    const sigma_color: f32 = map_in.getValue(f32, "sigma_color") orelse 0.02;
    const radius: ?i32 = map_in.getValue(i32, "radius");
    d.sigma_spatial = -0.5 / (sigma_spatial * sigma_spatial) * math.log2e;
    d.sigma_color = (-0.5 / (sigma_color * sigma_color)) * math.log2e;
    d.radius = radius orelse @max(1, @as(i32, @intFromFloat(@round(sigma_spatial * 3))));

    initOpenCL(&d) catch |err| {
        map_out.setError("Bilateral: OpenCL initialization failed.");
        std.log.err("OpenCL initialization failed: {}", .{err});
        zapi.freeNode(d.node);
        return;
    };

    var pad_size: u32 = std.simd.suggestVectorLength(u8) orelse 32;
    pad_size = @divTrunc(@min(pad_size, 64), @as(u32, @intCast(d.vi.format.bytesPerSample)));
    const stride: usize = vsh.ceilN(@intCast(d.vi.width), pad_size);
    const height: usize = @intCast(d.vi.height);
    d.buff_size = stride * height;

    d.pool = .{};

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var info: vs.CoreInfo = .{};
    zapi.getCoreInfo(core, &info);
    const threads: usize = if (info.numThreads > 0) @intCast(info.numThreads) else 1;
    // Run a few frames concurrently so their host<->device transfers overlap.
    // ~4 streams is the measured sweet spot on PoCL-CUDA (more adds contention),
    // capped by the core thread count so low-thread cores don't over-allocate.
    const streams: usize = @min(max_streams, threads);
    data.pool.prime(data, streams);
    data.pool.prewarm(streams) catch |err| {
        map_out.setError("Bilateral: OpenCL stream init failed.");
        std.log.err("OpenCL stream init failed: {}", .{err});
        data.pool.deinit();
        zapi.freeNode(d.node);
        allocator.destroy(data);
        return;
    };

    var dep = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    zapi.createVideoFilter(out, "Bilateral", d.vi, getFrame, free, .Unordered, &dep, data);
}

const max_streams: usize = 4;
