const std = @import("std");
const vszipcl = @import("vszipcl.zig");

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

    src: cl.Buffer(f32),
    dst: cl.Buffer(f32),

    context: cl.Context,
    queue: cl.CommandQueue,
    bilateral: cl.Kernel,
    program: cl.Program,
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

    const device = devices[0];

    d.context = try cl.createContext(&.{device}, .{ .platform = platform });
    d.queue = try cl.createCommandQueue(d.context, device, .{});

    d.program = try cl.createProgramWithSource(d.context, bilateral);
    d.program.build(&.{device}, "-cl-std=CL3.0") catch |err| {
        if (err == error.BuildProgramFailure) {
            const log = try d.program.getBuildLog(allocator, device);
            defer allocator.free(log);
            std.log.err("OpenCL kernel build failed: {s}", .{log});
        }
        return err;
    };

    // const platform_name = try platform.getName(allocator);
    // const device_name = try device.getName(allocator);
    // std.log.info("selected platform '{s}' and device '{s}'", .{ platform_name, device_name });

    d.bilateral = try cl.createKernel(d.program, "bilateral");
}

fn process(d: *Data, dstp: anytype, srcp: anytype, w: i32, h: i32, stride: i32) !void {
    const local_work_size: [2]usize = .{ 16, 16 };
    const global_work_size: [2]usize = .{
        vsh.ceilN(@intCast(w), local_work_size[0]),
        vsh.ceilN(@intCast(h), local_work_size[1]),
    };

    const whrite_complete = try d.queue.enqueueWriteBuffer(
        f32,
        d.src,
        false,
        0,
        srcp,
        &.{},
    );
    defer whrite_complete.release();

    try d.bilateral.setArg(@TypeOf(d.dst), 0, d.dst);
    try d.bilateral.setArg(@TypeOf(d.src), 1, d.src);
    try d.bilateral.setArg(c_int, 2, w);
    try d.bilateral.setArg(c_int, 3, h);
    try d.bilateral.setArg(c_int, 4, @intCast(stride));
    try d.bilateral.setArg(f32, 5, d.sigma_spatial);
    try d.bilateral.setArg(f32, 6, d.sigma_color);
    try d.bilateral.setArg(c_int, 7, d.radius);

    const bilateral_complete = try d.queue.enqueueNDRangeKernel(
        d.bilateral,
        null,
        &global_work_size,
        &local_work_size,
        &.{whrite_complete},
    );
    defer bilateral_complete.release();

    const read_complete = try d.queue.enqueueReadBuffer(
        f32,
        d.dst,
        false,
        0,
        dstp,
        &.{bilateral_complete},
    );
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

        var plane: u32 = 0;
        while (plane < d.vi.format.numPlanes) : (plane += 1) {
            const srcp = src.getReadSlice2(f32, plane);
            const dstp = dst.getWriteSlice2(f32, plane);

            const w = src.getWidthSigned(plane);
            const h = src.getHeightSigned(plane);
            const stride = src.getStride2(f32, plane);

            process(d, dstp, srcp, w, h, @intCast(stride)) catch |err| {
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

    d.src.release();
    d.dst.release();
    d.bilateral.release();
    d.program.release();
    d.queue.release();
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

    const buff_size: usize = stride * height;
    d.src = cl.createBuffer(f32, d.context, .{ .read_only = true }, buff_size) catch |err| {
        map_out.setError("Bilateral: cl.createBuffer failed.");
        std.log.err("OpenCL cl.createBuffer failed: {}", .{err});
        zapi.freeNode(d.node);
        return;
    };
    d.dst = cl.createBuffer(f32, d.context, .{ .read_write = true }, buff_size) catch |err| {
        map_out.setError("Bilateral: cl.createBuffer failed.");
        std.log.err("OpenCL cl.createBuffer failed: {}", .{err});
        zapi.freeNode(d.node);
        return;
    };

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var dep = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    zapi.createVideoFilter(out, "Bilateral", d.vi, getFrame, free, .Unordered, &dep, data);
}
