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

    src: cl.Buffer(f32),
    dst: cl.Buffer(f32),
    tmp: cl.Buffer(f32),
    blur_kernel: cl.Buffer(f32),
    ksize: i32,

    context: cl.Context,
    queue: cl.CommandQueue,
    vertical_blur: cl.Kernel,
    horizontal_blur: cl.Kernel,
    program: cl.Program,
};

const vertical_blur =
    \\kernel void vertical_blur(global float *dst, global const float *src,
    \\                          global const float *blur_kernel, int kernel_len,
    \\                          const int w, const int h, const int stride) {
    \\
    \\    const int x = get_global_id(0);
    \\    const int y = get_global_id(1);
    \\    if (x >= w || y >= h) {
    \\        return;
    \\    }
    \\
    \\    const int radius = kernel_len / 2;
    \\    float sum = 0.0f;
    \\
    \\    // First loop: Handles the top edge of the image where src_y < 0.
    \\    for (int k = 0; k < radius - y && k < kernel_len; k++) {
    \\        int src_y = y + k - radius;
    \\        src_y = -src_y;
    \\        sum += src[src_y * stride + x] * blur_kernel[k];
    \\    }
    \\    
    \\    // Second loop: Handles the main body of the image where src_y is in-bounds.
    \\    for (int k = max(0, radius - y); k < min(kernel_len, h + radius - y); k++) {
    \\        const int src_y = y + k - radius;
    \\        sum += src[src_y * stride + x] * blur_kernel[k];
    \\    }
    \\    
    \\    // Third loop: Handles the bottom edge of the image where src_y >= h.
    \\    for (int k = max(0, h + radius - y); k < kernel_len; k++) {
    \\        int src_y = y + k - radius;
    \\        src_y = 2 * (h - 1) - src_y;
    \\        sum += src[src_y * stride + x] * blur_kernel[k];
    \\    }
    \\    
    \\    dst[y * stride + x] = sum;
    \\}
    \\
;

const horizontal_blur =
    \\kernel void horizontal_blur(global float *dst, global const float *src,
    \\                            global const float *blur_kernel, int kernel_len,
    \\                            const int w, const int h, const int stride) {
    \\
    \\    const int x = get_global_id(0);
    \\    const int y = get_global_id(1);
    \\    if (x >= w || y >= h) {
    \\        return;
    \\    }
    \\
    \\    const int radius = kernel_len / 2;
    \\    float sum = 0.0f;
    \\
    \\    // First loop: Handles the left edge of the image where src_x < 0.
    \\    for (int k = 0; k < radius - x && k < kernel_len; k++) {
    \\        int src_x = x + k - radius;
    \\        src_x = -src_x;
    \\        sum += src[y * stride + src_x] * blur_kernel[k];
    \\    }
    \\    
    \\    // Second loop: Handles the main body of the image where src_x is in-bounds.
    \\    for (int k = max(0, radius - x); k < min(kernel_len, w + radius - x); k++) {
    \\        const int src_x = x + k - radius;
    \\        sum += src[y * stride + src_x] * blur_kernel[k];
    \\    }
    \\    
    \\    // Third loop: Handles the right edge of the image where src_x >= w.
    \\    for (int k = max(0, w + radius - x); k < kernel_len; k++) {
    \\        int src_x = x + k - radius;
    \\        src_x = 2 * (w - 1) - src_x;
    \\        sum += src[y * stride + src_x] * blur_kernel[k];
    \\    }
    \\
    \\    dst[y * stride + x] = sum;
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

    d.program = try cl.createProgramWithSource(d.context, vertical_blur ++ horizontal_blur);
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

    d.vertical_blur = try cl.createKernel(d.program, "vertical_blur");
    d.horizontal_blur = try cl.createKernel(d.program, "horizontal_blur");
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

    // Vertical pass: src -> tmp
    try d.vertical_blur.setArg(@TypeOf(d.tmp), 0, d.tmp);
    try d.vertical_blur.setArg(@TypeOf(d.src), 1, d.src);
    try d.vertical_blur.setArg(@TypeOf(d.blur_kernel), 2, d.blur_kernel);
    try d.vertical_blur.setArg(c_int, 3, d.ksize);
    try d.vertical_blur.setArg(c_int, 4, w);
    try d.vertical_blur.setArg(c_int, 5, h);
    try d.vertical_blur.setArg(c_int, 6, stride);

    const vertical_complete = try d.queue.enqueueNDRangeKernel(
        d.vertical_blur,
        null,
        &global_work_size,
        &local_work_size,
        &.{whrite_complete},
    );
    defer vertical_complete.release();

    // Horizontal pass: tmp -> dst
    try d.horizontal_blur.setArg(@TypeOf(d.dst), 0, d.dst);
    try d.horizontal_blur.setArg(@TypeOf(d.tmp), 1, d.tmp);
    try d.horizontal_blur.setArg(@TypeOf(d.blur_kernel), 2, d.blur_kernel);
    try d.horizontal_blur.setArg(c_int, 3, d.ksize);
    try d.horizontal_blur.setArg(c_int, 4, w);
    try d.horizontal_blur.setArg(c_int, 5, h);
    try d.horizontal_blur.setArg(c_int, 6, stride);

    const horizontal_complete = try d.queue.enqueueNDRangeKernel(
        d.horizontal_blur,
        null,
        &global_work_size,
        &local_work_size,
        &.{vertical_complete},
    );
    defer horizontal_complete.release();

    const read_complete = try d.queue.enqueueReadBuffer(
        f32,
        d.dst,
        false,
        0,
        dstp,
        &.{horizontal_complete},
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
                zapi.setFilterError("GaussBlur: process frame failed.");
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
    d.tmp.release();
    d.blur_kernel.release();
    d.vertical_blur.release();
    d.horizontal_blur.release();
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
        map_out.setError("GaussBlur: test build, GRAYS format only.");
        zapi.freeNode(d.node);
        return;
    }

    initOpenCL(&d) catch |err| {
        map_out.setError("GaussBlur: OpenCL initialization failed.");
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
        map_out.setError("GaussBlur: cl.createBuffer failed.");
        std.log.err("OpenCL cl.createBuffer failed: {}", .{err});
        zapi.freeNode(d.node);
        return;
    };
    d.dst = cl.createBuffer(f32, d.context, .{ .read_write = true }, buff_size) catch |err| {
        map_out.setError("GaussBlur: cl.createBuffer failed.");
        std.log.err("OpenCL cl.createBuffer failed: {}", .{err});
        zapi.freeNode(d.node);
        return;
    };
    d.tmp = cl.createBuffer(f32, d.context, .{ .read_write = true }, buff_size) catch |err| {
        map_out.setError("GaussBlur: cl.createBuffer failed.");
        std.log.err("OpenCL cl.createBuffer failed: {}", .{err});
        zapi.freeNode(d.node);
        return;
    };

    const sigma = map_in.getValue(f32, "sigma") orelse 0.5;
    const blur_kernel = getGaussKernel(sigma) catch unreachable;
    defer allocator.free(blur_kernel);
    d.ksize = @intCast(blur_kernel.len);
    d.blur_kernel = cl.createBufferWithData(f32, d.context, .{ .read_only = true }, blur_kernel) catch unreachable;

    const data: *Data = allocator.create(Data) catch unreachable;
    data.* = d;

    var dep = [_]vs.FilterDependency{
        .{ .source = d.node, .requestPattern = .StrictSpatial },
    };

    zapi.createVideoFilter(out, "GaussBlur", d.vi, getFrame, free, .Unordered, &dep, data);
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
