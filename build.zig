const std = @import("std");
const zon = @import("build.zig.zon");

pub fn build(b: *std.Build) !void {
    ensureZigVersion(try .parse(zon.minimum_zig_version)) catch return;
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const os = target.result.os.tag;
    const is_cross = !target.query.isNative();

    const mod = b.createModule(.{
        .root_source_file = b.path("src/vszipcl.zig"),
        .target = target,
        .optimize = optimize,
    });

    const options = b.addOptions();
    const version = try std.SemanticVersion.parse(zon.version);
    options.addOption(std.SemanticVersion, "version", version);
    mod.addOptions("zon", options);

    const lib = b.addLibrary(.{
        .name = "vszipcl",
        .linkage = .dynamic,
        .root_module = mod,
    });

    // Explicit -Dtarget (CI glibc 2.17 / macOS 11 minos pins) skips native OpenCL loader discovery.
    if (os == .windows) {
        lib.root_module.addLibraryPath(.{ .cwd_relative = "C:/WINDOWS/system32" });
    } else if (os == .linux and is_cross) {
        const triple = b.fmt("{s}-linux-gnu", .{@tagName(target.result.cpu.arch)});
        lib.root_module.addLibraryPath(.{ .cwd_relative = b.fmt("/usr/lib/{s}", .{triple}) });
        lib.root_module.addLibraryPath(.{ .cwd_relative = b.fmt("/lib/{s}", .{triple}) });
        lib.root_module.addLibraryPath(.{ .cwd_relative = "/usr/lib64" });
        lib.root_module.addLibraryPath(.{ .cwd_relative = "/usr/lib" });
    } else if (os == .macos and is_cross and b.sysroot == null) {
        if (macosSdkPath(b)) |sdk| b.sysroot = sdk;
    }

    const vapoursynth_dep = b.dependency("vapoursynth", .{
        .target = target,
        .optimize = optimize,
    }).module("vapoursynth");

    // opencl-zig hardcodes -lOpenCL; macOS ships OpenCL as a framework.
    const opencl_dep = b.dependency("opencl", .{ .target = target, .optimize = optimize });
    const opencl = b.createModule(.{
        .root_source_file = opencl_dep.path("src/opencl.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    if (os == .macos) {
        opencl.linkFramework("OpenCL", .{});
        addMacosOpenCLFrameworkPaths(b, opencl);
    } else {
        opencl.linkSystemLibrary("OpenCL", .{});
    }

    lib.root_module.addImport("opencl", opencl);
    lib.root_module.addImport("vapoursynth", vapoursynth_dep);
    lib.root_module.link_libc = true;

    if (lib.root_module.optimize == .ReleaseFast) {
        lib.root_module.strip = true;
    }

    b.installArtifact(lib);
}

fn macosSdkPath(b: *std.Build) ?[]const u8 {
    if (b.graph.environ_map.get("SDKROOT")) |sdk| {
        const trimmed = std.mem.trim(u8, sdk, " \t\r\n");
        if (trimmed.len > 0) return trimmed;
    }
    var code: u8 = undefined;
    const stdout = b.runAllowFail(
        &.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" },
        &code,
        .ignore,
    ) catch return null;
    const sdk = std.mem.trim(u8, stdout, " \t\r\n");
    return if (sdk.len > 0) sdk else null;
}

fn addMacosOpenCLFrameworkPaths(b: *std.Build, opencl: *std.Build.Module) void {
    const host_fw = "/System/Library/Frameworks";
    opencl.addFrameworkPath(.{ .cwd_relative = host_fw });

    if (b.sysroot) |sdk| {
        opencl.addFrameworkPath(.{ .cwd_relative = b.fmt("{s}/System/Library/Frameworks", .{sdk}) });
    }

    // Prefer an older SDK that still bundles OpenCL when the active one does not.
    const legacy_sdks = [_][]const u8{
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.sdk",
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX13.sdk",
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
    };
    const io = b.graph.io;
    for (legacy_sdks) |sdk| {
        const fw = b.fmt("{s}/System/Library/Frameworks", .{sdk});
        std.Io.Dir.accessAbsolute(io, b.fmt("{s}/OpenCL.framework", .{fw}), .{}) catch continue;
        opencl.addFrameworkPath(.{ .cwd_relative = fw });
    }
}

fn ensureZigVersion(min_zig_version: std.SemanticVersion) !void {
    var installed_ver = @import("builtin").zig_version;
    installed_ver.build = null;

    if (installed_ver.order(min_zig_version) == .lt) {
        std.log.err("\n" ++
            \\---------------------------------------------------------------------------
            \\
            \\Installed Zig compiler version is too old.
            \\
            \\Min. required version: {any}
            \\Installed version: {any}
            \\
            \\Please install newer version and try again.
            \\Latest version can be found here: https://ziglang.org/download/
            \\
            \\---------------------------------------------------------------------------
            \\
        , .{ min_zig_version, installed_ver });
        return error.ZigIsTooOld;
    }
}
