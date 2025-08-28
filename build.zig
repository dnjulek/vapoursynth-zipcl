const std = @import("std");
const zon = @import("build.zig.zon");

//NOTE: read https://github.com/ziglang/zig/blob/master/lib/init/build.zig

pub fn build(b: *std.Build) !void {
    ensureZigVersion(try .parse(zon.minimum_zig_version)) catch return;
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const os = target.result.os.tag;

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

    if (os == .windows) {
        lib.addLibraryPath(.{ .cwd_relative = "C:/WINDOWS/system32" });
    }

    const vapoursynth_dep = b.dependency("vapoursynth", .{
        .target = target,
        .optimize = optimize,
    }).module("vapoursynth");

    const opencl = b.dependency("opencl", .{
        .target = target,
        .optimize = optimize,
    }).module("opencl");

    lib.root_module.addImport("opencl", opencl);
    lib.root_module.addImport("vapoursynth", vapoursynth_dep);
    lib.linkLibC();

    if (lib.root_module.optimize == .ReleaseFast) {
        lib.root_module.strip = true;
    }

    b.installArtifact(lib);
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
