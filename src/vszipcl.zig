pub const cl = @import("opencl");
pub const vapoursynth = @import("vapoursynth");
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
const zon = @import("zon");

const bilateral = @import("bilateral.zig");
const gaussglur = @import("gaussglur.zig");

export fn VapourSynthPluginInit2(plugin: *vs.Plugin, vspapi: *const vs.PLUGINAPI) void {
    ZAPI.Plugin.config("com.julek.vszipcl", "vszipcl", "VapourSynth Zig Image Process OpenCL", zon.version, plugin, vspapi);
    ZAPI.Plugin.function("Bilateral", "clip:vnode;sigma_spatial:float[]:opt;sigma_color:float[]:opt;radius:int[]:opt;", "clip:vnode;", bilateral.create, plugin, vspapi);
    ZAPI.Plugin.function("GaussBlur", "clip:vnode;sigma:float[]:opt;", "clip:vnode;", gaussglur.create, plugin, vspapi);
}
