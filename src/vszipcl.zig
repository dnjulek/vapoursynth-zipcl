pub const cl = @import("opencl");
pub const vapoursynth = @import("vapoursynth");
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
const zon = @import("zon");

const bilateral = @import("bilateral.zig");
const gaussglur = @import("gaussglur.zig");
const eedi3 = @import("eedi3.zig");

const eedi3_sig = "clip:vnode;field:int;dh:int:opt;mdis:int:opt;nrad:int:opt;alpha:float:opt;beta:float:opt;gamma:float:opt;hp:int:opt;vcheck:int:opt;vthresh0:float:opt;vthresh1:float:opt;vthresh2:float:opt;sclip:vnode:opt;";

export fn VapourSynthPluginInit2(plugin: *vs.Plugin, vspapi: *const vs.PLUGINAPI) void {
    ZAPI.Plugin.config("com.julek.vszipcl", "vszipcl", "VapourSynth Zig Image Process OpenCL", zon.version, plugin, vspapi);
    ZAPI.Plugin.function("Bilateral", "clip:vnode;sigma_spatial:float[]:opt;sigma_color:float[]:opt;radius:int[]:opt;", "clip:vnode;", bilateral.create, plugin, vspapi);
    ZAPI.Plugin.function("GaussBlur", "clip:vnode;sigma:float[]:opt;", "clip:vnode;", gaussglur.create, plugin, vspapi);
    ZAPI.Plugin.function("EEDI3", eedi3_sig, "clip:vnode;", eedi3.createEEDI3, plugin, vspapi);
    ZAPI.Plugin.function("EEDI3H", eedi3_sig, "clip:vnode;", eedi3.createEEDI3H, plugin, vspapi);
}
