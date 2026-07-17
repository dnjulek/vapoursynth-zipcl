const std = @import("std");
const vszipcl = @import("vszipcl.zig");
const cl = vszipcl.cl;

const allocator = std.heap.c_allocator;

const threaded_io: std.Io = std.Io.Threaded.global_single_threaded.io();

pub const CacheSlot = struct {
    key: i64 = -1,
    buf: cl.Buffer(u8) = undefined,
    has_buf: bool = false,
    ev: ?cl.c.cl_event = null,
    refs: u32 = 0,
    stamp: u64 = 0,
    ready: bool = false,
};

pub const FrameCache = struct {
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    slots: []CacheSlot = &.{},
    clock: u64 = 0,

    pub fn deinit(self: *FrameCache) void {
        for (self.slots) |*s| {
            if (s.ev) |ev| _ = cl.c.clReleaseEvent(ev);
            if (s.has_buf) s.buf.release();
        }
        if (self.slots.len > 0) allocator.free(self.slots);
        self.slots = &.{};
    }

    pub fn acquire(self: *FrameCache, keys: []const i64, idx: []usize, load: []bool) void {
        self.mutex.lockUncancelable(threaded_io);
        defer self.mutex.unlock(threaded_io);

        outer: while (true) {
            for (keys) |k| {
                for (self.slots) |*s| {
                    if (s.key == k and !s.ready) {
                        self.cond.waitUncancelable(threaded_io, &self.mutex);
                        continue :outer;
                    }
                }
            }

            var claimed: usize = 0;
            while (claimed < keys.len) : (claimed += 1) {
                const k = keys[claimed];
                load[claimed] = false;

                var found: ?usize = null;
                for (self.slots, 0..) |*s, si| {
                    if (s.key == k) {
                        found = si;
                        break;
                    }
                }
                if (found == null) {
                    var victim: ?usize = null;
                    for (self.slots, 0..) |*s, si| {
                        if (s.refs != 0) continue;
                        if (victim == null or s.stamp < self.slots[victim.?].stamp) victim = si;
                    }
                    if (victim) |v| {
                        self.slots[v].key = k;
                        self.slots[v].ready = false;
                        found = v;
                        load[claimed] = true;
                    } else {
                        for (0..claimed) |j| {
                            self.slots[idx[j]].refs -= 1;
                            if (load[j]) self.slots[idx[j]].key = -1;
                        }
                        self.cond.broadcast(threaded_io);
                        self.cond.waitUncancelable(threaded_io, &self.mutex);
                        continue :outer;
                    }
                }
                idx[claimed] = found.?;
                self.clock += 1;
                self.slots[found.?].refs += 1;
                self.slots[found.?].stamp = self.clock;
            }
            return;
        }
    }

    pub fn publish(self: *FrameCache, si: usize) void {
        self.mutex.lockUncancelable(threaded_io);
        self.slots[si].ready = true;
        self.mutex.unlock(threaded_io);
        self.cond.broadcast(threaded_io);
    }

    pub fn abandon(self: *FrameCache, si: usize) void {
        self.mutex.lockUncancelable(threaded_io);
        if (!self.slots[si].ready) self.slots[si].key = -1;
        self.mutex.unlock(threaded_io);
        self.cond.broadcast(threaded_io);
    }

    pub fn release(self: *FrameCache, idx: []const usize) void {
        self.mutex.lockUncancelable(threaded_io);
        for (idx) |si| self.slots[si].refs -= 1;
        self.mutex.unlock(threaded_io);
        self.cond.broadcast(threaded_io);
    }
};
