const std = @import("std");

const allocator = std.heap.c_allocator;
const threaded_io: std.Io = std.Io.Threaded.global_single_threaded.io();

pub fn Pool(comptime Stream: type, comptime Owner: type) type {
    return struct {
        const Self = @This();

        lock: std.atomic.Mutex = .unlocked,
        sem: std.Io.Semaphore = .{},
        free: std.ArrayListUnmanaged(*Stream) = .empty,
        all: std.ArrayListUnmanaged(*Stream) = .empty,
        owner: *Owner = undefined,

        fn enter(self: *Self) void {
            while (!self.lock.tryLock()) std.atomic.spinLoopHint();
        }

        pub fn prime(self: *Self, owner: *Owner, hint: usize) void {
            self.owner = owner;
            self.free.ensureTotalCapacity(allocator, hint) catch {};
            self.all.ensureTotalCapacity(allocator, hint) catch {};
        }

        pub fn prewarm(self: *Self, n: usize) !void {
            var i: usize = 0;
            while (i < n) : (i += 1) {
                const s = try allocator.create(Stream);
                s.init(self.owner) catch |err| {
                    allocator.destroy(s);
                    if (self.all.items.len > 0 and (err == error.OutOfDeviceMemory or err == error.OutOfResources or err == error.OutOfMemory)) break;
                    return err;
                };
                self.all.append(allocator, s) catch {
                    s.deinit();
                    allocator.destroy(s);
                    break;
                };
                self.free.appendAssumeCapacity(s);
            }
            if (self.all.items.len == 0) return error.OutOfResources;
            self.sem.permits = self.all.items.len;
        }

        pub fn acquire(self: *Self) *Stream {
            self.sem.waitUncancelable(threaded_io);
            self.enter();
            defer self.lock.unlock();
            return self.free.pop().?;
        }

        pub fn release(self: *Self, s: *Stream) void {
            self.enter();
            self.free.appendAssumeCapacity(s);
            self.lock.unlock();
            self.sem.post(threaded_io);
        }

        pub fn deinit(self: *Self) void {
            for (self.all.items) |s| {
                s.deinit();
                allocator.destroy(s);
            }
            self.all.deinit(allocator);
            self.free.deinit(allocator);
        }
    };
}
