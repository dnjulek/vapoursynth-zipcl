//! Thread-safe pool of per-frame OpenCL "streams".
//!
//! VapourSynth calls getFrame concurrently from up to `core.numThreads` worker
//! threads. A single shared command queue + device buffers + kernel would race
//! (clSetKernelArg and buffer contents are not safe to share across concurrent
//! enqueues), and would also serialize all work on one queue. Each Stream owns
//! its own queue, buffers and kernel(s), so frames run independently and their
//! host<->device transfers and compute overlap across the PCIe bus.
//!
//! Streams are created lazily on first use (one-time cost) and recycled via a
//! free stack. `Stream` must expose `fn init(*Stream, *Owner) !void` and
//! `fn deinit(*Stream) void`.

const std = @import("std");

const allocator = std.heap.c_allocator;

pub fn Pool(comptime Stream: type, comptime Owner: type) type {
    return struct {
        const Self = @This();

        // Spinlock guards the tiny free-list pop/append. A counting semaphore
        // (initialized to the stream count) throttles concurrency: when all
        // streams are in use, extra VS worker threads block in sem_wait (kernel
        // sleep) instead of busy-spinning — busy-spinning would steal CPU from
        // the productive thread and slow it down.
        lock: std.atomic.Mutex = .unlocked,
        sem: std.c.sem_t = undefined,
        free: std.ArrayListUnmanaged(*Stream) = .empty,
        all: std.ArrayListUnmanaged(*Stream) = .empty,
        owner: *Owner = undefined,

        fn enter(self: *Self) void {
            while (!self.lock.tryLock()) std.atomic.spinLoopHint();
        }

        /// `hint` reserves capacity for the expected number of concurrent
        /// streams (typically core.numThreads) to avoid reallocations.
        pub fn prime(self: *Self, owner: *Owner, hint: usize) void {
            self.owner = owner;
            self.free.ensureTotalCapacity(allocator, hint) catch {};
            self.all.ensureTotalCapacity(allocator, hint) catch {};
        }

        /// Eagerly create up to `n` streams on the calling thread. PoCL-CUDA binds
        /// a command queue's fast-DMA path to the thread/context it was created on,
        /// so streams MUST be built on the same thread as the CL context (in
        /// create()), never lazily on a worker thread. Creation stops early (without
        /// error) if the device runs out of memory, as long as at least one stream
        /// was made — fewer streams just means less frame-level concurrency.
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
            _ = std.c.sem_init(&self.sem, 0, @intCast(self.all.items.len));
        }

        /// Borrow a prewarmed stream, blocking (kernel sleep) until one is free.
        pub fn acquire(self: *Self) *Stream {
            while (std.c.sem_wait(&self.sem) != 0) {} // retry on EINTR
            self.enter();
            defer self.lock.unlock();
            return self.free.pop().?; // a permit guarantees a free stream
        }

        pub fn release(self: *Self, s: *Stream) void {
            self.enter();
            self.free.appendAssumeCapacity(s); // capacity reserved by prime()
            self.lock.unlock();
            _ = std.c.sem_post(&self.sem);
        }

        pub fn deinit(self: *Self) void {
            _ = std.c.sem_destroy(&self.sem);
            for (self.all.items) |s| {
                s.deinit();
                allocator.destroy(s);
            }
            self.all.deinit(allocator);
            self.free.deinit(allocator);
        }
    };
}
