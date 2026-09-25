/**
 * @file   platform_win.h
 * @brief  The platform.h interface for Windows.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * Needs Windows Vista or later (SRW locks, condition variables, InitOnce).
 * plat_discard uses DiscardVirtualMemory on 8.1 and later, MEM_RESET before.
 * <windows.h> is included lean: WIN32_LEAN_AND_MEAN skips the parts nothing
 * here uses (about 27k lines, a quarter of the compile's front end), and
 * NOMINMAX stops it from defining min / max macros, which would rewrite any
 * later use of those names.
 */

#pragma once
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>

/** Reserves bytes of address space (no memory behind it yet); NULL on failure. */
static inline void *plat_reserve(size_t bytes) {
    return VirtualAlloc(NULL, bytes, MEM_RESERVE, PAGE_READWRITE);
}

/** Commits [p, p+bytes) of a reservation, making it usable; 0 on failure. */
static inline int plat_commit(void *p, size_t bytes) {
    return VirtualAlloc(p, bytes, MEM_COMMIT, PAGE_READWRITE) != NULL;
}

/** DiscardVirtualMemory's type, for looking it up at run time. */
typedef DWORD(WINAPI *PlatDiscardFn)(PVOID, SIZE_T);

/** Gives the pages of [p, p+bytes) back to the OS, keeping them committed and
    reserved; their contents are lost. DiscardVirtualMemory (8.1+) is looked
    up on first use, without a lock: a racing thread at worst takes the
    MEM_RESET fallback once, which is also correct. */
static inline void plat_discard(void *p, size_t bytes) {
    static PlatDiscardFn discard;
    static volatile LONG looked;
    if (!looked) {
        discard = (PlatDiscardFn)(void *)GetProcAddress(GetModuleHandleA("kernel32.dll"), "DiscardVirtualMemory");
        looked = 1;
    }
    if (!discard || discard(p, bytes) != 0) VirtualAlloc(p, bytes, MEM_RESET, PAGE_READWRITE);
}

/** Returns bytes of heap memory aligned to align (a power of 2); NULL on failure. */
static inline void *plat_aligned_alloc(size_t bytes, size_t align) {
    return _aligned_malloc(bytes, align);
}

/** Frees memory from plat_aligned_alloc. */
static inline void plat_aligned_free(void *p) {
    _aligned_free(p);
}

/** A mutex: a slim reader/writer lock, used exclusively (no kernel object). */
typedef SRWLOCK plat_lock;
#define PLAT_LOCK_INIT SRWLOCK_INIT

/** Locks l. */
static inline void plat_lock_acquire(plat_lock *l) { AcquireSRWLockExclusive(l); }

/** Unlocks l. */
static inline void plat_lock_release(plat_lock *l) { ReleaseSRWLockExclusive(l); }

/** A condition variable, paired with a plat_lock. */
typedef CONDITION_VARIABLE plat_cond;
#define PLAT_COND_INIT CONDITION_VARIABLE_INIT

/** Releases l, sleeps until c is signaled, and locks l again. */
static inline void plat_cond_wait(plat_cond *c, plat_lock *l) { SleepConditionVariableSRW(c, l, INFINITE, 0); }

/** Wakes one thread waiting on c. */
static inline void plat_cond_signal(plat_cond *c) { WakeConditionVariable(c); }

/** Wakes every thread waiting on c. */
static inline void plat_cond_broadcast(plat_cond *c) { WakeAllConditionVariable(c); }

/** A run-once flag for plat_once_run. */
typedef INIT_ONCE plat_once;
#define PLAT_ONCE_INIT INIT_ONCE_STATIC_INIT

/** InitOnce callback: calls the void (*)(void) passed through as fn. */
static BOOL CALLBACK plat_once_cb(PINIT_ONCE once, PVOID fn, PVOID *ctx) {
    (void)once; (void)ctx;
    ((void (*)(void))fn)();
    return TRUE;
}

/** Calls fn() the first time any thread gets here with o; later callers wait
    until it has finished, then return. */
static inline void plat_once_run(plat_once *o, void (*fn)(void)) {
    InitOnceExecuteOnce(o, plat_once_cb, (PVOID)fn, NULL);
}

/** What a new thread runs: fn(arg), passed through CreateThread's one pointer. */
typedef struct {
    void (*fn)(void *);
    void *arg;
} PlatThreadStart;

/** Thread entry point: copies and frees the PlatThreadStart, then runs fn(arg). */
static DWORD WINAPI plat_thread_main(LPVOID p) {
    PlatThreadStart s = *(PlatThreadStart *)p;
    free(p);
    s.fn(s.arg);
    return 0;
}

/** Starts a detached thread running fn(arg); 0 on failure. The handle is
    closed at once: nothing ever joins the pool's threads. */
static inline int plat_thread_start(void (*fn)(void *), void *arg) {
    PlatThreadStart *s = (PlatThreadStart *)malloc(sizeof *s);
    if (!s) return 0;
    s->fn = fn;
    s->arg = arg;
    HANDLE h = CreateThread(NULL, 0, plat_thread_main, s, 0, NULL);
    if (!h) {
        free(s);
        return 0;
    }
    CloseHandle(h);
    return 1;
}

/** Atomically adds 1 to *x and returns the new value. */
static inline long plat_atomic_inc(volatile long *x) { return InterlockedIncrement(x); }

/** Atomically subtracts 1 from *x and returns the new value. */
static inline long plat_atomic_dec(volatile long *x) { return InterlockedDecrement(x); }

/** Reads *x atomically (acquire). Windows has no plain interlocked load, so
    this is the compiler builtin: a plain mov on x86-64. */
static inline long plat_atomic_load(volatile long *x) { return __atomic_load_n(x, __ATOMIC_ACQUIRE); }

/** Writes v to *x atomically (release); a plain mov on x86-64. */
static inline void plat_atomic_store(volatile long *x, long v) { __atomic_store_n(x, v, __ATOMIC_RELEASE); }

/** Returns the number of logical CPUs in this process's processor group (at
    most 64, which is also all one group's threads can run on). */
static inline int plat_cpu_count(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
}

/** Returns a monotonic clock in seconds (the performance counter). */
static inline double plat_seconds(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
