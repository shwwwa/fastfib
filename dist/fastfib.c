/* ======== fastfib.c ======== */
/**
 * @file   fastfib.c
 * @brief  fastfib: exact Fibonacci numbers F(n) for huge n.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * F(10^8) (69 million bits) in ~45 ms and F(10^10) in ~18 s on an i9-9900K,
 * using fast doubling on top of a 3-prime AVX2 number-theoretic transform
 * multiply. No external libraries. Needs gcc or clang on x86-64 with AVX2;
 * the OS layer exists for Windows so far (src/platform.h).
 *
 *   Build:  gcc -O3 -march=native -o fastfib.exe fastfib.c
 *   Run:    fastfib.exe 100000000 [--print]    (--help for usage)
 *
 * This file is the whole program: it includes the modules in src/ in order
 * (a "unity build"), so the compiler sees one translation unit and inlines
 * across modules. Do not compile the src/ files on their own. (In
 * dist/fastfib.c the modules follow inline, in the same order.) Tunables are
 * #ifndef macros next to the code that uses them; override with -DNAME=value.
 * Define UNIT_TEST_NO_MAIN to include this file from a test harness.
 *
 * Layers, bottom to top:
 *   common.h        includes, the Big type, u128
 *   platform.h      the OS interface (memory, threads, locks, time)
 *   alloc.c         arena allocator: xmalloc / xfree
 *   pool.c          thread pool: pool_submit / pool_wait / pool_run
 *   bigint.c        add, sub, shift, compare, schoolbook
 *   mul_basic.c     big_mul / big_sqr dispatch, Karatsuba, Toom-3
 *   ntt_arith.c     primes and AVX2 Montgomery arithmetic
 *   ntt_tables.c    root tables (ntt_init)
 *   ntt_kernels.c   butterflies and transform passes
 *   ntt_conv.c      power-of-2 convolution, ntt_fwd / ntt_inv, thread settings
 *   ntt_radix3.c    3 * 2^k lengths, ntt_conv_any
 *   crt.c           three primes -> one bignum
 *   mul_ntt.c       big_mul_ntt, coefficient width choice
 *   mul_split.c     products past the longest transform
 *   decimal.c       big_to_string
 *   fib.c           fibonacci(n): fast doubling
 *   main.c          command line
 */

/* ======== src/common.h ======== */
/**
 * @file   common.h
 * @brief  Includes, core types and settings shared by every module.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * fastfib.c is a unity build: it includes every module in a fixed order
 * and the compiler sees one unit, so it can inline across modules (the speed
 * depends on that). Modules are never compiled on their own; each includes
 * the earlier ones it uses only so that an editor can analyze it alone.
 */


// Stop early with one clear message instead of many errors deep in the code
#if !defined(__SIZEOF_INT128__) || !defined(__x86_64__)
#error "fastfib.c needs gcc or clang on x86-64 (it uses __int128, gcc builtins and AVX2)"
#elif !defined(__AVX2__)
#error "fastfib.c needs AVX2: build with -march=native on an AVX2 CPU (or -mavx2)"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h> /* AVX2 intrinsics and _addcarry_u64 (x86-64 only) */

/** Arbitrary-precision unsigned integer: value = sum of d[i] * 2^(64*i). */
typedef struct {
    uint64_t *d; /* limbs, lowest first */
    size_t n;    /* number of limbs, always >= 1; normalized: no leading zero limb, zero is {0} */
} Big;

/* 128-bit unsigned integer: holds a full 64 x 64-bit product */
typedef unsigned __int128 u128;

/* The limb of the shared zero: empty views point here (see view_slice).
   Never written and never freed (big_free skips it). */
static uint64_t g_zero_limb = 0;

/* Longest NTT length is 2^NTT_MAX_LG. A length-2^k transform mod p needs 2^k
   to divide p - 1; of the three primes, p1 - 1 = 63 * 2^25 has the smallest
   power of 2. Here and not in the NTT modules: mul_basic.c (ntt_fits) comes
   before them. */
#ifndef NTT_MAX_LG
#define NTT_MAX_LG 25
#endif
/* ======== src/platform.h ======== */
/**
 * @file   platform.h
 * @brief  Everything the program needs from the operating system.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * All OS-specific code lives in one file per OS that implements this
 * interface; the rest of the program uses only these names. Only Windows
 * exists so far (platform_win.h). Porting to another OS means writing its
 * platform_<os>.h with the same names and adding it below; nothing else
 * changes. The wrappers are static inline, so they cost nothing.
 *
 * Memory (the arena allocator, alloc.c):
 *   void *plat_reserve(size_t bytes)       address space only, no memory yet;
 *                                          NULL on failure
 *   int   plat_commit(void *p, size_t n)   make [p, p+n) of a reservation
 *                                          usable; 0 on failure
 *   void  plat_discard(void *p, size_t n)  give the pages of [p, p+n) back to
 *                                          the OS but keep the addresses (p
 *                                          and n page-aligned); contents lost
 *   void *plat_aligned_alloc(size_t n, size_t align), plat_aligned_free(p)
 *
 * Threads and synchronization (the thread pool, pool.c):
 *   plat_lock   a mutex; PLAT_LOCK_INIT; plat_lock_acquire / plat_lock_release
 *   plat_cond   a condition variable; PLAT_COND_INIT;
 *               plat_cond_wait(cond, lock), plat_cond_signal, plat_cond_broadcast
 *   plat_once   PLAT_ONCE_INIT; plat_once_run(once, fn) calls fn() exactly once
 *   int  plat_thread_start(void (*fn)(void *), void *arg)
 *               starts a detached thread running fn(arg); 0 on failure
 *   long plat_atomic_inc(volatile long *), plat_atomic_dec(volatile long *)
 *               return the new value (full barriers)
 *   long plat_atomic_load(volatile long *)            acquire
 *   void plat_atomic_store(volatile long *, long)     release
 *   int  plat_cpu_count(void)   logical CPUs the pool can use (>= 1)
 *
 * Time (main.c):
 *   double plat_seconds(void)   a monotonic clock in seconds
 */


#if defined(_WIN32)
/* ======== src/platform_win.h ======== */
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
#else
#error "no platform layer for this OS yet: write src/platform_<os>.h implementing the interface in src/platform.h"
#endif
/* ======== src/alloc.c ======== */
/**
 * @file   alloc.c
 * @brief  Memory allocation: xmalloc, xfree and xcalloc.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * Blocks of ARENA_MIN_BYTES and up come from one reserved address range (the
 * arena) with an address-ordered, coalescing free list, so each doubling step
 * reuses the pages the earlier ones touched. Smaller blocks come from the C
 * heap. Every block is 32-byte aligned for AVX2; running out of memory ends
 * the program with a message.
 */


/* Big-block arena. Faulting in a fresh page costs about as much as a pass of
   NTT work over it, and each doubling step needs bigger blocks than the last
   one freed. So all blocks from ARENA_MIN_BYTES up share one reserved range
   with a coalescing, address-ordered free list: new blocks land on pages
   already touched, and a run faults in about its peak footprint once. */
#ifndef ARENA_MIN_BYTES
#define ARENA_MIN_BYTES 4096
#endif
#ifndef ARENA_RESERVE
/* address space only; 64 GB is far past what fits in RAM (F(10^10) peaks near
   7 GB) and blocks beyond it fall back to the heap. Reserving 256 GB cost
   ~0.12 ms at startup, a quarter of an F(10^5) run. */
#define ARENA_RESERVE ((size_t)1 << 36)
#endif
#ifndef ARENA_COMMIT_STEP
/* pages are committed in steps of this size as the arena grows */
#define ARENA_COMMIT_STEP ((size_t)2 << 20)
#endif
/* Header size and block alignment. AVX2 needs 32; 64 = one cache line, so
   no two blocks share a line (threads writing neighbouring blocks don't
   slow each other down) */
#define ARENA_ALIGN 64
/* free-list capacity; a run uses a few dozen entries */
#define ARENA_MAX_FREE 512

/** A free range of the arena, as byte offsets from g_arena. */
typedef struct {
    size_t off, len;
} ArenaRange;

/* guards everything below */
static plat_lock g_arena_lock = PLAT_LOCK_INIT;
static unsigned char *g_arena;
static int g_arena_failed;
static size_t g_arena_top, g_arena_committed;
static ArenaRange g_arena_free[ARENA_MAX_FREE];
static int g_arena_nfree;

/* When the arena grows past its high-water mark, free ranges this big are
   fragmentation holes: their pages go back to the OS (addresses kept). Saves
   ~1 GB at F(10^10). Only then, so blocks reused in place keep warm pages. */
#ifndef ARENA_DISCARD_MIN
#define ARENA_DISCARD_MIN ((size_t)256 << 20)
#endif
static size_t g_arena_hw; /* highest top so far */

/** Gives the pages fully inside [lo, hi) back to the OS: their contents are
    dropped, the addresses stay reserved. Partial pages at the ends are kept. */
static void arena_discard(unsigned char *lo, unsigned char *hi) {
    // round lo up and hi down to 4 KB page boundaries
    uintptr_t a = ((uintptr_t)lo + 4095) & ~(uintptr_t)4095, b = (uintptr_t)hi & ~(uintptr_t)4095;
    if (b <= a) return;
    plat_discard((void *)a, b - a);
}

/** Returns the offset of a new block of len bytes, or SIZE_MAX if the arena
    is full; lock held. Best fit: the smallest free range that is big enough,
    so big ranges stay whole for big blocks. Otherwise the arena grows at the
    top, committing more pages as needed. */
static size_t arena_take(size_t len) {
    int best = -1;
    for (int i = 0; i < g_arena_nfree; i++)
        if (g_arena_free[i].len >= len && (best < 0 || g_arena_free[i].len < g_arena_free[best].len)) best = i;
    if (best >= 0) {
        size_t off = g_arena_free[best].off;
        g_arena_free[best].off += len;
        g_arena_free[best].len -= len;
        if (!g_arena_free[best].len) {
            memmove(&g_arena_free[best], &g_arena_free[best + 1], (g_arena_nfree - best - 1) * sizeof(ArenaRange));
            g_arena_nfree--;
        }
        return off;
    }

    /* grow at the top, starting inside a free range that ends there */
    size_t off = g_arena_top;
    int tail = g_arena_nfree > 0 && g_arena_free[g_arena_nfree - 1].off + g_arena_free[g_arena_nfree - 1].len == g_arena_top;
    if (tail) off = g_arena_free[g_arena_nfree - 1].off;
    size_t top = off + len;
    if (top > ARENA_RESERVE) return SIZE_MAX;
    if (top > g_arena_committed) {
        size_t c = (top + ARENA_COMMIT_STEP - 1) / ARENA_COMMIT_STEP * ARENA_COMMIT_STEP;
        if (c > ARENA_RESERVE) c = ARENA_RESERVE;
        if (!plat_commit(g_arena + g_arena_committed, c - g_arena_committed)) return SIZE_MAX;
        g_arena_committed = c;
    }

    if (tail) g_arena_nfree--;
    if (top > g_arena_hw) { /* growing the footprint: drop the pages of big holes */
        for (int i = 0; i < g_arena_nfree; i++)
            if (g_arena_free[i].len >= ARENA_DISCARD_MIN)
                arena_discard(g_arena + g_arena_free[i].off, g_arena + g_arena_free[i].off + g_arena_free[i].len);
        g_arena_hw = top;
    }
    g_arena_top = top;
    return off;
}

/** Returns the block at off to the free list, merging it with free
   neighbours; lock held. If the list is full the block is not recorded
   (it leaks, which is harmless: at most a few blocks in rare cases). */
static void arena_give(size_t off, size_t len) {
    int i = 0;
    while (i < g_arena_nfree && g_arena_free[i].off < off) i++;
    int left = i > 0 && g_arena_free[i - 1].off + g_arena_free[i - 1].len == off;
    int right = i < g_arena_nfree && off + len == g_arena_free[i].off;

    if (left && right) {
        g_arena_free[i - 1].len += len + g_arena_free[i].len;
        memmove(&g_arena_free[i], &g_arena_free[i + 1], (g_arena_nfree - i - 1) * sizeof(ArenaRange));
        g_arena_nfree--;
    } else if (left) {
        g_arena_free[i - 1].len += len;
    } else if (right) {
        g_arena_free[i].off = off;
        g_arena_free[i].len += len;
    } else if (g_arena_nfree < ARENA_MAX_FREE) {
        memmove(&g_arena_free[i + 1], &g_arena_free[i], (g_arena_nfree - i) * sizeof(ArenaRange));
        g_arena_free[i] = (ArenaRange){off, len};
        g_arena_nfree++;
    }
}

/** Ends the program: an allocation of `bytes` failed. */
static void out_of_memory(size_t bytes) {
    fprintf(stderr, "out of memory (%zu bytes)\n", bytes);
    exit(1);
}

/** Every block is 32-byte aligned (for AVX2). Arena blocks sit behind an
   ARENA_ALIGN-byte header holding their length; heap blocks have no header
   (xfree tells the two apart by address). */
static void *xmalloc(size_t bytes) {
    // nothing real gets near half the address space; this keeps the rounding below from wrapping
    if (bytes > SIZE_MAX / 2) out_of_memory(bytes);

    if (bytes >= ARENA_MIN_BYTES) {
        // header (ARENA_ALIGN bytes) + the block, rounded up to ARENA_ALIGN
        size_t len = (bytes + 2 * ARENA_ALIGN - 1) / ARENA_ALIGN * ARENA_ALIGN, off = SIZE_MAX;
        plat_lock_acquire(&g_arena_lock);
        // the first big block reserves the whole arena (address space only)
        if (!g_arena && !g_arena_failed) {
            g_arena = plat_reserve(ARENA_RESERVE);
            g_arena_failed = !g_arena;
        }
        if (g_arena) off = arena_take(len);
        plat_lock_release(&g_arena_lock);
        if (off != SIZE_MAX) {
            *(size_t *)(g_arena + off) = len;
            return g_arena + off + ARENA_ALIGN;
        }
    }

    void *p = plat_aligned_alloc(bytes ? bytes : 1, 32);
    if (!p) out_of_memory(bytes);
    return p;
}

/** Frees a block from xmalloc. Arena blocks are recognized by their address
    and go back to the free list (their length is in the header); others go
    back to the heap. NULL is ignored. */
static void xfree(void *ptr) {
    if (!ptr) return;
    uintptr_t p = (uintptr_t)ptr, base = (uintptr_t)g_arena;
    if (g_arena && p > base && p < base + ARENA_RESERVE) {
        size_t off = p - ARENA_ALIGN - base;
        plat_lock_acquire(&g_arena_lock);
        arena_give(off, *(size_t *)(g_arena + off));
        plat_lock_release(&g_arena_lock);
        return;
    }
    plat_aligned_free(ptr);
}

/** Returns n * size zeroed bytes, like calloc. Requires freeing with xfree. */
static void *xcalloc(size_t n, size_t size) {
    if (size && n > SIZE_MAX / size) out_of_memory(SIZE_MAX); /* n * size would wrap */
    void *ptr = xmalloc(n * size);
    memset(ptr, 0, n * size);
    return ptr;
}
/* ======== src/pool.c ======== */
/**
 * @file   pool.c
 * @brief  A fixed thread pool with fork-join task groups.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * pool_submit(group, fn, arg) queues a task, pool_wait(group) runs queued
 * tasks until the group's tasks are done. pool_run does both for an array
 * of tasks.
 *
 * Creating an OS thread costs tens of microseconds and one big multiply used
 * to create around twenty, so a fixed set of workers pulls tasks off a shared
 * stack instead. A thread waiting on its tasks runs queued tasks itself in the
 * meantime, so nested fork-join (tasks that spawn and wait for subtasks) can
 * never deadlock with every worker blocked. Idle threads spin a while before
 * sleeping on a condition variable, since the transforms fork and join
 * constantly. The workers start on the first pool_submit.
 */


#ifndef POOL_WORKERS
#define POOL_WORKERS -1 /* worker threads besides the main one; -1: one per other logical CPU */
#endif
#define POOL_QUEUE 1024 /* task slots; a submit to a full stack runs the task at once */

/** A task: fn(arg). */
typedef void (*TaskFn)(void *);

/** A fork-join group: pool_wait(g) returns once all its tasks have finished.
    Zero-initialize it ({0}) before the first pool_submit. */
typedef struct {
    volatile long pending; /* tasks submitted and not finished yet; read with plat_atomic_load */
} TaskGroup;

/** A queued task and the group it counts against. */
typedef struct {
    TaskFn fn;
    void *arg;
    TaskGroup *g;
} PoolTask;

/* Idle threads spin this many pause rounds (~30 ns each) before sleeping:
   waking a sleeping thread costs tens of microseconds, as long as a whole
   small task, and the transforms fork and join constantly. */
#ifndef POOL_SPIN
#define POOL_SPIN 4000
#endif

static plat_lock g_tp_lock = PLAT_LOCK_INIT;  /* guards the stack and g_tp_sleepers */
static plat_cond g_tp_cv = PLAT_COND_INIT;    /* sleepers wait here for tasks or finished groups */
static PoolTask g_tp_queue[POOL_QUEUE];       /* a stack: last in, first out */
static volatile long g_tp_count = 0;          /* tasks on the stack; written under the lock, peeked without */
static int g_tp_sleepers = 0; /* threads asleep on g_tp_cv, guarded by g_tp_lock */
static plat_once g_tp_once = PLAT_ONCE_INIT;
static int g_tp_cpus, g_tp_workers;           /* set once by tp_start */

/** Runs task t and counts it done. The thread that finishes a group's last
    task wakes the sleepers, since a pool_wait may be asleep on it. The
    decrement is a full barrier, so a waiter that sees pending == 0 also
    sees everything the tasks wrote. */
static void tp_run(PoolTask t) {
    t.fn(t.arg);
    if (plat_atomic_dec(&t.g->pending) == 0) {
        /* under the lock, a waiter has either not checked pending yet or is
           already counted as asleep */
        plat_lock_acquire(&g_tp_lock);
        int wake = g_tp_sleepers > 0;
        plat_lock_release(&g_tp_lock);
        if (wake) plat_cond_broadcast(&g_tp_cv);
    }
}

/** Takes the newest task off the stack into *t; returns 0 if it was empty.
    An unlocked peek first keeps spinning threads off the lock. */
static int tp_try_pop(PoolTask *t) {
    if (plat_atomic_load(&g_tp_count) == 0) return 0; /* unlocked peek */
    plat_lock_acquire(&g_tp_lock);
    long n = g_tp_count;
    if (n > 0) {
        *t = g_tp_queue[n - 1];
        plat_atomic_store(&g_tp_count, n - 1);
    }
    plat_lock_release(&g_tp_lock);
    return n > 0;
}

/** Worker thread: runs tasks forever, spinning POOL_SPIN rounds when idle
    before it sleeps. The first worker (spawner != NULL) starts the others,
    so the thread that first submits work pays for starting one thread
    rather than all of them. */
static void tp_worker(void *spawner) {
    if (spawner)
        for (int i = 1; i < g_tp_workers; i++) plat_thread_start(tp_worker, NULL);
    for (;;) {
        PoolTask t;
        int got = 0;
        for (int s = 0; s < POOL_SPIN && !(got = tp_try_pop(&t)); s++) _mm_pause();
        if (!got) {
            plat_lock_acquire(&g_tp_lock);
            g_tp_sleepers++;
            while (g_tp_count == 0) plat_cond_wait(&g_tp_cv, &g_tp_lock);
            g_tp_sleepers--;
            t = g_tp_queue[g_tp_count - 1];
            plat_atomic_store(&g_tp_count, g_tp_count - 1);
            plat_lock_release(&g_tp_lock);
        }
        tp_run(t);
    }
}

/** Starts the pool (once, from pool_submit or pool_cpus): one spawner
    worker, which starts the rest. With POOL_WORKERS 0, tasks run in
    pool_wait instead. */
static void tp_start(void) {
    g_tp_cpus = plat_cpu_count();
    g_tp_workers = POOL_WORKERS >= 0 ? POOL_WORKERS : g_tp_cpus - 1;
    if (g_tp_workers > 0) plat_thread_start(tp_worker, (void *)1);
}

/** Returns the number of logical CPUs (16 on the i9-9900K). Starts the pool. */
static int pool_cpus(void) {
    plat_once_run(&g_tp_once, tp_start);
    return g_tp_cpus;
}

/** Queues fn(arg) as a task of group g, and wakes one sleeping thread if
    there is one (spinning threads find the task themselves). */
static void pool_submit(TaskGroup *g, TaskFn fn, void *arg) {
    plat_once_run(&g_tp_once, tp_start);
    plat_atomic_inc(&g->pending);
    plat_lock_acquire(&g_tp_lock);
    if (g_tp_count == POOL_QUEUE) { /* full: just run it here */
        plat_lock_release(&g_tp_lock);
        tp_run((PoolTask){fn, arg, g});
        return;
    }
    g_tp_queue[g_tp_count] = (PoolTask){fn, arg, g};
    plat_atomic_store(&g_tp_count, g_tp_count + 1);
    int wake = g_tp_sleepers > 0; /* spinning threads will find it themselves */
    plat_lock_release(&g_tp_lock);
    if (wake) plat_cond_signal(&g_tp_cv);
}

/** Returns once every task of g has finished. Meanwhile this thread runs
    queued tasks (any group's), which is what keeps nested fork-join from
    deadlocking; with nothing to run it spins, then sleeps until a task
    arrives or a group finishes. It sleeps only after checking, under the
    lock, that g is still pending: tp_run takes the same lock before waking,
    so the wakeup can't be missed. */
static void pool_wait(TaskGroup *g) {
    int spins = 0;
    while (plat_atomic_load(&g->pending) > 0) {
        PoolTask t;
        if (tp_try_pop(&t)) {
            tp_run(t);
            spins = 0;
        } else if (++spins < POOL_SPIN) {
            _mm_pause();
        } else {
            plat_lock_acquire(&g_tp_lock);
            if (plat_atomic_load(&g->pending) > 0 && g_tp_count == 0) {
                g_tp_sleepers++;
                plat_cond_wait(&g_tp_cv, &g_tp_lock);
                g_tp_sleepers--;
            }
            plat_lock_release(&g_tp_lock);
            spins = 0;
        }
    }

    // pass on a task wake this thread may have absorbed; taking the lock also
    // keeps the caller's reads of the results after the pending check
    plat_lock_acquire(&g_tp_lock);
    int wake = g_tp_count > 0 && g_tp_sleepers > 0;
    plat_lock_release(&g_tp_lock);
    if (wake) plat_cond_signal(&g_tp_cv);
}

/** Runs fn on each of count >= 1 items (item i at items + i * size) and
    returns when all are done: items 1.. as pool tasks, item 0 on this
    thread. Always inlined, so fn is a constant at each call site and item 0
    runs as a direct call. */
static inline __attribute__((always_inline)) void pool_run(TaskFn fn, void *items, int count, size_t size) {
    TaskGroup g = {0};
    for (int i = 1; i < count; i++) pool_submit(&g, fn, (char *)items + i * size);
    fn(items);
    pool_wait(&g);
}
/* ======== src/bigint.c ======== */
/**
 * @file   bigint.c
 * @brief  Basic arithmetic on Big: unsigned numbers in 64-bit limbs.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * Add, subtract (also in place, at a limb offset), shifts, compare,
 * schoolbook multiply and square, division by a small number, and SBig
 * (sign + magnitude) for Toom-3. Every function keeps a Big normalized:
 * n >= 1 and the top limb nonzero, except zero itself, which is {0}, n = 1.
 */


/* ==================== Basics ==================== */

/** Returns the length of d[0..n) without its leading zero limbs (at least 1). */
static size_t normalize_len(const uint64_t *d, size_t n) {
    while (n > 1 && d[n - 1] == 0) n--;
    return n;
}

/** Frees x and marks it dead (d = NULL, n = 0). The shared zero limb is never freed. */
static void big_free(Big *x) {
    if (x->d != &g_zero_limb) xfree(x->d);
    x->d = NULL;
    x->n = 0;
}

/** Returns v as a new one-limb Big. Requires freeing with big_free. */
static Big big_from_u64(uint64_t v) {
    Big r = {xmalloc(sizeof(uint64_t)), 1};
    r.d[0] = v;
    return r;
}

/** Returns true if a is 0 (relies on normalization: zero is exactly {0}). */
static int big_is_zero(const Big *a) {
    return a->n == 1 && a->d[0] == 0;
}

/** Returns the number of bits of a: the top limb's bits + 64 per limb below.
    a must not be 0 (clz of 0 is undefined). */
static size_t big_bits(const Big *a) {
    return 64 * (a->n - 1) + (64 - (size_t)__builtin_clzll(a->d[a->n - 1]));
}

/** Returns a view of a's limbs [lo, hi): points into a, copies nothing.
    Never pass it to big_free. An empty range gives the shared zero. */
static Big view_slice(const Big *a, size_t lo, size_t hi) {
    if (hi > a->n) hi = a->n;
    if (lo >= hi) {
        Big z = {&g_zero_limb, 1};
        return z;
    }
    Big v = {a->d + lo, normalize_len(a->d + lo, hi - lo)};
    return v;
}

/** Returns a copy of a (its own limbs, also for a view). Requires freeing with big_free. */
static Big big_copy(const Big *a) {
    Big r = {xmalloc(a->n * sizeof(uint64_t)), a->n};
    memcpy(r.d, a->d, a->n * sizeof(uint64_t));
    return r;
}

/* ==================== Addition and subtraction ==================== */

/** Adds one limb v into d at position pos and ripples the carry up
    (usually it dies after one limb). Returns the carry out of d[len-1]. */
static uint64_t limbs_add_u64(uint64_t *d, size_t len, size_t pos, uint64_t v) {
    for (; v && pos < len; pos++) {
        d[pos] += v;
        v = d[pos] < v;
    }
    return v;
}

/** Subtracts one limb v from d at position pos and ripples the borrow up.
    Returns the borrow out of d[len-1]. */
static uint64_t limbs_sub_u64(uint64_t *d, size_t len, size_t pos, uint64_t v) {
    for (; v && pos < len; pos++) {
        uint64_t old = d[pos];
        d[pos] = old - v;
        v = old < v;
    }
    return v;
}

/** Returns a+b as a new Big. Requires freeing with big_free.
    The common limbs go through one add-with-carry chain (adc); the rest of
    the longer number is copied and the carry rippled into it. */
static Big big_add(const Big *a, const Big *b) {
    if (a->n < b->n) { const Big *t = a; a = b; b = t; } /* a is the longer one */
    size_t n = a->n + 1;
    uint64_t *d = xmalloc(n * sizeof(uint64_t));
    unsigned char c = 0;
    for (size_t i = 0; i < b->n; i++) c = _addcarry_u64(c, a->d[i], b->d[i], (unsigned long long *)&d[i]);
    memcpy(d + b->n, a->d + b->n, (a->n - b->n) * sizeof(uint64_t));
    d[a->n] = 0;
    limbs_add_u64(d, n, b->n, c);
    Big r = {d, normalize_len(d, n)};
    return r;
}

/** Returns a-b as a new Big (requires a >= b). Requires freeing with big_free.
    Same as big_add with a subtract-with-borrow chain (sbb). */
static Big big_sub(const Big *a, const Big *b) {
    size_t bn = b->n < a->n ? b->n : a->n;
    uint64_t *d = xmalloc(a->n * sizeof(uint64_t));
    unsigned char c = 0;
    for (size_t i = 0; i < bn; i++) c = _subborrow_u64(c, a->d[i], b->d[i], (unsigned long long *)&d[i]);
    memcpy(d + bn, a->d + bn, (a->n - bn) * sizeof(uint64_t));
    limbs_sub_u64(d, a->n, bn, c);
    Big r = {d, normalize_len(d, a->n)};
    return r;
}

/** Adds x into d at limb offset off (d += x * 2^(64*off)), in place: no
    shifted copy of x. Returns the carry out of d[dn-1]: 0 when the caller
    made room for the sum. */
static uint64_t big_add_at(uint64_t *d, size_t dn, const Big *x, size_t off) {
    if (big_is_zero(x)) return 0;
    size_t xn = normalize_len(x->d, x->n);
    unsigned char c = 0;
    for (size_t i = 0; i < xn; i++) c = _addcarry_u64(c, d[off + i], x->d[i], (unsigned long long *)&d[off + i]);
    return limbs_add_u64(d, dn, off + xn, c);
}

/** Subtracts x from d at limb offset off, in place. The caller guarantees
    the result is >= 0. */
static void big_sub_at(uint64_t *d, size_t dn, const Big *x, size_t off) {
    if (big_is_zero(x)) return;
    size_t xn = normalize_len(x->d, x->n);
    unsigned char c = 0;
    for (size_t i = 0; i < xn; i++) c = _subborrow_u64(c, d[off + i], x->d[i], (unsigned long long *)&d[off + i]);
    limbs_sub_u64(d, dn, off + xn, c);
}

/* ==================== In-place addition and subtraction ==================== */

/** Subtracts y from x in place (requires x >= y). */
static void big_sub_inplace(Big *x, const Big *y) {
    big_sub_at(x->d, x->n, y, 0);
    x->n = normalize_len(x->d, x->n);
}

/** Adds y to x in place. x gets a new buffer only when y is longer, x is
    the shared zero (must never be written), or the sum carries out of the top. */
static void big_add_inplace(Big *x, const Big *y) {
    if (y->n > x->n || x->d == &g_zero_limb) {
        Big t = big_add(x, y);
        big_free(x);
        *x = t;
        return;
    }
    if (big_add_at(x->d, x->n, y, 0)) {
        // carry out of the top: one limb longer, and the new top limb is 1
        size_t n = x->n;
        uint64_t *d = xmalloc((n + 1) * sizeof(uint64_t));
        memcpy(d, x->d, n * sizeof(uint64_t));
        d[n] = 1;
        big_free(x);
        x->d = d;
        x->n = n + 1;
    }
}

/** Adds one limb v to x in place. v is passed as a one-limb view of the
    local variable, so nothing is allocated. */
static void big_add_u64_inplace(Big *x, uint64_t v) {
    Big t = {&v, 1};
    big_add_inplace(x, &t);
}

/** Subtracts one limb v from x in place (requires x >= v); see big_add_u64_inplace. */
static void big_sub_u64_inplace(Big *x, uint64_t v) {
    Big t = {&v, 1};
    big_sub_inplace(x, &t);
}

/* ==================== Shifts ==================== */

/** Returns 2a as a new Big (one limb longer for the bit that comes out of the top).
    Requires freeing with big_free. */
static Big big_shl1(const Big *a) {
    uint64_t *d = xmalloc((a->n + 1) * sizeof(uint64_t));
    uint64_t carry = 0;
    for (size_t i = 0; i < a->n; i++) {
        uint64_t v = a->d[i];
        // own bits one up + the top bit of the limb below
        d[i] = (v << 1) | carry;
        // the top bit falls out into the next limb
        carry = v >> 63;
    }
    d[a->n] = carry;
    Big r = {d, normalize_len(d, a->n + 1)};
    return r;
}

/** Returns a >> k = floor(a / 2^k) as a new Big, for any k: s = k/64 whole
    limbs are skipped, the remaining b = k%64 bits taken from two limbs.
    Requires freeing with big_free. */
static Big big_shr_bits(const Big *a, size_t k) {
    size_t s = k / 64, b = k % 64;
    if (s >= a->n) return big_from_u64(0);
    size_t n = a->n - s;
    uint64_t *d = xmalloc(n * sizeof(uint64_t));
    for (size_t i = 0; i < n; i++) {
        uint64_t lo = a->d[i + s], hi = i + s + 1 < a->n ? a->d[i + s + 1] : 0;
        // b == 0 separately: hi << 64 is undefined in C (x86 would shift by 0)
        d[i] = b ? (lo >> b) | (hi << (64 - b)) : lo;
    }
    Big r = {d, normalize_len(d, n)};
    return r;
}

/** Returns a << k = a * 2^k as a new Big, for any k: s = k/64 zero limbs at
    the bottom, then the limbs moved up by b = k%64 bits.
    Requires freeing with big_free. */
static Big big_shl_bits(const Big *a, size_t k) {
    if (big_is_zero(a)) return big_from_u64(0);
    size_t s = k / 64, b = k % 64, n = a->n + s + 1;
    uint64_t *d = xmalloc(n * sizeof(uint64_t));
    memset(d, 0, s * sizeof(uint64_t));

    // whole limbs only: a copy (and prev >> 64 would be undefined)
    if (b == 0) {
        memcpy(d + s, a->d, a->n * sizeof(uint64_t));
        d[n - 1] = 0;
    } else {
        // each limb = its own bits moved up + the top bits of the limb below
        uint64_t prev = 0;
        for (size_t i = 0; i < a->n; i++) {
            d[i + s] = (a->d[i] << b) | (prev >> (64 - b));
            prev = a->d[i];
        }
        d[n - 1] = prev >> (64 - b);
    }
    Big r = {d, normalize_len(d, n)};
    return r;
}

/** Returns 2^k as a new Big: a single 1 bit. Requires freeing with big_free. */
static Big big_pow2(size_t k) {
    Big r = {xcalloc(k / 64 + 1, sizeof(uint64_t)), k / 64 + 1};
    r.d[k / 64] = 1ull << (k % 64);
    return r;
}

/* ==================== Schoolbook multiplication ==================== */

/** Returns a*b as a new Big by long multiplication (n^2): each limb of a
    times all of b, added in shifted by i limbs. Requires freeing with big_free. */
static Big big_mul_schoolbook(const Big *a, const Big *b) {
    if (big_is_zero(a) || big_is_zero(b)) return big_from_u64(0);
    size_t n = a->n + b->n;
    uint64_t *d = xcalloc(n, sizeof(uint64_t));
    for (size_t i = 0; i < a->n; i++) {
        uint64_t ai = a->d[i];
        if (ai == 0) continue;
        uint64_t carry = 0;
        for (size_t j = 0; j < b->n; j++) {
            u128 cur = (u128)d[i + j] + (u128)ai * b->d[j] + carry;
            d[i + j] = (uint64_t)cur;
            carry = (uint64_t)(cur >> 64);
        }
        /* no earlier row reached d[i + b->n] (row i-1 ended at i-1 + b->n),
           so it is still 0: the carry is stored, never propagated */
        d[i + b->n] = carry;
    }
    Big r = {d, normalize_len(d, n)};
    return r;
}

/** Returns a*a as a new Big with about half the products of big_mul_schoolbook:
    a^2 = sum a[i]^2 * B^(2i) + 2 * sum(i<j) a[i]*a[j] * B^(i+j).
    Pass 1: the cross products once. Pass 2: double them and add the squares.
    Requires freeing with big_free. */
static Big big_sqr_schoolbook(const Big *a) {
    if (big_is_zero(a)) return big_from_u64(0);
    size_t n = a->n;
    size_t cap = 2 * n + 1;
    uint64_t *d = xcalloc(cap, sizeof(uint64_t));

    /* sum of cross terms a[i]*a[j], i<j */
    for (size_t i = 0; i < n; i++) {
        uint64_t ai = a->d[i];
        if (ai == 0) continue;
        uint64_t carry = 0;
        for (size_t j = i + 1; j < n; j++) {
            u128 cur = (u128)d[i + j] + (u128)ai * a->d[j] + carry;
            d[i + j] = (uint64_t)cur;
            carry = (uint64_t)(cur >> 64);
        }
        d[i + n] = carry; /* still 0 before this: row i-1 ended at i-1 + n */
    }

    /* one pass: double the cross-term sum and add the squares a[i]^2.
       Limbs go in pairs d[2i], d[2i+1], the positions of a[i]^2. */
    uint64_t shift_in = 0; /* top bit of the previous limb, shifted in by the doubling */
    uint64_t carry = 0;    /* carry of the additions */
    for (size_t i = 0; i < n; i++) {
        u128 sq = (u128)a->d[i] * a->d[i];
        uint64_t lo = d[2 * i], hi = d[2 * i + 1];
        uint64_t lo2 = (lo << 1) | shift_in, hi2 = (hi << 1) | (lo >> 63);
        shift_in = hi >> 63;
        u128 cur = (u128)lo2 + (uint64_t)sq + carry;
        d[2 * i] = (uint64_t)cur;
        cur = (u128)hi2 + (uint64_t)(sq >> 64) + (uint64_t)(cur >> 64);
        d[2 * i + 1] = (uint64_t)cur;
        carry = (uint64_t)(cur >> 64);
    }
    d[2 * n] = shift_in + carry; /* always 0: a^2 fits in 2n limbs */

    Big r = {d, normalize_len(d, cap)};
    return r;
}

/* ==================== Division and comparison ==================== */

/** Divides a by a small divisor in place (long division from the top limb
    down) and returns the remainder. The divisor is 32-bit so every quotient
    limb fits in 64 bits. */
static uint32_t big_divmod_small_inplace(Big *a, uint32_t divisor) {
    u128 rem = 0;
    for (size_t i = a->n; i-- > 0;) {
        // the remainder so far, followed by the next limb, divided
        u128 cur = (rem << 64) | a->d[i];
        a->d[i] = (uint64_t)(cur / divisor);
        rem = cur % divisor;
    }
    a->n = normalize_len(a->d, a->n);
    return (uint32_t)rem;
}

/** Returns -1, 0 or 1 as a < b, a == b, a > b. Normalized numbers: more limbs
    means bigger, else the highest differing limb decides. */
static int big_cmp(const Big *a, const Big *b) {
    if (a->n != b->n) return a->n < b->n ? -1 : 1;
    for (size_t i = a->n; i-- > 0;) {
        if (a->d[i] != b->d[i]) return a->d[i] < b->d[i] ? -1 : 1;
    }
    return 0;
}

/* ==================== Signed numbers (Toom-3) ==================== */

/** Signed number: sign + magnitude. Needed because Toom-3 evaluates at
    negative points. Zero is always stored with neg = 0 (no "-0"). */
typedef struct {
    int neg; /* canonical: mag==0 implies neg==0 */
    Big mag;
} SBig;

/** Returns mag with the sign neg; takes ownership of mag. Zero gets neg = 0. */
static SBig sbig_wrap(Big mag, int neg) {
    SBig r = {big_is_zero(&mag) ? 0 : neg, mag};
    return r;
}

/** Frees the magnitude of x. */
static void sbig_free(SBig *x) {
    big_free(&x->mag);
}

/** Frees x and puts v in its place: x = f(x, ...) without a temporary,
    as in sbig_replace(&x, sbig_sub(&x, &y)) (v is computed before x is freed). */
static void sbig_replace(SBig *x, SBig v) {
    sbig_free(x);
    *x = v;
}

/** Returns x+y as a new SBig. Same signs: magnitudes add. Different signs:
    the smaller magnitude is subtracted from the larger, which gives the sign. */
static SBig sbig_add(const SBig *x, const SBig *y) {
    if (x->neg == y->neg) {
        return sbig_wrap(big_add(&x->mag, &y->mag), x->neg);
    }
    int cmp = big_cmp(&x->mag, &y->mag);
    if (cmp == 0) return sbig_wrap(big_from_u64(0), 0);
    if (cmp > 0) return sbig_wrap(big_sub(&x->mag, &y->mag), x->neg);
    return sbig_wrap(big_sub(&y->mag, &x->mag), y->neg);
}

/** Returns x-y as a new SBig, as x + (-y). -y is a shallow copy (shares y's
    limbs, flipped sign), so nothing is copied, and it must not be freed. */
static SBig sbig_sub(const SBig *x, const SBig *y) {
    SBig negy = {big_is_zero(&y->mag) ? 0 : !y->neg, y->mag};
    return sbig_add(x, &negy);
}

/** Returns x-y as a new SBig for two unsigned numbers. */
static SBig sbig_diff(const Big *x, const Big *y) {
    if (big_cmp(x, y) >= 0) return sbig_wrap(big_sub(x, y), 0);
    return sbig_wrap(big_sub(y, x), 1);
}

/** Returns x / divisor as a new SBig. The division must be exact (the
    remainder is dropped): Toom-3's interpolation divides by 2 and 3. */
static SBig sbig_div_small_exact(const SBig *x, uint32_t divisor) {
    Big t = big_copy(&x->mag);
    big_divmod_small_inplace(&t, divisor);
    return sbig_wrap(t, x->neg);
}
/* ======== src/mul_basic.c ======== */
/**
 * @file   mul_basic.c
 * @brief  Multiplication dispatch and the algorithms below the NTT threshold.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * big_mul and big_sqr choose by operand size: schoolbook up to
 * KARATSUBA_THRESHOLD_LIMBS, then Karatsuba, then Toom-3 above
 * TOOM3_THRESHOLD_LIMBS.
 * Products past NTT_THRESHOLD_LIMBS go to the NTT
 * (mul_ntt.c) when its tables are large enough (ntt_fits). The recursive
 * algorithms call back into big_mul / big_sqr, so each subproduct is
 * dispatched again.
 */


// Forward declaration
static Big big_mul_toom3(const Big *a, const Big *b);
static Big big_mul_ntt(const Big *a, const Big *b);

/* Longest transform the NTT root tables are built for (0 = None)
   Used to decide if the NTT method actually can be used. */
static size_t g_ntt_max_L = 0;

/* Measured via benchmarking on the i9-9900K.
   Can be overridden with -DNAME=value on-demand. */
#ifndef KARATSUBA_THRESHOLD_LIMBS
#define KARATSUBA_THRESHOLD_LIMBS 64
#endif
#ifndef NTT_THRESHOLD_LIMBS
#define NTT_THRESHOLD_LIMBS 200
#endif
#ifndef TOOM3_THRESHOLD_LIMBS
#define TOOM3_THRESHOLD_LIMBS 1500 
#endif

/** Returns true if the NTT can do a*b with the root tables built at the moment. */
static int ntt_fits(const Big *a, const Big *b) {
    return 2 * (a->n + b->n) - 1 <= g_ntt_max_L || g_ntt_max_L >= ((size_t)1 << NTT_MAX_LG);
}

/** Returns the product in a new Big of at most n limbs; frees z0, z1_full and z2. */
static Big karatsuba_combine(Big *z0, Big *z1_full, Big *z2, size_t half, size_t n) {
    // Middle term, in place: z1 = z1_full - z2 - z0
    big_sub_at(z1_full->d, z1_full->n, z2, 0);
    big_sub_at(z1_full->d, z1_full->n, z0, 0);
    z1_full->n = normalize_len(z1_full->d, z1_full->n);

    // result = z2*B^2 + z1*B + z0, each term added straight at its limb offset
    uint64_t *d = xcalloc(n, sizeof(uint64_t));
    big_add_at(d, n, z0, 0);
    big_add_at(d, n, z1_full, half);
    big_add_at(d, n, z2, 2 * half);

    big_free(z0); big_free(z1_full); big_free(z2);
    Big r = {d, normalize_len(d, n)};
    return r;
}

/** Returns a*b as a new Big. Requires freeing with big_free. */
static Big big_mul(const Big *a, const Big *b) {
    size_t minlen = a->n < b->n ? a->n : b->n;

    /* This answers the question: what algorithm are we using for multiplication? */
    // Schoolbook method (n^2): multiplies every limb of a by every limb of b and add the results.
    // Each limb product is one CPU instruction so nothing beats this for small numbers.
    // Karatsuba can be used instead, but for small margins this is faster.
    if (minlen <= KARATSUBA_THRESHOLD_LIMBS) return big_mul_schoolbook(a, b);
    // Our main work-and-power-horse, NTT method (n*logn):
    // Numbers are polynomials; instead of multiplying every coefficient by every coefficient,
    // evaluates both at L special points (roots of unity), multiply the L values pairwise, and
    // convert back. The special points let evaluation and conversion be done
    // by halving again and again (FFT), so the whole multiply becomes n log n.
    if (minlen >= NTT_THRESHOLD_LIMBS && ntt_fits(a, b)) return big_mul_ntt(a, b);
    // Toom-3 method (n^1.465): only when the NTT tables are too small or missing.
    // Never happens in a normal run; needed for tests.
    if (minlen >= TOOM3_THRESHOLD_LIMBS) return big_mul_toom3(a, b);

    // Karatsuba method (n^1.585): optimizations for middle term
    // (less half-size multiplies), but
    // paid for with a few extra additions, substractions and allocations.
    
    size_t half = (a->n > b->n ? a->n : b->n) / 2;

    Big a_lo = view_slice(a, 0, half);
    Big a_hi = view_slice(a, half, a->n);
    Big b_lo = view_slice(b, 0, half);
    Big b_hi = view_slice(b, half, b->n);

    // Recursion here: each separate multiply goes back through big_mul
    // and chooses fastest algorithm to work with.
    Big z0 = big_mul(&a_lo, &b_lo);
    Big z2 = big_mul(&a_hi, &b_hi);

    Big sa = big_add(&a_lo, &a_hi);
    Big sb = big_add(&b_lo, &b_hi);
    Big z1_full = big_mul(&sa, &sb);

    /* a_lo, a_hi, b_lo, b_hi are views: we do not free them. */
    big_free(&sa); big_free(&sb);

    return karatsuba_combine(&z0, &z1_full, &z2, half, a->n + b->n);
}

/** Returns a*a as a new Big. Requires freeing with big_free. */
static Big big_sqr(const Big *a) {
    // Separate function to not compute redundant cross products.
    if (a->n <= KARATSUBA_THRESHOLD_LIMBS) return big_sqr_schoolbook(a);
    if (a->n >= NTT_THRESHOLD_LIMBS && ntt_fits(a, a)) return big_mul_ntt(a, a);
    if (a->n >= TOOM3_THRESHOLD_LIMBS) return big_mul_toom3(a, a);

    size_t half = a->n / 2;

    Big a_lo = view_slice(a, 0, half);
    Big a_hi = view_slice(a, half, a->n);

    // Recursion here: each separate multiply goes back through big_sqr
    // and chooses fastest algorithm to work with.
    Big z0 = big_sqr(&a_lo);
    Big z2 = big_sqr(&a_hi);

    Big sa = big_add(&a_lo, &a_hi);
    Big z1_full = big_sqr(&sa);

    /* a_lo, a_hi are views: we do not free them. */
    big_free(&sa);

    return karatsuba_combine(&z0, &z1_full, &z2, half, 2 * a->n);
}

/** A Toom-3 operand x0 + x1*t + x2*t^2 evaluated at t = 1, -1, -2 and stored here.
    No need to store t = 0 and t = inf: they are x0 and x2 themselves. */
typedef struct {
    Big at1;          /* x0 + x1 + x2: only additions, never negative */
    SBig atm1, atm2;  /* x0 - x1 + x2 and x0 - 2x1 + 4x2: can be negative */
} Toom3Eval;

/** Returns x0 + x1*t + x2*t^2 at t = 1, -1, -2. Requires freeing with toom3_eval_free. */
static Toom3Eval toom3_eval(const Big *x0, const Big *x1, const Big *x2) {
    Toom3Eval e;
    // x0 + x2 is shared by t = 1 and t = -1: computed once
    Big s = big_add(x0, x2);
    e.at1 = big_add(&s, x1);
    e.atm1 = sbig_diff(&s, x1);
    big_free(&s);

    Big x2x4 = big_shl_bits(x2, 2), x1x2 = big_shl1(x1);
    Big x0p4x2 = big_add(x0, &x2x4);
    e.atm2 = sbig_diff(&x0p4x2, &x1x2);
    
    big_free(&x2x4); big_free(&x1x2); big_free(&x0p4x2);
    return e;
}

/** Helper function to free the three values of a Toom3Eval. */
static void toom3_eval_free(Toom3Eval *e) {
    big_free(&e->at1); sbig_free(&e->atm1); sbig_free(&e->atm2);
}

/** Returns x*y as a new Big; when squaring, x*x via the cheaper big_sqr (y is ignored). */
static Big toom3_mul(const Big *x, const Big *y, int sqr) {
    return sqr ? big_sqr(x) : big_mul(x, y);
}

/** Returns a*b as a new Big by Toom-Cook-3. Requires freeing with big_free.
    Unused in normal build of the program: fib.c builds NTT table big enough
    for every product, so the NTT always fits.
    Splits each operand into 3 blocks (a polynomial in t = 2^(64k)), evaluates
    both at t = 0, 1, -1, -2, inf, does 5 multiplies of ~1/3 size instead of 9,
    then interpolates the 5 coefficients of the product: O(n^1.465).
    Squaring (a == b) evaluates once and squares the 5 values. */
static Big big_mul_toom3(const Big *a, const Big *b) {
    // detects squaring, compares by pointers
    int sqr = (a == b);
    size_t maxlen = a->n > b->n ? a->n : b->n;
    // Block size: maxlen/3 rounded up, so 3 blocks cover the longer operand
    size_t k = (maxlen + 2) / 3;

    // a = a0 + a1*B + a2*B^2 with B = 2^(64k);
    Big a0 = view_slice(a, 0, k), a1 = view_slice(a, k, 2 * k), a2 = view_slice(a, 2 * k, a->n);
    Big b0 = view_slice(b, 0, k), b1 = view_slice(b, k, 2 * k), b2 = view_slice(b, 2 * k, b->n);

    // Values at t = 1, -1, -2; a square reuses a's values for b
    Toom3Eval ea = toom3_eval(&a0, &a1, &a2);
    Toom3Eval eb = sqr ? ea : toom3_eval(&b0, &b1, &b2);

    // 5 pointwise products of ~1/3 size: R(t) = a(t)*b(t) at t = 0, inf, 1, -1, -2.
    // Magnitudes multiply; the sign is the XOR of the two signs.
    Big R0_mag = toom3_mul(&a0, &b0, sqr);
    Big Rinf_mag = toom3_mul(&a2, &b2, sqr);
    Big R1_mag = toom3_mul(&ea.at1, &eb.at1, sqr);
    SBig Rm1 = sbig_wrap(toom3_mul(&ea.atm1.mag, &eb.atm1.mag, sqr), ea.atm1.neg ^ eb.atm1.neg);
    SBig Rm2 = sbig_wrap(toom3_mul(&ea.atm2.mag, &eb.atm2.mag, sqr), ea.atm2.neg ^ eb.atm2.neg);
    toom3_eval_free(&ea);
    if (!sqr) toom3_eval_free(&eb);

    // Never negative, wrapped so interpolation works in SBig
    SBig R0 = sbig_wrap(R0_mag, 0);
    SBig R1 = sbig_wrap(R1_mag, 0);
    SBig Rinf = sbig_wrap(Rinf_mag, 0);

    // Interpolation: R0 = r0, Rinf = r4 directly; the other three values
    // are solved for r1, r2, r3. The r1..r3 variables hold intermediates
    // first (in r = coefficients of the product):
    //   r3 = (Rm2 - R1)/3        = -r1 + r2 - 3r3 + 5r4
    //   r1 = (R1 - Rm1)/2        = r1 + r3
    //   r2 = Rm1 - R0            = -r1 + r2 - r3 + r4
    //   r3 = (r2 - r3)/2 + 2*r4  = r3
    //   r2 = r2 + r1 - r4        = r2
    //   r1 = r1 - r3             = r1
    // The divisions are always exact and intermediates can be negative, hence SBig.
    SBig t = sbig_sub(&Rm2, &R1);
    SBig r3 = sbig_div_small_exact(&t, 3);
    sbig_free(&t);

    t = sbig_sub(&R1, &Rm1);
    SBig r1 = sbig_div_small_exact(&t, 2);
    sbig_free(&t);

    SBig r2 = sbig_sub(&Rm1, &R0);
    sbig_free(&Rm1); sbig_free(&Rm2); sbig_free(&R1);

    t = sbig_sub(&r2, &r3);
    sbig_replace(&r3, sbig_div_small_exact(&t, 2));
    sbig_free(&t);
    t = sbig_wrap(big_shl1(&Rinf.mag), 0);
    sbig_replace(&r3, sbig_add(&r3, &t));
    sbig_free(&t);

    sbig_replace(&r2, sbig_add(&r2, &r1));
    sbig_replace(&r2, sbig_sub(&r2, &Rinf));

    sbig_replace(&r1, sbig_sub(&r1, &r3));

    // result = r0 + r1*B + ... + r4*B^4 with B = 2^(64k), r0 = R0, r4 = Rinf.
    // All r's are >= 0 (sums of products of nonnegative pieces): .mag only.
    // Terms overlap (~2k limbs, k apart), so each is added with carries.
    size_t n = a->n + b->n;
    uint64_t *d = xcalloc(n, sizeof(uint64_t));
    big_add_at(d, n, &R0.mag, 0);
    big_add_at(d, n, &r1.mag, k);
    big_add_at(d, n, &r2.mag, 2 * k);
    big_add_at(d, n, &r3.mag, 3 * k);
    big_add_at(d, n, &Rinf.mag, 4 * k);

    sbig_free(&R0); sbig_free(&r1); sbig_free(&r2); sbig_free(&r3); sbig_free(&Rinf);
    Big result = {d, normalize_len(d, n)};
    return result;
}
/* ======== src/ntt_arith.c ======== */
/**
 * @file   ntt_arith.c
 * @brief  The three NTT primes and modular arithmetic on 8 AVX2 lanes.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * The NTT multiply convolves the operands as arrays of b-bit coefficients
 * (b = 32..35) with number-theoretic transforms (FFTs over Z/pZ) modulo three
 * primes p = c*2^k+1 < 2^31, then combines them with the CRT (crt.c). Each
 * convolution coefficient is a sum of at most min(ca, cb) products of two
 * coefficients; b is chosen so that this stays below p1*p2*p3 (~2^92.6), the
 * largest value the CRT can recover (ntt_width_ok). O(n log n) against
 * Toom-3's O(n^1.465).
 *
 * 32-bit lanes let AVX2 do 8 butterflies at once: vpmuludq gives four
 * 32x32->64 products per instruction, and Montgomery reduction (R = 2^32)
 * needs no division. Values stay canonical in [0, p); p < 2^31 means u + v
 * fits in 32 bits and reducing [0, 2p) -> [0, p) is a single min(x, x - p).
 * The inverse transform reuses the forward roots: that yields L * x[-n mod L],
 * so the CRT simply reads the result backwards. One root table per prime.
 */


#ifndef NTT_BASE_LEN
#define NTT_BASE_LEN 8192       /* at or below this, a block stays in L1/L2: plain loops */
#endif
_Static_assert(NTT_BASE_LEN >= 16, "the base kernels end in a 16-element in-register step (dif_last3)");

/** One NTT prime and everything precomputed for it (tables filled by ntt_init). */
typedef struct {
    uint32_t p;           /* the prime, c * 2^k + 1 */
    uint32_t g;           /* a generator mod p: its powers give the roots of unity */
    uint32_t pinv;        /* p^-1 mod 2^32 */
    uint32_t *rt;         /* rt[h + j] = w_{2h}^j * R mod p (Montgomery form) */
    uint32_t *rtp;        /* rtp[i] = rt[i] * pinv mod 2^32 (see vmontw), i < NTT_BASE_LEN */
    uint32_t scale[NTT_MAX_LG + 1];  /* R^2 / L mod p for L = 2^lg */
    uint32_t scale3[NTT_MAX_LG + 1]; /* R^2 / L mod p for L = 3 * 2^lg */
    uint32_t r2;          /* R^2 mod p: vmont(h, r2) = h * 2^32 mod p */
} NttPrime;

/* The three primes: below 2^31, and p - 1 divisible by 3 * 2^25 (63, 15 and
   27 are multiples of 3), so every length 2^k and 3 * 2^k up to
   2^NTT_MAX_LG has its roots of unity.
   Only p and g are set here; ntt_init computes the rest. */
static NttPrime g_ntt[3] = {
    {.p = 2113929217u, .g = 5},  /* 63*2^25+1 */
    {.p = 2013265921u, .g = 31}, /* 15*2^27+1 */
    {.p = 1811939329u, .g = 13}, /* 27*2^26+1 */
};

/* Garner's inverses for the CRT (pi^-1 mod pj), in Montgomery form; set by ntt_init */
static uint32_t g_inv_p1_mod_p2, g_inv_p1_mod_p3, g_inv_p2_mod_p3;

/** Returns a^e mod p (square and multiply). Scalar, for table setup only. */
static uint32_t powmod32(uint32_t a, uint64_t e, uint32_t p) {
    uint64_t r = 1, x = a % p;
    while (e) {
        if (e & 1) r = r * x % p;
        x = x * x % p;
        e >>= 1;
    }
    return (uint32_t)r;
}

/** Returns x in Montgomery form, x * R mod p with R = 2^32. Scalar, for setup. */
static uint32_t to_mont(uint32_t x, uint32_t p) {
    return (uint32_t)(((uint64_t)x << 32) % p);
}

/* Short names for the AVX2 loads and stores: LD / ST need 32-byte aligned
   addresses, LDU does not. SHUF_PS is the float shuffle used on integer
   vectors (it can pick lanes from two vectors, the integer one cannot). */
#define LD(ptr) _mm256_load_si256((const __m256i *)(ptr))
#define LDU(ptr) _mm256_loadu_si256((const __m256i *)(ptr))
#define ST(ptr, v) _mm256_store_si256((__m256i *)(ptr), (v))
#define SHUF_PS(a, b, imm) \
    _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), (imm)))

/** Returns x reduced from [0, 2p) to [0, p): if x >= p then x - p < x, else
    x - p wraps and is larger, so the unsigned minimum is right either way. */
static inline __m256i vred(__m256i x, __m256i P) {
    return _mm256_min_epu32(x, _mm256_sub_epi32(x, P));
}

/** Returns x with its odd 32-bit lanes moved into the even slots vpmuludq reads. */
static inline __m256i vodd(__m256i x) {
    return _mm256_shuffle_epi32(x, 0xF5);
}

/** Returns r moved from (-p, p) to [0, p): a negative r is huge as unsigned
    while r + p is small; a non-negative r is smaller than r + p. */
static inline __m256i vfix(__m256i r, __m256i P) {
    return _mm256_min_epu32(r, _mm256_add_epi32(r, P));
}

/** Returns (x + y) mod p for x, y in [0, p). */
static inline __m256i vaddm(__m256i x, __m256i y, __m256i P) {
    return vred(_mm256_add_epi32(x, y), P);
}

/** Returns (x - y) mod p for x, y in [0, p). */
static inline __m256i vsubm(__m256i x, __m256i y, __m256i P) {
    return vfix(_mm256_sub_epi32(x, y), P);
}

/** Returns (T - m*p) / 2^32 in [0, p): the shared end of vmont2 and vmontw,
    from the products te, to (even and odd lanes) and their m = T * pinv
    mod 2^32 (me, mo). */
static inline __m256i vmont_reduce(__m256i te, __m256i to, __m256i me, __m256i mo, __m256i P) {
    __m256i de = _mm256_sub_epi64(te, _mm256_mul_epi32(me, P));
    __m256i dodd = _mm256_sub_epi64(to, _mm256_mul_epi32(mo, P));
    /* high halves of de into the even slots: a shuffle (port 5) rather than a
       64-bit shift, which would compete with the multiplies for ports 0/1 */
    return vfix(_mm256_blend_epi32(_mm256_shuffle_epi32(de, 0xF5), dodd, 0xAA), P);
}

/** Returns a*b*R^-1 mod p in [0, p) (signed Montgomery), for a in (-p, p) as
    int32 and b in [0, p); ao, bo hold the odd lanes of a, b in the even slots.
    |T| < p^2 and |m*p| < 2^31 p, so (T - m*p) / 2^32 lies in (-p, p). Signed
    inputs let callers pass a difference x - y without adding p first. */
static inline __m256i vmont2(__m256i a, __m256i ao, __m256i b, __m256i bo, __m256i P, __m256i PINV) {
    __m256i te = _mm256_mul_epi32(a, b), to = _mm256_mul_epi32(ao, bo);
    __m256i me = _mm256_mul_epu32(te, PINV), mo = _mm256_mul_epu32(to, PINV);
    return vmont_reduce(te, to, me, mo, P);
}

/** Returns a*b*R^-1 mod p in [0, p): vmont2 with the odd lanes taken here. */
static inline __m256i vmont(__m256i a, __m256i b, __m256i P, __m256i PINV) {
    return vmont2(a, vodd(a), b, vodd(b), P, PINV);
}

/** Returns vmont(a, w) for a table twiddle w, with wp = w * pinv mod 2^32
    precomputed: m = a * w * pinv = a * wp (mod 2^32) comes straight from a,
    in parallel with a * w, instead of after it. */
static inline __m256i vmontw(__m256i a, __m256i w, __m256i wp, __m256i P) {
    __m256i ao = vodd(a);
    __m256i te = _mm256_mul_epi32(a, w), to = _mm256_mul_epi32(ao, vodd(w));
    __m256i me = _mm256_mul_epu32(a, wp), mo = _mm256_mul_epu32(ao, vodd(wp));
    return vmont_reduce(te, to, me, mo, P);
}

/** Returns c * pinv mod 2^32 in all lanes: the wp that goes with a constant c in vmontw. */
static inline __m256i vpre(uint32_t c, __m256i PINV) {
    return _mm256_set1_epi32((int)(c * (uint32_t)_mm256_cvtsi256_si32(PINV)));
}
/* ======== src/ntt_tables.c ======== */
/**
 * @file   ntt_tables.c
 * @brief  ntt_init: the root-of-unity tables, scale factors and Garner constants.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * Each prime gets one table of L words holding every level's twiddles:
 * rt[h + j] = w_2h^j * R mod p for h = 1, 2, 4, .., L/2 and j < h (w_2h a
 * primitive 2h-th root of unity; rt[0] is unused). Level h is rt[h .. 2h),
 * so a transform of any length up to L reads its levels from the same table.
 * rtp holds rt * pinv mod 2^32 for the first NTT_BASE_LEN entries only, the
 * part the in-cache kernels use with vmontw.
 *
 * Every level is its own running product, 8 lanes at a time (lane i starts
 * at w^(j0+i) and steps by w^8), cut into chunks spread over the pool, so
 * building the tables for a big product takes a fraction of a millisecond.
 * The tables only grow: ntt_init rebuilds them when a bigger product needs
 * longer ones, and does nothing otherwise.
 */


/** One chunk of one level's table, for the thread pool. */
typedef struct {
    NttPrime *Pr;
    size_t h;      /* the level: entries rt[h + j] */
    size_t j0, j1; /* this chunk: j in [j0, j1) */
} TableJob;

/** Pool task: fills rt[h + j] (and rtp, inside the base size) for j in
    [j0, j1) of one level. Levels below 8 are too short for a vector and go
    one entry at a time; the others start 8 lanes at w^j0 .. w^(j0+7) and
    step them all by w^8 with one vmontw per vector. */
static void table_proc(void *arg) {
    TableJob *t = (TableJob *)arg;
    const NttPrime *Pr = t->Pr;
    const uint32_t p = Pr->p;
    uint32_t w = powmod32(Pr->g, (p - 1) / (2 * t->h), p);
    uint32_t *out = Pr->rt + t->h, cur = powmod32(w, t->j0, p);
    uint32_t *outp = 2 * t->h <= NTT_BASE_LEN ? Pr->rtp + t->h : NULL;

    if (t->h < 8) {
        for (size_t j = t->j0; j < t->j1; j++, cur = (uint32_t)((uint64_t)cur * w % p)) {
            out[j] = to_mont(cur, p);
            outp[j] = out[j] * Pr->pinv;
        }
        return;
    }

    uint32_t start[8];
    for (int i = 0; i < 8; i++, cur = (uint32_t)((uint64_t)cur * w % p)) start[i] = to_mont(cur, p);
    const __m256i P = _mm256_set1_epi32((int)p), PINV = _mm256_set1_epi32((int)Pr->pinv);
    const uint32_t w8 = to_mont(powmod32(w, 8, p), p);
    const __m256i W8 = _mm256_set1_epi32((int)w8), W8p = vpre(w8, PINV);
    __m256i v = LDU(start);
    for (size_t j = t->j0; j < t->j1; j += 8) {
        _mm256_storeu_si256((__m256i *)(out + j), v);
        if (outp) _mm256_storeu_si256((__m256i *)(outp + j), _mm256_mullo_epi32(v, PINV));
        v = vmontw(v, W8, W8p, P);
    }
}

/** Builds the tables for products of up to max_limbs 64-bit limbs (at most
    2 * max_limbs coefficients, since b >= 32), capped at 2^NTT_MAX_LG; the
    split multiply handles longer products with these. Also sets each
    prime's pinv, r2 and scale factors, and the Garner inverses. Must run
    before any thread calls big_mul_ntt; the tables are read-only after. */
static void ntt_init(size_t max_limbs) {
    size_t L = 16;
    while (L < 2 * max_limbs && L < ((size_t)1 << NTT_MAX_LG)) L <<= 1;
    if (L <= g_ntt_max_L) return;

    // jobs of at least 32768 entries, at most ~64 per prime for the big
    // levels; cap bounds the count: 2L/chunk for those, one per small level
    size_t chunk = L / 64 > 32768 ? L / 64 : 32768;
    size_t njobs = 0, cap = 3 * (2 * L / chunk + NTT_MAX_LG + 2);
    TableJob *jobs = xmalloc(cap * sizeof(TableJob));
    for (int i = 0; i < 3; i++) {
        NttPrime *Pr = &g_ntt[i];
        // pinv = p^-1 mod 2^32 by Newton: p is its own inverse mod 8 (3 bits),
        // and each step doubles the correct bits: 3 -> 6 -> 12 -> 24 -> 48
        uint32_t inv = Pr->p;
        for (int it = 0; it < 4; it++) inv *= 2 - Pr->p * inv;
        Pr->pinv = inv;
        xfree(Pr->rt);
        Pr->rt = xmalloc(L * sizeof(uint32_t));
        xfree(Pr->rtp);
        Pr->rtp = xmalloc((L < NTT_BASE_LEN ? L : NTT_BASE_LEN) * sizeof(uint32_t));
        Pr->rt[0] = Pr->rtp[0] = 0; /* unused slot */
        for (size_t h = 1; h < L; h <<= 1)
            for (size_t j = 0; j < h; j += chunk) jobs[njobs++] = (TableJob){Pr, h, j, j + chunk < h ? j + chunk : h};
    }
    // not pool_run: this thread takes jobs[0], then computes the constants
    // below while the workers still fill tables, and only then waits
    TaskGroup g = {0};
    for (size_t k = 1; k < njobs; k++) pool_submit(&g, table_proc, &jobs[k]);
    table_proc(&jobs[0]);

    // scale[lg] = R^2 / L mod p for L = 2^lg and 3 * 2^lg (Fermat inverse):
    // the CRT multiplies by it to undo the pointwise R^-1 and the inverse's L
    for (int i = 0; i < 3; i++) {
        NttPrime *Pr = &g_ntt[i];
        uint32_t p = Pr->p, R = (uint32_t)((1ULL << 32) % p);
        uint32_t R2 = (uint32_t)((uint64_t)R * R % p);
        Pr->r2 = R2;
        for (int lg = 0; lg <= NTT_MAX_LG; lg++) {
            Pr->scale[lg] = (uint32_t)((uint64_t)R2 * powmod32((uint32_t)((1ULL << lg) % p), p - 2, p) % p);
            Pr->scale3[lg] = (uint32_t)((uint64_t)R2 * powmod32((uint32_t)((3ULL << lg) % p), p - 2, p) % p);
        }
    }
    // Garner's inverses pi^-1 mod pj, times R (Montgomery form, for vmontw)
    uint32_t p1 = g_ntt[0].p, p2 = g_ntt[1].p, p3 = g_ntt[2].p;
    uint64_t R_p2 = (1ULL << 32) % p2, R_p3 = (1ULL << 32) % p3;
    g_inv_p1_mod_p2 = (uint32_t)(powmod32(p1 % p2, p2 - 2, p2) * R_p2 % p2);
    g_inv_p1_mod_p3 = (uint32_t)(powmod32(p1 % p3, p3 - 2, p3) * R_p3 % p3);
    g_inv_p2_mod_p3 = (uint32_t)(powmod32(p2 % p3, p3 - 2, p3) * R_p3 % p3);
    pool_wait(&g);
    xfree(jobs);
    g_ntt_max_L = L;
}
/* ======== src/ntt_kernels.c ======== */
/**
 * @file   ntt_kernels.c
 * @brief  The transform passes: butterflies, in-cache base transforms, radix-16 passes.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * A level h of the transform pairs every element with the one h words away.
 * Forward transforms are DIF (Gentleman-Sande: add/subtract, then twiddle,
 * levels L/2 down to 1); inverses are DIT (Cooley-Tukey: twiddle, then
 * add/subtract, levels 1 up to L/2), which undoes the DIF order without any
 * bit-reversal pass. Twiddles for level h are rt[h .. 2h) (ntt_tables.c).
 *
 * Blocks up to NTT_BASE_LEN are transformed in cache by ntt_forward_base /
 * ntt_inverse_base: fused pairs of levels (radix 4), then the last three
 * levels in registers. Bigger blocks get one top pass first (ntt_conv.c):
 * radix 2, 4 or 16 levels per trip through memory. The first forward pass
 * can read its input straight from the bignum (Src, src_load8), so no
 * separate load pass is needed. Past NTT_TWD_MIN_LEN the radix-16 passes
 * derive most twiddles instead of streaming them from the big tables.
 */


/** A bignum read as nc coefficients of b bits, for src_load8. b is 32..35
    in practice; the split multiply goes narrower (down to 8) only for
    operands too big for 32. */
typedef struct {
    const uint64_t *d; /* the limbs */
    size_t nl, nc;     /* limbs, coefficients */
    unsigned b;        /* coefficient width in bits (<= 35) */
} Src;

/** Returns coefficients [k, k+8) of a bignum reduced to [0, p), zero past
    the end; k is a multiple of 8. b = 32 reads the halves of the limbs
    directly (each < 2^32 < 3p, so two vred calls reduce it). Other widths
    cut the fields out one by one and split them as hi * 2^32 + lo, with
    hi < 2^(b-32) <= 8 brought in as vmont(hi, R^2) = hi * 2^32 mod p. */
static inline __m256i src_load8(const Src *s, size_t k, __m256i P, __m256i PINV, __m256i R2) {
    if (k >= s->nc) return _mm256_setzero_si256();
    if (s->b == 32) {
        const uint32_t *w = (const uint32_t *)s->d;
        __m256i v;
        if (k + 8 <= s->nc) {
            v = LDU(w + k);
        } else {
            uint32_t tmp[8] = {0};
            memcpy(tmp, w + k, (s->nc - k) * sizeof(uint32_t));
            v = LDU(tmp);
        }
        return vred(vred(v, P), P);
    }

    uint32_t lo[8], hi[8];
    const unsigned b = s->b;
    const uint64_t mask = (1ULL << b) - 1;
    size_t bit = k * b;
    for (int t = 0; t < 8; t++, bit += b) {
        uint64_t v = 0;
        if (k + t < s->nc) {
            size_t wi = bit >> 6;
            unsigned sh = bit & 63;
            v = s->d[wi] >> sh;
            if (sh + b > 64 && wi + 1 < s->nl) v |= s->d[wi + 1] << (64 - sh);
            v &= mask;
        }
        lo[t] = (uint32_t)v;
        hi[t] = (uint32_t)(v >> 32);
    }

    __m256i l = vred(vred(LDU(lo), P), P);
    return vaddm(l, vmont(LDU(hi), R2, P, PINV), P);
}

/** Returns a * w[i] (Montgomery) for a table twiddle. The kernels below
    come in two builds, chosen by their `pre` flag: in-cache ones (base
    blocks) pass wp = the matching rtp entries and get the shorter vmontw
    chain; the top passes stream their twiddles from memory and pass
    wp = NULL, since a second table there costs more bandwidth than the
    multiply saves. */
static inline __attribute__((always_inline)) __m256i vtw(__m256i a, const uint32_t *w, const uint32_t *wp, size_t i,
                                                         __m256i P, __m256i PINV) {
    return wp ? vmontw(a, LD(w + i), LD(wp + i), P) : vmont(a, LD(w + i), P, PINV);
}

/** Runs DIF level h (Gentleman-Sande) on lanes [j0, j1) of a block of 2h:
    x, y = x + y, (x - y) * w_2h^j. h >= 8 (whole vectors); pre selects
    vmontw with rtp (in-cache) or plain vmont (streaming). */
static inline __attribute__((always_inline)) void dif2_body(uint32_t *a, size_t h, size_t j0, size_t j1,
                                                            const NttPrime *Pr, int pre) {
    const __m256i P = _mm256_set1_epi32((int)Pr->p), PINV = _mm256_set1_epi32((int)Pr->pinv);
    const uint32_t *w = Pr->rt + h, *wp = pre ? Pr->rtp + h : NULL;
    uint32_t *x = a, *y = a + h;
    for (size_t j = j0; j < j1; j += 8) {
        __m256i u = LD(x + j), v = LD(y + j);
        ST(x + j, vaddm(u, v, P));
        ST(y + j, vtw(_mm256_sub_epi32(u, v), w, wp, j, P, PINV));
    }
}

/** Runs DIT level h (Cooley-Tukey) on lanes [j0, j1) of a block of 2h:
    t = y * w_2h^j, then x, y = x + t, x - t. The mirror of dif2_body. */
static inline __attribute__((always_inline)) void dit2_body(uint32_t *a, size_t h, size_t j0, size_t j1,
                                                            const NttPrime *Pr, int pre) {
    const __m256i P = _mm256_set1_epi32((int)Pr->p), PINV = _mm256_set1_epi32((int)Pr->pinv);
    const uint32_t *w = Pr->rt + h, *wp = pre ? Pr->rtp + h : NULL;
    uint32_t *x = a, *y = a + h;
    for (size_t j = j0; j < j1; j += 8) {
        __m256i u = LD(x + j);
        __m256i t = vtw(LD(y + j), w, wp, j, P, PINV);
        ST(x + j, vaddm(u, t, P));
        ST(y + j, vsubm(u, t, P));
    }
}

/** Runs a radix-2 DIF top pass (streaming twiddles): one compiled copy of
    dif2_body for ntt_conv.c. */
static void dif_stage(uint32_t *a, size_t h, size_t j0, size_t j1, const NttPrime *Pr) {
    dif2_body(a, h, j0, j1, Pr, 0);
}

/** Runs a radix-2 DIT top pass (streaming twiddles): the mirror of dif_stage. */
static void dit_stage(uint32_t *a, size_t h, size_t j0, size_t j1, const NttPrime *Pr) {
    dit2_body(a, h, j0, j1, Pr, 0);
}

/** Twiddles for the in-register levels 4 and 2, laid out to match
    dif_last3 / dit_first3's lanes, with their rtp entries. */
typedef struct {
    __m256i w4, w2, w4p, w2p; /* w4 = rt[4..7] per 128-bit half; w2 = rt[2], rt[3] repeated */
    __m256i P;
} SmallTw;

/** Returns the SmallTw of one prime. */
static SmallTw small_twiddles(const NttPrime *Pr) {
    SmallTw t;
    const uint32_t *rt = Pr->rt, *rp = Pr->rtp;
    t.w4 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(rt + 4)));
    t.w2 = _mm256_setr_epi32(rt[2], rt[3], rt[2], rt[3], rt[2], rt[3], rt[2], rt[3]);
    t.w4p = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(rp + 4)));
    t.w2p = _mm256_setr_epi32(rp[2], rp[3], rp[2], rp[3], rp[2], rp[3], rp[2], rp[3]);
    t.P = _mm256_set1_epi32((int)Pr->p);
    return t;
}

/** Runs DIF levels h = 4, 2, 1 on 16 elements (two 8-blocks A, B), in registers.
   The output layout is permuted; dit_first3 reads exactly this layout back,
   and the pointwise product in between doesn't care about order.
     h=4: x = [A0..3 | B0..3], y = [A4..7 | B4..7]
     h=2: unpack 64-bit halves -> x = [A0 A1 A4 A5 | ..], y = [A2 A3 A6 A7 | ..]
     h=1: shuffle -> x = even elements, y = odd elements (twiddle is 1) */
static inline void dif_last3(uint32_t *a, const SmallTw *t) {
    const __m256i P = t->P;
    __m256i A = LD(a), B = LD(a + 8);
    __m256i x = _mm256_permute2x128_si256(A, B, 0x20), y = _mm256_permute2x128_si256(A, B, 0x31);
    __m256i s = vaddm(x, y, P);
    __m256i d = vmontw(_mm256_sub_epi32(x, y), t->w4, t->w4p, P);

    x = _mm256_unpacklo_epi64(s, d);
    y = _mm256_unpackhi_epi64(s, d);
    s = vaddm(x, y, P);
    d = vmontw(_mm256_sub_epi32(x, y), t->w2, t->w2p, P);

    x = SHUF_PS(s, d, 0x88); /* _MM_SHUFFLE(2,0,2,0) */
    y = SHUF_PS(s, d, 0xDD); /* _MM_SHUFFLE(3,1,3,1) */
    ST(a, vaddm(x, y, P));
    ST(a + 8, vsubm(x, y, P));
}

/** Runs DIT levels h = 1, 2, 4 on 16 elements: the exact mirror of
    dif_last3, undoing each shuffle. */
static inline void dit_first3(uint32_t *a, const SmallTw *t) {
    const __m256i P = t->P;
    __m256i x = LD(a), y = LD(a + 8);
    __m256i s = vaddm(x, y, P);
    __m256i d = vsubm(x, y, P);

    x = _mm256_unpacklo_epi32(s, d);
    y = _mm256_unpackhi_epi32(s, d);
    __m256i tt = vmontw(y, t->w2, t->w2p, P);
    s = vaddm(x, tt, P);
    d = vsubm(x, tt, P);

    x = _mm256_unpacklo_epi64(s, d);
    y = _mm256_unpackhi_epi64(s, d);
    tt = vmontw(y, t->w4, t->w4p, P);
    s = vaddm(x, tt, P);
    d = vsubm(x, tt, P);

    ST(a, _mm256_permute2x128_si256(s, d, 0x20));
    ST(a + 8, _mm256_permute2x128_si256(s, d, 0x31));
}

/** Runs two DIF levels fused (radix 4): levels h = 2Q then h = Q on a block
    of 4Q, lanes [j0, j1) of Q. Same arithmetic as two radix-2 passes, but
    each element is loaded and stored once instead of twice, which halves
    memory traffic when the block is bigger than the caches. With src, the
    input is read from a bignum instead of a (see src_load8), so loading
    costs no separate pass over memory. */
static inline __attribute__((always_inline)) void dif4_body(uint32_t *a, const Src *src, size_t Q, size_t j0,
                                                            size_t j1, const NttPrime *Pr, int pre) {
    const __m256i P = _mm256_set1_epi32((int)Pr->p), PINV = _mm256_set1_epi32((int)Pr->pinv);
    const uint32_t *w1 = Pr->rt + 2 * Q, *w2 = Pr->rt + Q;
    const uint32_t *w1p = pre ? Pr->rtp + 2 * Q : NULL, *w2p = pre ? Pr->rtp + Q : NULL;
    uint32_t *a0 = a, *a1 = a + Q, *a2 = a + 2 * Q, *a3 = a + 3 * Q;
    for (size_t j = j0; j < j1; j += 8) {
        __m256i x0, x1, x2, x3;
        if (src) {
            const __m256i R2 = _mm256_set1_epi32((int)Pr->r2);
            x0 = src_load8(src, j, P, PINV, R2);
            x1 = src_load8(src, Q + j, P, PINV, R2);
            x2 = src_load8(src, 2 * Q + j, P, PINV, R2);
            x3 = src_load8(src, 3 * Q + j, P, PINV, R2);
        } else {
            x0 = LD(a0 + j), x1 = LD(a1 + j), x2 = LD(a2 + j), x3 = LD(a3 + j);
        }
        __m256i b0 = vaddm(x0, x2, P);
        __m256i b2 = vtw(_mm256_sub_epi32(x0, x2), w1, w1p, j, P, PINV);
        __m256i b1 = vaddm(x1, x3, P);
        __m256i b3 = vtw(_mm256_sub_epi32(x1, x3), w1, w1p, Q + j, P, PINV);
        __m256i d0 = _mm256_sub_epi32(b0, b1), d2 = _mm256_sub_epi32(b2, b3);
        ST(a0 + j, vaddm(b0, b1, P));
        ST(a1 + j, vtw(d0, w2, w2p, j, P, PINV));
        ST(a2 + j, vaddm(b2, b3, P));
        ST(a3 + j, vtw(d2, w2, w2p, j, P, PINV));
    }
}

/** Runs a radix-4 DIF top pass (streaming twiddles) on lanes [j0, j1) of Q. */
static void dif_stage4(uint32_t *a, size_t Q, size_t j0, size_t j1, const NttPrime *Pr) {
    dif4_body(a, NULL, Q, j0, j1, Pr, 0);
}

/** Runs dif_stage4 reading its input from src: the first pass of a transform. */
static void dif_stage4_src(uint32_t *a, const Src *src, size_t Q, size_t j0, size_t j1, const NttPrime *Pr) {
    dif4_body(a, src, Q, j0, j1, Pr, 0);
}

/** Runs two DIT levels fused: h = Q then h = 2Q, the mirror of dif4_body. */
static inline __attribute__((always_inline)) void dit4_body(uint32_t *a, size_t Q, size_t j0, size_t j1,
                                                            const NttPrime *Pr, int pre) {
    const __m256i P = _mm256_set1_epi32((int)Pr->p), PINV = _mm256_set1_epi32((int)Pr->pinv);
    const uint32_t *w1 = Pr->rt + 2 * Q, *w2 = Pr->rt + Q;
    const uint32_t *w1p = pre ? Pr->rtp + 2 * Q : NULL, *w2p = pre ? Pr->rtp + Q : NULL;
    uint32_t *a0 = a, *a1 = a + Q, *a2 = a + 2 * Q, *a3 = a + 3 * Q;
    for (size_t j = j0; j < j1; j += 8) {
        __m256i x0 = LD(a0 + j), x1 = LD(a1 + j), x2 = LD(a2 + j), x3 = LD(a3 + j);
        __m256i t = vtw(x1, w2, w2p, j, P, PINV);
        __m256i b0 = vaddm(x0, t, P);
        __m256i b1 = vsubm(x0, t, P);
        t = vtw(x3, w2, w2p, j, P, PINV);
        __m256i b2 = vaddm(x2, t, P);
        __m256i b3 = vsubm(x2, t, P);
        t = vtw(b2, w1, w1p, j, P, PINV);
        ST(a0 + j, vaddm(b0, t, P));
        ST(a2 + j, vsubm(b0, t, P));
        t = vtw(b3, w1, w1p, Q + j, P, PINV);
        ST(a1 + j, vaddm(b1, t, P));
        ST(a3 + j, vsubm(b1, t, P));
    }
}

/** Runs a radix-4 DIT top pass (streaming twiddles) on lanes [j0, j1) of Q. */
static void dit_stage4(uint32_t *a, size_t Q, size_t j0, size_t j1, const NttPrime *Pr) {
    dit4_body(a, Q, j0, j1, Pr, 0);
}

/** Transforms a block of L (16 <= L <= NTT_BASE_LEN) forward in cache:
    levels L/2 .. 8 in fused pairs (plus one radix-2 level if the count is
    odd), then the in-register kernel for levels 4, 2, 1. Twiddles here come
    from the start of the tables, which stays in cache, so the kernels use
    the precomputed rtp as well (vmontw). */
static void ntt_forward_base(uint32_t *a, size_t L, const NttPrime *Pr) {
    size_t h = L >> 1;
    for (; h >= 16; h >>= 2)
        for (size_t s = 0; s < L; s += 2 * h) dif4_body(a + s, NULL, h / 2, 0, h / 2, Pr, 1);
    if (h == 8)
        for (size_t s = 0; s < L; s += 16) dif2_body(a + s, 8, 0, 8, Pr, 1);
    SmallTw t = small_twiddles(Pr);
    for (size_t s = 0; s < L; s += 16) dif_last3(a + s, &t);
}

/** Transforms a block of L back in cache: the mirror of ntt_forward_base.
    DIT levels must run in increasing order; how they're paired doesn't
    matter, so the odd radix-2 level (if any) comes last here. */
static void ntt_inverse_base(uint32_t *a, size_t L, const NttPrime *Pr) {
    SmallTw t = small_twiddles(Pr);
    for (size_t s = 0; s < L; s += 16) dit_first3(a + s, &t);
    size_t h = 8;
    for (; 4 * h <= L; h <<= 2)
        for (size_t s = 0; s < L; s += 4 * h) dit4_body(a + s, h, 0, h, Pr, 1);
    if (h < L)
        for (size_t s = 0; s < L; s += 2 * h) dit2_body(a + s, h, 0, h, Pr, 1);
}

#ifndef NTT_R16_MIN_LEN
#define NTT_R16_MIN_LEN (1 << 16) /* from this block length on, radix-16 top passes */
#endif
#define R16_CHUNK 128 /* columns per radix-16 step: 16 rows x 128 words = 8 KB */

/** Returns the radix of the top pass of a block of L above the base size,
    which leaves that many independent sub-transforms: 2 if the halves
    already fit the base size, 16 for big blocks (4 levels per trip through
    memory), else 4. */
static int ntt_radix(size_t L) {
    if (L / 2 <= NTT_BASE_LEN) return 2;
    return L >= NTT_R16_MIN_LEN ? 16 : 4;
}

/* The outer radix-4 step of a radix-16 pass (levels h = L/2 and L/4, lanes
   j = c + m Q2 with c < Q2 = L/16) needs twiddles from the two biggest
   level tables, 3/4 of the pass's twiddle traffic. Each is a small-table
   entry times a constant root:
     rt[L/2 + c + k Q2] = w_L^c w_16^k,   rt[L/4 + c + m Q2] = w_(L/2)^c w_8^m,
   with w_L^c = rt[L/2 + c], w_(L/2)^c = rt[L/4 + c], w_16^k = rt[8 + k] and
   w_8^m = rt[4 + m]. Past NTT_TWD_MIN_LEN, where the tables no longer fit
   in cache, the pass reads only the first L/16 entries of the two levels
   and makes the rest with one constant multiply each. */
#ifndef NTT_TWD_MIN_LEN
#define NTT_TWD_MIN_LEN (1 << 19)
#endif

/** The constant roots of row group m for the derived twiddles. */
typedef struct {
    __m256i ka, kap, kb, kbp, kc, kcp; /* w_16^m, w_16^(m+4), w_8^m and their vpre */
} Tw16;

/** Returns the Tw16 of row group m (0..3). */
static Tw16 tw16_consts(const NttPrime *Pr, int m) {
    const __m256i PINV = _mm256_set1_epi32((int)Pr->pinv);
    uint32_t a = Pr->rt[8 + m], b = Pr->rt[12 + m], c = Pr->rt[4 + m];
    return (Tw16){_mm256_set1_epi32((int)a), vpre(a, PINV), _mm256_set1_epi32((int)b), vpre(b, PINV),
                  _mm256_set1_epi32((int)c), vpre(c, PINV)};
}

/** Runs dif4_body at Q = L/4 on columns [c, e) of row group m (lanes
    m*L/16 + [c, e)), with derived twiddles: one table read per level and
    one vmontw by a Tw16 constant (none for m = 0, whose constant is 1). */
static inline __attribute__((always_inline)) void dif4_derived(uint32_t *a, const Src *src, size_t L, size_t c,
                                                               size_t e, const NttPrime *Pr, int m) {
    const __m256i P = _mm256_set1_epi32((int)Pr->p), PINV = _mm256_set1_epi32((int)Pr->pinv);
    const size_t Q = L / 4, off = m * (L / 16);
    const uint32_t *t1 = Pr->rt + L / 2, *t2 = Pr->rt + L / 4;
    const Tw16 k = tw16_consts(Pr, m);
    uint32_t *a0 = a + off, *a1 = a0 + Q, *a2 = a0 + 2 * Q, *a3 = a0 + 3 * Q;
    for (size_t j = c; j < e; j += 8) {
        __m256i x0, x1, x2, x3;
        if (src) {
            const __m256i R2 = _mm256_set1_epi32((int)Pr->r2);
            x0 = src_load8(src, off + j, P, PINV, R2);
            x1 = src_load8(src, off + Q + j, P, PINV, R2);
            x2 = src_load8(src, off + 2 * Q + j, P, PINV, R2);
            x3 = src_load8(src, off + 3 * Q + j, P, PINV, R2);
        } else {
            x0 = LD(a0 + j), x1 = LD(a1 + j), x2 = LD(a2 + j), x3 = LD(a3 + j);
        }
        __m256i u1 = LD(t1 + j), u2 = LD(t2 + j);
        __m256i wa = m ? vmontw(u1, k.ka, k.kap, P) : u1;
        __m256i wb = vmontw(u1, k.kb, k.kbp, P);
        __m256i wc = m ? vmontw(u2, k.kc, k.kcp, P) : u2;
        __m256i b0 = vaddm(x0, x2, P);
        __m256i b2 = vmont(_mm256_sub_epi32(x0, x2), wa, P, PINV);
        __m256i b1 = vaddm(x1, x3, P);
        __m256i b3 = vmont(_mm256_sub_epi32(x1, x3), wb, P, PINV);
        __m256i wco = vodd(wc);
        __m256i d0 = _mm256_sub_epi32(b0, b1), d2 = _mm256_sub_epi32(b2, b3);
        ST(a0 + j, vaddm(b0, b1, P));
        ST(a1 + j, vmont2(d0, vodd(d0), wc, wco, P, PINV));
        ST(a2 + j, vaddm(b2, b3, P));
        ST(a3 + j, vmont2(d2, vodd(d2), wc, wco, P, PINV));
    }
}

/** Runs dit4_body at Q = L/4 on columns [c, e) of row group m, with derived
    twiddles: the mirror of dif4_derived. */
static inline __attribute__((always_inline)) void dit4_derived(uint32_t *a, size_t L, size_t c, size_t e,
                                                               const NttPrime *Pr, int m) {
    const __m256i P = _mm256_set1_epi32((int)Pr->p), PINV = _mm256_set1_epi32((int)Pr->pinv);
    const size_t Q = L / 4, off = m * (L / 16);
    const uint32_t *t1 = Pr->rt + L / 2, *t2 = Pr->rt + L / 4;
    const Tw16 k = tw16_consts(Pr, m);
    uint32_t *a0 = a + off, *a1 = a0 + Q, *a2 = a0 + 2 * Q, *a3 = a0 + 3 * Q;
    for (size_t j = c; j < e; j += 8) {
        __m256i x0 = LD(a0 + j), x1 = LD(a1 + j), x2 = LD(a2 + j), x3 = LD(a3 + j);
        __m256i u1 = LD(t1 + j), u2 = LD(t2 + j);
        __m256i wa = m ? vmontw(u1, k.ka, k.kap, P) : u1;
        __m256i wb = vmontw(u1, k.kb, k.kbp, P);
        __m256i wc = m ? vmontw(u2, k.kc, k.kcp, P) : u2;
        __m256i wco = vodd(wc);
        __m256i t = vmont2(x1, vodd(x1), wc, wco, P, PINV);
        __m256i b0 = vaddm(x0, t, P);
        __m256i b1 = vsubm(x0, t, P);
        t = vmont2(x3, vodd(x3), wc, wco, P, PINV);
        __m256i b2 = vaddm(x2, t, P);
        __m256i b3 = vsubm(x2, t, P);
        t = vmont(b2, wa, P, PINV);
        ST(a0 + j, vaddm(b0, t, P));
        ST(a2 + j, vsubm(b0, t, P));
        t = vmont(b3, wb, P, PINV);
        ST(a1 + j, vaddm(b1, t, P));
        ST(a3 + j, vsubm(b1, t, P));
    }
}

/** Runs dif4_derived on all four row groups. The two branches look alike on
    purpose: with src a constant NULL in one of them, gcc compiles a copy of
    the loop without the "read from a bignum?" test. */
static void dif4_derived_all(uint32_t *a, const Src *src, size_t L, size_t c, size_t e, const NttPrime *Pr) {
    if (src) {
        dif4_derived(a, src, L, c, e, Pr, 0);
        dif4_derived(a, src, L, c, e, Pr, 1);
        dif4_derived(a, src, L, c, e, Pr, 2);
        dif4_derived(a, src, L, c, e, Pr, 3);
    } else {
        dif4_derived(a, NULL, L, c, e, Pr, 0);
        dif4_derived(a, NULL, L, c, e, Pr, 1);
        dif4_derived(a, NULL, L, c, e, Pr, 2);
        dif4_derived(a, NULL, L, c, e, Pr, 3);
    }
}

/** Runs dit4_derived on all four row groups. */
static void dit4_derived_all(uint32_t *a, size_t L, size_t c, size_t e, const NttPrime *Pr) {
    dit4_derived(a, L, c, e, Pr, 0);
    dit4_derived(a, L, c, e, Pr, 1);
    dit4_derived(a, L, c, e, Pr, 2);
    dit4_derived(a, L, c, e, Pr, 3);
}

/** Runs four DIF levels (h = L/2 .. L/16) on columns [c, e) of L/16. Those
    levels only combine words in the same column mod L/16, so a chunk of
    columns is a self-contained job: two radix-4 steps over 16 rows x
    R16_CHUNK words that stay in L1. The outer step (levels L/2, L/4) uses
    derived twiddles past NTT_TWD_MIN_LEN; the inner one (L/8, L/16) runs
    on each quarter r. */
static void dif16_chunk(uint32_t *a, size_t L, size_t c, size_t e, const NttPrime *Pr, const Src *src) {
    size_t Q = L / 4, Q2 = L / 16;
    if (L >= NTT_TWD_MIN_LEN) {
        dif4_derived_all(a, src, L, c, e, Pr);
    } else {
        for (size_t m = 0; m < 4; m++) {
            if (src) dif_stage4_src(a, src, Q, c + m * Q2, e + m * Q2, Pr);
            else dif_stage4(a, Q, c + m * Q2, e + m * Q2, Pr);
        }
    }
    for (size_t r = 0; r < 4; r++) dif_stage4(a + r * Q, Q2, c, e, Pr);
}

/** Runs four DIT levels (h = L/16 .. L/2) on columns [c, e) of L/16: the
    mirror of dif16_chunk. */
static void dit16_chunk(uint32_t *a, size_t L, size_t c, size_t e, const NttPrime *Pr) {
    size_t Q = L / 4, Q2 = L / 16;
    for (size_t r = 0; r < 4; r++) dit_stage4(a + r * Q, Q2, c, e, Pr);
    if (L >= NTT_TWD_MIN_LEN) {
        dit4_derived_all(a, L, c, e, Pr);
    } else {
        for (size_t m = 0; m < 4; m++) dit_stage4(a, Q, c + m * Q2, e + m * Q2, Pr);
    }
}

/** Runs the radix-16 DIF top pass on columns [c0, c1) of L/16 (one
    thread's share), R16_CHUNK columns at a time. */
static void dif_pass16(uint32_t *a, size_t L, size_t c0, size_t c1, const NttPrime *Pr, const Src *src) {
    for (size_t c = c0; c < c1; c += R16_CHUNK)
        dif16_chunk(a, L, c, c + R16_CHUNK < c1 ? c + R16_CHUNK : c1, Pr, src);
}

/** Runs the radix-16 DIT top pass on columns [c0, c1): the mirror of
    dif_pass16. */
static void dit_pass16(uint32_t *a, size_t L, size_t c0, size_t c1, const NttPrime *Pr) {
    for (size_t c = c0; c < c1; c += R16_CHUNK)
        dit16_chunk(a, L, c, c + R16_CHUNK < c1 ? c + R16_CHUNK : c1, Pr);
}
/* ======== src/ntt_conv.c ======== */
/**
 * @file   ntt_conv.c
 * @brief  Cyclic convolution of power-of-2 length, and standalone transforms.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * A block too big for the caches gets one top pass (radix 2, 4 or 16, see
 * ntt_radix), which leaves nb independent sub-blocks of L/nb. Each sub-block
 * is then convolved on its own, recursively: forward, pointwise product and
 * inverse while it is still in cache. A final inverse top pass puts the
 * block back together. So only the top passes stream through main memory.
 * Blocks up to NTT_BASE_LEN use the in-cache base kernels directly. Top
 * passes split their lanes across threads; sub-blocks are dealt out to
 * threads whole.
 *
 * ntt_fwd and ntt_inv are the two halves of ntt_conv on their own, for the
 * split multiply (mul_split.c), which combines forward results before
 * taking them back.
 */


#ifndef NTT_PAR_MIN_LEN
#define NTT_PAR_MIN_LEN 65536   /* don't spawn threads for blocks smaller than this */
#endif
#ifndef NTT_THREADS
#define NTT_THREADS 4           /* at most this many threads per transform (a power of 2) */
#endif
#ifndef NTT_SEQ_MIN_LEN
#define NTT_SEQ_MIN_LEN (1 << 19) /* from this length on, one prime at a time */
#endif
#ifndef NTT_THREADS_SEQ
#define NTT_THREADS_SEQ 64      /* at most this many then (a power of 2); see ntt_threads */
#endif
#ifndef NTT_SEQ_PIECES
#define NTT_SEQ_PIECES 4        /* then top-pass pieces per thread (a power of 2), see ntt_pieces */
#endif
/* the most tasks one pass forks: sizes the task arrays here and in ntt_radix3.c */
#define NTT_THREADS_MAX (NTT_THREADS > NTT_THREADS_SEQ ? NTT_THREADS : NTT_THREADS_SEQ)
#define NTT_TASKS_MAX (NTT_THREADS_MAX * NTT_SEQ_PIECES)
_Static_assert(NTT_THREADS > 0 && (NTT_THREADS & (NTT_THREADS - 1)) == 0 && NTT_THREADS_SEQ > 0 &&
                   (NTT_THREADS_SEQ & (NTT_THREADS_SEQ - 1)) == 0,
               "NTT_THREADS and NTT_THREADS_SEQ must be powers of 2 (ntt_sub_blocks splits blocks evenly)");
_Static_assert(NTT_SEQ_PIECES > 0 && (NTT_SEQ_PIECES & (NTT_SEQ_PIECES - 1)) == 0,
               "NTT_SEQ_PIECES must be a power of 2 (pieces stay multiples of 8 lanes)");

/** Returns the threads per transform: the logical CPUs rounded down to a
    power of 2 (16 on the i9-9900K), at most NTT_THREADS_SEQ for transforms
    of NTT_SEQ_MIN_LEN and longer (seq) and NTT_THREADS otherwise. */
static int ntt_threads(int seq) {
    int cap = seq ? NTT_THREADS_SEQ : NTT_THREADS, n = pool_cpus(), t = 1;
    while (t * 2 <= n && t * 2 <= cap) t *= 2;
    return t;
}

/** Returns how many pieces a top pass of `lanes` lanes on nt threads is cut
    into (at least 8 lanes, one AVX2 vector, each). Transforms of
    NTT_SEQ_MIN_LEN and longer get NTT_SEQ_PIECES per thread: other programs
    always hold a CPU or two, so with one piece per thread the pass waited on
    whichever piece started late (measured: ~600 us of a 1.2 ms pass at
    2^21). Spare pieces go to whichever threads finish first. Shorter
    passes keep one piece per thread: there the extra tasks cost more than
    they save. */
static int ntt_pieces(size_t L, size_t lanes, int nt) {
    if (L >= NTT_SEQ_MIN_LEN && lanes >= 8 * (size_t)nt * NTT_SEQ_PIECES) return nt * NTT_SEQ_PIECES;
    return nt;
}

/** Multiplies a by b pointwise mod p (Montgomery): a[i] = a[i] * b[i] * R^-1;
    squares when b == a (one load per element). */
static void ntt_pointwise(uint32_t *a, const uint32_t *b, size_t L, const NttPrime *Pr) {
    const __m256i P = _mm256_set1_epi32((int)Pr->p), PINV = _mm256_set1_epi32((int)Pr->pinv);
    if (a == b) {
        for (size_t i = 0; i < L; i += 8) {
            __m256i v = LD(a + i);
            ST(a + i, vmont(v, v, P, PINV));
        }
    } else {
        for (size_t i = 0; i < L; i += 8) ST(a + i, vmont(LD(a + i), LD(b + i), P, PINV));
    }
}

/** Fills f with the first L coefficients of a bignum, reduced mod p (zero
    past its end). */
static void ntt_load(uint32_t *f, const Src *s, size_t L, const NttPrime *Pr) {
    const __m256i P = _mm256_set1_epi32((int)Pr->p), PINV = _mm256_set1_epi32((int)Pr->pinv),
                  R2 = _mm256_set1_epi32((int)Pr->r2);
    for (size_t k = 0; k < L; k += 8) ST(f + k, src_load8(s, k, P, PINV, R2));
}

/** What an NttTask does. */
typedef enum {
    NTT_TOP_FWD,  /* forward top pass over lanes j0..j1 */
    NTT_TOP_INV,  /* inverse top pass over lanes j0..j1 */
    NTT_SUB_CONV, /* nblocks sub-block convolutions */
    NTT_SUB_FWD,  /* nblocks sub-block forward transforms */
    NTT_SUB_INV,  /* nblocks sub-block inverse transforms */
} NttKind;

/** One piece of work for the thread pool: part of a top pass, or a run of
    whole sub-blocks. */
typedef struct {
    int kind;        /* an NttKind; int because an enum field changes gcc's switch code */
    uint32_t *a, *b; /* the block(s); b == a squares (sub-block convolutions) */
    size_t len;      /* block length */
    size_t j0, j1;   /* top passes: lanes (columns) [j0, j1) */
    size_t nblocks;  /* sub-block kinds: consecutive blocks of len */
    const NttPrime *P;
    const Src *src;  /* NTT_TOP_FWD: input read from here instead of a */
    int nt;          /* sub-block kinds: threads for each block */
} NttTask;

// ntt_task_run recurses into these through the sub-block kinds
static void ntt_conv(uint32_t *a, uint32_t *b, size_t L, const NttPrime *P, int nt, const Src *sa, const Src *sb);
static void ntt_fwd(uint32_t *a, size_t L, const NttPrime *P, int nt, const Src *src);
static void ntt_inv(uint32_t *a, size_t L, const NttPrime *P, int nt);

/** Runs one task: a top pass with the kernel for the block's radix, or
    nblocks sub-blocks one after another. */
static void ntt_task_run(NttTask *t) {
    int radix = t->kind >= NTT_SUB_CONV ? 0 : ntt_radix(t->len);
    switch (t->kind) {
    case NTT_TOP_FWD:
        if (radix == 2) dif_stage(t->a, t->len / 2, t->j0, t->j1, t->P);
        else if (radix == 16) dif_pass16(t->a, t->len, t->j0, t->j1, t->P, t->src);
        else if (t->src) dif_stage4_src(t->a, t->src, t->len / 4, t->j0, t->j1, t->P);
        else dif_stage4(t->a, t->len / 4, t->j0, t->j1, t->P);
        break;
    case NTT_TOP_INV:
        if (radix == 2) dit_stage(t->a, t->len / 2, t->j0, t->j1, t->P);
        else if (radix == 16) dit_pass16(t->a, t->len, t->j0, t->j1, t->P);
        else dit_stage4(t->a, t->len / 4, t->j0, t->j1, t->P);
        break;
    case NTT_SUB_CONV:
        for (size_t k = 0; k < t->nblocks; k++)
            ntt_conv(t->a + k * t->len, t->b + k * t->len, t->len, t->P, t->nt, NULL, NULL);
        break;
    case NTT_SUB_FWD:
        for (size_t k = 0; k < t->nblocks; k++) ntt_fwd(t->a + k * t->len, t->len, t->P, t->nt, NULL);
        break;
    case NTT_SUB_INV:
        for (size_t k = 0; k < t->nblocks; k++) ntt_inv(t->a + k * t->len, t->len, t->P, t->nt);
        break;
    }
}

/** Pool task: runs one NttTask. */
static void ntt_task_proc(void *arg) {
    ntt_task_run((NttTask *)arg);
}

/** Runs the top pass (NTT_TOP_FWD or NTT_TOP_INV) of block a[0..L), its
    lanes split across nt threads (in ntt_pieces pieces) when the block is
    big enough to pay for it (at least 8 lanes, one AVX2 vector, per
    thread). */
static void ntt_top_stage(uint32_t *a, size_t L, const NttPrime *P, int nt, NttKind kind, const Src *src) {
    size_t lanes = L / ntt_radix(L);
    if (nt > 1 && L >= NTT_PAR_MIN_LEN && lanes >= 8 * (size_t)nt) {
        NttTask tasks[NTT_TASKS_MAX];
        int np = ntt_pieces(L, lanes, nt);
        for (int i = 0; i < np; i++)
            tasks[i] = (NttTask){kind, a, a, L, lanes * i / np, lanes * (i + 1) / np, 0, P, src, 1};
        pool_run(ntt_task_proc, tasks, np, sizeof tasks[0]);
    } else {
        NttTask t = {kind, a, a, L, 0, lanes, 0, P, src, 1};
        ntt_task_run(&t);
    }
}

/** Processes the nb sub-blocks under a top pass (convolutions, forward or
    inverse transforms, by kind). With nt threads, min(nt, nb) workers take
    nb / workers consecutive blocks each and nt / workers threads for each
    block's own passes; both counts are powers of 2, so this is exact. */
static void ntt_sub_blocks(uint32_t *a, uint32_t *b, size_t L, const NttPrime *P, int nt, NttKind kind) {
    size_t nb = (size_t)ntt_radix(L), len = L / nb;
    if (nt > 1 && L >= NTT_PAR_MIN_LEN) {
        int workers = nt < (int)nb ? nt : (int)nb;
        NttTask tasks[NTT_THREADS_MAX];
        for (int i = 0; i < workers; i++) {
            size_t off = i * (nb / workers) * len;
            tasks[i] = (NttTask){kind, a + off, b + off, len, 0, 0, nb / workers, P, NULL, nt / workers};
        }
        pool_run(ntt_task_proc, tasks, workers, sizeof tasks[0]);
    } else {
        NttTask t = {kind, a, b, len, 0, 0, nb, P, NULL, 1};
        ntt_task_run(&t);
    }
}

/** Convolves the sub-blocks of a and b under their top passes (also used by
    ntt_radix3.c's fused path). */
static void ntt_sub_convs(uint32_t *a, uint32_t *b, size_t L, const NttPrime *P, int nt) {
    ntt_sub_blocks(a, b, L, P, nt, NTT_SUB_CONV);
}

/** Replaces a with L * (a (*) b)[-n mod L]: the cyclic convolution,
    index-reversed (the inverse uses the forward roots), by DIF forward,
    pointwise product and DIT inverse; b == a squares. The recursion is
    fused: a sub-block goes forward, pointwise and back while it is still in
    cache, so only the top passes of big blocks touch main memory. With
    sa/sb the inputs come straight from bignums and a, b start
    uninitialized. */
static void ntt_conv(uint32_t *a, uint32_t *b, size_t L, const NttPrime *P, int nt, const Src *sa, const Src *sb) {
    // the radix-2 pass and the base kernels can't read a Src: load first
    // (such blocks are small, so the extra pass is cheap)
    if (sa && ntt_radix(L) == 2) {
        ntt_load(a, sa, L, P);
        if (b != a) ntt_load(b, sb, L, P);
        sa = sb = NULL;
    }

    if (L <= NTT_BASE_LEN) {
        ntt_forward_base(a, L, P);
        if (b != a) ntt_forward_base(b, L, P);
        ntt_pointwise(a, b, L, P);
        ntt_inverse_base(a, L, P);
        return;
    }

    ntt_top_stage(a, L, P, nt, NTT_TOP_FWD, sa);
    if (b != a) ntt_top_stage(b, L, P, nt, NTT_TOP_FWD, sb);
    ntt_sub_convs(a, b, L, P, nt);
    ntt_top_stage(a, L, P, nt, NTT_TOP_INV, NULL);
}

/** Transforms a forward in place (read from src when given, like ntt_conv's
    sa), for a power-of-2 L. The result is in the same permuted order
    ntt_conv uses between its halves, so pointwise combinations of forward
    results can be taken back with ntt_inv. For the split multiply. */
static void ntt_fwd(uint32_t *a, size_t L, const NttPrime *P, int nt, const Src *src) {
    if (src && ntt_radix(L) == 2) {
        ntt_load(a, src, L, P);
        src = NULL;
    }
    if (L <= NTT_BASE_LEN) {
        ntt_forward_base(a, L, P);
        return;
    }
    ntt_top_stage(a, L, P, nt, NTT_TOP_FWD, src);
    ntt_sub_blocks(a, a, L, P, nt, NTT_SUB_FWD);
}

/** Transforms a back in place from ntt_fwd's order: sub-blocks first, then
    the top pass (ntt_fwd in reverse). Like ntt_conv's result, it is scaled
    by L and index-reversed. */
static void ntt_inv(uint32_t *a, size_t L, const NttPrime *P, int nt) {
    if (L <= NTT_BASE_LEN) {
        ntt_inverse_base(a, L, P);
        return;
    }
    ntt_sub_blocks(a, a, L, P, nt, NTT_SUB_INV);
    ntt_top_stage(a, L, P, nt, NTT_TOP_INV, NULL);
}
/* ======== src/ntt_radix3.c ======== */
/**
 * @file   ntt_radix3.c
 * @brief  Convolutions of length 3 * 2^k: a radix-3 layer over three power-of-2 ones.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * Power-of-2 lengths can overshoot the needed size by almost 2x; allowing
 * L = 3M (M a power of 2) as well caps the waste at 1.5x. All three primes
 * have 3 | p - 1, so a cube root of unity w3 exists. The DIF layer, for each
 * n1 < M with a_r = x[n1 + rM]:
 *   y0 = a0 + a1 + a2
 *   y1 = (a0 + w3 a1 + w3^2 a2) * wL^n1
 *   y2 = (a0 + w3^2 a1 + w3 a2) * wL^2n1
 * leaves three independent length-M transforms (X[3k + r] = DFT_M(y_r)[k]),
 * done by ntt_conv. Since w3^2 = -1 - w3, both middle sums share
 * d = w3 (a1 - a2):
 *   a0 + w3 a1 + w3^2 a2 = a0 - a2 + d,   a0 + w3^2 a1 + w3 a2 = a0 - a1 - d
 * so a butterfly costs one multiply for d plus the two twiddles. The DIT
 * mirror twiddles first and combines after; with forward roots it again
 * yields L * x[-n mod L]. Twiddles wL^n1 are generated on the fly, 8 lanes
 * at a time, rather than stored.
 *
 * For big M the layer is fused with the sub-blocks' radix-16 top passes
 * (r3_fused), saving a trip through memory. ntt_conv_any picks the path for
 * any supported length.
 */


/** Constants of the radix-3 layer for one prime and one M. */
typedef struct {
    __m256i P, PINV;
    __m256i W3, W3p; /* w3 = wL^M (Montgomery form) and its vpre */
    __m256i W8, W8p; /* wL^8: steps a twiddle vector by 8 lanes */
    __m256i R2;      /* for src_load8 */
    uint32_t wL;     /* a primitive 3M-th root of unity (plain form) */
} R3K;

/** Returns the R3K of prime Pr for L = 3M. */
static R3K r3_consts(const NttPrime *Pr, size_t M) {
    const uint32_t p = Pr->p;
    R3K k;
    k.P = _mm256_set1_epi32((int)p);
    k.PINV = _mm256_set1_epi32((int)Pr->pinv);
    k.R2 = _mm256_set1_epi32((int)Pr->r2);
    k.wL = powmod32(Pr->g, (p - 1) / (3 * M), p);
    const uint32_t w3 = to_mont(powmod32(k.wL, M, p), p), w8 = to_mont(powmod32(k.wL, 8, p), p);
    k.W3 = _mm256_set1_epi32((int)w3), k.W3p = vpre(w3, k.PINV);
    k.W8 = _mm256_set1_epi32((int)w8), k.W8p = vpre(w8, k.PINV);
    return k;
}

/** Returns wL^(n + i) for the 8 lanes i, in Montgomery form: the start of a
    twiddle chain at lane n. Scalar (8 multiplies), once per chain. */
static __m256i r3_tw_start(uint32_t wL, size_t n, uint32_t p) {
    uint32_t start[8], cur = powmod32(wL, n, p);
    for (int i = 0; i < 8; i++, cur = (uint32_t)((uint64_t)cur * wL % p)) start[i] = to_mont(cur, p);
    return LDU(start);
}

/** Runs the radix-3 butterflies (DIF, or DIT if inverse) on lanes [j0, j1),
    reading the input from src when given. *tw holds wL^j0.. on entry and
    the twiddles for j1 on return, so a caller can continue the chain; each
    step multiplies it by wL^8, and tw2 = tw1^2 gives wL^2n1. */
static inline __attribute__((always_inline)) void r3_run(uint32_t *a, size_t M, size_t j0, size_t j1, const R3K *k,
                                                         __m256i *tw, int inverse, const Src *src) {
    const __m256i P = k->P, PINV = k->PINV;
    __m256i tw1 = *tw;
    uint32_t *x0 = a, *x1 = a + M, *x2 = a + 2 * M;
    for (size_t j = j0; j < j1; j += 8) {
        __m256i tw2 = vmont(tw1, tw1, P, PINV);
        __m256i a0, a1, a2;
        if (src) {
            a0 = src_load8(src, j, P, PINV, k->R2);
            a1 = src_load8(src, M + j, P, PINV, k->R2);
            a2 = src_load8(src, 2 * M + j, P, PINV, k->R2);
        } else {
            a0 = LD(x0 + j), a1 = LD(x1 + j), a2 = LD(x2 + j);
        }
        if (!inverse) {
            __m256i y0 = vaddm(vaddm(a0, a1, P), a2, P);
            __m256i d = vmontw(_mm256_sub_epi32(a1, a2), k->W3, k->W3p, P);
            __m256i y1 = vaddm(vsubm(a0, a2, P), d, P);
            __m256i y2 = vsubm(vsubm(a0, a1, P), d, P);
            ST(x0 + j, y0);
            ST(x1 + j, vmont(y1, tw1, P, PINV));
            ST(x2 + j, vmont(y2, tw2, P, PINV));
        } else {
            __m256i t1 = vmont(a1, tw1, P, PINV), t2 = vmont(a2, tw2, P, PINV);
            __m256i d = vmontw(_mm256_sub_epi32(t1, t2), k->W3, k->W3p, P);
            ST(x0 + j, vaddm(vaddm(a0, t1, P), t2, P));
            ST(x1 + j, vaddm(vsubm(a0, t2, P), d, P));
            ST(x2 + j, vsubm(vsubm(a0, t1, P), d, P));
        }
        tw1 = vmontw(tw1, k->W8, k->W8p, P);
    }
    *tw = tw1;
}

/** Runs the plain radix-3 layer on lanes [j0, j1) of M. The two r3_run
    calls look alike on purpose, see dif4_derived_all. */
static void r3_stage(uint32_t *a, size_t M, size_t j0, size_t j1, const NttPrime *Pr, int inverse, const Src *src) {
    R3K k = r3_consts(Pr, M);
    __m256i tw = r3_tw_start(k.wL, j0, Pr->p);
    if (src) r3_run(a, M, j0, j1, &k, &tw, inverse, src);
    else r3_run(a, M, j0, j1, &k, &tw, inverse, NULL);
}

#ifndef NTT_R3_FUSE_MIN_LEN
#define NTT_R3_FUSE_MIN_LEN (1 << 17) /* from this M on (with radix 16), use r3_fused */
#endif

/** Runs the radix-3 layer fused with the radix-16 top passes of the three
    sub-blocks, on columns [c0, c1) of M/16, chunk by chunk. A chunk's
    radix-3 butterflies on 16 rows produce exactly the 16 rows x chunk that
    each sub-block's radix-16 step needs, so those run right away from L1
    instead of in a second trip through memory. Each of the 16 rows carries
    its own running twiddle. The inverse mirrors: radix-16 steps first, then
    the radix-3 butterflies. */
static void r3_fused(uint32_t *a, size_t M, size_t c0, size_t c1, const NttPrime *Pr, int inverse, const Src *src) {
    const R3K k = r3_consts(Pr, M);
    const size_t Q2 = M / 16;
    __m256i tw[16];
    for (int r = 0; r < 16; r++) tw[r] = r3_tw_start(k.wL, r * Q2 + c0, Pr->p);
    for (size_t c = c0; c < c1; c += R16_CHUNK) {
        size_t e = c + R16_CHUNK < c1 ? c + R16_CHUNK : c1;
        if (!inverse) {
            for (int r = 0; r < 16; r++) {
                if (src) r3_run(a, M, r * Q2 + c, r * Q2 + e, &k, &tw[r], 0, src);
                else r3_run(a, M, r * Q2 + c, r * Q2 + e, &k, &tw[r], 0, NULL);
            }
            for (int s = 0; s < 3; s++) dif16_chunk(a + s * M, M, c, e, Pr, NULL);
        } else {
            for (int s = 0; s < 3; s++) dit16_chunk(a + s * M, M, c, e, Pr);
            for (int r = 0; r < 16; r++) r3_run(a, M, r * Q2 + c, r * Q2 + e, &k, &tw[r], 1, NULL);
        }
    }
}

/** One thread's share of a radix-3 layer, for the thread pool. */
typedef struct {
    uint32_t *a;
    size_t M, j0, j1;  /* sub-length; lanes (or columns, when fused) [j0, j1) */
    const NttPrime *P;
    int inverse, fused;
    const Src *src;    /* forward only: input read from here instead of a */
} R3Task;

/** Pool task: runs one R3Task, fused or plain. */
static void r3_proc(void *arg) {
    R3Task *t = (R3Task *)arg;
    if (t->fused) r3_fused(t->a, t->M, t->j0, t->j1, t->P, t->inverse, t->src);
    else r3_stage(t->a, t->M, t->j0, t->j1, t->P, t->inverse, t->src);
}

/** Runs the radix-3 layer (fused: plus the sub-blocks' top passes), its
    lanes split across nt threads (in ntt_pieces pieces): M lanes, or M/16
    columns when fused. */
static void r3_layer(uint32_t *a, size_t M, const NttPrime *P, int nt, int inverse, const Src *src, int fused) {
    size_t lanes = fused ? M / 16 : M;
    if (nt > 1 && 3 * M >= NTT_PAR_MIN_LEN && lanes >= 8 * (size_t)nt) {
        R3Task tasks[NTT_TASKS_MAX];
        int np = ntt_pieces(3 * M, lanes, nt);
        for (int i = 0; i < np; i++)
            tasks[i] = (R3Task){a, M, lanes * i / np, lanes * (i + 1) / np, P, inverse, fused, src};
        pool_run(r3_proc, tasks, np, sizeof tasks[0]);
    } else {
        R3Task t = {a, M, 0, lanes, P, inverse, fused, src};
        r3_proc(&t);
    }
}

#ifndef NTT_R3_SEQ_MIN_LEN
#define NTT_R3_SEQ_MIN_LEN (1 << 18) /* from this M on, sub-convolutions one after another */
#endif

/** Runs the three length-M sub-convolutions. Big ones run one after
    another, each on all nt threads (one block's data stays in cache, and
    three blocks don't split evenly over a power-of-2 thread count); smaller
    ones run side by side with nt/2 threads each. Blocks too small for
    threads just run in a loop. */
static void r3_sub_convs(uint32_t *a, uint32_t *b, size_t M, const NttPrime *P, int nt) {
    int par = nt > 1 && 3 * M >= NTT_PAR_MIN_LEN;
    if (M >= NTT_R3_SEQ_MIN_LEN || !par) {
        for (int r = 0; r < 3; r++) ntt_conv(a + r * M, b + r * M, M, P, nt, NULL, NULL);
        return;
    }
    int sub = nt >= 4 ? nt / 2 : 1;
    NttTask t[3];
    for (int r = 0; r < 3; r++) t[r] = (NttTask){NTT_SUB_CONV, a + r * M, b + r * M, M, 0, 0, 1, P, NULL, sub};
    pool_run(ntt_task_proc, t, 3, sizeof t[0]);
}

/** Runs ntt_conv for any supported length: 2^k directly, or 3 * 2^k via the
    radix-3 layer (fused with the sub-blocks' radix-16 passes when M is big
    enough, else plain layer + r3_sub_convs). Same result contract as
    ntt_conv: a <- L * (a (*) b)[-n mod L]. */
static void ntt_conv_any(uint32_t *a, uint32_t *b, size_t L, const NttPrime *P, int nt, const Src *sa, const Src *sb) {
    if ((L & (L - 1)) == 0) {
        ntt_conv(a, b, L, P, nt, sa, sb);
        return;
    }

    size_t M = L / 3;
    if (M >= NTT_R3_FUSE_MIN_LEN && ntt_radix(M) == 16) {
        r3_layer(a, M, P, nt, 0, sa, 1);
        if (b != a) r3_layer(b, M, P, nt, 0, sb, 1);
        for (int s = 0; s < 3; s++) ntt_sub_convs(a + s * M, b + s * M, M, P, nt);
        r3_layer(a, M, P, nt, 1, NULL, 1);
        return;
    }

    r3_layer(a, M, P, nt, 0, sa, 0);
    if (b != a) r3_layer(b, M, P, nt, 0, sb, 0);
    r3_sub_convs(a, b, M, P, nt);
    r3_layer(a, M, P, nt, 1, NULL, 0);
}
/* ======== src/crt.c ======== */
/**
 * @file   crt.c
 * @brief  Chinese remainder theorem: the three primes' results into one bignum.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * The NTT gives every coefficient of the product modulo three primes.
 * Garner's method combines the three residues into the exact coefficient
 * X < p1*p2*p3 (~2^93), 8 coefficients at a time in AVX2. Coefficient i is
 * then added at bit i*b with one running carry. Big results are split into
 * CRT_CHUNKS chunks converted in parallel; their carries are added afterwards.
 */


/* Transform buffers carry NTT_PAD spare words on each side so crt_garner8 can
   read the index-reversed result 8 lanes at a time without bounds checks: it
   reads up to 7 words before the start and the word at index L. */
#define NTT_PAD 8
_Static_assert(NTT_PAD >= 7, "crt_garner8 reads up to 7 words before a residue array");

/** Constants for crt_garner8, broadcast to all 8 lanes: the primes, the
    scale factors SC (see ntt_crt_range) and Garner's inverses INVij = pi^-1
    mod pj. The p-suffixed copies are premultiplied for vmontw. REV reverses
    the 8 lanes; LO32 keeps the low 32 bits of each 64-bit lane. */
typedef struct {
    __m256i P1, P2, P3, SC1, SC2, SC3, SC1p, SC2p, SC3p, INV12, INV13, INV23, INV12p, INV13p, INV23p, REV, LO32;
} CrtK;

/** Garner's method on coefficients i .. i+7. With residues r1, r2, r3:
      x2 = (r2 - x1) / p1 mod p2,  x3 = ((r3 - x1) / p1 - x2) / p2 mod p3,
      X  = x1 + x2*p1 + x3*p1*p2   (x1 = r1).
    Stores x1 + x2*p1 as 64-bit values split by parity (even[]: coefficients
    i, i+2, ..; odd[]: i+1, i+3, ..; that's how _mm256_mul_epu32 pairs lanes)
    and x3 in hi[]. Division means multiplying by the inverse INVij. */
static inline __attribute__((always_inline)) void crt_garner8(const CrtK *k, uint32_t *const y[3], size_t L, size_t i,
                                                              uint64_t *even, uint64_t *odd, uint32_t *hi) {
    ptrdiff_t pos = (ptrdiff_t)L - (ptrdiff_t)i - 7; /* may dip into the front pad */
    __m256i r1 = vmontw(_mm256_permutevar8x32_epi32(LDU(y[0] + pos), k->REV), k->SC1, k->SC1p, k->P1);
    __m256i r2 = vmontw(_mm256_permutevar8x32_epi32(LDU(y[1] + pos), k->REV), k->SC2, k->SC2p, k->P2);
    __m256i r3 = vmontw(_mm256_permutevar8x32_epi32(LDU(y[2] + pos), k->REV), k->SC3, k->SC3p, k->P3);

    __m256i x1m2 = vred(r1, k->P2); /* p1 < 2*p2 and p1 < 2*p3 */
    __m256i x2 = vmontw(_mm256_sub_epi32(r2, x1m2), k->INV12, k->INV12p, k->P2);
    __m256i x1m3 = vred(r1, k->P3);
    __m256i t3 = vmontw(_mm256_sub_epi32(r3, x1m3), k->INV13, k->INV13p, k->P3);
    __m256i x2m3 = vred(x2, k->P3);
    __m256i x3 = vmontw(_mm256_sub_epi32(t3, x2m3), k->INV23, k->INV23p, k->P3);

    /* x1 + x2*p1 < 2^62, as 64-bit lanes for even and odd coefficients */
    __m256i le = _mm256_add_epi64(_mm256_mul_epu32(x2, k->P1), _mm256_and_si256(r1, k->LO32));
    __m256i lo = _mm256_add_epi64(_mm256_mul_epu32(vodd(x2), k->P1), _mm256_srli_epi64(r1, 32));
    _mm256_storeu_si256((__m256i *)even, le);
    _mm256_storeu_si256((__m256i *)odd, lo);
    _mm256_storeu_si256((__m256i *)hi, x3);
}

/** Returns coefficient k of a crt_garner8 group: X = x3*p1*p2 + (x1 + x2*p1) (< 2^93). */
static inline __attribute__((always_inline)) u128 crt_value(const uint64_t *even, const uint64_t *odd,
                                                            const uint32_t *hi, size_t k, uint64_t P12) {
    return (u128)hi[k] * P12 + (k & 1 ? odd[k >> 1] : even[k >> 1]);
}

/** Packs 64 coefficients (plus the carry c) into b limbs at w and returns the
    carry into the next block. 64 coefficients of a compile-time width b fill
    exactly b limbs, so after unrolling every bit position is a constant:
    immediate shifts, no branches, whole limbs written once. */
static inline __attribute__((always_inline)) uint64_t crt_pack64(uint64_t *w, const uint64_t *even, const uint64_t *odd,
                                                                 const uint32_t *hi, uint64_t P12, uint64_t c,
                                                                 const unsigned b) {
    const uint64_t mask = (1ULL << b) - 1;
    uint64_t o[35];
#pragma GCC unroll 64
    for (unsigned k = 0; k < 64; k++) {
        u128 acc = crt_value(even, odd, hi, k, P12) + c;
        uint64_t v = (uint64_t)acc & mask;
        c = (uint64_t)(acc >> b);
        unsigned q = k * b / 64, off = k * b % 64;
        if (off == 0) o[q] = v;
        else o[q] |= v << off;
        if (off + b > 64) o[q + 1] = v >> (64 - off);
    }
    memcpy(w, o, b * sizeof(uint64_t));
    return c;
}

/** Converts whole blocks of 64 coefficients from i while they fit below i1,
    with b a compile-time constant (see crt_pack64). Returns where it stopped:
    fewer than 64 coefficients are left for the general loop. */
static inline __attribute__((always_inline)) size_t crt_blocks(const CrtK *k, uint32_t *const y[3], size_t L,
                                                               uint64_t P12, size_t i, size_t i1, uint64_t **w,
                                                               uint64_t *c, const unsigned b) {
    uint64_t even[32], odd[32];
    uint32_t hi[64];
    for (; i + 64 <= i1; i += 64) {
        for (unsigned g = 0; g < 8; g++) crt_garner8(k, y, L, i + 8 * g, even + 4 * g, odd + 4 * g, hi + 8 * g);
        *c = crt_pack64(*w, even, odd, hi, P12, *c, b);
        *w += b;
    }
    return i;
}

/** Converts coefficients [i0, i1) into limbs of out; returns the carry out
    (0 for the last chunk). Residues are read backwards (y[L - i]) and scaled
    by SC = R^2/L. Coefficient i adds X at bit i*b. i0 is a multiple of 64, so
    chunks start on a limb boundary; the last one flushes up to limb `end`. */
static uint64_t ntt_crt_range(uint64_t *out, size_t end, uint32_t *const y[3], size_t L, const uint32_t sc[3],
                              unsigned b, size_t i0, size_t i1, int last) {
    const uint32_t p1 = g_ntt[0].p, p2 = g_ntt[1].p, p3 = g_ntt[2].p;
    const __m256i PINV1 = _mm256_set1_epi32((int)g_ntt[0].pinv), PINV2 = _mm256_set1_epi32((int)g_ntt[1].pinv),
                  PINV3 = _mm256_set1_epi32((int)g_ntt[2].pinv);
    const CrtK k = {_mm256_set1_epi32((int)p1), _mm256_set1_epi32((int)p2), _mm256_set1_epi32((int)p3),
                    _mm256_set1_epi32((int)sc[0]), _mm256_set1_epi32((int)sc[1]), _mm256_set1_epi32((int)sc[2]),
                    vpre(sc[0], PINV1), vpre(sc[1], PINV2), vpre(sc[2], PINV3),
                    _mm256_set1_epi32((int)g_inv_p1_mod_p2), _mm256_set1_epi32((int)g_inv_p1_mod_p3),
                    _mm256_set1_epi32((int)g_inv_p2_mod_p3),
                    vpre(g_inv_p1_mod_p2, PINV2), vpre(g_inv_p1_mod_p3, PINV3), vpre(g_inv_p2_mod_p3, PINV3),
                    _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0), _mm256_set1_epi64x(0xFFFFFFFF)};
    const uint64_t P12 = (uint64_t)p1 * p2, mask = (1ULL << b) - 1;
    uint64_t c = 0, *w = out + i0 * b / 64;
    size_t i = i0; /* a multiple of 64: blocks start on a limb boundary */
    switch (b) {   /* the widths big_mul_ntt picks; others take the loop below */
    case 32: i = crt_blocks(&k, y, L, P12, i, i1, &w, &c, 32); break;
    case 33: i = crt_blocks(&k, y, L, P12, i, i1, &w, &c, 33); break;
    case 34: i = crt_blocks(&k, y, L, P12, i, i1, &w, &c, 34); break;
    case 35: i = crt_blocks(&k, y, L, P12, i, i1, &w, &c, 35); break;
    }
    u128 buf = 0;      /* pending output bits */
    unsigned nbits = 0; /* how many, < 64 between coefficients */
    for (; i < i1; i += 8) {
        uint64_t even[4], odd[4];
        uint32_t hi[8];
        crt_garner8(&k, y, L, i, even, odd, hi);

        size_t m = i1 - i < 8 ? i1 - i : 8;
        for (size_t j = 0; j < m; j++) {
            u128 acc = crt_value(even, odd, hi, j, P12) + c;
            buf |= (u128)((uint64_t)acc & mask) << nbits;
            c = (uint64_t)(acc >> b);
            nbits += b;
            if (nbits >= 64) {
                *w++ = (uint64_t)buf;
                buf >>= 64;
                nbits -= 64;
            }
        }
    }
    if (!last) return c;
    buf |= (u128)c << nbits; /* c < 2^62, nbits < 64 */
    for (; w < out + end; buf >>= 64) *w++ = (uint64_t)buf;
    return 0;
}

#ifndef CRT_CHUNKS
/* chunks converted in parallel (16 and 32 measured: no gain) */
#define CRT_CHUNKS 8
#endif

/** One chunk for the thread pool: ntt_crt_range's arguments and its carry out. */
typedef struct {
    uint64_t *out;
    size_t end;
    uint32_t *const *y;
    size_t L;
    const uint32_t *sc;
    unsigned b;
    size_t i0, i1;
    int last;
    uint64_t carry;
} CrtJob;

/** Pool task: converts one chunk and stores its carry out. */
static void crt_proc(void *arg) {
    CrtJob *j = (CrtJob *)arg;
    j->carry = ntt_crt_range(j->out, j->end, j->y, j->L, j->sc, j->b, j->i0, j->i1, j->last);
}

/** Converts the whole convolution (conv coefficients of b bits) into out,
    which has `end` limbs. Chunks with boundaries at multiples of 64
    coefficients (= limb boundaries) are converted in parallel; afterwards
    each chunk's carry out is added where the next chunk starts. */
static void ntt_crt(uint64_t *out, size_t end, uint32_t *const y[3], size_t L, const uint32_t sc[3], unsigned b,
                    size_t conv) {
    int chunks = conv >= NTT_PAR_MIN_LEN ? CRT_CHUNKS : 1;
    CrtJob jobs[CRT_CHUNKS];
    for (int k = 0; k < chunks; k++) {
        size_t i0 = conv * k / chunks / 64 * 64, i1 = k + 1 == chunks ? conv : conv * (k + 1) / chunks / 64 * 64;
        jobs[k] = (CrtJob){out, end, y, L, sc, b, i0, i1, k + 1 == chunks, 0};
    }
    pool_run(crt_proc, jobs, chunks, sizeof jobs[0]);
    for (int k = 0; k + 1 < chunks; k++) limbs_add_u64(out, end, jobs[k].i1 * b / 64, jobs[k].carry);
}
/* ======== src/mul_ntt.c ======== */
/**
 * @file   mul_ntt.c
 * @brief  big_mul_ntt: one multiplication by number-theoretic transform.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * Cuts both operands into coefficients of bw bits (32..35), picks the
 * shortest supported transform length (2^k or 3 * 2^k) for their product,
 * computes the cyclic convolution modulo each of the three primes, and
 * combines the three results with the CRT (crt.c). Products longer than the
 * NTT tables allow go to the split multiply (mul_split.c).
 */


/** One prime's convolution for the thread pool. */
typedef struct {
    const Src *a, *b; /* the operands as coefficients; b == a squares */
    size_t L;         /* transform length */
    int prime, nt;    /* which prime (index into g_ntt), threads to use */
    uint32_t *out;    /* out: the residue array, NTT_PAD words inside its buffer */
} NttJob;

/** Pool task: the convolution a*b modulo one prime, as a residue array with
    NTT_PAD spare words on each side (ntt_result frees it). */
static void ntt_proc(void *arg) {
    NttJob *job = (NttJob *)arg;
    const NttPrime *Pr = &g_ntt[job->prime];
    size_t L = job->L;

    uint32_t *fa = (uint32_t *)xmalloc((L + 2 * NTT_PAD) * sizeof(uint32_t)) + NTT_PAD;
    // a square needs one buffer and one forward transform
    if (job->a == job->b) {
        ntt_conv_any(fa, fa, L, Pr, job->nt, job->a, job->a);
    } else {
        uint32_t *fb = xmalloc(L * sizeof(uint32_t));
        ntt_conv_any(fa, fb, L, Pr, job->nt, job->a, job->b);
        xfree(fb);
    }
    /* coefficient i lives at index (L - i) mod L; mirror index 0 to L */
    memset(fa - NTT_PAD, 0, NTT_PAD * sizeof(uint32_t));
    memset(fa + L, 0, NTT_PAD * sizeof(uint32_t));
    fa[L] = fa[0];
    job->out = fa;
}

static Big big_mul_ntt_split(const Big *a, const Big *b, size_t bits_a, size_t bits_b); /* mul_split.c */

/** Returns true if coefficients of w bits can be used for a*b. The CRT
    recovers each convolution sum only below p1 p2 p3 (about 2^92.6), and a
    sum has at most min(ca, cb) products, each below 2^2w. */
static int ntt_width_ok(size_t bits_a, size_t bits_b, unsigned w) {
    const u128 M = (u128)g_ntt[0].p * g_ntt[1].p * g_ntt[2].p;
    size_t ca = (bits_a + w - 1) / w, cb = (bits_b + w - 1) / w;
    return ((u128)(ca < cb ? ca : cb) << (2 * w)) < M;
}

/** Returns the product a*b as a new Big from the three primes' residue arrays
    y (as ntt_proc or ntt_split_prime left them; freed here): one CRT. conv*bw
    bits can overshoot the product's limbs by up to bw bits (zeros), hence
    the extra limb. */
static Big ntt_result(const Big *a, const Big *b, uint32_t *y[3], size_t L, const uint32_t sc[3], unsigned bw,
                      size_t conv) {
    size_t limbs = a->n + b->n + 1;
    uint64_t *d = xmalloc(limbs * sizeof(uint64_t));
    ntt_crt(d, limbs, y, L, sc, bw, conv);
    for (int i = 0; i < 3; i++) xfree(y[i] - NTT_PAD);
    Big r = {d, normalize_len(d, limbs)};
    return r;
}

/** Returns the smallest supported transform length >= conv (at least 16):
    2^lg, or 3 * 2^lg when that is shorter (*r3_out = 1). Stores lg and r3,
    which select the scale factor for the CRT. */
static size_t ntt_length(size_t conv, int *lg_out, int *r3_out) {
    size_t L = 16;
    int lg = 4, r3 = 0;
    while (L < conv) {
        L <<= 1;
        lg++;
    }
    if (L / 4 * 3 >= conv && L / 4 >= 16) { /* 3 * 2^(lg-2) is enough and smaller */
        L = L / 4 * 3;
        lg -= 2;
        r3 = 1;
    }
    *lg_out = lg;
    *r3_out = r3;
    return L;
}

/** Returns a*b as a new Big by NTT (a == b squares, with one transform less).
    Requires freeing with big_free. */
static Big big_mul_ntt(const Big *a, const Big *b) {
    if (big_is_zero(a) || big_is_zero(b)) return big_from_u64(0);
    /* Coefficient width: wider coefficients mean fewer of them, which can
       drop the transform to a smaller length. Widths are tried up to 35 bits
       while the CRT bound allows them (ntt_width_ok); the shortest transform
       wins, the narrowest width among equals. */
    size_t bits_a = big_bits(a), bits_b = big_bits(b);
    unsigned bw = 32;
    int lg = 0, r3 = 0;
    size_t L = 0, conv = 0;
    for (unsigned w = 32; w <= 35; w++) {
        if (!ntt_width_ok(bits_a, bits_b, w)) break;
        size_t ca = (bits_a + w - 1) / w, cb = (bits_b + w - 1) / w;
        int lg_w, r3_w;
        size_t L_w = ntt_length(ca + cb - 1, &lg_w, &r3_w);
        if (L == 0 || L_w < L) {
            L = L_w, lg = lg_w, r3 = r3_w, bw = w, conv = ca + cb - 1;
        }
    }
    /* L == 0: not even 32-bit coefficients fit the CRT bound (operands of
       ~10^10 bits); the split multiply can go narrower */
    if (L == 0 || L > g_ntt_max_L) return big_mul_ntt_split(a, b, bits_a, bits_b);
    // per prime: R^2/L, which undoes Montgomery's R^-1 and the inverse's factor L
    uint32_t sc[3];
    for (int i = 0; i < 3; i++) sc[i] = r3 ? g_ntt[i].scale3[lg] : g_ntt[i].scale[lg];

    // the operands as bw-bit coefficients, read straight from their limbs
    Src sa = {a->d, a->n, (bits_a + bw - 1) / bw, bw}, sb = {b->d, b->n, (bits_b + bw - 1) / bw, bw};
    const Src *pb = a == b ? &sa : &sb;

    /* Small products run the three primes side by side. Big ones run them one
       after another, each on all threads, so one prime's buffers can stay in
       L3 instead of three primes' worth streaming through DRAM. */
    int seq = L >= NTT_SEQ_MIN_LEN;
    NttJob jobs[3];
    int nt = ntt_threads(seq);
    for (int i = 0; i < 3; i++) jobs[i] = (NttJob){&sa, pb, L, i, nt, NULL};
    if (seq) {
        for (int i = 0; i < 3; i++) ntt_proc(&jobs[i]);
    } else {
        pool_run(ntt_proc, jobs, 3, sizeof jobs[0]);
    }

    uint32_t *y[3] = {jobs[0].out, jobs[1].out, jobs[2].out};
    return ntt_result(a, b, y, L, sc, bw, conv);
}
/* ======== src/mul_split.c ======== */
/**
 * @file   mul_split.c
 * @brief  big_mul_ntt_split: products longer than the longest transform.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * The primes allow transforms up to 2^NTT_MAX_LG. Past that, each operand is
 * cut into pieces of H = Lp/2 coefficients (Lp the piece transform length),
 * and every piece is transformed once. Output piece m is the sum over
 * i + j = m of the pointwise products A_i B_j, taken back with one inverse
 * transform; the pieces overlap by H and are added mod p into one residue
 * array laid out like a single transform's output (coefficient g at index
 * Lbig - g), which the ordinary CRT then reads. The sums are the full
 * product's coefficients, so the CRT bound is the same as for one big
 * transform. Costs about 4 transforms per piece, against ~8 for Toom-3 split
 * twice.
 */


#ifndef NTT_SPLIT_LG
/* upper bound for the piece transform length 2^lg (the cost model in
   big_mul_ntt_split picks from here down to 4 steps below) */
#define NTT_SPLIT_LG NTT_MAX_LG
#endif

/** One part of a par_range for the thread pool: fn(ctx, i0, i1). */
typedef struct {
    void (*fn)(void *ctx, size_t i0, size_t i1);
    void *ctx;
    size_t i0, i1;
} RangeJob;

/** Pool task: runs one part. */
static void range_proc(void *arg) {
    RangeJob *r = (RangeJob *)arg;
    r->fn(r->ctx, r->i0, r->i1);
}

/** Runs fn(ctx, i0, i1) over [0, n) in nt parts on nt threads, cut at
    multiples of 64 so every part keeps the 8-lane loops aligned. */
static void par_range(size_t n, int nt, void (*fn)(void *, size_t, size_t), void *ctx) {
    RangeJob jobs[NTT_THREADS_SEQ]; /* nt is NTT_THREADS_SEQ at most */
    for (int i = 0; i < nt; i++)
        jobs[i] = (RangeJob){fn, ctx, n * i / nt / 64 * 64, i + 1 == nt ? n : n * (i + 1) / nt / 64 * 64};
    pool_run(range_proc, jobs, nt, sizeof jobs[0]);
}

/** Arguments of split_acc_range: output piece m's pointwise sum. */
typedef struct {
    uint32_t *C;           /* out: Lp words */
    const uint32_t *A, *B; /* ka, kb transformed pieces of Lp; B == A squares */
    size_t Lp;             /* piece transform length */
    int m, ka, kb;         /* output piece index, piece counts */
    const NttPrime *Pr;
} SplitAcc;

/** C[x0..x1) = sum over i + j = m of A_i * B_j, pointwise mod p (Montgomery).
    Only pieces that exist take part: max(0, m - kb + 1) <= i <= min(m, ka - 1). */
static void split_acc_range(void *arg, size_t x0, size_t x1) {
    const SplitAcc *s = (const SplitAcc *)arg;
    const __m256i P = _mm256_set1_epi32((int)s->Pr->p), PINV = _mm256_set1_epi32((int)s->Pr->pinv);
    const int sq = s->A == s->B;
    int i0 = s->m - s->kb + 1 > 0 ? s->m - s->kb + 1 : 0, i1 = s->m < s->ka - 1 ? s->m : s->ka - 1;
    if (sq && i1 > s->m / 2) i1 = s->m / 2; /* squares: pairs i <= j, the others by symmetry */
    for (size_t x = x0; x < x1; x += 8) {
        __m256i acc = _mm256_setzero_si256();
        for (int i = i0; i <= i1; i++) {
            int j = s->m - i;
            __m256i t = vmont(LD(s->A + i * s->Lp + x), LD(s->B + j * s->Lp + x), P, PINV);
            acc = vaddm(acc, t, P);
            if (sq && i < j) acc = vaddm(acc, t, P);
        }
        ST(s->C + x, acc);
    }
}

/** Arguments of split_add_range: add one inverse-transformed piece into Y. */
typedef struct {
    uint32_t *dst;     /* where the piece lands in Y */
    const uint32_t *C; /* the piece, Lp words */
    uint32_t p;
} SplitAdd;

/** dst[u] += C[u] mod p over [u0, u1). For u = 0 this is corrected
    afterwards (C[0] belongs at dst[Lp]). */
static void split_add_range(void *arg, size_t u0, size_t u1) {
    const SplitAdd *s = (const SplitAdd *)arg;
    const __m256i P = _mm256_set1_epi32((int)s->p);
    for (size_t u = u0; u < u1; u += 8) ST(s->dst + u, vaddm(LD(s->dst + u), LD(s->C + u), P));
}

/** Returns one prime's residues of the whole product: an array of Lbig words
    (plus NTT_PAD on each side) that reads like ntt_proc's output, for
    ntt_result to free. pb == pa squares. Uses nt threads throughout. */
static uint32_t *ntt_split_prime(const NttPrime *Pr, const Src *pa, int ka, const Src *pb, int kb, size_t Lp,
                                 size_t Lbig, int nt) {
    const size_t H = Lp / 2;
    const uint32_t p = Pr->p;
    // every piece transformed once, kept for all the pairs it is part of
    uint32_t *A = xmalloc(ka * Lp * sizeof(uint32_t));
    for (int i = 0; i < ka; i++) ntt_fwd(A + i * Lp, Lp, Pr, nt, &pa[i]);
    uint32_t *B = A;
    if (pb != pa) {
        B = xmalloc(kb * Lp * sizeof(uint32_t));
        for (int j = 0; j < kb; j++) ntt_fwd(B + j * Lp, Lp, Pr, nt, &pb[j]);
    }
    uint32_t *Y = (uint32_t *)xcalloc(Lbig + 2 * NTT_PAD, sizeof(uint32_t)) + NTT_PAD;
    uint32_t *C = xmalloc(Lp * sizeof(uint32_t));
    // one output piece at a time: pointwise sum, one inverse, add into Y
    for (int m = 0; m < ka + kb - 1; m++) {
        SplitAcc acc = {C, A, B, Lp, m, ka, kb, Pr};
        par_range(Lp, nt, split_acc_range, &acc);
        ntt_inv(C, Lp, Pr, nt); /* coefficient t of piece m at C[(Lp - t) mod Lp] */
        /* coefficient m*H + t goes to Y[Lbig - m*H - t] = dst[Lp - t] */
        uint32_t *dst = Y + Lbig - m * H - Lp;
        SplitAdd add = {dst, C, p};
        par_range(Lp, nt, split_add_range, &add);
        // C[0] (t = 0) was added at dst[0]; move it to dst[Lp]
        dst[0] = (uint32_t)(((uint64_t)dst[0] + p - C[0]) % p);
        dst[Lp] = (uint32_t)(((uint64_t)dst[Lp] + C[0]) % p);
    }
    xfree(C);
    if (B != A) xfree(B);
    xfree(A);
    return Y;
}

/** Returns the pieces of a as an xmalloc'ed array of *count Srcs: piece i is
    the coefficients [i*H, (i+1)*H) of bw bits, read in place from a's limbs. */
static Src *split_src(const Big *a, size_t bits, unsigned bw, size_t H, int *count) {
    size_t nc = (bits + bw - 1) / bw;
    int k = (int)((nc + H - 1) / H);
    Src *s = xmalloc(k * sizeof(Src));
    for (int i = 0; i < k; i++) {
        size_t limb0 = i * H * bw / 64; /* H is a multiple of 64: exact */
        size_t left = nc - i * H;
        s[i] = (Src){a->d + limb0, a->n - limb0, left < H ? left : H, bw};
    }
    *count = k;
    return s;
}

/** Returns a*b as a new Big, for products past the NTT tables' longest
    transform (big_mul_ntt calls it). bits_a, bits_b: the operands' bit
    lengths. a == b squares. Requires freeing with big_free. */
static Big big_mul_ntt_split(const Big *a, const Big *b, size_t bits_a, size_t bits_b) {
    /* widest width the CRT bound allows: fewest coefficients, fewest pieces */
    unsigned bw = 35;
    while (bw > 8 && !ntt_width_ok(bits_a, bits_b, bw)) bw--;

    /* Piece length: 2ka + 2kb - 1 transforms of Lp (ka + kb fewer for a
       square) against ka*kb pointwise products (about half for a square)
       that stream from RAM at ~5x a transform level's cost per element.
       Pieces are whole, so the best length depends on how well the operands
       fill them: at F(2*10^9) 2^23 beats 2^25 by 17% (operands of 1.2 pieces
       of 2^25), at F(10^10) 2^25 beats 2^23 by 24%. */
    int lgmax = __builtin_ctzll(g_ntt_max_L < ((size_t)1 << NTT_SPLIT_LG) ? g_ntt_max_L : (size_t)1 << NTT_SPLIT_LG);
    int lgp = lgmax;
    double best = 0;
    for (int lg = lgmax; lg >= lgmax - 4 && lg >= 10; lg--) {
        size_t h = (size_t)1 << (lg - 1);
        size_t ka_lg = ((bits_a + bw - 1) / bw + h - 1) / h, kb_lg = ((bits_b + bw - 1) / bw + h - 1) / h;
        double ca = (double)ka_lg, cb = (double)kb_lg;
        double tr = a == b ? 3 * ca - 1 : 2 * (ca + cb) - 1, pairs = a == b ? ca * (ca + 1) / 2 : ca * cb;
        double cost = (tr * lg + 5 * pairs) * (double)((size_t)1 << lg);
        if (lg == lgmax || cost < best) best = cost, lgp = lg;
    }

    // pieces of H coefficients: a piece product has up to 2H - 1, so fits Lp
    const size_t Lp = (size_t)1 << lgp, H = Lp / 2;
    int ka, kb;
    Src *pa = split_src(a, bits_a, bw, H, &ka), *pb = a == b ? pa : split_src(b, bits_b, bw, H, &kb);
    if (a == b) kb = ka;
    size_t conv = (bits_a + bw - 1) / bw + (bits_b + bw - 1) / bw - 1, Lbig = (size_t)(ka + kb) * H;

    uint32_t *y[3], sc[3];
    for (int i = 0; i < 3; i++) {
        y[i] = ntt_split_prime(&g_ntt[i], pa, ka, pb, kb, Lp, Lbig, ntt_threads(1));
        sc[i] = g_ntt[i].scale[lgp];
    }
    if (pb != pa) xfree(pb);
    xfree(pa);
    return ntt_result(a, b, y, Lbig, sc, bw, conv);
}
/* ======== src/decimal.c ======== */
/**
 * @file   decimal.c
 * @brief  big_to_string: binary to decimal digits.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * Divide and conquer with P_i = 10^(19 * 2^i): P_0 = 10^19 is the largest
 * power of ten in a limb, each next one is the square of the previous. A
 * number below P_i^2 is split as q * P_i + r, and q and r each become
 * 19 * 2^i digits of the answer, written straight into their place in one
 * buffer. The halves are independent and run in parallel once small. Each
 * division is Barrett reduction with a reciprocal of P_i computed once by
 * Newton's method, so it costs two multiplications (the NTT at large sizes).
 * Pieces below P_DEC_LEAF_LG^2 are converted by repeated division by 10^19.
 * Near-linear instead of the quadratic digit-by-digit method: F(10^8) takes
 * well under a second instead of hours.
 */


#ifndef DEC_LEAF_LG
/* leaves: numbers below P_4^2 = 10^608 (<= 32 limbs) go to dec_leaf */
#define DEC_LEAF_LG 4
#endif
#ifndef DEC_PAR_MAX_LIMBS
/* halves of numbers below this size run as parallel tasks; bigger ones run
   one after the other (their multiplications are parallel already) */
#define DEC_PAR_MAX_LIMBS 65536
#endif
#define DEC_P0 10000000000000000000ull /* 10^19 */

/** Returns floor(2^(2s) / P) minus at most 4, never more, for P of exactly
    s bits (enough for Barrett, and cheaper than exact). Newton from the
    reciprocal of P's top half: one step y + y (2^2s - P y) / 2^2s squares
    the error; taking 2 off keeps the result at or below the true value. */
static Big big_recip(const Big *P, size_t s) {
    // small enough for one 128-bit division
    if (s <= 62) {
        u128 num = (u128)1 << (2 * s);
        return big_from_u64((uint64_t)(num / P->d[0]));
    }

    // y0 from the top h bits (recursion: half the size each time)
    size_t h = (s + 1) / 2 + 16;
    Big Ph = big_shr_bits(P, s - h);
    Big mh = big_recip(&Ph, h);
    big_free(&Ph);
    Big y = big_shl_bits(&mh, s - h);
    big_free(&mh);

    // one Newton step: y += y * (2^2s - P y) / 2^2s, the error term signed
    Big Z = big_pow2(2 * s);
    Big d = big_mul(P, &y);
    int below = big_cmp(&d, &Z) <= 0;
    Big e = below ? big_sub(&Z, &d) : big_sub(&d, &Z);
    big_free(&d);
    Big ye = big_mul(&y, &e);
    big_free(&e);
    Big corr = big_shr_bits(&ye, 2 * s);
    big_free(&ye);
    if (below) big_add_inplace(&y, &corr);
    else big_sub_inplace(&y, &corr);
    big_free(&corr);
    big_free(&Z);

    big_sub_u64_inplace(&y, 2); /* y >= 2^(s-1) here: no underflow */
    return y;
}

/** Computes q = floor(x / P) and r = x mod P for x < 2^(2s), P of s bits,
    m = big_recip(P, s). Barrett: q ~ ((x >> (s-1)) * m) >> (s+1) is at most
    2 below the true quotient, so r is corrected by a few subtractions. */
static void big_divmod_barrett(const Big *x, const Big *P, const Big *m, size_t s, Big *q, Big *r) {
    Big q1 = big_shr_bits(x, s - 1);
    Big t = big_mul(&q1, m);
    big_free(&q1);
    *q = big_shr_bits(&t, s + 1);
    big_free(&t);
    Big qp = big_mul(q, P);
    *r = big_sub(x, &qp);
    big_free(&qp);
    while (big_cmp(r, P) >= 0) {
        big_sub_inplace(r, P);
        big_add_u64_inplace(q, 1);
    }
}

/** Computes q and r of x / P for any x. The quotient's t bits depend only on
    the top ~t + 64 bits of x and P, so those are divided exactly (with a
    reciprocal of that size) and the result corrected by a unit or two. Used
    once, at the top, where the quotient is much shorter than P. */
static void big_divmod_any(const Big *x, const Big *P, size_t s, Big *q, Big *r) {
    size_t bx = big_is_zero(x) ? 0 : big_bits(x);
    if (bx < s) {
        *q = big_from_u64(0);
        *r = big_copy(x);
        return;
    }

    // divide the top k bits of both
    size_t t = bx - s + 1, k = t + 64 < s ? t + 64 : s;
    Big Pk = big_shr_bits(P, s - k), Xk = big_shr_bits(x, s - k);
    Big mk = big_recip(&Pk, k), rk;
    big_divmod_barrett(&Xk, &Pk, &mk, k, q, &rk);
    big_free(&Pk);
    big_free(&Xk);
    big_free(&mk);
    big_free(&rk);

    // q may be a unit too high or too low for the full x: correct both ways
    Big qp = big_mul(q, P);
    while (big_cmp(&qp, x) > 0) {
        big_sub_u64_inplace(q, 1);
        big_sub_inplace(&qp, P);
    }
    *r = big_sub(x, &qp);
    big_free(&qp);
    while (big_cmp(r, P) >= 0) {
        big_sub_inplace(r, P);
        big_add_u64_inplace(q, 1);
    }
}

/** Returns (hi * 2^64 + lo) / d and stores the remainder; requires hi < d.
    One divq instruction: gcc's u128 division would call a slow library routine. */
static inline uint64_t div_u128(uint64_t hi, uint64_t lo, uint64_t d, uint64_t *rem) {
    uint64_t q, r;
    __asm__("divq %4" : "=a"(q), "=d"(r) : "a"(lo), "d"(hi), "rm"(d));
    *rem = r;
    return q;
}

/** Writes exactly 19 * chunks digits of x (< 10^(19 * chunks)) to out, zero
    padded. Each pass divides x by 10^19; the remainder is the next 19 digits
    from the right. */
static void dec_leaf(const Big *x, size_t chunks, char *out) {
    uint64_t t[(((size_t)38 << DEC_LEAF_LG) * 10 / 3) / 64 + 2]; /* x < 10^(38 * 2^LEAF) */
    size_t n = x->n;
    memcpy(t, x->d, n * sizeof(uint64_t));
    for (size_t c = chunks; c-- > 0;) {
        uint64_t rem = 0;
        for (size_t i = n; i-- > 0;) t[i] = div_u128(rem, t[i], DEC_P0, &rem);
        while (n > 1 && t[n - 1] == 0) n--;
        char *o = out + 19 * c;
        for (int j = 18; j >= 0; j--, rem /= 10) o[j] = (char)('0' + rem % 10);
    }
}

/** Shared by all tasks of one conversion, indexed by level i. */
typedef struct {
    const Big *pw;      /* P_i */
    const Big *inv;     /* big_recip(P_i) */
    const size_t *bits; /* bit length of P_i */
} DecCtx;

/** One half for the thread pool: dec_emit's arguments. */
typedef struct {
    const DecCtx *ctx;
    Big x;
    int lvl;
    char *out;
} DecJob;

static void dec_emit(const DecCtx *c, Big x, int lvl, char *out);

/** Pool task: converts one half. */
static void dec_proc(void *arg) {
    DecJob *j = (DecJob *)arg;
    dec_emit(j->ctx, j->x, j->lvl, j->out);
}

/** Writes q and r (x = q * P_lvl + r) as the two halves of x's 2 * 19 * 2^lvl
    digits, in parallel or one after the other; frees both. */
static void dec_emit_halves(const DecCtx *c, Big q, Big r, int lvl, char *out, int parallel) {
    char *lo = out + ((size_t)19 << lvl);
    if (parallel) {
        DecJob j = {c, q, lvl - 1, out};
        TaskGroup g = {0};
        pool_submit(&g, dec_proc, &j);
        dec_emit(c, r, lvl - 1, lo);
        pool_wait(&g);
    } else {
        dec_emit(c, q, lvl - 1, out);
        dec_emit(c, r, lvl - 1, lo);
    }
}

/** Writes exactly 2 * 19 * 2^lvl digits of x (< P_lvl^2) to out; frees x. */
static void dec_emit(const DecCtx *c, Big x, int lvl, char *out) {
    if (lvl <= DEC_LEAF_LG) {
        dec_leaf(&x, (size_t)2 << lvl, out);
        big_free(&x);
        return;
    }
    Big q, r;
    big_divmod_barrett(&x, &c->pw[lvl], &c->inv[lvl], c->bits[lvl], &q, &r);
    size_t n = x.n;
    big_free(&x);
    dec_emit_halves(c, q, r, lvl, out, n < DEC_PAR_MAX_LIMBS);
}

/** One reciprocal for the thread pool: computes inv[lvl] = big_recip(pw[lvl]). */
typedef struct {
    Big *inv;
    const Big *pw;
    const size_t *bits;
    int lvl;
} RecipJob;

/** Pool task: computes one reciprocal. */
static void recip_proc(void *arg) {
    RecipJob *j = (RecipJob *)arg;
    j->inv[j->lvl] = big_recip(&j->pw[j->lvl], j->bits[j->lvl]);
}

/** Returns a buffer for the result string. It is malloc'ed, not xmalloc'ed:
    the caller frees it with free(). */
static char *dec_string_alloc(size_t bytes) {
    char *s = malloc(bytes);
    if (!s) out_of_memory(bytes);
    return s;
}

/** Returns the decimal digits of a as a new string (no leading zeros).
    The caller frees it with free(). */
static char *big_to_string(const Big *a) {
    // zero separately: big_bits needs a nonzero number
    if (big_is_zero(a)) {
        char *s = dec_string_alloc(2);
        s[0] = '0';
        s[1] = '\0';
        return s;
    }

    /* P_0 .. P_T with P_T^2 > a: 2 (bits(P_T) - 1) >= bits(a) is enough */
    Big pw[48], inv[48];
    size_t bits[48];
    size_t ba = big_bits(a);
    int T = 0;
    pw[0] = big_from_u64(DEC_P0);
    bits[0] = big_bits(&pw[0]);
    while (2 * (bits[T] - 1) < ba) {
        pw[T + 1] = big_sqr(&pw[T]);
        bits[T + 1] = big_bits(&pw[T + 1]);
        T++;
    }
    /* reciprocals for the levels below the top (the top split is unbalanced
       and uses big_divmod_any); the biggest on this thread, the rest as tasks */
    RecipJob rj[48];
    TaskGroup g = {0};
    for (int i = DEC_LEAF_LG + 1; i < T - 1; i++) {
        rj[i] = (RecipJob){inv, pw, bits, i};
        pool_submit(&g, recip_proc, &rj[i]);
    }
    if (T - 1 > DEC_LEAF_LG) inv[T - 1] = big_recip(&pw[T - 1], bits[T - 1]);
    pool_wait(&g);

    // all 2 * 19 * 2^T digits are written, leading zeros included
    size_t half = (size_t)19 << T, total = 2 * half;
    char *buf = dec_string_alloc(total + 1);
    DecCtx c = {pw, inv, bits};
    if (T <= DEC_LEAF_LG) {
        dec_leaf(a, (size_t)2 << T, buf);
    } else {
        Big q, r;
        big_divmod_any(a, &pw[T], bits[T], &q, &r);
        dec_emit_halves(&c, q, r, T, buf, 1);
    }
    for (int i = 0; i <= T; i++) big_free(&pw[i]);
    for (int i = DEC_LEAF_LG + 1; i < T; i++) big_free(&inv[i]);

    // drop the leading zeros (keeping at least one digit)
    size_t z = 0;
    while (z + 1 < total && buf[z] == '0') z++;
    memmove(buf, buf + z, total - z);
    buf[total - z] = '\0';
    return buf;
}
/* ======== src/fib.c ======== */
/**
 * @file   fib.c
 * @brief  fibonacci(n): F(n) by fast doubling.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * Walks the bits of n from the top, keeping the pair (F(k-1), F(k)). Each
 * bit costs two squarings (run side by side) and one fused linear pass that
 * builds the next pair; the last bit needs only F(n), which is one multiply.
 * The NTT tables are sized once, up front, for that final product.
 */


/** One chunk of the linear pass. From s1 = F(k)^2 and s2 = F(k-1)^2, limb by
    limb with three carry chains:
      A = s1 + s2                 = F(2k-1)
      B = 4 s1 - s2 + 2(-1)^k     = F(2k+1)
      C = B - A                   = F(2k)
    Only two of them are kept. Chunks run in parallel from zero carries; a
    short pass afterwards adds each chunk's carry-out (for C: cC + cB - cA)
    where the next chunk starts. */
typedef struct {
    const uint64_t *s1, *s2; /* F(k)^2 and F(k-1)^2 */
    size_t n1, n2;           /* their lengths in limbs */
    uint64_t *A, *B, *C;     /* A or B NULL: not kept */
    size_t i0, i1;           /* this chunk: limbs [i0, i1) */
    int64_t cb0;             /* in: carry into B at i0 (the +-2 for chunk 0) */
    int64_t eA, eB, eC;      /* out: carry-outs at i1 */
} StepJob;

/** Pool task: computes A, B and C over the chunk's limbs. */
static void step_proc(void *arg) {
    StepJob *j = (StepJob *)arg;
    uint64_t cA = 0, prev = j->i0 ? j->s1[j->i0 - 1] : 0;
    // B and C subtract, so their carries are signed
    int64_t cB = j->cb0, cC = 0;
    for (size_t i = j->i0; i < j->i1; i++) {
        uint64_t x = i < j->n1 ? j->s1[i] : 0, y = i < j->n2 ? j->s2[i] : 0;
        // limb i of 4*s1: shifted up 2 bits, with the top 2 bits of the limb below
        uint64_t x4 = (x << 2) | (prev >> 62);
        prev = x;
        u128 a = (u128)x + y + cA;
        uint64_t ai = (uint64_t)a;
        cA = (uint64_t)(a >> 64);
        __int128 b = (__int128)x4 - y + cB;
        uint64_t bi = (uint64_t)b;
        cB = (int64_t)(b >> 64);
        __int128 c = (__int128)bi - ai + cC;
        cC = (int64_t)(c >> 64);
        if (j->A) j->A[i] = ai;
        if (j->B) j->B[i] = bi;
        j->C[i] = (uint64_t)c;
    }
    j->eA = (int64_t)cA;
    j->eB = cB;
    j->eC = cC + cB - (int64_t)cA;
}

/** Adds a small signed v into d at limb pos (the total is known to stay >= 0).
    d NULL (a result that is not kept) is ignored. */
static void add_small_at(uint64_t *d, size_t len, size_t pos, int64_t v) {
    if (!d) return;
    if (v > 0) limbs_add_u64(d, len, pos, (uint64_t)v);
    else if (v < 0) limbs_sub_u64(d, len, pos, 0 - (uint64_t)v);
}

#ifndef STEP_PAR_MIN_LIMBS
/* linear passes this long or longer are split into chunks run in parallel */
#define STEP_PAR_MIN_LIMBS 16384
#endif
#define STEP_CHUNKS 8

/** Builds the next pair from s1 = F(k)^2 and s2 = F(k-1)^2: (F(2k), F(2k+1))
    if bit is 1, else (F(2k-1), F(2k)), as new Bigs in lo and hi. k_odd gives
    the sign of 2(-1)^k. */
static void fib_step_linear(const Big *s1, const Big *s2, int k_odd, int bit, Big *lo, Big *hi) {
    size_t len = s1->n + 1; /* s2 <= s1, and each result is < 4 s1 + 2 */
    uint64_t *A = bit ? NULL : xmalloc(len * sizeof(uint64_t));
    uint64_t *B = bit ? xmalloc(len * sizeof(uint64_t)) : NULL;
    uint64_t *C = xmalloc(len * sizeof(uint64_t));

    int chunks = len >= STEP_PAR_MIN_LIMBS ? STEP_CHUNKS : 1;
    StepJob jobs[STEP_CHUNKS];
    for (int k = 0; k < chunks; k++)
        jobs[k] = (StepJob){s1->d, s2->d, s1->n, s2->n, A, B, C, len * k / chunks, len * (k + 1) / chunks,
                            k ? 0 : (k_odd ? -2 : 2), 0, 0, 0};
    pool_run(step_proc, jobs, chunks, sizeof jobs[0]);

    // each chunk's carry-out goes in where the next chunk starts
    for (int k = 0; k + 1 < chunks; k++) {
        add_small_at(A, len, jobs[k].i1, jobs[k].eA);
        add_small_at(B, len, jobs[k].i1, jobs[k].eB);
        add_small_at(C, len, jobs[k].i1, jobs[k].eC);
    }

    Big c = {C, normalize_len(C, len)};
    if (bit) {
        *lo = c;
        *hi = (Big){B, normalize_len(B, len)};
    } else {
        *lo = (Big){A, normalize_len(A, len)};
        *hi = c;
    }
}

/** One chunk of the last step's factors, computed in one pass and chunked
    like step_proc:
      X = 2u + v,   Y = 2u - v (when kept; u >= v there, so Y >= 0) */
typedef struct {
    const uint64_t *u, *v;
    size_t nu, nv;   /* their lengths in limbs */
    uint64_t *X, *Y; /* Y NULL: not kept */
    size_t i0, i1;   /* this chunk: limbs [i0, i1) */
    int64_t eX, eY;  /* out: carry-outs at i1 */
} LastJob;

/** Pool task: computes X and Y over the chunk's limbs. */
static void last_proc(void *arg) {
    LastJob *j = (LastJob *)arg;
    uint64_t cX = 0, prev = j->i0 && j->i0 - 1 < j->nu ? j->u[j->i0 - 1] : 0;
    int64_t cY = 0;
    for (size_t i = j->i0; i < j->i1; i++) {
        uint64_t x = i < j->nu ? j->u[i] : 0, y = i < j->nv ? j->v[i] : 0;
        // limb i of 2u: shifted up 1 bit, with the top bit of the limb below
        uint64_t x2 = (x << 1) | (prev >> 63);
        prev = x;
        u128 a = (u128)x2 + y + cX;
        j->X[i] = (uint64_t)a;
        cX = (uint64_t)(a >> 64);
        if (j->Y) {
            __int128 b = (__int128)x2 - y + cY;
            j->Y[i] = (uint64_t)b;
            cY = (int64_t)(b >> 64);
        }
    }
    j->eX = (int64_t)cX;
    j->eY = cY;
}

/** Computes x = 2u + v and, if y is not NULL, y = 2u - v (requires 2u >= v),
    as new Bigs. */
static void last_factors(const Big *u, const Big *v, Big *x, Big *y) {
    size_t len = (u->n > v->n ? u->n : v->n) + 1;
    uint64_t *X = xmalloc(len * sizeof(uint64_t)), *Y = y ? xmalloc(len * sizeof(uint64_t)) : NULL;
    int chunks = len >= STEP_PAR_MIN_LIMBS ? STEP_CHUNKS : 1;
    LastJob jobs[STEP_CHUNKS];
    for (int k = 0; k < chunks; k++)
        jobs[k] = (LastJob){u->d, v->d, u->n, v->n, X, Y, len * k / chunks, len * (k + 1) / chunks, 0, 0};
    pool_run(last_proc, jobs, chunks, sizeof jobs[0]);
    for (int k = 0; k + 1 < chunks; k++) {
        add_small_at(X, len, jobs[k].i1, jobs[k].eX);
        add_small_at(Y, len, jobs[k].i1, jobs[k].eY);
    }
    *x = (Big){X, normalize_len(X, len)};
    if (y) *y = (Big){Y, normalize_len(Y, len)};
}

#ifndef SQR_PAR_MIN_LIMBS
/* from this size on, the two squarings of each step run on separate threads */
#define SQR_PAR_MIN_LIMBS 512
#endif

/** One squaring for the thread pool. */
typedef struct {
    const Big *x;
    Big result;
} SqrJob;

/** Pool task: result = x^2. */
static void sqr_proc(void *arg) {
    SqrJob *job = (SqrJob *)arg;
    job->result = big_sqr(job->x);
}

/** Computes xx = x^2 and yy = y^2 (y <= x), side by side from SQR_PAR_MIN_LIMBS
    on. Squares past one transform (the split multiply) run one after the
    other: each already uses every thread, and side by side they would double
    the peak memory, gigabytes at those sizes. */
static void sqr_pair(const Big *x, const Big *y, Big *xx, Big *yy) {
    // x^2 needs a longer transform than the tables' maximum: the split multiply
    int huge = 4 * x->n - 1 > g_ntt_max_L && g_ntt_max_L >= ((size_t)1 << NTT_MAX_LG);
    if (x->n >= SQR_PAR_MIN_LIMBS && !huge) {
        SqrJob job = {y, {NULL, 0}};
        TaskGroup g = {0};
        pool_submit(&g, sqr_proc, &job);
        *xx = big_sqr(x);
        pool_wait(&g);
        *yy = job.result;
    } else {
        *xx = big_sqr(x);
        *yy = big_sqr(y);
    }
}

#define LOG2_PHI 0.6942419136306174 /* log2 of the golden ratio: F(n) has ~n * LOG2_PHI bits */

/** Returns F(n) as a new Big. Requires freeing with big_free.
    Fast doubling on the pair (F(k-1), F(k)), as GMP does: 2 squarings per bit
    instead of 2 squarings + 1 multiply.
      F(2k-1) = F(k)^2 + F(k-1)^2
      F(2k+1) = 4F(k)^2 - F(k-1)^2 + 2(-1)^k
      F(2k)   = F(2k+1) - F(2k-1)
    The last bit needs only F(n), which is one multiply:
      F(2k)   = F(k) * (F(k) + 2F(k-1))
      F(2k+1) = (2F(k) + F(k-1)) * (2F(k) - F(k-1)) + 2(-1)^k */
static Big fibonacci(uint64_t n) {
    if (n == 0) return big_from_u64(0);

    /* F(n) has about n*log2(phi) bits; size the NTT tables for the final product */
    ntt_init((size_t)((double)n * LOG2_PHI / 64.0) + 8);

    Big fkm1 = big_from_u64(0), fk = big_from_u64(1); /* k = 1 */
    int k_odd = 1;
    int top = 63 - __builtin_clzll(n);

    // one doubling step per bit of n below the top one, except the last bit (i = 0)
    for (int i = top - 1; i >= 1; i--) {
        Big s1, s2;
        sqr_pair(&fk, &fkm1, &s1, &s2);
        big_free(&fkm1);
        big_free(&fk);
        int bit = (int)((n >> i) & 1);
        fib_step_linear(&s1, &s2, k_odd, bit, &fkm1, &fk);
        big_free(&s1);
        big_free(&s2);
        k_odd = bit;
    }

    if (top == 0) { /* n == 1 */
        big_free(&fkm1);
        return fk;
    }

    /* the factors are built first and everything else freed before the
       multiply: at the largest sizes that is gigabytes of peak memory */
    Big result;
    if (n & 1) {
        Big x, y;
        last_factors(&fk, &fkm1, &x, &y); /* 2F(k) + F(k-1), 2F(k) - F(k-1) */
        big_free(&fk);
        big_free(&fkm1);
        result = big_mul(&x, &y);
        big_free(&x);
        big_free(&y);
        if (k_odd) big_sub_u64_inplace(&result, 2); /* + 2(-1)^k, in place */
        else big_add_u64_inplace(&result, 2);
    } else {
        Big x;
        last_factors(&fkm1, &fk, &x, NULL); /* F(k) + 2F(k-1) */
        big_free(&fkm1);
        result = big_mul(&fk, &x);
        big_free(&x);
        big_free(&fk);
    }
    return result;
}
#ifndef UNIT_TEST_NO_MAIN
/* ======== src/main.c ======== */
/**
 * @file   main.c
 * @brief  Command line: fastfib [n] [--print] [--help].
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * Computes F(n) and prints the time it took; with --print also the decimal
 * digits and the conversion time. n comes from the command line or, if
 * missing, from a prompt. Bad input is rejected before any work starts.
 */


/** Exits with a message if the CPU has no AVX2, instead of crashing on the
    first AVX2 instruction. Runs before main (which may already use AVX) and
    is itself compiled without AVX. There is no fallback path on purpose. */
__attribute__((constructor, target("no-avx"))) static void require_avx2(void) {
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2")) {
        fputs("fastfib: this CPU does not support AVX2, which fastfib requires\n", stderr);
        exit(1);
    }
}

/** Prints the usage text to f (stdout for --help, stderr after an error). */
static void usage(FILE *f) {
    fprintf(f, "usage: fastfib [n] [--print]\n"
               "  n        which Fibonacci number (asked for if missing)\n"
               "  --print  also print its decimal digits\n");
}

/** Parses s as a decimal n into *n. Returns 1 if valid, 0 otherwise: digits
    only (no sign, no spaces, not empty), and it must fit in 64 bits. Stricter
    than strtoull, which reads "-5" as 2^64 - 5 and "1e9" as 1. */
static int parse_n(const char *s, uint64_t *n) {
    uint64_t v = 0;
    if (!*s) return 0;
    for (; *s; s++) {
        if (*s < '0' || *s > '9') return 0;
        unsigned d = (unsigned)(*s - '0');
        // v * 10 + d would pass 2^64 - 1
        if (v > (UINT64_MAX - d) / 10) return 0;
        v = v * 10 + d;
    }
    *n = v;
    return 1;
}

/** Returns 0 on success, 1 on bad input. */
int main(int argc, char **argv) {
    uint64_t n = 0;
    int have_n = 0, print_result = 0;

    // options in any order; one number; anything else is an error
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--print") == 0) {
            print_result = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout);
            return 0;
        } else if (!have_n && parse_n(argv[i], &n)) {
            have_n = 1;
        } else {
            fprintf(stderr, "invalid argument: %s\n", argv[i]);
            usage(stderr);
            return 1;
        }
    }
    // no n on the command line: ask for it, trimming spaces around the number
    if (!have_n) {
        char line[64];
        printf("Enter n: ");
        fflush(stdout);
        if (!fgets(line, sizeof line, stdin)) return 1;
        char *s = line;
        if (strncmp(s, "\xEF\xBB\xBF", 3) == 0) s += 3; /* UTF-8 byte order mark (piped input) */
        s += strspn(s, " \t");
        s[strcspn(s, " \t\r\n")] = '\0';
        if (!parse_n(s, &n)) {
            fprintf(stderr, "invalid n: %s\n", s);
            return 1;
        }
    }

    // the time covers F(n) only; the decimal conversion is timed separately
    double start = plat_seconds();
    Big result = fibonacci(n);
    double elapsed = plat_seconds() - start;

    double conv = 0;
    if (print_result) {
        start = plat_seconds();
        char *s = big_to_string(&result);
        conv = plat_seconds() - start;
        printf("fibonacci(%llu) = %s\n", (unsigned long long)n, s);
        free(s);
    }
    printf("Time: %.6f seconds\n", elapsed);
    if (print_result) printf("Decimal conversion: %.6f seconds\n", conv);

    big_free(&result);
    return 0;
}
#endif
