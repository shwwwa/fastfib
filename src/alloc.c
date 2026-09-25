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

#pragma once
#include "common.h"
#include "platform.h"

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
