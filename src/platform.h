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

#pragma once

#if defined(_WIN32)
#include "platform_win.h"
#else
#error "no platform layer for this OS yet: write src/platform_<os>.h implementing the interface in src/platform.h"
#endif
