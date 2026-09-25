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

#pragma once
#include "common.h"
#include "platform.h"

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
