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

#pragma once
#include "mul_basic.c"
#include "ntt_tables.c"
#include "pool.c"

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
