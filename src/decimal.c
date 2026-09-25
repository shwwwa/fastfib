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

#pragma once
#include "mul_basic.c"
#include "pool.c"

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
