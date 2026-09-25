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

#pragma once
#include "bigint.c"

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
