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

#pragma once
#include "ntt_radix3.c"
#include "crt.c"

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
