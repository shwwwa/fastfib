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

#pragma once
#include "ntt_conv.c"

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
