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

#pragma once
#include "mul_ntt.c"

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
