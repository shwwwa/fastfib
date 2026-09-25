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

#pragma once
#include "ntt_kernels.c"

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
