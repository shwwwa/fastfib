/**
 * @file   ntt_tables.c
 * @brief  ntt_init: the root-of-unity tables, scale factors and Garner constants.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * Each prime gets one table of L words holding every level's twiddles:
 * rt[h + j] = w_2h^j * R mod p for h = 1, 2, 4, .., L/2 and j < h (w_2h a
 * primitive 2h-th root of unity; rt[0] is unused). Level h is rt[h .. 2h),
 * so a transform of any length up to L reads its levels from the same table.
 * rtp holds rt * pinv mod 2^32 for the first NTT_BASE_LEN entries only, the
 * part the in-cache kernels use with vmontw.
 *
 * Every level is its own running product, 8 lanes at a time (lane i starts
 * at w^(j0+i) and steps by w^8), cut into chunks spread over the pool, so
 * building the tables for a big product takes a fraction of a millisecond.
 * The tables only grow: ntt_init rebuilds them when a bigger product needs
 * longer ones, and does nothing otherwise.
 */

#pragma once
#include "ntt_arith.c"
#include "mul_basic.c"
#include "pool.c"

/** One chunk of one level's table, for the thread pool. */
typedef struct {
    NttPrime *Pr;
    size_t h;      /* the level: entries rt[h + j] */
    size_t j0, j1; /* this chunk: j in [j0, j1) */
} TableJob;

/** Pool task: fills rt[h + j] (and rtp, inside the base size) for j in
    [j0, j1) of one level. Levels below 8 are too short for a vector and go
    one entry at a time; the others start 8 lanes at w^j0 .. w^(j0+7) and
    step them all by w^8 with one vmontw per vector. */
static void table_proc(void *arg) {
    TableJob *t = (TableJob *)arg;
    const NttPrime *Pr = t->Pr;
    const uint32_t p = Pr->p;
    uint32_t w = powmod32(Pr->g, (p - 1) / (2 * t->h), p);
    uint32_t *out = Pr->rt + t->h, cur = powmod32(w, t->j0, p);
    uint32_t *outp = 2 * t->h <= NTT_BASE_LEN ? Pr->rtp + t->h : NULL;

    if (t->h < 8) {
        for (size_t j = t->j0; j < t->j1; j++, cur = (uint32_t)((uint64_t)cur * w % p)) {
            out[j] = to_mont(cur, p);
            outp[j] = out[j] * Pr->pinv;
        }
        return;
    }

    uint32_t start[8];
    for (int i = 0; i < 8; i++, cur = (uint32_t)((uint64_t)cur * w % p)) start[i] = to_mont(cur, p);
    const __m256i P = _mm256_set1_epi32((int)p), PINV = _mm256_set1_epi32((int)Pr->pinv);
    const uint32_t w8 = to_mont(powmod32(w, 8, p), p);
    const __m256i W8 = _mm256_set1_epi32((int)w8), W8p = vpre(w8, PINV);
    __m256i v = LDU(start);
    for (size_t j = t->j0; j < t->j1; j += 8) {
        _mm256_storeu_si256((__m256i *)(out + j), v);
        if (outp) _mm256_storeu_si256((__m256i *)(outp + j), _mm256_mullo_epi32(v, PINV));
        v = vmontw(v, W8, W8p, P);
    }
}

/** Builds the tables for products of up to max_limbs 64-bit limbs (at most
    2 * max_limbs coefficients, since b >= 32), capped at 2^NTT_MAX_LG; the
    split multiply handles longer products with these. Also sets each
    prime's pinv, r2 and scale factors, and the Garner inverses. Must run
    before any thread calls big_mul_ntt; the tables are read-only after. */
static void ntt_init(size_t max_limbs) {
    size_t L = 16;
    while (L < 2 * max_limbs && L < ((size_t)1 << NTT_MAX_LG)) L <<= 1;
    if (L <= g_ntt_max_L) return;

    // jobs of at least 32768 entries, at most ~64 per prime for the big
    // levels; cap bounds the count: 2L/chunk for those, one per small level
    size_t chunk = L / 64 > 32768 ? L / 64 : 32768;
    size_t njobs = 0, cap = 3 * (2 * L / chunk + NTT_MAX_LG + 2);
    TableJob *jobs = xmalloc(cap * sizeof(TableJob));
    for (int i = 0; i < 3; i++) {
        NttPrime *Pr = &g_ntt[i];
        // pinv = p^-1 mod 2^32 by Newton: p is its own inverse mod 8 (3 bits),
        // and each step doubles the correct bits: 3 -> 6 -> 12 -> 24 -> 48
        uint32_t inv = Pr->p;
        for (int it = 0; it < 4; it++) inv *= 2 - Pr->p * inv;
        Pr->pinv = inv;
        xfree(Pr->rt);
        Pr->rt = xmalloc(L * sizeof(uint32_t));
        xfree(Pr->rtp);
        Pr->rtp = xmalloc((L < NTT_BASE_LEN ? L : NTT_BASE_LEN) * sizeof(uint32_t));
        Pr->rt[0] = Pr->rtp[0] = 0; /* unused slot */
        for (size_t h = 1; h < L; h <<= 1)
            for (size_t j = 0; j < h; j += chunk) jobs[njobs++] = (TableJob){Pr, h, j, j + chunk < h ? j + chunk : h};
    }
    // not pool_run: this thread takes jobs[0], then computes the constants
    // below while the workers still fill tables, and only then waits
    TaskGroup g = {0};
    for (size_t k = 1; k < njobs; k++) pool_submit(&g, table_proc, &jobs[k]);
    table_proc(&jobs[0]);

    // scale[lg] = R^2 / L mod p for L = 2^lg and 3 * 2^lg (Fermat inverse):
    // the CRT multiplies by it to undo the pointwise R^-1 and the inverse's L
    for (int i = 0; i < 3; i++) {
        NttPrime *Pr = &g_ntt[i];
        uint32_t p = Pr->p, R = (uint32_t)((1ULL << 32) % p);
        uint32_t R2 = (uint32_t)((uint64_t)R * R % p);
        Pr->r2 = R2;
        for (int lg = 0; lg <= NTT_MAX_LG; lg++) {
            Pr->scale[lg] = (uint32_t)((uint64_t)R2 * powmod32((uint32_t)((1ULL << lg) % p), p - 2, p) % p);
            Pr->scale3[lg] = (uint32_t)((uint64_t)R2 * powmod32((uint32_t)((3ULL << lg) % p), p - 2, p) % p);
        }
    }
    // Garner's inverses pi^-1 mod pj, times R (Montgomery form, for vmontw)
    uint32_t p1 = g_ntt[0].p, p2 = g_ntt[1].p, p3 = g_ntt[2].p;
    uint64_t R_p2 = (1ULL << 32) % p2, R_p3 = (1ULL << 32) % p3;
    g_inv_p1_mod_p2 = (uint32_t)(powmod32(p1 % p2, p2 - 2, p2) * R_p2 % p2);
    g_inv_p1_mod_p3 = (uint32_t)(powmod32(p1 % p3, p3 - 2, p3) * R_p3 % p3);
    g_inv_p2_mod_p3 = (uint32_t)(powmod32(p2 % p3, p3 - 2, p3) * R_p3 % p3);
    pool_wait(&g);
    xfree(jobs);
    g_ntt_max_L = L;
}
