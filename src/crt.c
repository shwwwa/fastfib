/**
 * @file   crt.c
 * @brief  Chinese remainder theorem: the three primes' results into one bignum.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * The NTT gives every coefficient of the product modulo three primes.
 * Garner's method combines the three residues into the exact coefficient
 * X < p1*p2*p3 (~2^93), 8 coefficients at a time in AVX2. Coefficient i is
 * then added at bit i*b with one running carry. Big results are split into
 * CRT_CHUNKS chunks converted in parallel; their carries are added afterwards.
 */

#pragma once
#include "ntt_conv.c"

/* Transform buffers carry NTT_PAD spare words on each side so crt_garner8 can
   read the index-reversed result 8 lanes at a time without bounds checks: it
   reads up to 7 words before the start and the word at index L. */
#define NTT_PAD 8
_Static_assert(NTT_PAD >= 7, "crt_garner8 reads up to 7 words before a residue array");

/** Constants for crt_garner8, broadcast to all 8 lanes: the primes, the
    scale factors SC (see ntt_crt_range) and Garner's inverses INVij = pi^-1
    mod pj. The p-suffixed copies are premultiplied for vmontw. REV reverses
    the 8 lanes; LO32 keeps the low 32 bits of each 64-bit lane. */
typedef struct {
    __m256i P1, P2, P3, SC1, SC2, SC3, SC1p, SC2p, SC3p, INV12, INV13, INV23, INV12p, INV13p, INV23p, REV, LO32;
} CrtK;

/** Garner's method on coefficients i .. i+7. With residues r1, r2, r3:
      x2 = (r2 - x1) / p1 mod p2,  x3 = ((r3 - x1) / p1 - x2) / p2 mod p3,
      X  = x1 + x2*p1 + x3*p1*p2   (x1 = r1).
    Stores x1 + x2*p1 as 64-bit values split by parity (even[]: coefficients
    i, i+2, ..; odd[]: i+1, i+3, ..; that's how _mm256_mul_epu32 pairs lanes)
    and x3 in hi[]. Division means multiplying by the inverse INVij. */
static inline __attribute__((always_inline)) void crt_garner8(const CrtK *k, uint32_t *const y[3], size_t L, size_t i,
                                                              uint64_t *even, uint64_t *odd, uint32_t *hi) {
    ptrdiff_t pos = (ptrdiff_t)L - (ptrdiff_t)i - 7; /* may dip into the front pad */
    __m256i r1 = vmontw(_mm256_permutevar8x32_epi32(LDU(y[0] + pos), k->REV), k->SC1, k->SC1p, k->P1);
    __m256i r2 = vmontw(_mm256_permutevar8x32_epi32(LDU(y[1] + pos), k->REV), k->SC2, k->SC2p, k->P2);
    __m256i r3 = vmontw(_mm256_permutevar8x32_epi32(LDU(y[2] + pos), k->REV), k->SC3, k->SC3p, k->P3);

    __m256i x1m2 = vred(r1, k->P2); /* p1 < 2*p2 and p1 < 2*p3 */
    __m256i x2 = vmontw(_mm256_sub_epi32(r2, x1m2), k->INV12, k->INV12p, k->P2);
    __m256i x1m3 = vred(r1, k->P3);
    __m256i t3 = vmontw(_mm256_sub_epi32(r3, x1m3), k->INV13, k->INV13p, k->P3);
    __m256i x2m3 = vred(x2, k->P3);
    __m256i x3 = vmontw(_mm256_sub_epi32(t3, x2m3), k->INV23, k->INV23p, k->P3);

    /* x1 + x2*p1 < 2^62, as 64-bit lanes for even and odd coefficients */
    __m256i le = _mm256_add_epi64(_mm256_mul_epu32(x2, k->P1), _mm256_and_si256(r1, k->LO32));
    __m256i lo = _mm256_add_epi64(_mm256_mul_epu32(vodd(x2), k->P1), _mm256_srli_epi64(r1, 32));
    _mm256_storeu_si256((__m256i *)even, le);
    _mm256_storeu_si256((__m256i *)odd, lo);
    _mm256_storeu_si256((__m256i *)hi, x3);
}

/** Returns coefficient k of a crt_garner8 group: X = x3*p1*p2 + (x1 + x2*p1) (< 2^93). */
static inline __attribute__((always_inline)) u128 crt_value(const uint64_t *even, const uint64_t *odd,
                                                            const uint32_t *hi, size_t k, uint64_t P12) {
    return (u128)hi[k] * P12 + (k & 1 ? odd[k >> 1] : even[k >> 1]);
}

/** Packs 64 coefficients (plus the carry c) into b limbs at w and returns the
    carry into the next block. 64 coefficients of a compile-time width b fill
    exactly b limbs, so after unrolling every bit position is a constant:
    immediate shifts, no branches, whole limbs written once. */
static inline __attribute__((always_inline)) uint64_t crt_pack64(uint64_t *w, const uint64_t *even, const uint64_t *odd,
                                                                 const uint32_t *hi, uint64_t P12, uint64_t c,
                                                                 const unsigned b) {
    const uint64_t mask = (1ULL << b) - 1;
    uint64_t o[35];
#pragma GCC unroll 64
    for (unsigned k = 0; k < 64; k++) {
        u128 acc = crt_value(even, odd, hi, k, P12) + c;
        uint64_t v = (uint64_t)acc & mask;
        c = (uint64_t)(acc >> b);
        unsigned q = k * b / 64, off = k * b % 64;
        if (off == 0) o[q] = v;
        else o[q] |= v << off;
        if (off + b > 64) o[q + 1] = v >> (64 - off);
    }
    memcpy(w, o, b * sizeof(uint64_t));
    return c;
}

/** Converts whole blocks of 64 coefficients from i while they fit below i1,
    with b a compile-time constant (see crt_pack64). Returns where it stopped:
    fewer than 64 coefficients are left for the general loop. */
static inline __attribute__((always_inline)) size_t crt_blocks(const CrtK *k, uint32_t *const y[3], size_t L,
                                                               uint64_t P12, size_t i, size_t i1, uint64_t **w,
                                                               uint64_t *c, const unsigned b) {
    uint64_t even[32], odd[32];
    uint32_t hi[64];
    for (; i + 64 <= i1; i += 64) {
        for (unsigned g = 0; g < 8; g++) crt_garner8(k, y, L, i + 8 * g, even + 4 * g, odd + 4 * g, hi + 8 * g);
        *c = crt_pack64(*w, even, odd, hi, P12, *c, b);
        *w += b;
    }
    return i;
}

/** Converts coefficients [i0, i1) into limbs of out; returns the carry out
    (0 for the last chunk). Residues are read backwards (y[L - i]) and scaled
    by SC = R^2/L. Coefficient i adds X at bit i*b. i0 is a multiple of 64, so
    chunks start on a limb boundary; the last one flushes up to limb `end`. */
static uint64_t ntt_crt_range(uint64_t *out, size_t end, uint32_t *const y[3], size_t L, const uint32_t sc[3],
                              unsigned b, size_t i0, size_t i1, int last) {
    const uint32_t p1 = g_ntt[0].p, p2 = g_ntt[1].p, p3 = g_ntt[2].p;
    const __m256i PINV1 = _mm256_set1_epi32((int)g_ntt[0].pinv), PINV2 = _mm256_set1_epi32((int)g_ntt[1].pinv),
                  PINV3 = _mm256_set1_epi32((int)g_ntt[2].pinv);
    const CrtK k = {_mm256_set1_epi32((int)p1), _mm256_set1_epi32((int)p2), _mm256_set1_epi32((int)p3),
                    _mm256_set1_epi32((int)sc[0]), _mm256_set1_epi32((int)sc[1]), _mm256_set1_epi32((int)sc[2]),
                    vpre(sc[0], PINV1), vpre(sc[1], PINV2), vpre(sc[2], PINV3),
                    _mm256_set1_epi32((int)g_inv_p1_mod_p2), _mm256_set1_epi32((int)g_inv_p1_mod_p3),
                    _mm256_set1_epi32((int)g_inv_p2_mod_p3),
                    vpre(g_inv_p1_mod_p2, PINV2), vpre(g_inv_p1_mod_p3, PINV3), vpre(g_inv_p2_mod_p3, PINV3),
                    _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0), _mm256_set1_epi64x(0xFFFFFFFF)};
    const uint64_t P12 = (uint64_t)p1 * p2, mask = (1ULL << b) - 1;
    uint64_t c = 0, *w = out + i0 * b / 64;
    size_t i = i0; /* a multiple of 64: blocks start on a limb boundary */
    switch (b) {   /* the widths big_mul_ntt picks; others take the loop below */
    case 32: i = crt_blocks(&k, y, L, P12, i, i1, &w, &c, 32); break;
    case 33: i = crt_blocks(&k, y, L, P12, i, i1, &w, &c, 33); break;
    case 34: i = crt_blocks(&k, y, L, P12, i, i1, &w, &c, 34); break;
    case 35: i = crt_blocks(&k, y, L, P12, i, i1, &w, &c, 35); break;
    }
    u128 buf = 0;      /* pending output bits */
    unsigned nbits = 0; /* how many, < 64 between coefficients */
    for (; i < i1; i += 8) {
        uint64_t even[4], odd[4];
        uint32_t hi[8];
        crt_garner8(&k, y, L, i, even, odd, hi);

        size_t m = i1 - i < 8 ? i1 - i : 8;
        for (size_t j = 0; j < m; j++) {
            u128 acc = crt_value(even, odd, hi, j, P12) + c;
            buf |= (u128)((uint64_t)acc & mask) << nbits;
            c = (uint64_t)(acc >> b);
            nbits += b;
            if (nbits >= 64) {
                *w++ = (uint64_t)buf;
                buf >>= 64;
                nbits -= 64;
            }
        }
    }
    if (!last) return c;
    buf |= (u128)c << nbits; /* c < 2^62, nbits < 64 */
    for (; w < out + end; buf >>= 64) *w++ = (uint64_t)buf;
    return 0;
}

#ifndef CRT_CHUNKS
/* chunks converted in parallel (16 and 32 measured: no gain) */
#define CRT_CHUNKS 8
#endif

/** One chunk for the thread pool: ntt_crt_range's arguments and its carry out. */
typedef struct {
    uint64_t *out;
    size_t end;
    uint32_t *const *y;
    size_t L;
    const uint32_t *sc;
    unsigned b;
    size_t i0, i1;
    int last;
    uint64_t carry;
} CrtJob;

/** Pool task: converts one chunk and stores its carry out. */
static void crt_proc(void *arg) {
    CrtJob *j = (CrtJob *)arg;
    j->carry = ntt_crt_range(j->out, j->end, j->y, j->L, j->sc, j->b, j->i0, j->i1, j->last);
}

/** Converts the whole convolution (conv coefficients of b bits) into out,
    which has `end` limbs. Chunks with boundaries at multiples of 64
    coefficients (= limb boundaries) are converted in parallel; afterwards
    each chunk's carry out is added where the next chunk starts. */
static void ntt_crt(uint64_t *out, size_t end, uint32_t *const y[3], size_t L, const uint32_t sc[3], unsigned b,
                    size_t conv) {
    int chunks = conv >= NTT_PAR_MIN_LEN ? CRT_CHUNKS : 1;
    CrtJob jobs[CRT_CHUNKS];
    for (int k = 0; k < chunks; k++) {
        size_t i0 = conv * k / chunks / 64 * 64, i1 = k + 1 == chunks ? conv : conv * (k + 1) / chunks / 64 * 64;
        jobs[k] = (CrtJob){out, end, y, L, sc, b, i0, i1, k + 1 == chunks, 0};
    }
    pool_run(crt_proc, jobs, chunks, sizeof jobs[0]);
    for (int k = 0; k + 1 < chunks; k++) limbs_add_u64(out, end, jobs[k].i1 * b / 64, jobs[k].carry);
}
