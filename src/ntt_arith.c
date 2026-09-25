/**
 * @file   ntt_arith.c
 * @brief  The three NTT primes and modular arithmetic on 8 AVX2 lanes.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * The NTT multiply convolves the operands as arrays of b-bit coefficients
 * (b = 32..35) with number-theoretic transforms (FFTs over Z/pZ) modulo three
 * primes p = c*2^k+1 < 2^31, then combines them with the CRT (crt.c). Each
 * convolution coefficient is a sum of at most min(ca, cb) products of two
 * coefficients; b is chosen so that this stays below p1*p2*p3 (~2^92.6), the
 * largest value the CRT can recover (ntt_width_ok). O(n log n) against
 * Toom-3's O(n^1.465).
 *
 * 32-bit lanes let AVX2 do 8 butterflies at once: vpmuludq gives four
 * 32x32->64 products per instruction, and Montgomery reduction (R = 2^32)
 * needs no division. Values stay canonical in [0, p); p < 2^31 means u + v
 * fits in 32 bits and reducing [0, 2p) -> [0, p) is a single min(x, x - p).
 * The inverse transform reuses the forward roots: that yields L * x[-n mod L],
 * so the CRT simply reads the result backwards. One root table per prime.
 */

#pragma once
#include "common.h"

#ifndef NTT_BASE_LEN
#define NTT_BASE_LEN 8192       /* at or below this, a block stays in L1/L2: plain loops */
#endif
_Static_assert(NTT_BASE_LEN >= 16, "the base kernels end in a 16-element in-register step (dif_last3)");

/** One NTT prime and everything precomputed for it (tables filled by ntt_init). */
typedef struct {
    uint32_t p;           /* the prime, c * 2^k + 1 */
    uint32_t g;           /* a generator mod p: its powers give the roots of unity */
    uint32_t pinv;        /* p^-1 mod 2^32 */
    uint32_t *rt;         /* rt[h + j] = w_{2h}^j * R mod p (Montgomery form) */
    uint32_t *rtp;        /* rtp[i] = rt[i] * pinv mod 2^32 (see vmontw), i < NTT_BASE_LEN */
    uint32_t scale[NTT_MAX_LG + 1];  /* R^2 / L mod p for L = 2^lg */
    uint32_t scale3[NTT_MAX_LG + 1]; /* R^2 / L mod p for L = 3 * 2^lg */
    uint32_t r2;          /* R^2 mod p: vmont(h, r2) = h * 2^32 mod p */
} NttPrime;

/* The three primes: below 2^31, and p - 1 divisible by 3 * 2^25 (63, 15 and
   27 are multiples of 3), so every length 2^k and 3 * 2^k up to
   2^NTT_MAX_LG has its roots of unity.
   Only p and g are set here; ntt_init computes the rest. */
static NttPrime g_ntt[3] = {
    {.p = 2113929217u, .g = 5},  /* 63*2^25+1 */
    {.p = 2013265921u, .g = 31}, /* 15*2^27+1 */
    {.p = 1811939329u, .g = 13}, /* 27*2^26+1 */
};

/* Garner's inverses for the CRT (pi^-1 mod pj), in Montgomery form; set by ntt_init */
static uint32_t g_inv_p1_mod_p2, g_inv_p1_mod_p3, g_inv_p2_mod_p3;

/** Returns a^e mod p (square and multiply). Scalar, for table setup only. */
static uint32_t powmod32(uint32_t a, uint64_t e, uint32_t p) {
    uint64_t r = 1, x = a % p;
    while (e) {
        if (e & 1) r = r * x % p;
        x = x * x % p;
        e >>= 1;
    }
    return (uint32_t)r;
}

/** Returns x in Montgomery form, x * R mod p with R = 2^32. Scalar, for setup. */
static uint32_t to_mont(uint32_t x, uint32_t p) {
    return (uint32_t)(((uint64_t)x << 32) % p);
}

/* Short names for the AVX2 loads and stores: LD / ST need 32-byte aligned
   addresses, LDU does not. SHUF_PS is the float shuffle used on integer
   vectors (it can pick lanes from two vectors, the integer one cannot). */
#define LD(ptr) _mm256_load_si256((const __m256i *)(ptr))
#define LDU(ptr) _mm256_loadu_si256((const __m256i *)(ptr))
#define ST(ptr, v) _mm256_store_si256((__m256i *)(ptr), (v))
#define SHUF_PS(a, b, imm) \
    _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), (imm)))

/** Returns x reduced from [0, 2p) to [0, p): if x >= p then x - p < x, else
    x - p wraps and is larger, so the unsigned minimum is right either way. */
static inline __m256i vred(__m256i x, __m256i P) {
    return _mm256_min_epu32(x, _mm256_sub_epi32(x, P));
}

/** Returns x with its odd 32-bit lanes moved into the even slots vpmuludq reads. */
static inline __m256i vodd(__m256i x) {
    return _mm256_shuffle_epi32(x, 0xF5);
}

/** Returns r moved from (-p, p) to [0, p): a negative r is huge as unsigned
    while r + p is small; a non-negative r is smaller than r + p. */
static inline __m256i vfix(__m256i r, __m256i P) {
    return _mm256_min_epu32(r, _mm256_add_epi32(r, P));
}

/** Returns (x + y) mod p for x, y in [0, p). */
static inline __m256i vaddm(__m256i x, __m256i y, __m256i P) {
    return vred(_mm256_add_epi32(x, y), P);
}

/** Returns (x - y) mod p for x, y in [0, p). */
static inline __m256i vsubm(__m256i x, __m256i y, __m256i P) {
    return vfix(_mm256_sub_epi32(x, y), P);
}

/** Returns (T - m*p) / 2^32 in [0, p): the shared end of vmont2 and vmontw,
    from the products te, to (even and odd lanes) and their m = T * pinv
    mod 2^32 (me, mo). */
static inline __m256i vmont_reduce(__m256i te, __m256i to, __m256i me, __m256i mo, __m256i P) {
    __m256i de = _mm256_sub_epi64(te, _mm256_mul_epi32(me, P));
    __m256i dodd = _mm256_sub_epi64(to, _mm256_mul_epi32(mo, P));
    /* high halves of de into the even slots: a shuffle (port 5) rather than a
       64-bit shift, which would compete with the multiplies for ports 0/1 */
    return vfix(_mm256_blend_epi32(_mm256_shuffle_epi32(de, 0xF5), dodd, 0xAA), P);
}

/** Returns a*b*R^-1 mod p in [0, p) (signed Montgomery), for a in (-p, p) as
    int32 and b in [0, p); ao, bo hold the odd lanes of a, b in the even slots.
    |T| < p^2 and |m*p| < 2^31 p, so (T - m*p) / 2^32 lies in (-p, p). Signed
    inputs let callers pass a difference x - y without adding p first. */
static inline __m256i vmont2(__m256i a, __m256i ao, __m256i b, __m256i bo, __m256i P, __m256i PINV) {
    __m256i te = _mm256_mul_epi32(a, b), to = _mm256_mul_epi32(ao, bo);
    __m256i me = _mm256_mul_epu32(te, PINV), mo = _mm256_mul_epu32(to, PINV);
    return vmont_reduce(te, to, me, mo, P);
}

/** Returns a*b*R^-1 mod p in [0, p): vmont2 with the odd lanes taken here. */
static inline __m256i vmont(__m256i a, __m256i b, __m256i P, __m256i PINV) {
    return vmont2(a, vodd(a), b, vodd(b), P, PINV);
}

/** Returns vmont(a, w) for a table twiddle w, with wp = w * pinv mod 2^32
    precomputed: m = a * w * pinv = a * wp (mod 2^32) comes straight from a,
    in parallel with a * w, instead of after it. */
static inline __m256i vmontw(__m256i a, __m256i w, __m256i wp, __m256i P) {
    __m256i ao = vodd(a);
    __m256i te = _mm256_mul_epi32(a, w), to = _mm256_mul_epi32(ao, vodd(w));
    __m256i me = _mm256_mul_epu32(a, wp), mo = _mm256_mul_epu32(ao, vodd(wp));
    return vmont_reduce(te, to, me, mo, P);
}

/** Returns c * pinv mod 2^32 in all lanes: the wp that goes with a constant c in vmontw. */
static inline __m256i vpre(uint32_t c, __m256i PINV) {
    return _mm256_set1_epi32((int)(c * (uint32_t)_mm256_cvtsi256_si32(PINV)));
}
