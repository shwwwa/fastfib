/**
 * @file   ntt_kernels.c
 * @brief  The transform passes: butterflies, in-cache base transforms, radix-16 passes.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * A level h of the transform pairs every element with the one h words away.
 * Forward transforms are DIF (Gentleman-Sande: add/subtract, then twiddle,
 * levels L/2 down to 1); inverses are DIT (Cooley-Tukey: twiddle, then
 * add/subtract, levels 1 up to L/2), which undoes the DIF order without any
 * bit-reversal pass. Twiddles for level h are rt[h .. 2h) (ntt_tables.c).
 *
 * Blocks up to NTT_BASE_LEN are transformed in cache by ntt_forward_base /
 * ntt_inverse_base: fused pairs of levels (radix 4), then the last three
 * levels in registers. Bigger blocks get one top pass first (ntt_conv.c):
 * radix 2, 4 or 16 levels per trip through memory. The first forward pass
 * can read its input straight from the bignum (Src, src_load8), so no
 * separate load pass is needed. Past NTT_TWD_MIN_LEN the radix-16 passes
 * derive most twiddles instead of streaming them from the big tables.
 */

#pragma once
#include "ntt_tables.c"

/** A bignum read as nc coefficients of b bits, for src_load8. b is 32..35
    in practice; the split multiply goes narrower (down to 8) only for
    operands too big for 32. */
typedef struct {
    const uint64_t *d; /* the limbs */
    size_t nl, nc;     /* limbs, coefficients */
    unsigned b;        /* coefficient width in bits (<= 35) */
} Src;

/** Returns coefficients [k, k+8) of a bignum reduced to [0, p), zero past
    the end; k is a multiple of 8. b = 32 reads the halves of the limbs
    directly (each < 2^32 < 3p, so two vred calls reduce it). Other widths
    cut the fields out one by one and split them as hi * 2^32 + lo, with
    hi < 2^(b-32) <= 8 brought in as vmont(hi, R^2) = hi * 2^32 mod p. */
static inline __m256i src_load8(const Src *s, size_t k, __m256i P, __m256i PINV, __m256i R2) {
    if (k >= s->nc) return _mm256_setzero_si256();
    if (s->b == 32) {
        const uint32_t *w = (const uint32_t *)s->d;
        __m256i v;
        if (k + 8 <= s->nc) {
            v = LDU(w + k);
        } else {
            uint32_t tmp[8] = {0};
            memcpy(tmp, w + k, (s->nc - k) * sizeof(uint32_t));
            v = LDU(tmp);
        }
        return vred(vred(v, P), P);
    }

    uint32_t lo[8], hi[8];
    const unsigned b = s->b;
    const uint64_t mask = (1ULL << b) - 1;
    size_t bit = k * b;
    for (int t = 0; t < 8; t++, bit += b) {
        uint64_t v = 0;
        if (k + t < s->nc) {
            size_t wi = bit >> 6;
            unsigned sh = bit & 63;
            v = s->d[wi] >> sh;
            if (sh + b > 64 && wi + 1 < s->nl) v |= s->d[wi + 1] << (64 - sh);
            v &= mask;
        }
        lo[t] = (uint32_t)v;
        hi[t] = (uint32_t)(v >> 32);
    }

    __m256i l = vred(vred(LDU(lo), P), P);
    return vaddm(l, vmont(LDU(hi), R2, P, PINV), P);
}

/** Returns a * w[i] (Montgomery) for a table twiddle. The kernels below
    come in two builds, chosen by their `pre` flag: in-cache ones (base
    blocks) pass wp = the matching rtp entries and get the shorter vmontw
    chain; the top passes stream their twiddles from memory and pass
    wp = NULL, since a second table there costs more bandwidth than the
    multiply saves. */
static inline __attribute__((always_inline)) __m256i vtw(__m256i a, const uint32_t *w, const uint32_t *wp, size_t i,
                                                         __m256i P, __m256i PINV) {
    return wp ? vmontw(a, LD(w + i), LD(wp + i), P) : vmont(a, LD(w + i), P, PINV);
}

/** Runs DIF level h (Gentleman-Sande) on lanes [j0, j1) of a block of 2h:
    x, y = x + y, (x - y) * w_2h^j. h >= 8 (whole vectors); pre selects
    vmontw with rtp (in-cache) or plain vmont (streaming). */
static inline __attribute__((always_inline)) void dif2_body(uint32_t *a, size_t h, size_t j0, size_t j1,
                                                            const NttPrime *Pr, int pre) {
    const __m256i P = _mm256_set1_epi32((int)Pr->p), PINV = _mm256_set1_epi32((int)Pr->pinv);
    const uint32_t *w = Pr->rt + h, *wp = pre ? Pr->rtp + h : NULL;
    uint32_t *x = a, *y = a + h;
    for (size_t j = j0; j < j1; j += 8) {
        __m256i u = LD(x + j), v = LD(y + j);
        ST(x + j, vaddm(u, v, P));
        ST(y + j, vtw(_mm256_sub_epi32(u, v), w, wp, j, P, PINV));
    }
}

/** Runs DIT level h (Cooley-Tukey) on lanes [j0, j1) of a block of 2h:
    t = y * w_2h^j, then x, y = x + t, x - t. The mirror of dif2_body. */
static inline __attribute__((always_inline)) void dit2_body(uint32_t *a, size_t h, size_t j0, size_t j1,
                                                            const NttPrime *Pr, int pre) {
    const __m256i P = _mm256_set1_epi32((int)Pr->p), PINV = _mm256_set1_epi32((int)Pr->pinv);
    const uint32_t *w = Pr->rt + h, *wp = pre ? Pr->rtp + h : NULL;
    uint32_t *x = a, *y = a + h;
    for (size_t j = j0; j < j1; j += 8) {
        __m256i u = LD(x + j);
        __m256i t = vtw(LD(y + j), w, wp, j, P, PINV);
        ST(x + j, vaddm(u, t, P));
        ST(y + j, vsubm(u, t, P));
    }
}

/** Runs a radix-2 DIF top pass (streaming twiddles): one compiled copy of
    dif2_body for ntt_conv.c. */
static void dif_stage(uint32_t *a, size_t h, size_t j0, size_t j1, const NttPrime *Pr) {
    dif2_body(a, h, j0, j1, Pr, 0);
}

/** Runs a radix-2 DIT top pass (streaming twiddles): the mirror of dif_stage. */
static void dit_stage(uint32_t *a, size_t h, size_t j0, size_t j1, const NttPrime *Pr) {
    dit2_body(a, h, j0, j1, Pr, 0);
}

/** Twiddles for the in-register levels 4 and 2, laid out to match
    dif_last3 / dit_first3's lanes, with their rtp entries. */
typedef struct {
    __m256i w4, w2, w4p, w2p; /* w4 = rt[4..7] per 128-bit half; w2 = rt[2], rt[3] repeated */
    __m256i P;
} SmallTw;

/** Returns the SmallTw of one prime. */
static SmallTw small_twiddles(const NttPrime *Pr) {
    SmallTw t;
    const uint32_t *rt = Pr->rt, *rp = Pr->rtp;
    t.w4 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(rt + 4)));
    t.w2 = _mm256_setr_epi32(rt[2], rt[3], rt[2], rt[3], rt[2], rt[3], rt[2], rt[3]);
    t.w4p = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(rp + 4)));
    t.w2p = _mm256_setr_epi32(rp[2], rp[3], rp[2], rp[3], rp[2], rp[3], rp[2], rp[3]);
    t.P = _mm256_set1_epi32((int)Pr->p);
    return t;
}

/** Runs DIF levels h = 4, 2, 1 on 16 elements (two 8-blocks A, B), in registers.
   The output layout is permuted; dit_first3 reads exactly this layout back,
   and the pointwise product in between doesn't care about order.
     h=4: x = [A0..3 | B0..3], y = [A4..7 | B4..7]
     h=2: unpack 64-bit halves -> x = [A0 A1 A4 A5 | ..], y = [A2 A3 A6 A7 | ..]
     h=1: shuffle -> x = even elements, y = odd elements (twiddle is 1) */
static inline void dif_last3(uint32_t *a, const SmallTw *t) {
    const __m256i P = t->P;
    __m256i A = LD(a), B = LD(a + 8);
    __m256i x = _mm256_permute2x128_si256(A, B, 0x20), y = _mm256_permute2x128_si256(A, B, 0x31);
    __m256i s = vaddm(x, y, P);
    __m256i d = vmontw(_mm256_sub_epi32(x, y), t->w4, t->w4p, P);

    x = _mm256_unpacklo_epi64(s, d);
    y = _mm256_unpackhi_epi64(s, d);
    s = vaddm(x, y, P);
    d = vmontw(_mm256_sub_epi32(x, y), t->w2, t->w2p, P);

    x = SHUF_PS(s, d, 0x88); /* _MM_SHUFFLE(2,0,2,0) */
    y = SHUF_PS(s, d, 0xDD); /* _MM_SHUFFLE(3,1,3,1) */
    ST(a, vaddm(x, y, P));
    ST(a + 8, vsubm(x, y, P));
}

/** Runs DIT levels h = 1, 2, 4 on 16 elements: the exact mirror of
    dif_last3, undoing each shuffle. */
static inline void dit_first3(uint32_t *a, const SmallTw *t) {
    const __m256i P = t->P;
    __m256i x = LD(a), y = LD(a + 8);
    __m256i s = vaddm(x, y, P);
    __m256i d = vsubm(x, y, P);

    x = _mm256_unpacklo_epi32(s, d);
    y = _mm256_unpackhi_epi32(s, d);
    __m256i tt = vmontw(y, t->w2, t->w2p, P);
    s = vaddm(x, tt, P);
    d = vsubm(x, tt, P);

    x = _mm256_unpacklo_epi64(s, d);
    y = _mm256_unpackhi_epi64(s, d);
    tt = vmontw(y, t->w4, t->w4p, P);
    s = vaddm(x, tt, P);
    d = vsubm(x, tt, P);

    ST(a, _mm256_permute2x128_si256(s, d, 0x20));
    ST(a + 8, _mm256_permute2x128_si256(s, d, 0x31));
}

/** Runs two DIF levels fused (radix 4): levels h = 2Q then h = Q on a block
    of 4Q, lanes [j0, j1) of Q. Same arithmetic as two radix-2 passes, but
    each element is loaded and stored once instead of twice, which halves
    memory traffic when the block is bigger than the caches. With src, the
    input is read from a bignum instead of a (see src_load8), so loading
    costs no separate pass over memory. */
static inline __attribute__((always_inline)) void dif4_body(uint32_t *a, const Src *src, size_t Q, size_t j0,
                                                            size_t j1, const NttPrime *Pr, int pre) {
    const __m256i P = _mm256_set1_epi32((int)Pr->p), PINV = _mm256_set1_epi32((int)Pr->pinv);
    const uint32_t *w1 = Pr->rt + 2 * Q, *w2 = Pr->rt + Q;
    const uint32_t *w1p = pre ? Pr->rtp + 2 * Q : NULL, *w2p = pre ? Pr->rtp + Q : NULL;
    uint32_t *a0 = a, *a1 = a + Q, *a2 = a + 2 * Q, *a3 = a + 3 * Q;
    for (size_t j = j0; j < j1; j += 8) {
        __m256i x0, x1, x2, x3;
        if (src) {
            const __m256i R2 = _mm256_set1_epi32((int)Pr->r2);
            x0 = src_load8(src, j, P, PINV, R2);
            x1 = src_load8(src, Q + j, P, PINV, R2);
            x2 = src_load8(src, 2 * Q + j, P, PINV, R2);
            x3 = src_load8(src, 3 * Q + j, P, PINV, R2);
        } else {
            x0 = LD(a0 + j), x1 = LD(a1 + j), x2 = LD(a2 + j), x3 = LD(a3 + j);
        }
        __m256i b0 = vaddm(x0, x2, P);
        __m256i b2 = vtw(_mm256_sub_epi32(x0, x2), w1, w1p, j, P, PINV);
        __m256i b1 = vaddm(x1, x3, P);
        __m256i b3 = vtw(_mm256_sub_epi32(x1, x3), w1, w1p, Q + j, P, PINV);
        __m256i d0 = _mm256_sub_epi32(b0, b1), d2 = _mm256_sub_epi32(b2, b3);
        ST(a0 + j, vaddm(b0, b1, P));
        ST(a1 + j, vtw(d0, w2, w2p, j, P, PINV));
        ST(a2 + j, vaddm(b2, b3, P));
        ST(a3 + j, vtw(d2, w2, w2p, j, P, PINV));
    }
}

/** Runs a radix-4 DIF top pass (streaming twiddles) on lanes [j0, j1) of Q. */
static void dif_stage4(uint32_t *a, size_t Q, size_t j0, size_t j1, const NttPrime *Pr) {
    dif4_body(a, NULL, Q, j0, j1, Pr, 0);
}

/** Runs dif_stage4 reading its input from src: the first pass of a transform. */
static void dif_stage4_src(uint32_t *a, const Src *src, size_t Q, size_t j0, size_t j1, const NttPrime *Pr) {
    dif4_body(a, src, Q, j0, j1, Pr, 0);
}

/** Runs two DIT levels fused: h = Q then h = 2Q, the mirror of dif4_body. */
static inline __attribute__((always_inline)) void dit4_body(uint32_t *a, size_t Q, size_t j0, size_t j1,
                                                            const NttPrime *Pr, int pre) {
    const __m256i P = _mm256_set1_epi32((int)Pr->p), PINV = _mm256_set1_epi32((int)Pr->pinv);
    const uint32_t *w1 = Pr->rt + 2 * Q, *w2 = Pr->rt + Q;
    const uint32_t *w1p = pre ? Pr->rtp + 2 * Q : NULL, *w2p = pre ? Pr->rtp + Q : NULL;
    uint32_t *a0 = a, *a1 = a + Q, *a2 = a + 2 * Q, *a3 = a + 3 * Q;
    for (size_t j = j0; j < j1; j += 8) {
        __m256i x0 = LD(a0 + j), x1 = LD(a1 + j), x2 = LD(a2 + j), x3 = LD(a3 + j);
        __m256i t = vtw(x1, w2, w2p, j, P, PINV);
        __m256i b0 = vaddm(x0, t, P);
        __m256i b1 = vsubm(x0, t, P);
        t = vtw(x3, w2, w2p, j, P, PINV);
        __m256i b2 = vaddm(x2, t, P);
        __m256i b3 = vsubm(x2, t, P);
        t = vtw(b2, w1, w1p, j, P, PINV);
        ST(a0 + j, vaddm(b0, t, P));
        ST(a2 + j, vsubm(b0, t, P));
        t = vtw(b3, w1, w1p, Q + j, P, PINV);
        ST(a1 + j, vaddm(b1, t, P));
        ST(a3 + j, vsubm(b1, t, P));
    }
}

/** Runs a radix-4 DIT top pass (streaming twiddles) on lanes [j0, j1) of Q. */
static void dit_stage4(uint32_t *a, size_t Q, size_t j0, size_t j1, const NttPrime *Pr) {
    dit4_body(a, Q, j0, j1, Pr, 0);
}

/** Transforms a block of L (16 <= L <= NTT_BASE_LEN) forward in cache:
    levels L/2 .. 8 in fused pairs (plus one radix-2 level if the count is
    odd), then the in-register kernel for levels 4, 2, 1. Twiddles here come
    from the start of the tables, which stays in cache, so the kernels use
    the precomputed rtp as well (vmontw). */
static void ntt_forward_base(uint32_t *a, size_t L, const NttPrime *Pr) {
    size_t h = L >> 1;
    for (; h >= 16; h >>= 2)
        for (size_t s = 0; s < L; s += 2 * h) dif4_body(a + s, NULL, h / 2, 0, h / 2, Pr, 1);
    if (h == 8)
        for (size_t s = 0; s < L; s += 16) dif2_body(a + s, 8, 0, 8, Pr, 1);
    SmallTw t = small_twiddles(Pr);
    for (size_t s = 0; s < L; s += 16) dif_last3(a + s, &t);
}

/** Transforms a block of L back in cache: the mirror of ntt_forward_base.
    DIT levels must run in increasing order; how they're paired doesn't
    matter, so the odd radix-2 level (if any) comes last here. */
static void ntt_inverse_base(uint32_t *a, size_t L, const NttPrime *Pr) {
    SmallTw t = small_twiddles(Pr);
    for (size_t s = 0; s < L; s += 16) dit_first3(a + s, &t);
    size_t h = 8;
    for (; 4 * h <= L; h <<= 2)
        for (size_t s = 0; s < L; s += 4 * h) dit4_body(a + s, h, 0, h, Pr, 1);
    if (h < L)
        for (size_t s = 0; s < L; s += 2 * h) dit2_body(a + s, h, 0, h, Pr, 1);
}

#ifndef NTT_R16_MIN_LEN
#define NTT_R16_MIN_LEN (1 << 16) /* from this block length on, radix-16 top passes */
#endif
#define R16_CHUNK 128 /* columns per radix-16 step: 16 rows x 128 words = 8 KB */

/** Returns the radix of the top pass of a block of L above the base size,
    which leaves that many independent sub-transforms: 2 if the halves
    already fit the base size, 16 for big blocks (4 levels per trip through
    memory), else 4. */
static int ntt_radix(size_t L) {
    if (L / 2 <= NTT_BASE_LEN) return 2;
    return L >= NTT_R16_MIN_LEN ? 16 : 4;
}

/* The outer radix-4 step of a radix-16 pass (levels h = L/2 and L/4, lanes
   j = c + m Q2 with c < Q2 = L/16) needs twiddles from the two biggest
   level tables, 3/4 of the pass's twiddle traffic. Each is a small-table
   entry times a constant root:
     rt[L/2 + c + k Q2] = w_L^c w_16^k,   rt[L/4 + c + m Q2] = w_(L/2)^c w_8^m,
   with w_L^c = rt[L/2 + c], w_(L/2)^c = rt[L/4 + c], w_16^k = rt[8 + k] and
   w_8^m = rt[4 + m]. Past NTT_TWD_MIN_LEN, where the tables no longer fit
   in cache, the pass reads only the first L/16 entries of the two levels
   and makes the rest with one constant multiply each. */
#ifndef NTT_TWD_MIN_LEN
#define NTT_TWD_MIN_LEN (1 << 19)
#endif

/** The constant roots of row group m for the derived twiddles. */
typedef struct {
    __m256i ka, kap, kb, kbp, kc, kcp; /* w_16^m, w_16^(m+4), w_8^m and their vpre */
} Tw16;

/** Returns the Tw16 of row group m (0..3). */
static Tw16 tw16_consts(const NttPrime *Pr, int m) {
    const __m256i PINV = _mm256_set1_epi32((int)Pr->pinv);
    uint32_t a = Pr->rt[8 + m], b = Pr->rt[12 + m], c = Pr->rt[4 + m];
    return (Tw16){_mm256_set1_epi32((int)a), vpre(a, PINV), _mm256_set1_epi32((int)b), vpre(b, PINV),
                  _mm256_set1_epi32((int)c), vpre(c, PINV)};
}

/** Runs dif4_body at Q = L/4 on columns [c, e) of row group m (lanes
    m*L/16 + [c, e)), with derived twiddles: one table read per level and
    one vmontw by a Tw16 constant (none for m = 0, whose constant is 1). */
static inline __attribute__((always_inline)) void dif4_derived(uint32_t *a, const Src *src, size_t L, size_t c,
                                                               size_t e, const NttPrime *Pr, int m) {
    const __m256i P = _mm256_set1_epi32((int)Pr->p), PINV = _mm256_set1_epi32((int)Pr->pinv);
    const size_t Q = L / 4, off = m * (L / 16);
    const uint32_t *t1 = Pr->rt + L / 2, *t2 = Pr->rt + L / 4;
    const Tw16 k = tw16_consts(Pr, m);
    uint32_t *a0 = a + off, *a1 = a0 + Q, *a2 = a0 + 2 * Q, *a3 = a0 + 3 * Q;
    for (size_t j = c; j < e; j += 8) {
        __m256i x0, x1, x2, x3;
        if (src) {
            const __m256i R2 = _mm256_set1_epi32((int)Pr->r2);
            x0 = src_load8(src, off + j, P, PINV, R2);
            x1 = src_load8(src, off + Q + j, P, PINV, R2);
            x2 = src_load8(src, off + 2 * Q + j, P, PINV, R2);
            x3 = src_load8(src, off + 3 * Q + j, P, PINV, R2);
        } else {
            x0 = LD(a0 + j), x1 = LD(a1 + j), x2 = LD(a2 + j), x3 = LD(a3 + j);
        }
        __m256i u1 = LD(t1 + j), u2 = LD(t2 + j);
        __m256i wa = m ? vmontw(u1, k.ka, k.kap, P) : u1;
        __m256i wb = vmontw(u1, k.kb, k.kbp, P);
        __m256i wc = m ? vmontw(u2, k.kc, k.kcp, P) : u2;
        __m256i b0 = vaddm(x0, x2, P);
        __m256i b2 = vmont(_mm256_sub_epi32(x0, x2), wa, P, PINV);
        __m256i b1 = vaddm(x1, x3, P);
        __m256i b3 = vmont(_mm256_sub_epi32(x1, x3), wb, P, PINV);
        __m256i wco = vodd(wc);
        __m256i d0 = _mm256_sub_epi32(b0, b1), d2 = _mm256_sub_epi32(b2, b3);
        ST(a0 + j, vaddm(b0, b1, P));
        ST(a1 + j, vmont2(d0, vodd(d0), wc, wco, P, PINV));
        ST(a2 + j, vaddm(b2, b3, P));
        ST(a3 + j, vmont2(d2, vodd(d2), wc, wco, P, PINV));
    }
}

/** Runs dit4_body at Q = L/4 on columns [c, e) of row group m, with derived
    twiddles: the mirror of dif4_derived. */
static inline __attribute__((always_inline)) void dit4_derived(uint32_t *a, size_t L, size_t c, size_t e,
                                                               const NttPrime *Pr, int m) {
    const __m256i P = _mm256_set1_epi32((int)Pr->p), PINV = _mm256_set1_epi32((int)Pr->pinv);
    const size_t Q = L / 4, off = m * (L / 16);
    const uint32_t *t1 = Pr->rt + L / 2, *t2 = Pr->rt + L / 4;
    const Tw16 k = tw16_consts(Pr, m);
    uint32_t *a0 = a + off, *a1 = a0 + Q, *a2 = a0 + 2 * Q, *a3 = a0 + 3 * Q;
    for (size_t j = c; j < e; j += 8) {
        __m256i x0 = LD(a0 + j), x1 = LD(a1 + j), x2 = LD(a2 + j), x3 = LD(a3 + j);
        __m256i u1 = LD(t1 + j), u2 = LD(t2 + j);
        __m256i wa = m ? vmontw(u1, k.ka, k.kap, P) : u1;
        __m256i wb = vmontw(u1, k.kb, k.kbp, P);
        __m256i wc = m ? vmontw(u2, k.kc, k.kcp, P) : u2;
        __m256i wco = vodd(wc);
        __m256i t = vmont2(x1, vodd(x1), wc, wco, P, PINV);
        __m256i b0 = vaddm(x0, t, P);
        __m256i b1 = vsubm(x0, t, P);
        t = vmont2(x3, vodd(x3), wc, wco, P, PINV);
        __m256i b2 = vaddm(x2, t, P);
        __m256i b3 = vsubm(x2, t, P);
        t = vmont(b2, wa, P, PINV);
        ST(a0 + j, vaddm(b0, t, P));
        ST(a2 + j, vsubm(b0, t, P));
        t = vmont(b3, wb, P, PINV);
        ST(a1 + j, vaddm(b1, t, P));
        ST(a3 + j, vsubm(b1, t, P));
    }
}

/** Runs dif4_derived on all four row groups. The two branches look alike on
    purpose: with src a constant NULL in one of them, gcc compiles a copy of
    the loop without the "read from a bignum?" test. */
static void dif4_derived_all(uint32_t *a, const Src *src, size_t L, size_t c, size_t e, const NttPrime *Pr) {
    if (src) {
        dif4_derived(a, src, L, c, e, Pr, 0);
        dif4_derived(a, src, L, c, e, Pr, 1);
        dif4_derived(a, src, L, c, e, Pr, 2);
        dif4_derived(a, src, L, c, e, Pr, 3);
    } else {
        dif4_derived(a, NULL, L, c, e, Pr, 0);
        dif4_derived(a, NULL, L, c, e, Pr, 1);
        dif4_derived(a, NULL, L, c, e, Pr, 2);
        dif4_derived(a, NULL, L, c, e, Pr, 3);
    }
}

/** Runs dit4_derived on all four row groups. */
static void dit4_derived_all(uint32_t *a, size_t L, size_t c, size_t e, const NttPrime *Pr) {
    dit4_derived(a, L, c, e, Pr, 0);
    dit4_derived(a, L, c, e, Pr, 1);
    dit4_derived(a, L, c, e, Pr, 2);
    dit4_derived(a, L, c, e, Pr, 3);
}

/** Runs four DIF levels (h = L/2 .. L/16) on columns [c, e) of L/16. Those
    levels only combine words in the same column mod L/16, so a chunk of
    columns is a self-contained job: two radix-4 steps over 16 rows x
    R16_CHUNK words that stay in L1. The outer step (levels L/2, L/4) uses
    derived twiddles past NTT_TWD_MIN_LEN; the inner one (L/8, L/16) runs
    on each quarter r. */
static void dif16_chunk(uint32_t *a, size_t L, size_t c, size_t e, const NttPrime *Pr, const Src *src) {
    size_t Q = L / 4, Q2 = L / 16;
    if (L >= NTT_TWD_MIN_LEN) {
        dif4_derived_all(a, src, L, c, e, Pr);
    } else {
        for (size_t m = 0; m < 4; m++) {
            if (src) dif_stage4_src(a, src, Q, c + m * Q2, e + m * Q2, Pr);
            else dif_stage4(a, Q, c + m * Q2, e + m * Q2, Pr);
        }
    }
    for (size_t r = 0; r < 4; r++) dif_stage4(a + r * Q, Q2, c, e, Pr);
}

/** Runs four DIT levels (h = L/16 .. L/2) on columns [c, e) of L/16: the
    mirror of dif16_chunk. */
static void dit16_chunk(uint32_t *a, size_t L, size_t c, size_t e, const NttPrime *Pr) {
    size_t Q = L / 4, Q2 = L / 16;
    for (size_t r = 0; r < 4; r++) dit_stage4(a + r * Q, Q2, c, e, Pr);
    if (L >= NTT_TWD_MIN_LEN) {
        dit4_derived_all(a, L, c, e, Pr);
    } else {
        for (size_t m = 0; m < 4; m++) dit_stage4(a, Q, c + m * Q2, e + m * Q2, Pr);
    }
}

/** Runs the radix-16 DIF top pass on columns [c0, c1) of L/16 (one
    thread's share), R16_CHUNK columns at a time. */
static void dif_pass16(uint32_t *a, size_t L, size_t c0, size_t c1, const NttPrime *Pr, const Src *src) {
    for (size_t c = c0; c < c1; c += R16_CHUNK)
        dif16_chunk(a, L, c, c + R16_CHUNK < c1 ? c + R16_CHUNK : c1, Pr, src);
}

/** Runs the radix-16 DIT top pass on columns [c0, c1): the mirror of
    dif_pass16. */
static void dit_pass16(uint32_t *a, size_t L, size_t c0, size_t c1, const NttPrime *Pr) {
    for (size_t c = c0; c < c1; c += R16_CHUNK)
        dit16_chunk(a, L, c, c + R16_CHUNK < c1 ? c + R16_CHUNK : c1, Pr);
}
