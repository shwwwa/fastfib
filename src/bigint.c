/**
 * @file   bigint.c
 * @brief  Basic arithmetic on Big: unsigned numbers in 64-bit limbs.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * Add, subtract (also in place, at a limb offset), shifts, compare,
 * schoolbook multiply and square, division by a small number, and SBig
 * (sign + magnitude) for Toom-3. Every function keeps a Big normalized:
 * n >= 1 and the top limb nonzero, except zero itself, which is {0}, n = 1.
 */

#pragma once
#include "alloc.c"

/* ==================== Basics ==================== */

/** Returns the length of d[0..n) without its leading zero limbs (at least 1). */
static size_t normalize_len(const uint64_t *d, size_t n) {
    while (n > 1 && d[n - 1] == 0) n--;
    return n;
}

/** Frees x and marks it dead (d = NULL, n = 0). The shared zero limb is never freed. */
static void big_free(Big *x) {
    if (x->d != &g_zero_limb) xfree(x->d);
    x->d = NULL;
    x->n = 0;
}

/** Returns v as a new one-limb Big. Requires freeing with big_free. */
static Big big_from_u64(uint64_t v) {
    Big r = {xmalloc(sizeof(uint64_t)), 1};
    r.d[0] = v;
    return r;
}

/** Returns true if a is 0 (relies on normalization: zero is exactly {0}). */
static int big_is_zero(const Big *a) {
    return a->n == 1 && a->d[0] == 0;
}

/** Returns the number of bits of a: the top limb's bits + 64 per limb below.
    a must not be 0 (clz of 0 is undefined). */
static size_t big_bits(const Big *a) {
    return 64 * (a->n - 1) + (64 - (size_t)__builtin_clzll(a->d[a->n - 1]));
}

/** Returns a view of a's limbs [lo, hi): points into a, copies nothing.
    Never pass it to big_free. An empty range gives the shared zero. */
static Big view_slice(const Big *a, size_t lo, size_t hi) {
    if (hi > a->n) hi = a->n;
    if (lo >= hi) {
        Big z = {&g_zero_limb, 1};
        return z;
    }
    Big v = {a->d + lo, normalize_len(a->d + lo, hi - lo)};
    return v;
}

/** Returns a copy of a (its own limbs, also for a view). Requires freeing with big_free. */
static Big big_copy(const Big *a) {
    Big r = {xmalloc(a->n * sizeof(uint64_t)), a->n};
    memcpy(r.d, a->d, a->n * sizeof(uint64_t));
    return r;
}

/* ==================== Addition and subtraction ==================== */

/** Adds one limb v into d at position pos and ripples the carry up
    (usually it dies after one limb). Returns the carry out of d[len-1]. */
static uint64_t limbs_add_u64(uint64_t *d, size_t len, size_t pos, uint64_t v) {
    for (; v && pos < len; pos++) {
        d[pos] += v;
        v = d[pos] < v;
    }
    return v;
}

/** Subtracts one limb v from d at position pos and ripples the borrow up.
    Returns the borrow out of d[len-1]. */
static uint64_t limbs_sub_u64(uint64_t *d, size_t len, size_t pos, uint64_t v) {
    for (; v && pos < len; pos++) {
        uint64_t old = d[pos];
        d[pos] = old - v;
        v = old < v;
    }
    return v;
}

/** Returns a+b as a new Big. Requires freeing with big_free.
    The common limbs go through one add-with-carry chain (adc); the rest of
    the longer number is copied and the carry rippled into it. */
static Big big_add(const Big *a, const Big *b) {
    if (a->n < b->n) { const Big *t = a; a = b; b = t; } /* a is the longer one */
    size_t n = a->n + 1;
    uint64_t *d = xmalloc(n * sizeof(uint64_t));
    unsigned char c = 0;
    for (size_t i = 0; i < b->n; i++) c = _addcarry_u64(c, a->d[i], b->d[i], (unsigned long long *)&d[i]);
    memcpy(d + b->n, a->d + b->n, (a->n - b->n) * sizeof(uint64_t));
    d[a->n] = 0;
    limbs_add_u64(d, n, b->n, c);
    Big r = {d, normalize_len(d, n)};
    return r;
}

/** Returns a-b as a new Big (requires a >= b). Requires freeing with big_free.
    Same as big_add with a subtract-with-borrow chain (sbb). */
static Big big_sub(const Big *a, const Big *b) {
    size_t bn = b->n < a->n ? b->n : a->n;
    uint64_t *d = xmalloc(a->n * sizeof(uint64_t));
    unsigned char c = 0;
    for (size_t i = 0; i < bn; i++) c = _subborrow_u64(c, a->d[i], b->d[i], (unsigned long long *)&d[i]);
    memcpy(d + bn, a->d + bn, (a->n - bn) * sizeof(uint64_t));
    limbs_sub_u64(d, a->n, bn, c);
    Big r = {d, normalize_len(d, a->n)};
    return r;
}

/** Adds x into d at limb offset off (d += x * 2^(64*off)), in place: no
    shifted copy of x. Returns the carry out of d[dn-1]: 0 when the caller
    made room for the sum. */
static uint64_t big_add_at(uint64_t *d, size_t dn, const Big *x, size_t off) {
    if (big_is_zero(x)) return 0;
    size_t xn = normalize_len(x->d, x->n);
    unsigned char c = 0;
    for (size_t i = 0; i < xn; i++) c = _addcarry_u64(c, d[off + i], x->d[i], (unsigned long long *)&d[off + i]);
    return limbs_add_u64(d, dn, off + xn, c);
}

/** Subtracts x from d at limb offset off, in place. The caller guarantees
    the result is >= 0. */
static void big_sub_at(uint64_t *d, size_t dn, const Big *x, size_t off) {
    if (big_is_zero(x)) return;
    size_t xn = normalize_len(x->d, x->n);
    unsigned char c = 0;
    for (size_t i = 0; i < xn; i++) c = _subborrow_u64(c, d[off + i], x->d[i], (unsigned long long *)&d[off + i]);
    limbs_sub_u64(d, dn, off + xn, c);
}

/* ==================== In-place addition and subtraction ==================== */

/** Subtracts y from x in place (requires x >= y). */
static void big_sub_inplace(Big *x, const Big *y) {
    big_sub_at(x->d, x->n, y, 0);
    x->n = normalize_len(x->d, x->n);
}

/** Adds y to x in place. x gets a new buffer only when y is longer, x is
    the shared zero (must never be written), or the sum carries out of the top. */
static void big_add_inplace(Big *x, const Big *y) {
    if (y->n > x->n || x->d == &g_zero_limb) {
        Big t = big_add(x, y);
        big_free(x);
        *x = t;
        return;
    }
    if (big_add_at(x->d, x->n, y, 0)) {
        // carry out of the top: one limb longer, and the new top limb is 1
        size_t n = x->n;
        uint64_t *d = xmalloc((n + 1) * sizeof(uint64_t));
        memcpy(d, x->d, n * sizeof(uint64_t));
        d[n] = 1;
        big_free(x);
        x->d = d;
        x->n = n + 1;
    }
}

/** Adds one limb v to x in place. v is passed as a one-limb view of the
    local variable, so nothing is allocated. */
static void big_add_u64_inplace(Big *x, uint64_t v) {
    Big t = {&v, 1};
    big_add_inplace(x, &t);
}

/** Subtracts one limb v from x in place (requires x >= v); see big_add_u64_inplace. */
static void big_sub_u64_inplace(Big *x, uint64_t v) {
    Big t = {&v, 1};
    big_sub_inplace(x, &t);
}

/* ==================== Shifts ==================== */

/** Returns 2a as a new Big (one limb longer for the bit that comes out of the top).
    Requires freeing with big_free. */
static Big big_shl1(const Big *a) {
    uint64_t *d = xmalloc((a->n + 1) * sizeof(uint64_t));
    uint64_t carry = 0;
    for (size_t i = 0; i < a->n; i++) {
        uint64_t v = a->d[i];
        // own bits one up + the top bit of the limb below
        d[i] = (v << 1) | carry;
        // the top bit falls out into the next limb
        carry = v >> 63;
    }
    d[a->n] = carry;
    Big r = {d, normalize_len(d, a->n + 1)};
    return r;
}

/** Returns a >> k = floor(a / 2^k) as a new Big, for any k: s = k/64 whole
    limbs are skipped, the remaining b = k%64 bits taken from two limbs.
    Requires freeing with big_free. */
static Big big_shr_bits(const Big *a, size_t k) {
    size_t s = k / 64, b = k % 64;
    if (s >= a->n) return big_from_u64(0);
    size_t n = a->n - s;
    uint64_t *d = xmalloc(n * sizeof(uint64_t));
    for (size_t i = 0; i < n; i++) {
        uint64_t lo = a->d[i + s], hi = i + s + 1 < a->n ? a->d[i + s + 1] : 0;
        // b == 0 separately: hi << 64 is undefined in C (x86 would shift by 0)
        d[i] = b ? (lo >> b) | (hi << (64 - b)) : lo;
    }
    Big r = {d, normalize_len(d, n)};
    return r;
}

/** Returns a << k = a * 2^k as a new Big, for any k: s = k/64 zero limbs at
    the bottom, then the limbs moved up by b = k%64 bits.
    Requires freeing with big_free. */
static Big big_shl_bits(const Big *a, size_t k) {
    if (big_is_zero(a)) return big_from_u64(0);
    size_t s = k / 64, b = k % 64, n = a->n + s + 1;
    uint64_t *d = xmalloc(n * sizeof(uint64_t));
    memset(d, 0, s * sizeof(uint64_t));

    // whole limbs only: a copy (and prev >> 64 would be undefined)
    if (b == 0) {
        memcpy(d + s, a->d, a->n * sizeof(uint64_t));
        d[n - 1] = 0;
    } else {
        // each limb = its own bits moved up + the top bits of the limb below
        uint64_t prev = 0;
        for (size_t i = 0; i < a->n; i++) {
            d[i + s] = (a->d[i] << b) | (prev >> (64 - b));
            prev = a->d[i];
        }
        d[n - 1] = prev >> (64 - b);
    }
    Big r = {d, normalize_len(d, n)};
    return r;
}

/** Returns 2^k as a new Big: a single 1 bit. Requires freeing with big_free. */
static Big big_pow2(size_t k) {
    Big r = {xcalloc(k / 64 + 1, sizeof(uint64_t)), k / 64 + 1};
    r.d[k / 64] = 1ull << (k % 64);
    return r;
}

/* ==================== Schoolbook multiplication ==================== */

/** Returns a*b as a new Big by long multiplication (n^2): each limb of a
    times all of b, added in shifted by i limbs. Requires freeing with big_free. */
static Big big_mul_schoolbook(const Big *a, const Big *b) {
    if (big_is_zero(a) || big_is_zero(b)) return big_from_u64(0);
    size_t n = a->n + b->n;
    uint64_t *d = xcalloc(n, sizeof(uint64_t));
    for (size_t i = 0; i < a->n; i++) {
        uint64_t ai = a->d[i];
        if (ai == 0) continue;
        uint64_t carry = 0;
        for (size_t j = 0; j < b->n; j++) {
            u128 cur = (u128)d[i + j] + (u128)ai * b->d[j] + carry;
            d[i + j] = (uint64_t)cur;
            carry = (uint64_t)(cur >> 64);
        }
        /* no earlier row reached d[i + b->n] (row i-1 ended at i-1 + b->n),
           so it is still 0: the carry is stored, never propagated */
        d[i + b->n] = carry;
    }
    Big r = {d, normalize_len(d, n)};
    return r;
}

/** Returns a*a as a new Big with about half the products of big_mul_schoolbook:
    a^2 = sum a[i]^2 * B^(2i) + 2 * sum(i<j) a[i]*a[j] * B^(i+j).
    Pass 1: the cross products once. Pass 2: double them and add the squares.
    Requires freeing with big_free. */
static Big big_sqr_schoolbook(const Big *a) {
    if (big_is_zero(a)) return big_from_u64(0);
    size_t n = a->n;
    size_t cap = 2 * n + 1;
    uint64_t *d = xcalloc(cap, sizeof(uint64_t));

    /* sum of cross terms a[i]*a[j], i<j */
    for (size_t i = 0; i < n; i++) {
        uint64_t ai = a->d[i];
        if (ai == 0) continue;
        uint64_t carry = 0;
        for (size_t j = i + 1; j < n; j++) {
            u128 cur = (u128)d[i + j] + (u128)ai * a->d[j] + carry;
            d[i + j] = (uint64_t)cur;
            carry = (uint64_t)(cur >> 64);
        }
        d[i + n] = carry; /* still 0 before this: row i-1 ended at i-1 + n */
    }

    /* one pass: double the cross-term sum and add the squares a[i]^2.
       Limbs go in pairs d[2i], d[2i+1], the positions of a[i]^2. */
    uint64_t shift_in = 0; /* top bit of the previous limb, shifted in by the doubling */
    uint64_t carry = 0;    /* carry of the additions */
    for (size_t i = 0; i < n; i++) {
        u128 sq = (u128)a->d[i] * a->d[i];
        uint64_t lo = d[2 * i], hi = d[2 * i + 1];
        uint64_t lo2 = (lo << 1) | shift_in, hi2 = (hi << 1) | (lo >> 63);
        shift_in = hi >> 63;
        u128 cur = (u128)lo2 + (uint64_t)sq + carry;
        d[2 * i] = (uint64_t)cur;
        cur = (u128)hi2 + (uint64_t)(sq >> 64) + (uint64_t)(cur >> 64);
        d[2 * i + 1] = (uint64_t)cur;
        carry = (uint64_t)(cur >> 64);
    }
    d[2 * n] = shift_in + carry; /* always 0: a^2 fits in 2n limbs */

    Big r = {d, normalize_len(d, cap)};
    return r;
}

/* ==================== Division and comparison ==================== */

/** Divides a by a small divisor in place (long division from the top limb
    down) and returns the remainder. The divisor is 32-bit so every quotient
    limb fits in 64 bits. */
static uint32_t big_divmod_small_inplace(Big *a, uint32_t divisor) {
    u128 rem = 0;
    for (size_t i = a->n; i-- > 0;) {
        // the remainder so far, followed by the next limb, divided
        u128 cur = (rem << 64) | a->d[i];
        a->d[i] = (uint64_t)(cur / divisor);
        rem = cur % divisor;
    }
    a->n = normalize_len(a->d, a->n);
    return (uint32_t)rem;
}

/** Returns -1, 0 or 1 as a < b, a == b, a > b. Normalized numbers: more limbs
    means bigger, else the highest differing limb decides. */
static int big_cmp(const Big *a, const Big *b) {
    if (a->n != b->n) return a->n < b->n ? -1 : 1;
    for (size_t i = a->n; i-- > 0;) {
        if (a->d[i] != b->d[i]) return a->d[i] < b->d[i] ? -1 : 1;
    }
    return 0;
}

/* ==================== Signed numbers (Toom-3) ==================== */

/** Signed number: sign + magnitude. Needed because Toom-3 evaluates at
    negative points. Zero is always stored with neg = 0 (no "-0"). */
typedef struct {
    int neg; /* canonical: mag==0 implies neg==0 */
    Big mag;
} SBig;

/** Returns mag with the sign neg; takes ownership of mag. Zero gets neg = 0. */
static SBig sbig_wrap(Big mag, int neg) {
    SBig r = {big_is_zero(&mag) ? 0 : neg, mag};
    return r;
}

/** Frees the magnitude of x. */
static void sbig_free(SBig *x) {
    big_free(&x->mag);
}

/** Frees x and puts v in its place: x = f(x, ...) without a temporary,
    as in sbig_replace(&x, sbig_sub(&x, &y)) (v is computed before x is freed). */
static void sbig_replace(SBig *x, SBig v) {
    sbig_free(x);
    *x = v;
}

/** Returns x+y as a new SBig. Same signs: magnitudes add. Different signs:
    the smaller magnitude is subtracted from the larger, which gives the sign. */
static SBig sbig_add(const SBig *x, const SBig *y) {
    if (x->neg == y->neg) {
        return sbig_wrap(big_add(&x->mag, &y->mag), x->neg);
    }
    int cmp = big_cmp(&x->mag, &y->mag);
    if (cmp == 0) return sbig_wrap(big_from_u64(0), 0);
    if (cmp > 0) return sbig_wrap(big_sub(&x->mag, &y->mag), x->neg);
    return sbig_wrap(big_sub(&y->mag, &x->mag), y->neg);
}

/** Returns x-y as a new SBig, as x + (-y). -y is a shallow copy (shares y's
    limbs, flipped sign), so nothing is copied, and it must not be freed. */
static SBig sbig_sub(const SBig *x, const SBig *y) {
    SBig negy = {big_is_zero(&y->mag) ? 0 : !y->neg, y->mag};
    return sbig_add(x, &negy);
}

/** Returns x-y as a new SBig for two unsigned numbers. */
static SBig sbig_diff(const Big *x, const Big *y) {
    if (big_cmp(x, y) >= 0) return sbig_wrap(big_sub(x, y), 0);
    return sbig_wrap(big_sub(y, x), 1);
}

/** Returns x / divisor as a new SBig. The division must be exact (the
    remainder is dropped): Toom-3's interpolation divides by 2 and 3. */
static SBig sbig_div_small_exact(const SBig *x, uint32_t divisor) {
    Big t = big_copy(&x->mag);
    big_divmod_small_inplace(&t, divisor);
    return sbig_wrap(t, x->neg);
}
