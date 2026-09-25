/* CRT bound stress: all-ones squares (every coefficient at its maximum) up to
   huge sizes, checked against the closed form (2^N-1)^2 = 2^2N - 2^(N+1) + 1,
   plus unequal all-ones products (2^x-1)(2^y-1) = 2^(x+y) - 2^x - 2^y + 1. */
#define UNIT_TEST_NO_MAIN
#include "../fastfib.c"

/** Returns 2^bits - 1: bits one-bits. */
static Big ones_bits(size_t bits) {
    Big a = {xmalloc((bits + 63) / 64 * sizeof(uint64_t)), (bits + 63) / 64};
    for (size_t i = 0; i < a.n; i++) a.d[i] = ~0ULL;
    if (bits % 64) a.d[a.n - 1] = (1ULL << (bits % 64)) - 1;
    return a;
}

/** Returns (2^x - 1)(2^y - 1) = 2^(x+y) - 2^x - 2^y + 1, built without multiplying. */
static Big expect(size_t x, size_t y) {
    Big t = big_pow2(x + y), bx = big_pow2(x), by = big_pow2(y), one = big_from_u64(1);
    Big u = big_sub(&t, &bx), v = big_sub(&u, &by), r = big_add(&v, &one);
    big_free(&t);
    big_free(&bx);
    big_free(&by);
    big_free(&one);
    big_free(&u);
    big_free(&v);
    return r;
}

int main(void) {
    ntt_init(8000000);
    size_t sq[] = {64 * 1000, 64 * 100000, 64 * 542377, 64 * 1100000, 64 * 2000000 + 17, 64 * 4000000 - 5};
    int fails = 0;
    for (int i = 0; i < 6; i++) {
        Big a = ones_bits(sq[i]);
        Big got = big_mul_ntt(&a, &a), want = expect(sq[i], sq[i]);
        int ok = big_cmp(&got, &want) == 0;
        fails += !ok;
        printf("square of %zu one-bits: %s\n", sq[i], ok ? "ok" : "FAIL");
        big_free(&a);
        big_free(&got);
        big_free(&want);
    }

    size_t pr[][2] = {{64 * 1100000, 64 * 1000000 + 3}, {64 * 3000000, 64 * 50000}, {64 * 2500000 + 9, 64 * 2400000}};
    for (int i = 0; i < 3; i++) {
        Big a = ones_bits(pr[i][0]), b = ones_bits(pr[i][1]);
        Big got = big_mul_ntt(&a, &b), want = expect(pr[i][0], pr[i][1]);
        int ok = big_cmp(&got, &want) == 0;
        fails += !ok;
        printf("product %zu x %zu one-bits: %s\n", pr[i][0], pr[i][1], ok ? "ok" : "FAIL");
        big_free(&a);
        big_free(&b);
        big_free(&got);
        big_free(&want);
    }

    printf("9 products, %d failures\n", fails);
    return fails != 0;
}
