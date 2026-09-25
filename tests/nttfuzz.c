/* NTT multiply fuzzer: 600 random products (and squares) through big_mul_ntt,
   each compared with the same product by Karatsuba / Toom-3 (big_mul with the
   NTT tables hidden). Operands are random, all ones, or sparse (mostly zero
   limbs), up to 60000 limbs. Ends with the worst case for the CRT bound: the
   largest all-ones square the tables allow. Fixed seed, so runs repeat exactly.
   Exits 1 on any mismatch. */
#define UNIT_TEST_NO_MAIN
#include "../fastfib.c"

static uint64_t g_rng = 88172645463325252ULL;

/** Returns the next xorshift64 value. */
static uint64_t rnd(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

enum { RANDOM, ALL_ONES, SPARSE };

/** Returns an n-limb operand (top bit set) of the given kind. */
static Big make(size_t n, int kind) {
    Big r = {xmalloc(n * sizeof(uint64_t)), n};
    for (size_t i = 0; i < n; i++)
        r.d[i] = kind == RANDOM ? rnd() : kind == ALL_ONES ? ~0ULL : (rnd() % 7 == 0 ? rnd() : 0);
    r.d[n - 1] |= 1ULL << 63;
    r.n = normalize_len(r.d, n);
    return r;
}

/** Returns a*b without the NTT: g_ntt_max_L = 0 makes big_mul use Karatsuba / Toom-3. */
static Big reference_mul(const Big *a, const Big *b) {
    size_t saved = g_ntt_max_L;
    g_ntt_max_L = 0;
    Big r = big_mul(a, b);
    g_ntt_max_L = saved;
    return r;
}

int main(void) {
    ntt_init(1 << 18);
    int fails = 0, trials = 0;
    for (int t = 0; t < 600; t++) {
        size_t na = 1 + rnd() % (t < 500 ? 2000 : 60000);
        size_t nb = rnd() % 3 == 0 ? 1 + rnd() % na : na;
        int kind = (int)(rnd() % 3);
        Big a = make(na, kind), b = make(nb, (int)(rnd() % 3));
        const Big *y = rnd() % 3 == 0 ? &a : &b; /* a third are squares */
        Big got = big_mul_ntt(&a, y);
        Big want = reference_mul(&a, y);
        trials++;
        if (big_cmp(&got, &want) != 0) {
            fails++;
            printf("FAIL na=%zu nb=%zu kind=%d square=%d\n", na, nb, kind, y == &a);
        }
        big_free(&a);
        big_free(&b);
        big_free(&got);
        big_free(&want);
    }

    Big a = make(100000, ALL_ONES);
    Big got = big_mul_ntt(&a, &a);
    Big want = reference_mul(&a, &a);
    trials++;
    if (big_cmp(&got, &want) != 0) {
        fails++;
        puts("FAIL all-ones square, 100000 limbs");
    }
    big_free(&a);
    big_free(&got);
    big_free(&want);

    printf("%d trials, %d failures\n", trials, fails);
    return fails != 0;
}
