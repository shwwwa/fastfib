/* single-thread in-cache kernel timing: base forward + inverse, best of many */
#define UNIT_TEST_NO_MAIN
#include "../fastfib.c"
#include <x86intrin.h>
int main(void) {
    ntt_init(1 << 16);
    const NttPrime *P = &g_ntt[1];
    size_t sizes[] = {1024, 4096, 8192};
    uint32_t *a = _aligned_malloc(8192 * 4, 64);
    for (int s = 0; s < 3; s++) {
        size_t L = sizes[s];
        for (size_t i = 0; i < L; i++) a[i] = (uint32_t)(i * 2654435761u) % P->p;
        unsigned long long best = ~0ull;
        for (int r = 0; r < 3000; r++) {
            unsigned long long t = __rdtsc();
            ntt_forward_base(a, L, P);
            ntt_inverse_base(a, L, P);
            t = __rdtsc() - t;
            if (t < best) best = t;
        }
        printf("L=%5zu: %.2f ticks per butterfly-vector (fwd+inv)\n", L, (double)best / (L / 16 * __builtin_ctzll(L) * 2));
    }
    return 0;
}
