/* one cold fibonacci(n): time, peak working set (RAM) and peak commit */
#define UNIT_TEST_NO_MAIN
#include "../fastfib.c"
#include <psapi.h>
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s n\n", argv[0]);
        return 2;
    }
    uint64_t n = strtoull(argv[1], NULL, 10);
    double t0 = plat_seconds();
    Big r = fibonacci(n);
    double secs = plat_seconds() - t0;
    PROCESS_MEMORY_COUNTERS pm;
    GetProcessMemoryInfo(GetCurrentProcess(), &pm, sizeof pm);
    printf("n=%llu: %.2f s | result %.0f MB | peak RAM %.0f MB | peak commit %.0f MB\n", (unsigned long long)n,
           secs, r.n * 8 / 1e6, pm.PeakWorkingSetSize / 1e6,
           pm.PeakPagefileUsage / 1e6);
    /* same fingerprint lines as fingerprint.py */
    printf("bits %llu\n", (unsigned long long)big_bits(&r));
    printf("low64 %llu\n", (unsigned long long)r.d[0]);
    const uint64_t ps[] = {18446744073709551557ull, 18446744073709551533ull, 9223372036854775783ull, 4611686018427387847ull};
    for (int i = 0; i < 4; i++) {
        u128 m = 0;
        for (size_t k = r.n; k-- > 0;) m = ((m << 64) | r.d[k]) % ps[i];
        printf("mod %llu %llu\n", (unsigned long long)ps[i], (unsigned long long)m);
    }
    return 0;
}
