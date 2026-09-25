/* Prints F(n) in hex, one line per argument, for check.py and bigcheck.ps1.
   usage: hexfib n [n ...] */
#define UNIT_TEST_NO_MAIN
#include "../fastfib.c"
int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        uint64_t n = strtoull(argv[i], NULL, 10);
        Big r = fibonacci(n);
        printf("%llx", (unsigned long long)r.d[r.n - 1]);
        for (size_t j = r.n - 1; j-- > 0;) printf("%016llx", (unsigned long long)r.d[j]);
        printf("\n");
        big_free(&r);
    }
    return 0;
}
