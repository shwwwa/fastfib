/**
 * @file   main.c
 * @brief  Command line: fastfib [n] [--print] [--help].
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * Computes F(n) and prints the time it took; with --print also the decimal
 * digits and the conversion time. n comes from the command line or, if
 * missing, from a prompt. Bad input is rejected before any work starts.
 */

#pragma once
#include "fib.c"
#include "decimal.c"

/** Exits with a message if the CPU has no AVX2, instead of crashing on the
    first AVX2 instruction. Runs before main (which may already use AVX) and
    is itself compiled without AVX. There is no fallback path on purpose. */
__attribute__((constructor, target("no-avx"))) static void require_avx2(void) {
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2")) {
        fputs("fastfib: this CPU does not support AVX2, which fastfib requires\n", stderr);
        exit(1);
    }
}

/** Prints the usage text to f (stdout for --help, stderr after an error). */
static void usage(FILE *f) {
    fprintf(f, "usage: fastfib [n] [--print]\n"
               "  n        which Fibonacci number (asked for if missing)\n"
               "  --print  also print its decimal digits\n");
}

/** Parses s as a decimal n into *n. Returns 1 if valid, 0 otherwise: digits
    only (no sign, no spaces, not empty), and it must fit in 64 bits. Stricter
    than strtoull, which reads "-5" as 2^64 - 5 and "1e9" as 1. */
static int parse_n(const char *s, uint64_t *n) {
    uint64_t v = 0;
    if (!*s) return 0;
    for (; *s; s++) {
        if (*s < '0' || *s > '9') return 0;
        unsigned d = (unsigned)(*s - '0');
        // v * 10 + d would pass 2^64 - 1
        if (v > (UINT64_MAX - d) / 10) return 0;
        v = v * 10 + d;
    }
    *n = v;
    return 1;
}

/** Returns 0 on success, 1 on bad input. */
int main(int argc, char **argv) {
    uint64_t n = 0;
    int have_n = 0, print_result = 0;

    // options in any order; one number; anything else is an error
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--print") == 0) {
            print_result = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout);
            return 0;
        } else if (!have_n && parse_n(argv[i], &n)) {
            have_n = 1;
        } else {
            fprintf(stderr, "invalid argument: %s\n", argv[i]);
            usage(stderr);
            return 1;
        }
    }
    // no n on the command line: ask for it, trimming spaces around the number
    if (!have_n) {
        char line[64];
        printf("Enter n: ");
        fflush(stdout);
        if (!fgets(line, sizeof line, stdin)) return 1;
        char *s = line;
        if (strncmp(s, "\xEF\xBB\xBF", 3) == 0) s += 3; /* UTF-8 byte order mark (piped input) */
        s += strspn(s, " \t");
        s[strcspn(s, " \t\r\n")] = '\0';
        if (!parse_n(s, &n)) {
            fprintf(stderr, "invalid n: %s\n", s);
            return 1;
        }
    }

    // the time covers F(n) only; the decimal conversion is timed separately
    double start = plat_seconds();
    Big result = fibonacci(n);
    double elapsed = plat_seconds() - start;

    double conv = 0;
    if (print_result) {
        start = plat_seconds();
        char *s = big_to_string(&result);
        conv = plat_seconds() - start;
        printf("fibonacci(%llu) = %s\n", (unsigned long long)n, s);
        free(s);
    }
    printf("Time: %.6f seconds\n", elapsed);
    if (print_result) printf("Decimal conversion: %.6f seconds\n", conv);

    big_free(&result);
    return 0;
}
