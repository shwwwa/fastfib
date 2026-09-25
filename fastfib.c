/**
 * @file   fastfib.c
 * @brief  fastfib: exact Fibonacci numbers F(n) for huge n.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * F(10^8) (69 million bits) in ~45 ms and F(10^10) in ~18 s on an i9-9900K,
 * using fast doubling on top of a 3-prime AVX2 number-theoretic transform
 * multiply. No external libraries. Needs gcc or clang on x86-64 with AVX2;
 * the OS layer exists for Windows so far (src/platform.h).
 *
 *   Build:  gcc -O3 -march=native -o fastfib.exe fastfib.c
 *   Run:    fastfib.exe 100000000 [--print]    (--help for usage)
 *
 * This file is the whole program: it includes the modules in src/ in order
 * (a "unity build"), so the compiler sees one translation unit and inlines
 * across modules. Do not compile the src/ files on their own. (In
 * dist/fastfib.c the modules follow inline, in the same order.) Tunables are
 * #ifndef macros next to the code that uses them; override with -DNAME=value.
 * Define UNIT_TEST_NO_MAIN to include this file from a test harness.
 *
 * Layers, bottom to top:
 *   common.h        includes, the Big type, u128
 *   platform.h      the OS interface (memory, threads, locks, time)
 *   alloc.c         arena allocator: xmalloc / xfree
 *   pool.c          thread pool: pool_submit / pool_wait / pool_run
 *   bigint.c        add, sub, shift, compare, schoolbook
 *   mul_basic.c     big_mul / big_sqr dispatch, Karatsuba, Toom-3
 *   ntt_arith.c     primes and AVX2 Montgomery arithmetic
 *   ntt_tables.c    root tables (ntt_init)
 *   ntt_kernels.c   butterflies and transform passes
 *   ntt_conv.c      power-of-2 convolution, ntt_fwd / ntt_inv, thread settings
 *   ntt_radix3.c    3 * 2^k lengths, ntt_conv_any
 *   crt.c           three primes -> one bignum
 *   mul_ntt.c       big_mul_ntt, coefficient width choice
 *   mul_split.c     products past the longest transform
 *   decimal.c       big_to_string
 *   fib.c           fibonacci(n): fast doubling
 *   main.c          command line
 */

#include "src/common.h"
#include "src/platform.h"
#include "src/alloc.c"
#include "src/pool.c"
#include "src/bigint.c"
#include "src/mul_basic.c"
#include "src/ntt_arith.c"
#include "src/ntt_tables.c"
#include "src/ntt_kernels.c"
#include "src/ntt_conv.c"
#include "src/ntt_radix3.c"
#include "src/crt.c"
#include "src/mul_ntt.c"
#include "src/mul_split.c"
#include "src/decimal.c"
#include "src/fib.c"
#ifndef UNIT_TEST_NO_MAIN
#include "src/main.c"
#endif
