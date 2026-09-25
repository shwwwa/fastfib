/**
 * @file   common.h
 * @brief  Includes, core types and settings shared by every module.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 shwwwa
 *
 * fastfib.c is a unity build: it includes every module in a fixed order
 * and the compiler sees one unit, so it can inline across modules (the speed
 * depends on that). Modules are never compiled on their own; each includes
 * the earlier ones it uses only so that an editor can analyze it alone.
 */

#pragma once

// Stop early with one clear message instead of many errors deep in the code
#if !defined(__SIZEOF_INT128__) || !defined(__x86_64__)
#error "fastfib.c needs gcc or clang on x86-64 (it uses __int128, gcc builtins and AVX2)"
#elif !defined(__AVX2__)
#error "fastfib.c needs AVX2: build with -march=native on an AVX2 CPU (or -mavx2)"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h> /* AVX2 intrinsics and _addcarry_u64 (x86-64 only) */

/** Arbitrary-precision unsigned integer: value = sum of d[i] * 2^(64*i). */
typedef struct {
    uint64_t *d; /* limbs, lowest first */
    size_t n;    /* number of limbs, always >= 1; normalized: no leading zero limb, zero is {0} */
} Big;

/* 128-bit unsigned integer: holds a full 64 x 64-bit product */
typedef unsigned __int128 u128;

/* The limb of the shared zero: empty views point here (see view_slice).
   Never written and never freed (big_free skips it). */
static uint64_t g_zero_limb = 0;

/* Longest NTT length is 2^NTT_MAX_LG. A length-2^k transform mod p needs 2^k
   to divide p - 1; of the three primes, p1 - 1 = 63 * 2^25 has the smallest
   power of 2. Here and not in the NTT modules: mul_basic.c (ntt_fits) comes
   before them. */
#ifndef NTT_MAX_LG
#define NTT_MAX_LG 25
#endif
