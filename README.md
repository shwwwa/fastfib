# fastfib
Fast fibonacci finder program, written from scratch in C
(no libraries). Fast doubling on top of a three-prime
number-theoretic-transform multiply, vectorized with AVX2 and multithreaded.

<sub>inspired by https://www.youtube.com/watch?v=KzT9I1d-LlQ</sub>

## Speed

i9-9900K (8 cores / 16 threads), Windows 10, best of 15 runs:

| n | fastfib | GMP 6 (gmpy2) |
|---|---|---|
| 10^6 | 1.0 ms | 1.5 ms |
| 5·10^6 | 2.7 ms | 10 ms |
| 2·10^7 | 8 ms | 51 ms |
| 10^8 | 46 ms | 356 ms |
| 10^9 | 0.7 s | |
| 10^10 | 18 s (6.1 GB RAM) | |

Printing F(10^8) in decimal (20.9 million digits) takes a further 0.6 s
(GMP: 2.6 s).

## Build and run

Needs gcc (MinGW-w64) or clang (MinGW target), and an x86-64 CPU with AVX2.
It refuses to build without AVX2 enabled, and a build run on a CPU without
AVX2 exits with a message instead of crashing.

```
gcc -O3 -march=native -o fastfib.exe fastfib.c     # or: .\build.ps1
.\fastfib.exe 100000000            # time only
.\fastfib.exe 1000 --print         # also print the number
.\fastfib.exe                      # asks for n
.\fastfib.exe --help
```

```
> .\fastfib.exe 10 --print
fibonacci(10) = 55
Time: 0.000101 seconds
Decimal conversion: 0.000001 seconds
```

`.\build.ps1 -Test` builds and runs the tests; `.\build.ps1 -Single` also
writes `dist\fastfib.c`, the whole program as one self-contained file. The
scripts use `gcc` and `nm` from PATH, or the `CC` and `NM` environment
variables when set.

## How it works

- **Fast doubling.** From (F(k-1), F(k)) the next pair needs two squarings:
  F(2k-1) = F(k)² + F(k-1)², F(2k+1) = 4F(k)² - F(k-1)² ± 2. The last step
  is a single multiply.
- **NTT multiply.** Numbers are cut into 32-35-bit coefficients and
  convolved modulo three primes below 2^31 with transforms of length 2^k
  or 3·2^k, then recombined by the Chinese remainder theorem. Montgomery
  arithmetic on 8 lanes at a time (AVX2). Radix-16 passes and fused
  sub-transforms keep memory traffic low.
- **Past the longest transform** (2^25), operands are split into pieces that
  are transformed once and combined pointwise.
- **Decimal output** by divide and conquer with Barrett division by powers
  of 10^19.

Below the NTT sizes it uses schoolbook, Karatsuba and Toom-3 methods of calculation.

## Layout

`fastfib.c` is the unity build program that includes the modules in
`src/` in order, so the compiler sees one translation unit and inlines across
modules. Build only `fastfib.c`, never the `src/` files alone.

| file | contents |
|---|---|
| `src/common.h` | includes, the `Big` type |
| `src/platform.h`, `src/platform_win.h` | the OS interface and its Windows version |
| `src/alloc.c` | arena allocator |
| `src/pool.c` | thread pool |
| `src/bigint.c` | add, subtract, shift, compare, schoolbook |
| `src/mul_basic.c` | multiply dispatch, Karatsuba, Toom-3 |
| `src/ntt_arith.c` | primes, AVX2 Montgomery arithmetic |
| `src/ntt_tables.c` | root-of-unity tables |
| `src/ntt_kernels.c` | butterflies and transform passes |
| `src/ntt_conv.c` | power-of-2 convolution |
| `src/ntt_radix3.c` | 3·2^k lengths |
| `src/crt.c` | three primes back to one number |
| `src/mul_ntt.c` | NTT multiplication |
| `src/mul_split.c` | products past the longest transform |
| `src/decimal.c` | conversion to decimal |
| `src/fib.c` | fast doubling |
| `src/main.c` | command line |

Tunables are `#ifndef` macros next to the code that uses them; override with
`-DNAME=value`.

## Portability

Windows only for now. All OS-specific code (virtual memory, threads, locks,
atomics, the clock) is behind the small interface described in
`src/platform.h`; a port to another OS means writing one
`src/platform_<os>.h` implementing it. The program also assumes x86-64 with
AVX2 and gcc or clang (`__int128`, inline assembly).

Linux support will come if I get bored.

## Tests

In `tests/` (PowerShell; the reference values need Python with gmpy2).
Builds go to `tests\out\`. Each script exits non-zero if anything fails.

- `test.ps1`: 19 variants, one line each. Builds with forced thresholds, so
  every code path runs at small sizes, checked against Python; NTT fuzzing;
  the CRT bound; decimal conversion; and the command line (`cli.ps1`).
- `bigcheck.ps1`: F(n) up to n = 1.28·10^8 against gmpy2.
- `hugecheck.ps1 N`: time, memory and an independent fingerprint for huge n
  (needs mpmath too).
- `bench.ps1`, `profile.ps1`: timing and a sampling profiler.

## License

MIT, see [LICENSE](LICENSE).
