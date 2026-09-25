# Checks big_to_string against Python/GMP on edge cases and random numbers.
# usage: python dectest.py exe [seed]
import sys, random, subprocess, gmpy2

exe = sys.argv[1]
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
nums = [0, 1, 9, 10, 2**64 - 1, 2**64, 10**19 - 1, 10**19, 10**19 + 1]
for i in range(0, 12):          # around every split size 19 * 2^i digits
    for k in (19 << i, (19 << i) + 1, (19 << i) - 1, (38 << i), (38 << i) - 1):
        for d in (-1, 0, 1):
            nums.append(10**k + d)
for b in (64, 128, 1000, 2047, 2048, 2049, 4096, 10000, 65536, 200000):
    nums += [2**b - 1, 2**b, 2**b + 1]
for _ in range(300):             # random sizes up to ~1M bits, some with long zero runs
    bits = int(2 ** rng.uniform(1, 20))
    x = rng.getrandbits(bits)
    if rng.random() < 0.3:
        x = x * 10**rng.randint(1, 5000) + rng.getrandbits(rng.randint(1, 64))
    nums.append(x)
for n in (1000, 100000, 1000000, 3000000):
    nums.append(int(gmpy2.fib(n)))
inp = "".join(format(x, "x") + "\n" for x in nums)
out = subprocess.run([exe], input=inp, capture_output=True, text=True, check=True).stdout.split("\n")
bad = [i for i, x in enumerate(nums) if out[i] != gmpy2.mpz(x).digits(10)]
print(f"{len(nums)} numbers, {len(bad)} mismatches" + (f" (first: index {bad[0]})" if bad else ""))
sys.exit(1 if bad else 0)
