# Independent fingerprint of F(n): F(n) mod m by fast doubling (small numbers only),
# and the exact bit length from Binet's formula at high precision.
import sys
from mpmath import mp, mpf, sqrt, log, floor

def fib_mod(n, m):
    def fd(k):  # (F(k), F(k+1)) mod m
        if k == 0:
            return 0, 1
        a, b = fd(k >> 1)
        c = a * ((2 * b - a) % m) % m
        d = (a * a + b * b) % m
        return (d, (c + d) % m) if k & 1 else (c, d)
    return fd(n)[0]

n = int(sys.argv[1])
mp.dps = 60
phi = (1 + sqrt(5)) / 2
# F(n) = round(phi^n / sqrt5) and phi^n / sqrt5 is never near a power of two here
print("bits", int(floor(n * log(phi, 2) - log(sqrt(5), 2))) + 1)
print("low64", fib_mod(n, 2**64))
for p in (18446744073709551557, 18446744073709551533, 9223372036854775783, 4611686018427387847):
    print("mod", p, fib_mod(n, p))
