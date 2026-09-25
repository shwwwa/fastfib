# Checks a hexfib.c build: F(n) in hex for 245 values of n against the Python
# reference ..\fibonacci.py. The n are 0..199 (every small case), 40 seeded
# random n up to 3 million, and a few around 2^22 and 5 million.
# usage: python check.py hexfib.exe seed      (exits 1 on any mismatch)
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fibonacci import fibonacci  # noqa: E402  (needs the path above)


def main():
    exe, seed = sys.argv[1], int(sys.argv[2])
    rng = random.Random(seed)
    ns = list(range(200)) + [rng.randint(200, 3_000_000) for _ in range(40)]
    ns += [4_999_999, 5_000_000, 2**22 - 1, 2**22, 2**22 + 1]
    out = subprocess.run([exe, *map(str, ns)], capture_output=True, text=True, check=True).stdout.split()
    bad = [n for n, h in zip(ns, out) if h != format(fibonacci(n), "x")]
    if len(out) != len(ns):
        print(f"expected {len(ns)} results, got {len(out)}")
    print(f"{len(ns)} values, {len(bad)} mismatches" + (f" (first: n = {bad[0]})" if bad else ""))
    return 1 if bad or len(out) != len(ns) else 0


if __name__ == "__main__":
    sys.exit(main())
