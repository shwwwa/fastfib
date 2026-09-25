import argparse
import sys
import time

sys.set_int_max_str_digits(0)

def fibonacci(n):
    def fib_pair(k):
        if k == 0:
            return 0, 1
        a, b = fib_pair(k >> 1)
        c = a * (2 * b - a)
        d = a * a + b * b
        if k & 1:
            return d, c + d
        return c, d

    return fib_pair(n)[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the nth Fibonacci number")
    parser.add_argument("n", nargs="?", type=int, help="which Fibonacci number to compute")
    args = parser.parse_args()

    n = args.n if args.n is not None else int(input("Enter n: "))
    start = time.perf_counter()
    result = fibonacci(n)
    elapsed = time.perf_counter() - start
    print(f"Time: {elapsed:.6f} seconds")
