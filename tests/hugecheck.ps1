# One huge n (1e9 and up): time, peak RAM, and a fingerprint of F(n) (bit length,
# low 64 bits, F(n) mod four 64-bit primes) compared with fingerprint.py, which
# computes it independently (fast doubling mod p, Binet for the length).
# HEAVY: F(1e10) takes ~18 s and ~6 GB RAM and lags the PC. One n at a time.
# usage: .\hugecheck.ps1 2000000000
param([Parameter(Mandatory)][string]$N)
Set-Location $PSScriptRoot
$gcc = if ($env:CC) { $env:CC } else { 'gcc' }
New-Item -ItemType Directory -Force out | Out-Null
& $gcc -O3 -march=native -o out\memrun.exe memrun.c -lpsapi
if ($LASTEXITCODE) { throw 'build failed' }
$ours = @(& out\memrun.exe $N)
$ours[0]
$ref = @(python fingerprint.py $N)
if ((($ours | Select-Object -Skip 1) -join "`n") -eq ($ref -join "`n")) {
    "  fingerprints match ($($ref.Count) lines)"
} else {
    '  MISMATCH'; $ours; '-- reference:'; $ref
}
