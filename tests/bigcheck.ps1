# Large-n check: F(n) in hex from fastfib.c vs gmpy2, for n up to 128M
# (odd, even, and around the 2^25 transform limit). Needs gmpy2. ~1 minute.
Set-Location $PSScriptRoot
$gcc = if ($env:CC) { $env:CC } else { 'gcc' }
New-Item -ItemType Directory -Force out | Out-Null
& $gcc -O3 -march=native -o out\hex-default.exe hexfib.c
if ($LASTEXITCODE) { throw 'build failed' }
$ns = 20000000, 33554431, 50000001, 64000000, 77777777, 100000000, 128000001
$bad = 0
foreach ($n in $ns) {
    $a = (& out\hex-default.exe $n | Out-String).Trim()
    $b = (python -c "import gmpy2; print(gmpy2.fib($n).digits(16))" | Out-String).Trim()
    if ($a -ne $b -or $a.Length -eq 0) { $bad++ }
    "{0,10}: {1} ({2} hex digits)" -f $n, $(if ($a -eq $b) { 'match' } else { 'MISMATCH' }), $a.Length
}
"bigcheck: $($ns.Count - $bad) of $($ns.Count) match"
exit [int]($bad -ne 0)
