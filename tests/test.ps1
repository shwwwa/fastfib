# Full correctness sweep. Builds fastfib.c in about 20 variants and checks each:
#   hex   F(n) in hex for 245 values of n vs ..\fibonacci.py (check.py)
#   dec   big_to_string on 523 numbers vs gmpy2 (dectest.py)
#   self  the harness checks itself and exits non-zero on failure (nttfuzz.c, bound.c)
#   cli   fastfib.exe's command line (cli.ps1)
# Variants force tiny thresholds so every code path runs at sizes the checks reach
# quickly, force the split multiply, and vary threads and the arena. Builds go to
# tests\out (and ..\fastfib.exe). Prints one line per variant, then a count;
# exits 1 if anything failed.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$gcc = if ($env:CC) { $env:CC } else { 'gcc' }
New-Item -ItemType Directory -Force out | Out-Null

# small blocks, parallel passes and arena blocks from the first sizes on
$small = @('-DNTT_BASE_LEN=16', '-DNTT_PAR_MIN_LEN=32', '-DARENA_MIN_BYTES=64')
# plus every NTT pass kind and Karatsuba from the smallest products on
$forced = $small + @('-DNTT_THRESHOLD_LIMBS=1', '-DNTT_R16_MIN_LEN=256', '-DNTT_TWD_MIN_LEN=256',
                     '-DNTT_R3_SEQ_MIN_LEN=32', '-DNTT_R3_FUSE_MIN_LEN=256',
                     '-DKARATSUBA_THRESHOLD_LIMBS=1', '-DSQR_PAR_MIN_LIMBS=1', '-DSTEP_PAR_MIN_LIMBS=1')
# the fuzzer's operands are bigger: switch the pass kinds a little later
$forcedFuzz = $small + @('-DNTT_R16_MIN_LEN=512', '-DNTT_TWD_MIN_LEN=1024',
                         '-DNTT_R3_SEQ_MIN_LEN=128', '-DNTT_R3_FUSE_MIN_LEN=512')
$forcedDec = $small + @('-DNTT_THRESHOLD_LIMBS=1', '-DNTT_R16_MIN_LEN=256', '-DKARATSUBA_THRESHOLD_LIMBS=4',
                        '-DDEC_LEAF_LG=1', '-DDEC_PAR_MAX_LIMBS=64')
# thresholds between the forced and the default ones
$mixed = @('-DNTT_BASE_LEN=64', '-DNTT_PAR_MIN_LEN=256', '-DNTT_THRESHOLD_LIMBS=8', '-DNTT_R16_MIN_LEN=2048',
           '-DNTT_TWD_MIN_LEN=4096', '-DSTEP_PAR_MIN_LIMBS=64')
$split10 = @('-DNTT_MAX_LG=10')  # longest transform 2^10: the split multiply from small sizes on

# Check is 'hex:<seed>', 'dec:<seed>', 'self' or 'cli'.
$variants = @(
    @{ Name = 'hex-default';         Src = 'hexfib.c';  Check = 'hex:1'; Flags = @() }
    @{ Name = 'hex-forced';          Src = 'hexfib.c';  Check = 'hex:2'; Flags = $forced }
    @{ Name = 'hex-mixed';           Src = 'hexfib.c';  Check = 'hex:3'; Flags = $mixed }
    @{ Name = 'hex-no-workers';      Src = 'hexfib.c';  Check = 'hex:4'; Flags = @('-DPOOL_WORKERS=0') }
    @{ Name = 'hex-seq-primes';      Src = 'hexfib.c';  Check = 'hex:5'; Flags = @('-DNTT_SEQ_MIN_LEN=16', '-DNTT_THRESHOLD_LIMBS=8') }
    @{ Name = 'hex-split10';         Src = 'hexfib.c';  Check = 'hex:6'; Flags = $split10 }
    @{ Name = 'hex-split12-forced';  Src = 'hexfib.c';  Check = 'hex:7'; Flags = $forced + @('-DNTT_MAX_LG=12') }
    @{ Name = 'hex-arena-discard';   Src = 'hexfib.c';  Check = 'hex:8'; Flags = @('-DARENA_DISCARD_MIN=4096') }
    @{ Name = 'hex-2-threads';       Src = 'hexfib.c';  Check = 'hex:9'; Flags = @('-DNTT_THREADS=2', '-DNTT_THREADS_SEQ=2', '-DNTT_SEQ_MIN_LEN=4096') }
    @{ Name = 'fuzz-default';        Src = 'nttfuzz.c'; Check = 'self';  Flags = @() }
    @{ Name = 'fuzz-forced';         Src = 'nttfuzz.c'; Check = 'self';  Flags = $forcedFuzz }
    @{ Name = 'fuzz-seq-8-threads';  Src = 'nttfuzz.c'; Check = 'self';  Flags = @('-DNTT_SEQ_MIN_LEN=16', '-DNTT_THREADS_SEQ=8') }
    @{ Name = 'fuzz-split10';        Src = 'nttfuzz.c'; Check = 'self';  Flags = $split10 }
    @{ Name = 'fuzz-split12-forced'; Src = 'nttfuzz.c'; Check = 'self';  Flags = $forcedFuzz + @('-DNTT_MAX_LG=12') }
    @{ Name = 'crt-bound';           Src = 'bound.c';   Check = 'self';  Flags = @() }
    @{ Name = 'dec-default';         Src = 'dectest.c'; Check = 'dec:1'; Flags = @() }
    @{ Name = 'dec-forced';          Src = 'dectest.c'; Check = 'dec:2'; Flags = $forcedDec }
    @{ Name = 'dec-split10';         Src = 'dectest.c'; Check = 'dec:3'; Flags = $split10 + @('-DDEC_LEAF_LG=6') }
    @{ Name = 'fastfib';             Src = '..\fastfib.c'; Check = 'cli'; Flags = @(); Exe = '..\fastfib.exe' }
)
foreach ($v in $variants) { if (-not $v.Exe) { $v.Exe = "out\$($v.Name).exe" } }

# At most 8 compilers at once: each needs ~350 MB, and all of them together can
# run out of memory when other apps are open.
$maxPar = 8
$procs = New-Object System.Collections.Generic.List[object]
foreach ($v in $variants) {
    while (@($procs | Where-Object { -not $_.HasExited }).Count -ge $maxPar) { Start-Sleep -Milliseconds 100 }
    $gccArgs = @('-O3', '-march=native') + $v.Flags + @('-o', $v.Exe, $v.Src)
    $p = Start-Process -FilePath $gcc -ArgumentList $gccArgs -NoNewWindow -PassThru
    $null = $p.Handle  # keeps ExitCode readable after the process exits
    $procs.Add($p)
}
$procs | Wait-Process
for ($i = 0; $i -lt $variants.Count; $i++) {
    if ($procs[$i].ExitCode -ne 0) { throw "build failed: $($variants[$i].Name)" }
}

$runCheck = {
    param($dir, $exe, $check)
    Set-Location $dir
    $kind, $seed = $check -split ':'
    # stderr is merged in (a crash's message belongs in the report) and every
    # line made plain text: stderr lines arrive as error records, which would
    # otherwise stop the whole sweep when the job's output is received
    $text = switch ($kind) {
        'hex'  { python check.py $exe $seed 2>&1 }
        'dec'  { python dectest.py $exe $seed 2>&1 }
        'self' { & $exe 2>&1 }
        'cli'  { powershell -NoProfile -File cli.ps1 $exe 2>&1 }
    }
    [pscustomobject]@{ Text = @($text | ForEach-Object { "$_" }); Ok = $LASTEXITCODE -eq 0 }
}
$jobs = foreach ($v in $variants) { Start-Job $runCheck -ArgumentList $PSScriptRoot, $v.Exe, $v.Check }
$results = @(foreach ($j in $jobs) { Receive-Job $j -Wait })
$jobs | Remove-Job

$failed = 0
for ($i = 0; $i -lt $variants.Count; $i++) {
    $r = $results[$i]
    if ($r.Ok) {
        '{0,-20} ok    {1}' -f $variants[$i].Name, $r.Text[-1]
    } else {
        $failed++
        '{0,-20} FAIL' -f $variants[$i].Name
        $r.Text | ForEach-Object { "    $_" }
    }
}
"test.ps1: $($variants.Count - $failed) passed, $failed failed"
exit [int]($failed -ne 0)
