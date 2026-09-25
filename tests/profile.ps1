# Sampling profile: runs fibonacci(N) Reps times while a sampler thread records
# every thread's instruction pointer ~every 1 ms, then prints time per function
# (sprof.py) and how many threads were busy per sample (busy.py).
# tp_worker / pool_wait = threads spinning for work; "outside exe" = asleep.
# usage: .\profile.ps1 100000000 200
param([string]$N = '100000000', [int]$Reps = 200)
Set-Location $PSScriptRoot
$gcc = if ($env:CC) { $env:CC } else { 'gcc' }
New-Item -ItemType Directory -Force out | Out-Null
& $gcc -O3 -march=native -g -o out\sprof.exe sprof.c -lwinmm
if ($LASTEXITCODE) { throw 'build failed' }
# sprof.exe writes sprof_out.txt to the current folder; the .py scripts read it and sprof.exe from there
Push-Location out
try {
    .\sprof.exe $N $Reps
    python ..\sprof.py
    python ..\busy.py
} finally {
    Pop-Location
}
