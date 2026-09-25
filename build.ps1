# Builds fastfib.exe from fastfib.c (which includes src/).
# usage: .\build.ps1           build
#        .\build.ps1 -Test     build, then run the test sweep and the large-n check
#        .\build.ps1 -Single   also write dist\fastfib.c, the one-file version
param([switch]$Test, [switch]$Single)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$gcc = if ($env:CC) { $env:CC } else { 'gcc' }

& $gcc -O3 -march=native -Wall -o fastfib.exe fastfib.c
if ($LASTEXITCODE) { throw 'build failed' }
'built fastfib.exe'

if ($Single) {
    & "$PSScriptRoot\tools\amalgamate.ps1"
}
if ($Test) {
    & powershell -File "$PSScriptRoot\tests\test.ps1"
    if ($LASTEXITCODE) { throw 'tests failed' }
    & powershell -File "$PSScriptRoot\tests\bigcheck.ps1"
    if ($LASTEXITCODE) { throw 'bigcheck failed' }
}
