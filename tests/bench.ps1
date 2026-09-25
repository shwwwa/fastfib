# Best-of-N wall times of fastfib.exe builds.
# usage: bench.ps1 [-Exes a.exe,b.exe] [-Ns 1000000,...] [-Reps 7]
param(
    [string[]]$Exes = @((Join-Path $PSScriptRoot '..\fastfib.exe')),
    [string[]]$Ns = @(200000, 1000000, 5000000, 20000000, 100000000),
    [int]$Reps = 7
)
Set-Location $PSScriptRoot
# powershell -File passes a list as one comma-joined string
$Exes = $Exes | ForEach-Object { $_ -split ',' }
$Ns = $Ns | ForEach-Object { "$_" -split ',' } | ForEach-Object { [long]$_ }
foreach ($e in $Exes) {
    $row = foreach ($n in $Ns) {
        $best = [double]::MaxValue
        for ($r = 0; $r -lt $Reps; $r++) {
            $line = @(& $e $n) -match 'Time' | Select-Object -First 1
            $t = [double]($line -replace '[^0-9.]', '')
            if ($t -lt $best) { $best = $t }
        }
        '{0}: {1:N2} ms' -f $n, ($best * 1000)
    }
    '{0,-22} {1}' -f (Split-Path $e -Leaf), ($row -join ' | ')
}
