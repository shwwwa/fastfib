# Writes dist\fastfib.c: the whole program as one self-contained C file, by
# inlining every local #include "..." of fastfib.c once, in order (the
# same text the compiler sees in the normal build). System includes stay.
# usage: .\tools\amalgamate.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$seen = @{}
$out = New-Object System.Text.StringBuilder

function Inline([string]$path) {
    $full = [IO.Path]::GetFullPath($path)
    if ($seen.ContainsKey($full)) { return }
    $seen[$full] = $true
    $dir = Split-Path $full -Parent
    $rel = $full.Substring($root.Length + 1).Replace('\', '/')
    [void]$out.AppendLine("/* ======== $rel ======== */")
    foreach ($line in [IO.File]::ReadAllLines($full)) {
        if ($line -match '^\s*#pragma once\s*$') { continue }
        if ($line -match '^\s*#include "([^"]+)"') {
            Inline (Join-Path $dir $Matches[1])
            continue
        }
        [void]$out.AppendLine($line)
    }
}

Inline (Join-Path $root 'fastfib.c')
New-Item -ItemType Directory -Force (Join-Path $root 'dist') | Out-Null
$dst = Join-Path $root 'dist\fastfib.c'
[IO.File]::WriteAllText($dst, $out.ToString())
"wrote dist\fastfib.c ($(([IO.File]::ReadAllLines($dst)).Count) lines, $($seen.Count) files)"
