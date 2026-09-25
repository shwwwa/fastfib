# Command-line tests for fastfib.exe (src/main.c): argument parsing and order,
# --print and --help, the n prompt on stdin, the 64-bit edge (F(93) is the last
# Fibonacci number below 2^64), and exit code 1 with a message for bad input.
# usage: .\cli.ps1 [path\to\fastfib.exe]     (default ..\fastfib.exe)
param([string]$Exe = (Join-Path $PSScriptRoot '..\fastfib.exe'))

# Args: the arguments, split on spaces. Stdin: typed at the prompt. The output
# must contain Expect and must not contain Reject.
$cases = @(
    @{ Args = '10 --print';            Exit = 0; Expect = 'fibonacci(10) = 55' }
    @{ Args = '--print 10';            Exit = 0; Expect = 'fibonacci(10) = 55' }
    @{ Args = '0 --print';             Exit = 0; Expect = 'fibonacci(0) = 0' }
    @{ Args = '1 --print';             Exit = 0; Expect = 'fibonacci(1) = 1' }
    @{ Args = '2 --print';             Exit = 0; Expect = 'fibonacci(2) = 1' }
    @{ Args = '93 --print';            Exit = 0; Expect = 'fibonacci(93) = 12200160415121876738' }
    @{ Args = '94 --print';            Exit = 0; Expect = 'fibonacci(94) = 19740274219868223167' }
    @{ Args = '007 --print';           Exit = 0; Expect = 'fibonacci(7) = 13' }
    @{ Args = '1000';                  Exit = 0; Expect = 'Time:'; Reject = 'fibonacci(' }
    @{ Args = '--help';                Exit = 0; Expect = 'usage: fastfib' }
    @{ Args = '-h';                    Exit = 0; Expect = 'usage: fastfib' }
    @{ Args = '--print'; Stdin = '12';     Exit = 0; Expect = 'fibonacci(12) = 144' }
    @{ Args = '--print'; Stdin = '  12  '; Exit = 0; Expect = 'fibonacci(12) = 144' }
    @{ Args = '--print'; Stdin = 'abc';    Exit = 1; Expect = 'invalid n: abc' }
    @{ Args = '--print'; Stdin = '';       Exit = 1; Expect = 'invalid n' }
    @{ Args = '-5';                    Exit = 1; Expect = 'invalid argument: -5' }
    @{ Args = '+5';                    Exit = 1; Expect = 'invalid argument: +5' }
    @{ Args = '1e9';                   Exit = 1; Expect = 'invalid argument: 1e9' }
    @{ Args = '0x10';                  Exit = 1; Expect = 'invalid argument: 0x10' }
    @{ Args = '18446744073709551616';  Exit = 1; Expect = 'invalid argument' }  # 2^64
    @{ Args = '10 20';                 Exit = 1; Expect = 'invalid argument: 20' }
    @{ Args = '--bogus';               Exit = 1; Expect = 'usage: fastfib' }
)

$failed = 0
foreach ($c in $cases) {
    $argv = @($c.Args -split ' ')
    # stderr is merged in; in Windows PowerShell those lines arrive as error
    # records, so turn every line back into plain text
    $lines = if ($c.ContainsKey('Stdin')) { $c.Stdin | & $Exe @argv 2>&1 } else { & $Exe @argv 2>&1 }
    $code = $LASTEXITCODE
    $text = ($lines | ForEach-Object { "$_" }) -join "`n"
    $problems = @()
    if ($code -ne $c.Exit) { $problems += "exit code $code, expected $($c.Exit)" }
    if (-not $text.Contains($c.Expect)) { $problems += "missing '$($c.Expect)'" }
    if ($c.Reject -and $text.Contains($c.Reject)) { $problems += "unexpected '$($c.Reject)'" }
    if ($problems) {
        $failed++
        $stdin = if ($c.ContainsKey('Stdin')) { " (stdin '$($c.Stdin)')" } else { '' }
        "FAIL fastfib $($c.Args)$stdin : $($problems -join '; ')"
    }
}
"$($cases.Count) command lines, $failed failures"
exit [int]($failed -ne 0)
