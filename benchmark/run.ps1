<#
.SYNOPSIS
    Run every benchmark/*.vpy speed test with `vspipe file.vpy .` and print the fps details.

.DESCRIPTION
    Each *.vpy renders one (filter, implementation) on a filter-bound 1080p YUV420 source.
    This driver sweeps the sample format (u16/f16/f32) and, for filters that expose a
    num_streams knob (vszipcl / bilateralgpu / nlm_cuda), num_streams 1 and 2. Format and
    stream count are passed to the scripts via environment variables (BENCH_FMT / BENCH_NS /
    BENCH_FRAMES / BENCH_THREADS). Unsupported (impl, format) combos error in vspipe and are
    reported as N/A (e.g. bilateralgpu & nlm_cuda reject f16; the CPU vszip EEDI3 is f32-only).

.EXAMPLE
    pwsh benchmark/run.ps1
    pwsh benchmark/run.ps1 -Frames 1000 -Formats u16,f32
    pwsh benchmark/run.ps1 -Csv benchmark/results.csv
#>
[CmdletBinding()]
param(
    [string[]]$Formats = @('u16', 'f16', 'f32'),
    [int]$Frames = 2000,
    [int]$Threads = 8,
    [int[]]$Streams = @(1, 2),
    [string]$Csv
)

$ErrorActionPreference = 'Continue'
# Force '.' decimals regardless of the host locale (vspipe prints '.', and comma decimals would
# break the CSV export too).
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
$bench = $PSScriptRoot
if (-not (Get-Command vspipe -ErrorAction SilentlyContinue)) {
    throw "vspipe not found on PATH. Install VapourSynth (it ships vspipe) and retry."
}

# Implementations that accept a num_streams argument (swept at $Streams); everything else runs once.
$streamed = @('vszipcl', 'bilateralgpu', 'nlm_cuda')

$vpys = Get-ChildItem -Path $bench -Filter *.vpy | Sort-Object Name
Write-Host ("vszipcl speed test | 1080p YUV420 filter-bound | {0} frames | threads={1} | formats {2}" -f $Frames, $Threads, ($Formats -join ', ')) -ForegroundColor Cyan
Write-Host ("{0} scripts x formats x num_streams. Rendered with ``vspipe file.vpy .`` (output discarded); fps is vspipe's own.`n" -f $vpys.Count)

$results = New-Object System.Collections.Generic.List[object]

foreach ($vpy in $vpys) {
    $base = $vpy.BaseName
    $parts = $base -split '_', 2
    $filter = $parts[0]
    $impl = if ($parts.Count -gt 1) { $parts[1] } else { $base }
    # 'none' sentinel for non-streamed impls: a single-element @($null) collapses to scalar $null
    # through the if-expression, and `foreach ($x in $null)` iterates ZERO times (skipping the row).
    $nsList = if ($streamed -contains $impl) { $Streams } else { @('none') }

    foreach ($fmt in $Formats) {
        foreach ($ns in $nsList) {
            $hasNs = $ns -ne 'none'
            $env:BENCH_FMT = $fmt
            $env:BENCH_FRAMES = "$Frames"
            $env:BENCH_THREADS = "$Threads"
            $env:BENCH_NS = if ($hasNs) { "$ns" } else { '1' }
            $nsLabel = if ($hasNs) { "s$ns" } else { '-' }

            $out = & vspipe $vpy.FullName . 2>&1 | Out-String

            if ($out -match 'Output (\d+) frames in ([\d.]+) seconds \(([\d.]+) fps\)') {
                $fps = [double]$matches[3]
                $sec = [double]$matches[2]
                Write-Host ("{0,-13} {1,-13} {2,-4} {3,-3} -> {4,9:N1} fps   ({5} frames, {6:N2}s)" -f $filter, $impl, $fmt, $nsLabel, $fps, $matches[1], $sec)
            }
            else {
                $fps = $null
                $reason = if ($out -match 'Error:\s*(.+)') { $matches[1].Trim() } else { 'render failed' }
                if ($reason.Length -gt 50) { $reason = $reason.Substring(0, 50) + '...' }
                Write-Host ("{0,-13} {1,-13} {2,-4} {3,-3} -> {4,9}       ({5})" -f $filter, $impl, $fmt, $nsLabel, 'N/A', $reason) -ForegroundColor DarkGray
            }

            $results.Add([pscustomobject]@{ Filter = $filter; Impl = $impl; Format = $fmt; NS = $nsLabel; FPS = $fps })
        }
    }
}

Remove-Item Env:BENCH_FMT, Env:BENCH_NS, Env:BENCH_FRAMES, Env:BENCH_THREADS -ErrorAction SilentlyContinue

Write-Host "`n===== SUMMARY (fps, higher is better; N/A = format not accepted by that filter) =====" -ForegroundColor Cyan
$results | Sort-Object Filter, Impl, Format, NS |
    Format-Table Filter, Impl, Format, NS,
        @{ Name = 'FPS'; Align = 'Right'; Expression = { if ($null -eq $_.FPS) { 'N/A' } else { '{0:N1}' -f $_.FPS } } } -AutoSize

if ($Csv) {
    $results | Select-Object Filter, Impl, Format, NS, FPS | Export-Csv -Path $Csv -NoTypeInformation
    Write-Host ("Wrote {0}" -f $Csv) -ForegroundColor Green
}
