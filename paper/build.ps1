# ============================================================
# AURA paper build script (xelatex + bibtex, no perl required)
#
# Usage (from project root d:\py\AURA):
#     powershell -ExecutionPolicy Bypass -File paper\build.ps1
# Options:
#     -Clean    remove build artifacts
#     -Open     open the compiled PDF after build
# ============================================================
param(
    [string]$Main   = "main.tex",
    [string]$OutDir = "build",
    [switch]$Clean,
    [switch]$Open
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$stem = [IO.Path]::GetFileNameWithoutExtension($Main)
$outPath = Join-Path $here $OutDir

Push-Location $here
try {
    if (-not (Test-Path $outPath)) { New-Item -ItemType Directory -Path $outPath | Out-Null }

    if ($Clean) {
        Get-ChildItem -Path $outPath -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne ".gitkeep" } | Remove-Item -Force
        Write-Output "[clean] removed build artifacts in $outPath"
        return
    }

    $xArgs = @("-interaction=nonstopmode", "-file-line-error", "-output-directory=$OutDir")

    Write-Output "[1/4] xelatex pass 1 ..."
    & xelatex @xArgs $Main | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "xelatex pass 1 failed (exit $LASTEXITCODE)" }

    Write-Output "[2/4] bibtex ..."
    $hasBib = Test-Path (Join-Path $here "refs.bib")
    if ($hasBib) {
        Push-Location $outPath
        try {
            # bibtex reads build/$stem.aux; refs.bib is resolved via BIBINPUTS
            $env:BIBINPUTS = "$here;$here\..;$here\$OutDir"
            & bibtex $stem | Out-Null
            # bibtex returns 1 on warnings (e.g. empty bib), not fatal -> ignore
        }
        finally { Pop-Location }
    } else {
        Write-Output "      (no refs.bib, skipped)"
    }

    Write-Output "[3/4] xelatex pass 2 ..."
    & xelatex @xArgs $Main | Out-Null
    Write-Output "[4/4] xelatex pass 3 ..."
    & xelatex @xArgs $Main | Out-Null

    $pdf = Join-Path $outPath "$stem.pdf"
    if (Test-Path $pdf) {
        $size = [math]::Round((Get-Item $pdf).Length / 1KB, 1)
        Write-Output "[ok] $pdf ($size KB)"
        if ($Open) { Invoke-Item $pdf }
    } else {
        throw "build finished but $pdf not found - check $outPath\$stem.log"
    }
}
finally {
    Pop-Location
}
