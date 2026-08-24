# packaging/build.ps1 -- orchestrates the full build: PyInstaller (--onedir)
# then Inno Setup, producing dist\Jarvis-Setup.exe.
#
# Run from anywhere: powershell -File packaging\build.ps1
# Requires: pip install -r jarvis\requirements.txt (pulls in pyinstaller),
# and Inno Setup's iscc.exe on PATH or at its default install location
# (https://jrsoftware.org/isdl.php -- free, not a pip package).

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "== Building with PyInstaller (--onedir) =="
pyinstaller (Join-Path $repoRoot "packaging\jarvis_gui.spec") --noconfirm `
    --distpath (Join-Path $repoRoot "dist") --workpath (Join-Path $repoRoot "build")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed (exit $LASTEXITCODE)" }

$iscc = (Get-Command "iscc.exe" -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    $candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:LocalAppData "Programs\Inno Setup 6\ISCC.exe")  # winget --scope user default
    )
    $iscc = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $iscc) {
        throw "Inno Setup compiler (iscc.exe) not found. Install it (free) from " + `
              "https://jrsoftware.org/isdl.php, then re-run this script."
    }
}

Write-Host "== Compiling installer with Inno Setup =="
& $iscc (Join-Path $repoRoot "packaging\installer.iss")
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compile failed (exit $LASTEXITCODE)" }

Write-Host "== Done: dist\Jarvis-Setup.exe =="
