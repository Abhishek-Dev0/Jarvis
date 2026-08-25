# packaging/build.ps1 -- orchestrates the full build: PyInstaller (--onedir)
# then Inno Setup, producing dist\Jarvis-Setup.exe.
#
# Run from anywhere: powershell -File packaging\build.ps1
# Requires: pip install -r jarvis\requirements.txt (pulls in pyinstaller),
# and Inno Setup's iscc.exe on PATH or at its default install location
# (https://jrsoftware.org/isdl.php -- free, not a pip package).

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

# jarvis/data/models/ (Piper + Kokoro TTS weights, ~490MB) is gitignored --
# too large to commit to git history. It's hosted instead as a GitHub
# Release asset (tag "assets-tts-v1", not a "v*" app-version tag, so it
# never triggers release.yml). A dev machine that already has the weights
# on disk (the normal case) skips this entirely; only a fresh checkout
# (CI) needs to fetch them. Requires `gh` authenticated (CI's built-in
# GITHUB_TOKEN covers this; see release.yml).
$modelsDir = Join-Path $repoRoot "jarvis\data\models"
$havePiper = Test-Path (Join-Path $modelsDir "piper") -PathType Container
$haveKokoro = Test-Path (Join-Path $modelsDir "kokoro") -PathType Container
if (-not ($havePiper -and $haveKokoro)) {
    Write-Host "== Voice model weights missing locally -- fetching from release asset =="
    $zipPath = Join-Path $repoRoot "jarvis-voice-models.zip"
    gh release download assets-tts-v1 --repo Abhishek-Dev0/Jarvis --pattern "*.zip" --dir $repoRoot --clobber
    if ($LASTEXITCODE -ne 0) { throw "Failed to download voice model weights (exit $LASTEXITCODE)" }
    New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
    Expand-Archive -Path $zipPath -DestinationPath $modelsDir -Force
    Remove-Item $zipPath
}

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
