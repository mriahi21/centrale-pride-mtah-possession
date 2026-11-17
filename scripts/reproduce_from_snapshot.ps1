# Regenerate the submission CSV from a CLEAN COPY of the code-only snapshot, validate it, and compare SHA-256 hashes.
#   powershell -ExecutionPolicy Bypass -File scripts\reproduce_from_snapshot.ps1 -Snapshot public_snapshot `
#       -DataRoot <DATA> -Profile v3 -VideoPass -ExpectedSha <sha256 of the chosen candidate>
# -Snapshot may be a folder (public_snapshot) or a git URL (the public repo, to prove the published code reproduces the file).
# -FreshCache additionally moves <DataRoot>\cache aside so everything is recomputed from the raw inputs (slow: about an hour).
param(
    [Parameter(Mandatory = $true)][string]$Snapshot,
    [string]$DataRoot = $(if ($env:MTAH_DATA_DIR) { $env:MTAH_DATA_DIR } else { "data" }),
    [string]$Team = "Centrale Pride",
    [string]$Profile = "v3",
    [switch]$VideoPass,
    [string]$ExpectedSha = "",
    [switch]$FreshCache,
    [string]$OutDir = ""
)
$ErrorActionPreference = "Stop"
$PY = "python"
try { & python -c "import sys" } catch { $PY = "py" }
if ($LASTEXITCODE -ne 0) { $PY = "py" }

if ($OutDir) { New-Item -ItemType Directory -Force $OutDir | Out-Null; $OutDir = (Resolve-Path $OutDir).Path }
$work = Join-Path $env:TEMP ("mtah_repro_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
if ($Snapshot -match "^(https?://|git@)") {
    git clone --depth 1 $Snapshot $work
} else {
    Copy-Item -Recurse (Resolve-Path $Snapshot) $work
}
Set-Location $work
Write-Host "clean copy: $work"
if (Test-Path "$DataRoot\cache") { Write-Host "(cache present: reused)" }
if ($FreshCache -and (Test-Path "$DataRoot\cache")) {
    Move-Item "$DataRoot\cache" ("$DataRoot\cache_aside_" + (Get-Date -Format "HHmmss"))
    Write-Host "cache moved aside: recomputing from raw inputs"
}
& $PY -m pip install -r requirements-full.txt
$flags = @("--data-dir", $DataRoot, "--team-name", $Team, "--profile", $Profile, "--out", "outputs\repro")
if ($VideoPass) { $flags += "--video-pass" }
$env:PYTHONUNBUFFERED = "1"
& $PY -m poss.make_submission @flags
$csv = Join-Path $work "outputs\repro\$Team.csv"
& $PY evaluation\validate_submission.py $csv
$sha = (Get-FileHash $csv -Algorithm SHA256).Hash.ToLower()
Write-Host ""
Write-Host "regenerated: $csv"
Write-Host "SHA-256:     $sha"
if ($ExpectedSha) {
    if ($sha -ne $ExpectedSha.ToLower()) { Write-Host "HASH MISMATCH: do NOT mark reproduction complete; diagnose first" -ForegroundColor Red; exit 2 }
    Write-Host "HASH MATCH: reproduction confirmed" -ForegroundColor Green
}
if ($OutDir) { Copy-Item $csv (Join-Path $OutDir "$Team.csv") -Force; Write-Host "copied to $OutDir\$Team.csv" }
