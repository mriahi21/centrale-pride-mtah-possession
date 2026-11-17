# One-command LOCAL full-bundle workflow. Run it in your own PowerShell (outside Claude), from anywhere:
#   powershell -ExecutionPolicy Bypass -File scripts\run_full_bundle.ps1
# Optional: -DataRoot D:\mtah  -Team "Centrale Pride"  -SkipTests  -VideoBudgetMin 40  -NoVideoPass
# Re-runnable: completed work (alignment, held-out study, video activity, per-window detections) is cached under <DataRoot>\cache,
# so a second run only does what is missing; the broadcast-video pass resumes where its time budget stopped.
# What it does: backs up the validated v1 file, installs the pinned local tooling, adopts completed browser downloads
# from Downloads into the data root, verifies every asset (size), inspects/decodes them (tactical video, audio,
# homography, broadcast tracking), verifies each signal, runs v1 / v2 / v3 + ablations, validates + hashes every
# candidate, and writes a LOCAL report. Nothing leaves your machine; do not paste the report into hosted tools.
param(
    [string]$DataRoot = $(if ($env:MTAH_DATA_DIR) { $env:MTAH_DATA_DIR } else { "data" }),
    [string]$Team = "Centrale Pride",
    [switch]$SkipTests,
    [double]$VideoBudgetMin = 40,
    [switch]$NoVideoPass
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$reports = Join-Path $DataRoot "reports"
New-Item -ItemType Directory -Force $reports | Out-Null

Write-Host "== 0. free disk space and validated-v1 backup"
Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Name -eq (Split-Path -Qualifier $DataRoot).TrimEnd(':') } |
    ForEach-Object { "{0}: {1:N1} GB free" -f $_.Name, ($_.Free / 1GB) }
$v1 = "outputs\$Team.csv"
$bk = "outputs\frozen_v1\$Team.csv"
if ((Test-Path $v1) -and -not (Test-Path $bk)) {
    New-Item -ItemType Directory -Force outputs\frozen_v1 | Out-Null
    Copy-Item $v1 $bk
    Get-FileHash $bk -Algorithm SHA256 | Tee-Object outputs\frozen_v1\SHA256.txt
} elseif (Test-Path $bk) {
    Write-Host "backup already present:"; Get-FileHash $bk -Algorithm SHA256
}

Write-Host "== 1. local tooling (pinned)"
$PY = "python"                                   # fall back to the Windows launcher when `python` is only the Store stub
try { & python -c "import sys" } catch { $PY = "py" }
if ($LASTEXITCODE -ne 0) { $PY = "py" }
Write-Host "using interpreter: $PY"
& $PY -m pip install -r requirements-full.txt
if (-not $SkipTests) { & $PY -m pytest -q tests }

Write-Host "== 2. full-bundle run (inventory, signals, candidates, report)"
$env:PYTHONUNBUFFERED = "1"
$extra = @("--video-budget-min", "$VideoBudgetMin")
if ($NoVideoPass) { $extra += "--no-video-pass" }
& $PY -m poss.fullrun --data-root $DataRoot --team-name $Team --out outputs\full @extra |
    Tee-Object (Join-Path $reports "fullrun_console.log")

Write-Host ""
Write-Host "Report:   $reports\full_report.md  (and asset_report.csv, holdout_ball.md, video_pass.md)"
Write-Host "Candidates: outputs\full\<variant>\$Team.csv   (v1, v2, v3_final, ablations)"
Write-Host "Validate the exact file you will upload:  python evaluation\validate_submission.py `"outputs\full\v3_final\$Team.csv`""
