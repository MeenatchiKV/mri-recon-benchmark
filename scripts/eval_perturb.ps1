# Perturbation and distribution-shift evaluation on the fixed 25 % test-slice subset (every 4th slice of all
# 15 test volumes; identical for every method):
#   13 k-space perturbations (bench.perturb.CONDITIONS), equispaced masks, unseen R = 12.
# Under perturbations, map-based methods get ESPIRiT maps re-estimated from the perturbed ACS (bench.evaluate).
# Methods run fastest first; score-MRI is not included (113 s/slice; run separately on a smaller subset).
# Re-running is safe: finished result files are skipped; a failing method is reported and the next one starts.
# Usage:  powershell -ExecutionPolicy Bypass -File .\scripts\eval_perturb.ps1 [-Anatomy knee] [-Methods zf,unet]
#         score-MRI: ... -Methods scoremri -Stride 16 -Cache   (every 16th slice; resumable per slice)
param([string]$Anatomy = "knee",
      [string[]]$Methods = @("zf", "grappa", "unet", "varnet", "modl", "rim", "kcd", "swinmr", "sense", "cs"),
      [int]$Stride = 4,
      [switch]$Cache)
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force logs | Out-Null
$log = "eval_perturb_$Anatomy.log"
$classical = @("zf", "grappa", "sense", "cs")
$perturb = @("noise:0.005", "noise:0.01", "noise:0.02", "noise:0.04", "motion:1", "motion:2", "motion:4",
             "dropout:0.05", "dropout:0.1", "dropout:0.2", "spike:2", "spike:5", "spike:10")

function Done($m, $setting) { Test-Path "results\$Anatomy\per_slice\${m}__$setting.csv" }

function Run($m, [string[]]$cmd) {
    Write-Host "$(Get-Date -Format 'dd-MM HH:mm')  $m  python $($cmd -join ' ')" -ForegroundColor Cyan
    cmd /c "python -u $($cmd -join ' ') >> logs\$log 2>&1"
    if ($LASTEXITCODE -ne 0) { Write-Host "  FAILED (exit $LASTEXITCODE), see logs\$log" -ForegroundColor Red; return $false }
    return $true
}

foreach ($m in $Methods) {
    $base = @("-m", "bench.evaluate", "--method", $m, "--anatomy", $Anatomy, "--slice-stride", "$Stride")
    if ($Cache) { $base += @("--cache") }
    if ($classical -contains $m) { $base += @("--workers", "3") }
    $todo = @($perturb | Where-Object { -not (Done $m ("random_" + $_.Replace(":", "-"))) })
    if ($todo.Count) { if (-not (Run $m ($base + @("--conditions") + $todo))) { continue } }
    if (-not (Done $m "equispaced_nominal")) { if (-not (Run $m ($base + @("--mask-type", "equispaced")))) { continue } }
    if (-not (Done $m "random_nominal_R12")) { [void](Run $m ($base + @("--accels", "12"))) }
    Write-Host "$(Get-Date -Format 'dd-MM HH:mm')  $m done" -ForegroundColor Green
}
Write-Host "$(Get-Date -Format 'dd-MM HH:mm')  all done" -ForegroundColor Green
