# Evaluation stage. Runs alongside training: each learned method is evaluated as soon as its training has
# finished (checked every 10 min); classical methods start once validation tuning has written its parameters.
#   nominal (all test slices, reconstructions saved for figures)
#   perturbations + distribution shifts on a fixed 25 % slice subset (every 4th slice, all 15 test volumes),
#   identical for every method
#   diffusion samplers: step count chosen on validation first (bench.select_steps); score-MRI (~6 min/slice)
#   is evaluated on the 25 % subset only
# Re-running is safe: finished result files are skipped.
# Usage:  powershell -ExecutionPolicy Bypass -File .\scripts\evaluate_experiment.ps1 [-Anatomy knee]
param([string]$Anatomy = "knee")
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force logs | Out-Null
$log = "evaluate_$Anatomy.log"

function Run([string[]]$cmd) {
    Write-Host "$(Get-Date -Format 'dd-MM HH:mm')  python $($cmd -join ' ')" -ForegroundColor Cyan
    cmd /c "python -u $($cmd -join ' ') >> logs\$log 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "failed (exit $LASTEXITCODE): python $($cmd -join ' ')  (see logs\$log)" }
}

function Done($method, $setting) { Test-Path "results\$Anatomy\per_slice\${method}__$setting.csv" }

function TrainingDone($m) {
    $dir = "runs\$Anatomy\$m"
    if (-not (Test-Path "$dir\log.csv") -or -not (Test-Path "$dir\best.pt")) { return $false }
    $conf = Get-Content "$dir\config.json" -Raw | ConvertFrom-Json
    $rows = @(Import-Csv "$dir\log.csv")
    if ($m -eq "scoremri") { return [int]$rows[-1].step -ge [int]$conf.train.steps }
    return [int]$rows[-1].epoch -ge ([int]$conf.train.epochs - 1)
}

function WaitFor($what, [scriptblock]$ready) {
    while (-not (& $ready)) {
        Write-Host "$(Get-Date -Format 'dd-MM HH:mm')  waiting for $what ..." -ForegroundColor DarkGray
        Start-Sleep -Seconds 600
    }
}

$perturb = @("noise:0.005", "noise:0.01", "noise:0.02", "noise:0.04", "motion:1", "motion:2", "motion:4",
             "dropout:0.05", "dropout:0.1", "dropout:0.2", "spike:2", "spike:5", "spike:10")

function Evaluate($m, [string[]]$extra, [bool]$nominalFull = $true) {
    if (-not (Done $m "random_nominal")) {
        $stride = if ($nominalFull) { @() } else { @("--slice-stride", "4") }
        Run (@("-m", "bench.evaluate", "--method", $m, "--anatomy", $Anatomy, "--save-recon") + $stride + $extra)
    }
    $todo = @($perturb | Where-Object { -not (Done $m ("random_" + $_.Replace(":", "-"))) })
    if ($todo.Count) {
        Run (@("-m", "bench.evaluate", "--method", $m, "--anatomy", $Anatomy, "--slice-stride", "4", "--conditions") + $todo + $extra)
    }
    if (-not (Done $m "equispaced_nominal")) {
        Run (@("-m", "bench.evaluate", "--method", $m, "--anatomy", $Anatomy, "--slice-stride", "4", "--mask-type", "equispaced") + $extra)
    }
    if (-not (Done $m "random_nominal_R12")) {
        Run (@("-m", "bench.evaluate", "--method", $m, "--anatomy", $Anatomy, "--slice-stride", "4", "--accels", "12") + $extra)
    }
}

WaitFor "classical tuning" { Test-Path "results\$Anatomy\classical_params.json" }
foreach ($m in "zf", "grappa", "sense", "cs") { Evaluate $m @("--workers", "3") }

foreach ($m in "unet", "varnet", "modl", "rim", "swinmr", "kcd") {
    WaitFor "$m training" { TrainingDone $m }
    if ($m -eq "kcd" -and -not (Test-Path "runs\$Anatomy\kcd\inference.json")) {
        Run @("-m", "bench.select_steps", "--model", "kcd", "--anatomy", $Anatomy, "--steps", "25", "50", "100", "200", "1000")
    }
    Evaluate $m @()
}

WaitFor "score-MRI training" { TrainingDone "scoremri" }
if (-not (Test-Path "runs\$Anatomy\scoremri\inference.json")) {
    Run @("-m", "bench.select_steps", "--model", "scoremri", "--anatomy", $Anatomy, "--steps", "100", "200", "500")
}
if (-not (Done "scoremri" "random_nominal")) {
    Run @("-m", "bench.evaluate", "--method", "scoremri", "--anatomy", $Anatomy, "--slice-stride", "4", "--save-recon")
}

Run @("-m", "bench.stats", "--anatomy", $Anatomy)
Run @("-m", "bench.stats", "--anatomy", $Anatomy, "--reference", "sense")
Run @("-m", "bench.stats", "--anatomy", $Anatomy, "--reference", "whole")
Write-Host "`nEvaluation finished: results\$Anatomy\tables" -ForegroundColor Green
