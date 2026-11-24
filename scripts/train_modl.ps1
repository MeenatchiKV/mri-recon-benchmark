# MoDL two-stage training, as in the reference code (github.com/hkaggarwal/modl, trn.py):
#   stage 1: K = 1 unrolled iteration, 50 epochs            -> runs/<anatomy>/modl_k1
#   stage 2: K = 10, initialised from the stage-1 best.pt     -> runs/<anatomy>/modl  (the benchmarked model)
# Re-running is safe: both stages resume from last.pt.
# Usage:  powershell -ExecutionPolicy Bypass -File .\scripts\train_modl.ps1 [-Anatomy knee]
param([string]$Anatomy = "knee")
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force logs | Out-Null

function Run($log, [string]$cmd) {
    Write-Host "$(Get-Date -Format 'dd-MM HH:mm')  python $cmd  ->  logs\$log" -ForegroundColor Cyan
    cmd /c "python -u $cmd >> logs\$log 2>&1"
    if ($LASTEXITCODE -ne 0) { Write-Host "failed (see logs\$log)" -ForegroundColor Red; exit 1 }
    Write-Host "$(Get-Date -Format 'dd-MM HH:mm')  done" -ForegroundColor Green
}

Run "train_${Anatomy}_modl_k1.log" "-m bench.train --model modl --anatomy $Anatomy --tag k1 --model-args {\`"iterations\`":1} --resume --workers 3"
Run "train_${Anatomy}_modl.log" "-m bench.train --model modl --anatomy $Anatomy --init-from runs\$Anatomy\modl_k1 --resume --workers 3"
