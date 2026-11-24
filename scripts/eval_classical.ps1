# Evaluate one classical method (or ceiling row) on all knee test slices, nominal condition.
# Usage:  powershell -ExecutionPolicy Bypass -File .\scripts\eval_classical.ps1 zf [full_rss full_sense ...]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Methods)
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach ($m in $Methods) {
    Write-Host "$(Get-Date -Format 'dd-MM HH:mm')  evaluating $m  ->  logs\evaluate_knee_$m.log" -ForegroundColor Cyan
    $save = if ($m -like "full_*") { "" } else { "--save-recon" }
    cmd /c "python -u -m bench.evaluate --method $m --anatomy knee --workers 6 $save >> logs\evaluate_knee_$m.log 2>&1"
    if ($LASTEXITCODE -ne 0) { Write-Host "failed: $m (see logs\evaluate_knee_$m.log)" -ForegroundColor Red; exit 1 }
    Write-Host "$(Get-Date -Format 'dd-MM HH:mm')  done $m" -ForegroundColor Green
}
