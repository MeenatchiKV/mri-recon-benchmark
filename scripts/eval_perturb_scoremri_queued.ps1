# Waits for the main perturbation script (scripts/eval_perturb.ps1) to write its last file (CS at R12), then runs
# score-MRI on every 16th test slice with per-slice caching, so the two never share the GPU.
# If CS fails, this keeps waiting: press Ctrl-C and run the second command by hand.
Set-Location (Split-Path $PSScriptRoot -Parent)
$last = "results\knee\per_slice\cs__random_nominal_R12.csv"
Write-Host "$(Get-Date -Format 'dd-MM HH:mm')  waiting for $last" -ForegroundColor Cyan
while (-not (Test-Path $last)) { Start-Sleep 300 }
Start-Sleep 60
& "$PSScriptRoot\eval_perturb.ps1" -Anatomy knee -Methods scoremri -Stride 16 -Cache
