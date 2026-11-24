# Knee benchmark: data preparation + training queue (30/5/15 volume split).
# Classical-method tuning runs on the CPU in parallel with GPU training.
# Re-running is safe: ESPIRiT skips finished volumes and every training run resumes from last.pt.
# Usage:  powershell -ExecutionPolicy Bypass -File .\scripts\run_experiment.ps1 [-Anatomy knee]
param([string]$Anatomy = "knee")
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force logs | Out-Null

function Run($log, [string[]]$cmd) {
    # Output goes only to the log file: streaming tqdm progress through the console pipe stalls the process
    # once the pipe fills. Watch progress with:  Get-Content logs\<log> -Wait -Tail 20
    Write-Host "$(Get-Date -Format 'dd-MM HH:mm')  python $($cmd -join ' ')  ->  logs\$log" -ForegroundColor Cyan
    cmd /c "python -u $($cmd -join ' ') >> logs\$log 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "failed (exit $LASTEXITCODE): python $($cmd -join ' ')  (see logs\$log)" }
    Write-Host "$(Get-Date -Format 'dd-MM HH:mm')  done" -ForegroundColor Green
}

if (-not (Test-Path "data\$Anatomy\split.json")) {
    Run preprocess.log @("-m", "bench.preprocess", "--anatomy", $Anatomy, "--workers", "3")
}
Run espirit.log @("-m", "bench.espirit", "--anatomy", $Anatomy)
Run espirit.log @("-m", "bench.espirit", "--anatomy", $Anatomy, "--accel", "12", "--split", "test")

$tune = Start-Process python -ArgumentList "-u", "-m", "bench.tune", "--anatomy", $Anatomy, "--workers", "6" `
    -RedirectStandardOutput "logs\tune_$Anatomy.log" -RedirectStandardError "logs\tune_$Anatomy.err" -NoNewWindow -PassThru

foreach ($m in "unet", "varnet", "modl", "rim", "swinmr") {
    Run "train_${Anatomy}_${m}.log" @("-m", "bench.train", "--model", $m, "--anatomy", $Anatomy, "--resume")
}
# KCD validates with its 50-step sampler on 40 validation items per epoch (the full set would take ~45 min)
Run "train_${Anatomy}_kcd.log" @("-m", "bench.train", "--model", "kcd", "--anatomy", $Anatomy, "--resume", "--max-val", "40")
Run "train_${Anatomy}_scoremri.log" @("-m", "bench.train_score", "--anatomy", $Anatomy, "--resume")

$tune.WaitForExit()
Write-Host "`nAll $Anatomy training finished. Next: evaluation (see README)." -ForegroundColor Green
