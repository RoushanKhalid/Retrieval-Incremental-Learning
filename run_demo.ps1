param(
    [int]$Seed = 777,
    [int]$Epochs = 20,
    [string]$QueryImage = "",
    [int]$Port = 8501,
    [switch]$ForceRebuild = $false
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (!(Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runLog = "logs\thesis_benchmark_${timestamp}.log"
$latestRunLog = "logs\thesis_benchmark_latest.log"
$summaryPath = "tcontext\artifacts\benchmark_summary.json"
$cudaCsv = "tcontext\artifacts\benchmark_results_cuda_0.csv"
$vectorDbRoot = "tcontext\vector_db"

Write-Host "== Thesis Multi-Model Benchmarking & Demo Pipeline ==" -ForegroundColor Cyan
Write-Host "Random Seed: $Seed"
Write-Host ""

function Get-PythonPath {
    $candidates = @(
        "C:\Users\DSLAB\AppData\Local\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:ProgramFiles\Python311\python.exe"
    )
    foreach ($cand in $candidates) {
        if (Test-Path $cand) { return $cand }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notmatch "WindowsApps") {
        return $cmd.Source
    }
    return "python"
}

$pythonExe = Get-PythonPath
Write-Host "Using Python Runtime: $pythonExe" -ForegroundColor Gray

function Invoke-PythonScript {
    param(
        [string]$ScriptPath,
        [string[]]$Arguments,
        [string]$StdOutPath,
        [string]$StepName
    )
    Write-Host "Executing $StepName..." -ForegroundColor Yellow
    $stderrPath = "$StdOutPath.err.log"
    $allArgs = @($ScriptPath) + $Arguments
    $proc = Start-Process -FilePath $pythonExe -ArgumentList $allArgs -RedirectStandardOutput $StdOutPath -RedirectStandardError $stderrPath -Wait -PassThru
    $stderrText = if (Test-Path $stderrPath) { (Get-Content -Path $stderrPath -Raw).Trim() } else { "" }
    if ($stderrText) {
        Add-Content -Path $StdOutPath -Value "`n[stderr]`n$stderrText"
    }
    $null = Remove-Item -Path $stderrPath -Force -ErrorAction SilentlyContinue
    if ($proc.ExitCode -ne 0) {
        Get-Content -Path $StdOutPath -Tail 50
        throw "$StepName failed with exit code $($proc.ExitCode). Check log: $StdOutPath"
    }
}

$hasBenchmarkResults = (Test-Path $summaryPath) -and (Test-Path $cudaCsv)

if ($ForceRebuild -or -not $hasBenchmarkResults) {
    Write-Host "Running End-to-End Multi-Model Benchmark (Seed: $Seed)..." -ForegroundColor Cyan
    Invoke-PythonScript -ScriptPath "tcontext\experiments\run_benchmark.py" -Arguments @("--seed", "$Seed") -StdOutPath $runLog -StepName "multi-model benchmark pipeline"
    Copy-Item -Path $runLog -Destination $latestRunLog -Force
} else {
    Write-Host "Verified existing benchmark summary and artifacts in tcontext\artifacts." -ForegroundColor Green
}

# Free port 8501
$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listeners) {
    $pidsToStop = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($pidToStop in $pidsToStop) {
        Stop-Process -Id $pidToStop -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

Write-Host "Launching Streamlit Thesis Dashboard on Port $Port..." -ForegroundColor Cyan
$streamlitArgs = @("-m", "streamlit", "run", "tcontext\web_app.py", "--server.port", "$Port", "--server.headless", "false")
$streamlitProc = Start-Process -FilePath $pythonExe -ArgumentList $streamlitArgs -PassThru

$maxWaitSeconds = 25
$isReady = $false
for ($i = 0; $i -lt $maxWaitSeconds; $i++) {
    Start-Sleep -Seconds 1
    $ready = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($ready) {
        $isReady = $true
        break
    }
}

if ($isReady) {
    $url = "http://localhost:$Port"
    Write-Host "Dashboard online: $url" -ForegroundColor Green
    Start-Process $url | Out-Null
} else {
    Write-Warning "Dashboard did not open port within $maxWaitSeconds seconds. Check logs."
}

Wait-Process -Id $streamlitProc.Id
