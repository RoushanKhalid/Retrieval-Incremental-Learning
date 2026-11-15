# One-click launcher for the Thesis Multi-Model & Incremental Learning Streamlit demo
# Usage: Right-click -> "Run with PowerShell"  OR  .\scripts\launch_app.ps1

$ErrorActionPreference = "Stop"

# Robust working directory resolution: always resolve to repository root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (Test-Path (Join-Path $scriptDir "..\tcontext")) {
    $root = (Resolve-Path (Join-Path $scriptDir "..")).Path
} else {
    $root = (Resolve-Path $scriptDir).Path
}
Set-Location $root

# Suppress TF/oneDNN noise and slow initialisation logs
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:TRANSFORMERS_VERBOSITY = "error"
$env:TOKENIZERS_PARALLELISM = "false"

Write-Host ""
Write-Host "=== Thesis Multi-Model & Incremental Retrieval Demo Launcher ===" -ForegroundColor Cyan
Write-Host "Resolving Python runtime..."

function Get-PythonPath {
    # 1. Active virtual environment
    if ($env:VIRTUAL_ENV -and (Test-Path "$env:VIRTUAL_ENV\Scripts\python.exe")) {
        return "$env:VIRTUAL_ENV\Scripts\python.exe"
    }
    # 2. Standard Windows Python installations
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:ProgramFiles\Python311\python.exe",
        "$env:ProgramFiles\Python312\python.exe",
        "C:\Users\DSLAB\AppData\Local\Programs\Python\Python311\python.exe"
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
Write-Host "Using Python: $pythonExe" -ForegroundColor Gray
Write-Host "Launching Streamlit app on http://localhost:8501"

# Safely free port 8501 if held by previous Streamlit/Python process
$occupied = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
if ($occupied) {
    $occupied | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
        $procObj = Get-Process -Id $_ -ErrorAction SilentlyContinue
        if ($procObj -and ($procObj.ProcessName -match "python|streamlit")) {
            Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Seconds 1
}

# Launch streamlit using module invocation (-m streamlit) from resolved pythonExe
$proc = Start-Process -FilePath $pythonExe `
    -ArgumentList @("-m", "streamlit", "run", "tcontext\web_app.py", "--server.port", "8501", "--server.headless", "false") `
    -PassThru

# Wait up to 20 s for port to open, then open browser
$ready = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 1
    if (Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue) {
        $ready = $true
        break
    }
}

if ($ready) {
    Write-Host "App is ready -> opening http://localhost:8501" -ForegroundColor Green
    Start-Process "http://localhost:8501"
} else {
    Write-Host "App may still be starting. Check http://localhost:8501 in your browser." -ForegroundColor Yellow
}

# Keep the window open so Streamlit logs are visible
Wait-Process -Id $proc.Id

