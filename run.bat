@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: Thesis Research Framework: One-Click Unified Launcher (Windows)
:: "A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration"
:: ============================================================================

title Thesis Multi-Modal Incremental Learning Framework

cd /d "%~dp0"

:: Suppress framework noise
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=3
set TRANSFORMERS_VERBOSITY=error
set TOKENIZERS_PARALLELISM=false

echo.
echo =========================================================================
echo   Multi-Modal Retrieval-Incremental Learning Framework
echo   Interactive Research Dashboard and Experiment Launcher
echo =========================================================================
echo.

:: 1. Resolve Python executable
set "PY_CMD="

if defined VIRTUAL_ENV (
    if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "PY_CMD=%VIRTUAL_ENV%\Scripts\python.exe"
)

if not defined PY_CMD (
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    ) else if exist "%ProgramFiles%\Python311\python.exe" (
        set "PY_CMD=%ProgramFiles%\Python311\python.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    ) else if exist "%ProgramFiles%\Python312\python.exe" (
        set "PY_CMD=%ProgramFiles%\Python312\python.exe"
    ) else (
        where python >nul 2>&1
        if !errorlevel! equ 0 set "PY_CMD=python"
    )
)

if not defined PY_CMD (
    echo [ERROR] Python 3.11+ executable not found.
    echo Please install Python or ensure it is added to your PATH.
    pause
    exit /b 1
)

echo [OK] Using Python: %PY_CMD%

:: 2. Handle optional CLI flags
if "%~1"=="--benchmark" (
    echo [RUN] Executing Multi-Model Benchmark Suite - Seed 777
    "%PY_CMD%" tcontext\experiments\run_benchmark.py --seed 777
    goto :end
)

if "%~1"=="--test" (
    echo [RUN] Running Pytest Verification Suite
    "%PY_CMD%" -m pytest -v
    goto :end
)

if "%~1"=="--query" (
    echo [RUN] Executing Single Image Query Demo
    shift
    "%PY_CMD%" tcontext\query_demo.py %2 %3 %4 %5 %6 %7 %8 %9
    goto :end
)

:: 3. Default: Launch Interactive Streamlit Dashboard
echo.
echo [1/3] Checking Port 8501...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :8501 ^| findstr LISTENING') do (
    echo [INFO] Releasing Port 8501 - PID %%a
    taskkill /f /pid %%a >nul 2>&1
)

echo [2/3] Opening Browser at http://localhost:8501...
start "" http://localhost:8501

echo [3/3] Starting Streamlit Server - Press Ctrl+C to stop
echo.
"%PY_CMD%" -m streamlit run streamlit_app.py --server.port 8501 --server.headless false

:end
echo.
