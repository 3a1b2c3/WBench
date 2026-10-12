@echo off
REM Run one (or a few) WBench cases through the InSpatio-World 1.5 driver
REM (src\models\camera\inspatio_model.py), then optionally evaluate them. Windows counterpart of run_inspatio.sh.
REM Requires: ..\inspatio-world-v1.5\.venv (its setup_env.bat) and checkpoints (its download_models.bat).
REM Set WBENCH_PYTHON to a venv python with WBench's dependencies; default is %USERPROFILE%\wbench-venv.
REM
REM   run_inspatio.bat --cases data/cases/case_1.json
REM   run_inspatio.bat --cases data/cases/case_1.json --evaluate
REM   run_inspatio.bat --limit 3
setlocal enabledelayedexpansion

cd /d "%~dp0"

if "%WBENCH_PYTHON%"=="" set "WBENCH_PYTHON=%USERPROFILE%\wbench-venv\Scripts\python.exe"
if not exist "%WBENCH_PYTHON%" (
    echo ERROR: WBENCH_PYTHON not found: %WBENCH_PYTHON%
    echo Set WBENCH_PYTHON to a venv with WBench's dependencies installed.
    exit /b 1
)

set EVALUATE=0
set "GEN_ARGS="
for %%A in (%*) do (
    if /i "%%~A"=="--evaluate" (
        set EVALUATE=1
    ) else (
        set GEN_ARGS=!GEN_ARGS! %%A
    )
)

echo ==========================================
echo GPU memory before the run, MiB used and total:
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo ==========================================
echo WBench: generating with inspatio
echo ==========================================
"%WBENCH_PYTHON%" generate.py --model inspatio !GEN_ARGS!
if errorlevel 1 exit /b 1

if "!EVALUATE!"=="1" (
    echo.
    echo ==========================================
    echo WBench: evaluating inspatio
    echo ==========================================
    "%WBENCH_PYTHON%" main.py --model inspatio
    if errorlevel 1 exit /b 1
)

echo.
echo Done. Videos: work_dirs\inspatio\videos\
if "!EVALUATE!"=="1" echo Results: work_dirs\inspatio\evaluation\ and work_dirs\inspatio\report.json
exit /b 0
