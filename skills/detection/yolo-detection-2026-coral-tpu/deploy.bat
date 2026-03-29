@echo off
REM deploy.bat — Native local bootstrapper for Coral TPU Detection Skill (Windows)
REM
REM Builds a local Python virtual environment and installs the Google Coral Edge TPU
REM natively on the host OS. Safely triggers UAC elevation for driver installation.
REM
REM Exit codes:
REM   0 = success
REM   1 = fatal error (Python/pip not found or UAC denied)
REM   2 = partial success (no TPU detected, will use CPU fallback)

setlocal enabledelayedexpansion

set "SKILL_DIR=%~dp0"
set "LOG_PREFIX=[coral-tpu-deploy]"

REM Ensure we run inside the correct folder
cd /d "%SKILL_DIR%"

echo %LOG_PREFIX% Platform: Windows 1>&2
echo {"event": "progress", "stage": "platform", "message": "Windows native environment detected"}

REM ─── Step 1: Install Native OS TPU Drivers (UAC Promoted) ───────────────

echo %LOG_PREFIX% Downloading Google official x64 Windows installer... 1>&2

set "TMP_DIR=%TEMP%\coral_tpu_install"
if not exist "%TMP_DIR%" mkdir "%TMP_DIR%"
cd /d "%TMP_DIR%"

powershell -Command "Invoke-WebRequest -Uri 'https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip' -OutFile 'edgetpu_runtime_20221024.zip'"
if %errorlevel% neq 0 (
    echo %LOG_PREFIX% ERROR: Failed to download Edge TPU runtime. 1>&2
    exit /b 1
)

powershell -Command "Expand-Archive -Path 'edgetpu_runtime_20221024.zip' -DestinationPath '.' -Force"
cd edgetpu_runtime

echo %LOG_PREFIX% Prompting for Administrator rights to install drivers... 1>&2
echo {"event": "progress", "stage": "platform", "message": "A separate UAC prompt and blue terminal will appear. Please approve it."}
echo {"event": "progress", "stage": "platform", "message": "Waiting for Google install.bat to finish..."}

REM Start the official install script elevated. Wait for it to finish.
powershell -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', 'install.bat < nul' -Verb RunAs -Wait"

if %errorlevel% neq 0 (
    echo.
    echo [91m[MANUAL SETUP REQUIRED][0m UAC prompt was skipped or user aborted.
    echo Please execute the following fragile instructions manually in an Administrator console:
    echo.
    echo [96mpowershell -Command "Invoke-WebRequest -Uri 'https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip' -OutFile 'edgetpu_runtime_20221024.zip'"[0m
    echo [96mpowershell -Command "Expand-Archive -Path 'edgetpu_runtime_20221024.zip' -DestinationPath '.'"[0m
    echo [96mcd edgetpu_runtime[0m
    echo [96minstall.bat[0m
    echo.
    echo Once completed, re-run this deployment.
    echo {"event": "error", "stage": "platform", "message": "Manual OS setup required (UAC skipped)"}
    cd /d "%SKILL_DIR%"
    rmdir /S /Q "%TMP_DIR%"
    exit /b 1
)

cd /d "%SKILL_DIR%"
rmdir /S /Q "%TMP_DIR%"

REM ─── Step 2: Ensure Python 3 ──────────────────────────────────────────────

set "PYTHON_CMD=python"
python --version >nul 2>&1
if %errorlevel% neq 0 (
    python3 --version >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=python3"
    ) else (
        echo %LOG_PREFIX% ERROR: Python not found. 1>&2
        echo {"event": "error", "stage": "python", "message": "Python not found"}
        exit /b 1
    )
)

echo %LOG_PREFIX% Using Python: !PYTHON_CMD! 1>&2

REM ─── Step 3: Create Virtual Environment ───────────────────────────────────

set "VENV_DIR=%SKILL_DIR%venv"
echo %LOG_PREFIX% Setting up virtual environment in %VENV_DIR%... 1>&2
echo {"event": "progress", "stage": "build", "message": "Creating Python virtual environment..."}

!PYTHON_CMD! -m venv "%VENV_DIR%"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo %LOG_PREFIX% ERROR: Failed to create virtual environment. 1>&2
    echo {"event": "error", "stage": "build", "message": "Failed to create venv"}
    exit /b 1
)

REM ─── Step 4: Install PyCoral & Dependencies ───────────────────────────────

echo %LOG_PREFIX% Installing Python dependencies (including pycoral)... 1>&2
echo {"event": "progress", "stage": "build", "message": "Fetching pycoral specific wheels..."}

"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1

"%VENV_DIR%\Scripts\python.exe" "%SKILL_DIR%scripts\install_pycoral.py"
if %errorlevel% neq 0 (
    echo %LOG_PREFIX% WARNING: pycoral install script failed. 1>&2
)

"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%SKILL_DIR%requirements.txt"
if %errorlevel% neq 0 (
    echo %LOG_PREFIX% ERROR: Failed to install Python dependencies. 1>&2
    echo {"event": "error", "stage": "build", "message": "pip install failed"}
    exit /b 1
)

echo %LOG_PREFIX% Dependencies installed successfully. 1>&2

REM ─── Step 5: Download Pre-compiled Models ──────────────────────────────────

echo %LOG_PREFIX% Downloading Edge TPU models... 1>&2
echo {"event": "progress", "stage": "build", "message": "Downloading Edge TPU models..."}

if not exist "%SKILL_DIR%models" mkdir "%SKILL_DIR%models"
cd /d "%SKILL_DIR%models"

powershell -Command "Invoke-WebRequest -Uri 'https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess.tflite' -OutFile 'ssd_mobilenet_v2_coco_quant_postprocess.tflite'"
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite' -OutFile 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'"

cd /d "%SKILL_DIR%"

REM ─── Step 6: Probe for Edge TPU devices ────────────────────────────────────

echo %LOG_PREFIX% Probing for Edge TPU devices natively... 1>&2
echo {"event": "progress", "stage": "probe", "message": "Checking for physical Edge TPU..."}

set "TPU_FOUND=false"
set "TPU_COUNT=?"

for /f "delims=" %%I in ('"%VENV_DIR%\Scripts\python.exe" "%SKILL_DIR%scripts\tpu_probe.py" 2^>nul') do (
    set "PROBE_OUTPUT=%%I"
)

echo !PROBE_OUTPUT! | findstr /C:"\"available\": true" >nul
if %errorlevel% equ 0 (
    set "TPU_FOUND=true"
    REM Approximate counts natively
    echo {"event": "progress", "stage": "probe", "message": "Edge TPU device natively registered"}
) else (
    echo {"event": "progress", "stage": "probe", "message": "No Edge TPU detected -- CPU fallback available"}
)

REM ─── Step 7: Complete ──────────────────────────────────────────────────────

if "!TPU_FOUND!"=="true" (
    echo {"event": "complete", "status": "success", "tpu_found": true, "message": "Native Coral TPU skill installed -- Edge TPU ready"}
    exit /b 0
) else (
    echo {"event": "complete", "status": "partial", "tpu_found": false, "message": "Native Coral TPU skill installed -- no TPU detected (CPU fallback)"}
    exit /b 2
)
