@echo off
REM deploy.bat — Coral TPU Detection Skill installer for Windows
REM
REM What this does:
REM   1. Downloads + installs the Edge TPU runtime (edgetpu.dll) via UAC
REM   2. Creates a Python virtual environment (Python 3.9–3.11 recommended)
REM   3. Installs ai-edge-litert and image processing deps
REM   4. Verifies the compiled yolo26n_edgetpu.tflite model is present
REM   5. Probes for an Edge TPU device
REM
REM Note: pycoral is NOT used. detect.py uses ai-edge-litert directly,
REM       which supports Python 3.9–3.13 and does not require pycoral.
REM
REM Exit codes:
REM   0 = success (TPU detected and ready)
REM   1 = fatal error
REM   2 = partial success (no TPU detected, CPU fallback available)

setlocal enabledelayedexpansion

set "SKILL_DIR=%~dp0"
set "LOG_PREFIX=[coral-tpu-deploy]"

REM Ensure we run inside the skill folder
cd /d "%SKILL_DIR%"

echo %LOG_PREFIX% Platform: Windows 1>&2
echo {"event": "progress", "stage": "platform", "message": "Windows installer starting..."}

REM ─── Step 1: Edge TPU DLLs + WinUSB driver ──────────────────────────────────
REM Strategy (fastest-first):
REM   A) If edgetpu.dll is pre-bundled in driver\  → copy directly, skip download
REM   B) Otherwise → download runtime zip, extract, copy DLLs
REM Then always run pnputil (elevated) with the bundled coral_winusb.inf.

REM Check for VC++ 2019 Redistributable (required by edgetpu.dll)
echo %LOG_PREFIX% Checking for Visual C++ 2019 redistributable... 1>&2
powershell -NoProfile -Command "if (Test-Path 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64') { exit 0 } else { exit 1 }" >nul 2>&1
if %errorlevel% neq 0 (
    echo %LOG_PREFIX% Installing Visual C++ 2019 Redistributable... 1>&2
    echo {"event": "progress", "stage": "platform", "message": "Installing Visual C++ 2019 Redistributable (required for edgetpu.dll)..."}
    powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://aka.ms/vs/16/release/vc_redist.x64.exe' -OutFile '%TEMP%\vc_redist.x64.exe' -UseBasicParsing"
    "%TEMP%\vc_redist.x64.exe" /install /quiet /norestart
)

if not exist "%SKILL_DIR%lib" mkdir "%SKILL_DIR%lib"

REM ── A) Use pre-bundled DLLs if present ──────────────────────────────────────
if exist "%SKILL_DIR%driver\edgetpu.dll" (
    echo %LOG_PREFIX% Using pre-bundled edgetpu.dll from driver\. 1>&2
    echo {"event": "progress", "stage": "platform", "message": "Using pre-bundled Edge TPU DLLs (offline install)."}
    copy /Y "%SKILL_DIR%driver\edgetpu.dll" "%SKILL_DIR%lib\edgetpu.dll" >nul 2>&1
    copy /Y "%SKILL_DIR%driver\libusb-1.0.dll" "%SKILL_DIR%lib\libusb-1.0.dll" >nul 2>&1
    set "TMP_DIR="
    goto :dll_ready
)

REM ── B) Download runtime zip ──────────────────────────────────────────────────
echo %LOG_PREFIX% Downloading Edge TPU runtime (DLLs not pre-bundled)... 1>&2
echo {"event": "progress", "stage": "platform", "message": "Downloading Google Edge TPU runtime (edgetpu.dll)..."}

set "TMP_DIR=%TEMP%\coral_tpu_install_%RANDOM%"
mkdir "%TMP_DIR%"
cd /d "%TMP_DIR%"

powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip' -OutFile 'edgetpu_runtime_20221024.zip' -UseBasicParsing"
if %errorlevel% neq 0 (
    echo %LOG_PREFIX% ERROR: Failed to download Edge TPU runtime. Check internet connectivity. 1>&2
    echo {"event": "error", "stage": "platform", "message": "Download failed - check internet connectivity and retry"}
    cd /d "%SKILL_DIR%"
    rmdir /S /Q "%TMP_DIR%" 2>nul
    exit /b 1
)

powershell -NoProfile -Command "Expand-Archive -Path 'edgetpu_runtime_20221024.zip' -DestinationPath '.' -Force"

set "RUNTIME_DIR=%TMP_DIR%\edgetpu_runtime"
if not exist "%RUNTIME_DIR%\install.bat" (
    echo %LOG_PREFIX% ERROR: Runtime zip did not extract correctly. 1>&2
    echo {"event": "error", "stage": "platform", "message": "edgetpu_runtime_20221024.zip extraction failed"}
    cd /d "%SKILL_DIR%"
    rmdir /S /Q "%TMP_DIR%" 2>nul
    exit /b 1
)

copy /Y "%RUNTIME_DIR%\libedgetpu\direct\x64_windows\edgetpu.dll" "%SKILL_DIR%lib\edgetpu.dll" >nul 2>&1
copy /Y "%RUNTIME_DIR%\third_party\libusb_win\libusb-1.0.dll" "%SKILL_DIR%lib\libusb-1.0.dll" >nul 2>&1

cd /d "%SKILL_DIR%"
rmdir /S /Q "%TMP_DIR%" 2>nul
set "TMP_DIR="

:dll_ready
if not exist "%SKILL_DIR%lib\edgetpu.dll" (
    echo %LOG_PREFIX% ERROR: edgetpu.dll could not be placed in lib\. 1>&2
    echo {"event": "error", "stage": "platform", "message": "Failed to obtain edgetpu.dll"}
    exit /b 1
)
echo %LOG_PREFIX% edgetpu.dll ready in lib\. 1>&2
echo {"event": "progress", "stage": "platform", "message": "Edge TPU DLLs ready."}

REM ── Install UsbDk Driver (bundled MSI, required for Coral TPU on Windows) 
if exist "%SKILL_DIR%driver\UsbDk_1.0.22_x64.msi" (
    echo %LOG_PREFIX% Emitting Pause Modal for explicit driver installation... 1>&2
    echo {"event": "progress", "stage": "platform", "message": "Waiting for User to install UsbDk Driver..."}
    echo [AEGIS_PAUSE_MODAL] file=driver\UsbDk_1.0.22_x64.msi; msg=Google Coral TPU Requires the UsbDk system driver to run over USB. Click 'Launch Installer' to proceed.
    set /p "DUMMY_VAR=Press ENTER to continue deploy after installing UsbDk..."
    
    echo %LOG_PREFIX% Continued deployment. Assuming UsbDk driver was installed. 1>&2
    echo {"event": "progress", "stage": "platform", "message": "Driver installation complete. Unplug and replug your Coral USB Accelerator to activate."}
) else (
    echo %LOG_PREFIX% WARNING: UsbDk MSI not found in driver\. Skipping driver install. 1>&2
)



REM ─── Step 2: Find Python ─────────────────────────────────────────────────────
REM ai-edge-litert supports Python 3.9–3.13. We prefer the system default.
REM If only Python 3.12+ is available, it still works (no pycoral needed).

set "PYTHON_CMD="

REM Try common Python launchers in preference order
for %%P in (python python3 py) do (
    if not defined PYTHON_CMD (
        %%P --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=%%P"
        )
    )
)

if not defined PYTHON_CMD (
    echo %LOG_PREFIX% ERROR: Python not found on PATH. 1>&2
    echo {"event": "error", "stage": "python", "message": "Python not found — install Python 3.9-3.11 from python.org and re-run"}
    exit /b 1
)

REM Get Python version for info only (not blocking — ai-edge-litert works 3.9-3.13)
for /f "tokens=2" %%V in ('!PYTHON_CMD! --version 2^>^&1') do set "PY_VERSION=%%V"
echo %LOG_PREFIX% Python version: !PY_VERSION! 1>&2
echo {"event": "progress", "stage": "python", "message": "Using Python !PY_VERSION!"}

REM ─── Step 3: Create virtual environment ──────────────────────────────────────

set "VENV_DIR=%SKILL_DIR%venv"
echo %LOG_PREFIX% Creating virtual environment at %VENV_DIR%... 1>&2
echo {"event": "progress", "stage": "build", "message": "Creating Python virtual environment..."}

!PYTHON_CMD! -m venv "%VENV_DIR%"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo %LOG_PREFIX% ERROR: Failed to create virtual environment. 1>&2
    echo {"event": "error", "stage": "build", "message": "venv creation failed"}
    exit /b 1
)

REM ─── Step 4: Install Python dependencies ─────────────────────────────────────
REM ai-edge-litert: LiteRT runtime with Edge TPU delegate support (Python 3.9-3.13)
REM numpy + Pillow: image processing

echo %LOG_PREFIX% Upgrading pip... 1>&2
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip --quiet

echo %LOG_PREFIX% Installing dependencies (ai-edge-litert, numpy, Pillow)... 1>&2
echo {"event": "progress", "stage": "build", "message": "Installing ai-edge-litert and image processing libraries..."}

"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%SKILL_DIR%requirements.txt" --quiet
if %errorlevel% neq 0 (
    echo %LOG_PREFIX% ERROR: pip install failed. 1>&2
    echo {"event": "error", "stage": "build", "message": "pip install requirements.txt failed"}
    exit /b 1
)

echo %LOG_PREFIX% Dependencies installed. 1>&2
echo {"event": "progress", "stage": "build", "message": "Python dependencies installed successfully."}

REM ─── Step 5: Verify compiled EdgeTPU model ────────────────────────────────────
REM The yolo26n_edgetpu.tflite is pre-compiled via docker/compile.sh and committed
REM to the git repository. deploy.bat does NOT compile it — that requires Linux.

echo %LOG_PREFIX% Checking for compiled EdgeTPU model... 1>&2

set "MODEL_FOUND=false"
set "MODEL_FILE="

REM Accept either naming convention from edgetpu_compiler output
for %%M in (
    "%SKILL_DIR%models\yolo26n_int8_edgetpu.tflite"
    "%SKILL_DIR%models\yolo26n_edgetpu.tflite"
    "%SKILL_DIR%models\yolo26n_320_edgetpu.tflite"
) do (
    if exist %%M (
        set "MODEL_FOUND=true"
        set "MODEL_FILE=%%~M"
    )
)

if "!MODEL_FOUND!"=="false" (
    echo %LOG_PREFIX% WARNING: No pre-compiled EdgeTPU model found in models\. 1>&2
    echo {"event": "progress", "stage": "model", "message": "No EdgeTPU model found — will fall back to CPU inference (SSD MobileNet)"}
) else (
    echo %LOG_PREFIX% Found model: !MODEL_FILE! 1>&2
    echo {"event": "progress", "stage": "model", "message": "Edge TPU model ready: yolo26n_edgetpu.tflite"}
)

REM Download SSD MobileNet as a universal CPU fallback so the skill is unconditionally functional
if not exist "%SKILL_DIR%models\ssd_mobilenet_v2_coco_quant_postprocess.tflite" (
    echo %LOG_PREFIX% Downloading SSD MobileNet CPU fallback model... 1>&2
    if not exist "%SKILL_DIR%models" mkdir "%SKILL_DIR%models"
    powershell -NoProfile -Command ^
      "Invoke-WebRequest -Uri 'https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite' -OutFile '%SKILL_DIR%models\ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite' -UseBasicParsing" 2>nul
    powershell -NoProfile -Command ^
      "Invoke-WebRequest -Uri 'https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess.tflite' -OutFile '%SKILL_DIR%models\ssd_mobilenet_v2_coco_quant_postprocess.tflite' -UseBasicParsing" 2>nul
)

REM ─── Step 6: Probe for Edge TPU devices ──────────────────────────────────────

echo %LOG_PREFIX% Probing for Edge TPU devices... 1>&2
echo {"event": "progress", "stage": "probe", "message": "Checking for Coral USB Accelerator..."}

set "TPU_FOUND=false"
set "PROBE_JSON="

for /f "delims=" %%I in ('"%VENV_DIR%\Scripts\python.exe" "%SKILL_DIR%scripts\tpu_probe.py" 2^>nul') do (
    set "PROBE_JSON=%%I"
)

echo !PROBE_JSON! | findstr /C:"\"available\": true" >nul 2>&1
if %errorlevel% equ 0 (
    set "TPU_FOUND=true"
    echo %LOG_PREFIX% Edge TPU detected. 1>&2
    echo {"event": "progress", "stage": "probe", "message": "Coral USB Accelerator detected and ready."}
) else (
    echo %LOG_PREFIX% No Edge TPU detected (device may not be plugged in). 1>&2
    echo {"event": "progress", "stage": "probe", "message": "No Edge TPU detected. Plug in the Coral USB Accelerator and restart the skill."}
)

REM ─── Step 7: Done ────────────────────────────────────────────────────────────

if "!TPU_FOUND!"=="true" (
    echo {"event": "complete", "status": "success", "tpu_found": true, "message": "Coral TPU skill installed and Edge TPU is ready."}
    exit /b 0
) else (
    echo {"event": "complete", "status": "partial", "tpu_found": false, "message": "Coral TPU skill installed. Plug in your Coral USB Accelerator to enable hardware acceleration."}
    exit /b 0
)
