@echo off
REM deploy.bat — Docker-based bootstrapper for OpenVINO Detection Skill (Windows)

setlocal enabledelayedexpansion

set "SKILL_DIR=%~dp0"
set "IMAGE_NAME=aegis-openvino-detect"
set "IMAGE_TAG=latest"
set "LOG_PREFIX=[openvino-deploy]"

REM ─── Check Docker ────────────────────────────────────────────────────────
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo %LOG_PREFIX% ERROR: Docker not found. 1>&2
    echo {"event": "error", "stage": "docker", "message": "Docker not found"}
    exit /b 1
)

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo %LOG_PREFIX% ERROR: Docker daemon not running. 1>&2
    echo {"event": "error", "stage": "docker", "message": "Docker daemon not running"}
    exit /b 1
)

echo {"event": "progress", "stage": "docker", "message": "Docker ready"}

REM ─── Build Docker image ──────────────────────────────────────────────────
echo %LOG_PREFIX% Building Docker image... 1>&2
echo {"event": "progress", "stage": "build", "message": "Building Docker image..."}

docker build -t %IMAGE_NAME%:%IMAGE_TAG% "%SKILL_DIR%"
if %errorlevel% neq 0 (
    echo {"event": "error", "stage": "build", "message": "Docker image build failed"}
    exit /b 1
)

echo {"event": "progress", "stage": "build", "message": "Docker image ready"}

REM ─── Probe devices ──────────────────────────────────────────────────────
docker run --rm --privileged %IMAGE_NAME%:%IMAGE_TAG% python3 scripts/device_probe.py >nul 2>&1

REM ─── Set run command ─────────────────────────────────────────────────────
set "RUN_CMD=docker run -i --rm --privileged -v /tmp/aegis_detection:/tmp/aegis_detection --env AEGIS_SKILL_ID --env AEGIS_SKILL_PARAMS --env PYTHONUNBUFFERED=1 %IMAGE_NAME%:%IMAGE_TAG%"

echo {"event": "complete", "status": "success", "run_command": "%RUN_CMD%", "message": "OpenVINO skill installed"}
exit /b 0
