@echo off
REM deploy.bat — Docker-based bootstrapper for Coral TPU Detection Skill (Windows)
REM
REM Builds the Docker image locally and verifies Edge TPU connectivity.
REM Called by Aegis skill-runtime-manager during installation.
REM
REM Requires: Docker Desktop 4.35+ with USB/IP support

setlocal enabledelayedexpansion

set "SKILL_DIR=%~dp0"
set "IMAGE_NAME=aegis-coral-tpu"
set "IMAGE_TAG=latest"
set "LOG_PREFIX=[coral-tpu-deploy]"

REM ─── Step 1: Check Docker ────────────────────────────────────────────────

where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo %LOG_PREFIX% ERROR: Docker not found. Install Docker Desktop 4.35+ 1>&2
    echo {"event": "error", "stage": "docker", "message": "Docker not found. Install Docker Desktop 4.35+"}
    exit /b 1
)

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo %LOG_PREFIX% ERROR: Docker daemon not running. Start Docker Desktop. 1>&2
    echo {"event": "error", "stage": "docker", "message": "Docker daemon not running"}
    exit /b 1
)

for /f "tokens=*" %%v in ('docker version --format "{{.Server.Version}}" 2^>nul') do set "DOCKER_VER=%%v"
echo %LOG_PREFIX% Using Docker (version: %DOCKER_VER%) 1>&2
echo {"event": "progress", "stage": "docker", "message": "Docker ready (%DOCKER_VER%)"}

REM ─── Step 2: Build Docker image ──────────────────────────────────────────

echo %LOG_PREFIX% Building Docker image: %IMAGE_NAME%:%IMAGE_TAG% ... 1>&2
echo {"event": "progress", "stage": "build", "message": "Building Docker image..."}

docker build -t %IMAGE_NAME%:%IMAGE_TAG% "%SKILL_DIR%"
if %errorlevel% neq 0 (
    echo %LOG_PREFIX% ERROR: Docker build failed 1>&2
    echo {"event": "error", "stage": "build", "message": "Docker image build failed"}
    exit /b 1
)

echo {"event": "progress", "stage": "build", "message": "Docker image ready"}

REM ─── Step 3: Probe for Edge TPU ──────────────────────────────────────────

echo %LOG_PREFIX% Probing for Edge TPU devices... 1>&2
echo {"event": "progress", "stage": "probe", "message": "Checking for Edge TPU devices..."}

docker run --rm --privileged %IMAGE_NAME%:%IMAGE_TAG% python3 scripts/tpu_probe.py >nul 2>&1
if %errorlevel% equ 0 (
    echo %LOG_PREFIX% Edge TPU detected 1>&2
    echo {"event": "progress", "stage": "probe", "message": "Edge TPU detected"}
) else (
    echo %LOG_PREFIX% WARNING: No Edge TPU detected - CPU fallback 1>&2
    echo {"event": "progress", "stage": "probe", "message": "No Edge TPU detected - CPU fallback"}
)

REM ─── Step 4: Set run command ──────────────────────────────────────────────

set "RUN_CMD=docker run -i --rm --privileged -v /tmp/aegis_detection:/tmp/aegis_detection --env AEGIS_SKILL_ID --env AEGIS_SKILL_PARAMS --env PYTHONUNBUFFERED=1 %IMAGE_NAME%:%IMAGE_TAG%"

echo {"event": "complete", "status": "success", "run_command": "%RUN_CMD%", "message": "Coral TPU skill installed"}

echo %LOG_PREFIX% Done! 1>&2
exit /b 0
