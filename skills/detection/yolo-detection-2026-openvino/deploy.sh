#!/usr/bin/env bash
# deploy.sh — Docker-based bootstrapper for OpenVINO Detection Skill
#
# Builds the Docker image locally and verifies device availability.
# Called by Aegis skill-runtime-manager during installation.

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="aegis-openvino-detect"
IMAGE_TAG="latest"
LOG_PREFIX="[openvino-deploy]"

log()  { echo "$LOG_PREFIX $*" >&2; }
emit() { echo "$1"; }

# ─── Step 1: Check Docker ────────────────────────────────────────────────────

DOCKER_CMD=""
for cmd in docker podman; do
    if command -v "$cmd" &>/dev/null; then
        DOCKER_CMD="$cmd"
        break
    fi
done

if [ -z "$DOCKER_CMD" ]; then
    log "ERROR: Docker (or Podman) not found."
    emit '{"event": "error", "stage": "docker", "message": "Docker not found. Install Docker Desktop 4.35+"}'
    exit 1
fi

if ! "$DOCKER_CMD" info &>/dev/null; then
    log "ERROR: Docker daemon is not running."
    emit '{"event": "error", "stage": "docker", "message": "Docker daemon not running"}'
    exit 1
fi

DOCKER_VER=$("$DOCKER_CMD" version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
log "Using $DOCKER_CMD (version: $DOCKER_VER)"
emit "{\"event\": \"progress\", \"stage\": \"docker\", \"message\": \"Docker ready ($DOCKER_VER)\"}"

# ─── Step 2: Detect platform for device access ──────────────────────────────

PLATFORM="$(uname -s)"
DEVICE_FLAGS=""

case "$PLATFORM" in
    Linux)
        # Pass Intel GPU and USB devices
        [ -d /dev/dri ] && DEVICE_FLAGS="--device /dev/dri"
        [ -d /dev/bus/usb ] && DEVICE_FLAGS="$DEVICE_FLAGS --device /dev/bus/usb"
        [ -z "$DEVICE_FLAGS" ] && DEVICE_FLAGS="--privileged"
        log "Platform: Linux — devices: $DEVICE_FLAGS"
        ;;
    Darwin)
        log "Platform: macOS — Docker Desktop USB/IP for NCS2, CPU fallback available"
        DEVICE_FLAGS="--privileged"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        log "Platform: Windows — Docker Desktop USB/IP"
        DEVICE_FLAGS="--privileged"
        ;;
    *)
        DEVICE_FLAGS="--privileged"
        ;;
esac

emit "{\"event\": \"progress\", \"stage\": \"platform\", \"message\": \"Platform: $PLATFORM\"}"

# ─── Step 3: Build Docker image ─────────────────────────────────────────────

log "Building Docker image: $IMAGE_NAME:$IMAGE_TAG ..."
emit '{"event": "progress", "stage": "build", "message": "Building Docker image..."}'

if "$DOCKER_CMD" build -t "$IMAGE_NAME:$IMAGE_TAG" "$SKILL_DIR" 2>&1 | while read -r line; do
    log "$line"
done; then
    log "Docker image built successfully"
    emit '{"event": "progress", "stage": "build", "message": "Docker image ready"}'
else
    log "ERROR: Docker build failed"
    emit '{"event": "error", "stage": "build", "message": "Docker image build failed"}'
    exit 1
fi

# ─── Step 4: Probe devices ──────────────────────────────────────────────────

log "Probing OpenVINO devices..."
emit '{"event": "progress", "stage": "probe", "message": "Checking OpenVINO devices..."}'

PROBE_OUTPUT=$("$DOCKER_CMD" run --rm $DEVICE_FLAGS \
    "$IMAGE_NAME:$IMAGE_TAG" python3 scripts/device_probe.py 2>/dev/null) || true

if echo "$PROBE_OUTPUT" | grep -q '"accelerator_found": true'; then
    log "Hardware accelerator detected (GPU/NCS2)"
    emit '{"event": "progress", "stage": "probe", "message": "Hardware accelerator detected"}'
else
    log "No accelerator — CPU mode available"
    emit '{"event": "progress", "stage": "probe", "message": "CPU mode (no GPU/NCS2 detected)"}'
fi

# ─── Step 5: Build run command ───────────────────────────────────────────────

RUN_CMD="$DOCKER_CMD run -i --rm $DEVICE_FLAGS"
RUN_CMD="$RUN_CMD -v /tmp/aegis_detection:/tmp/aegis_detection"
RUN_CMD="$RUN_CMD --env AEGIS_SKILL_ID --env AEGIS_SKILL_PARAMS --env PYTHONUNBUFFERED=1"
RUN_CMD="$RUN_CMD $IMAGE_NAME:$IMAGE_TAG"

log "Runtime command: $RUN_CMD"

emit "{\"event\": \"complete\", \"status\": \"success\", \"run_command\": \"$RUN_CMD\", \"message\": \"OpenVINO skill installed\"}"
log "Done!"
exit 0
