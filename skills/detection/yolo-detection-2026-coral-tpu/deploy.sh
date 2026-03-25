#!/usr/bin/env bash
# deploy.sh — Docker-based bootstrapper for Coral TPU Detection Skill
#
# Builds the Docker image locally and verifies Edge TPU connectivity.
# Called by Aegis skill-runtime-manager during installation.
#
# Exit codes:
#   0 = success
#   1 = fatal error (Docker not found)
#   2 = partial success (no TPU detected, will use CPU fallback)

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="aegis-coral-tpu"
IMAGE_TAG="latest"
LOG_PREFIX="[coral-tpu-deploy]"

log()  { echo "$LOG_PREFIX $*" >&2; }
emit() { echo "$1"; }  # JSON to stdout for Aegis to parse

# ─── Step 1: Check Docker ────────────────────────────────────────────────────

find_docker() {
    for cmd in docker podman; do
        if command -v "$cmd" &>/dev/null; then
            echo "$cmd"
            return 0
        fi
    done
    return 1
}

DOCKER_CMD=$(find_docker) || {
    log "ERROR: Docker (or Podman) not found. Install Docker Desktop 4.35+ and retry."
    emit '{"event": "error", "stage": "docker", "message": "Docker not found. Install Docker Desktop 4.35+"}'
    exit 1
}

# Verify Docker is running
if ! "$DOCKER_CMD" info &>/dev/null; then
    log "ERROR: Docker daemon is not running. Start Docker Desktop and retry."
    emit '{"event": "error", "stage": "docker", "message": "Docker daemon not running"}'
    exit 1
fi

DOCKER_VER=$("$DOCKER_CMD" version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
log "Using $DOCKER_CMD (version: $DOCKER_VER)"
emit "{\"event\": \"progress\", \"stage\": \"docker\", \"message\": \"Docker ready ($DOCKER_VER)\"}"

# ─── Step 2: Detect platform for USB access hints ───────────────────────────

PLATFORM="$(uname -s)"
ARCH="$(uname -m)"
USB_FLAG=""

case "$PLATFORM" in
    Linux)
        USB_FLAG="--device /dev/bus/usb"
        log "Platform: Linux — will use --device /dev/bus/usb"
        ;;
    Darwin)
        log "Platform: macOS ($ARCH) — Docker Desktop 4.35+ USB/IP required"
        log "Ensure Docker Desktop Settings → Features → USB devices is enabled"
        # macOS Docker Desktop 4.35+ handles USB/IP transparently
        # No --device flag needed, but privileged may be required
        USB_FLAG="--privileged"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        log "Platform: Windows — Docker Desktop 4.35+ USB/IP or WSL2 backend"
        USB_FLAG="--privileged"
        ;;
    *)
        log "Platform: Unknown ($PLATFORM) — attempting with --privileged"
        USB_FLAG="--privileged"
        ;;
esac

emit "{\"event\": \"progress\", \"stage\": \"platform\", \"message\": \"Platform: $PLATFORM/$ARCH\"}"

# ─── Step 3: Build Docker image ─────────────────────────────────────────────

log "Building Docker image: $IMAGE_NAME:$IMAGE_TAG ..."
emit '{"event": "progress", "stage": "build", "message": "Building Docker image (this may take a few minutes)..."}'

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

# ─── Step 4: Probe for Edge TPU devices ──────────────────────────────────────

log "Probing for Edge TPU devices..."
emit '{"event": "progress", "stage": "probe", "message": "Checking for Edge TPU devices..."}'

TPU_FOUND=false
PROBE_OUTPUT=$("$DOCKER_CMD" run --rm $USB_FLAG \
    "$IMAGE_NAME:$IMAGE_TAG" python3 scripts/tpu_probe.py 2>/dev/null) || true

if echo "$PROBE_OUTPUT" | grep -q '"available": true'; then
    TPU_COUNT=$(echo "$PROBE_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['count'])" 2>/dev/null || echo "?")
    TPU_FOUND=true
    log "Edge TPU detected: $TPU_COUNT device(s)"
    emit "{\"event\": \"progress\", \"stage\": \"probe\", \"message\": \"Found $TPU_COUNT Edge TPU device(s)\"}"
else
    log "WARNING: No Edge TPU detected — skill will run in CPU fallback mode"
    emit '{"event": "progress", "stage": "probe", "message": "No Edge TPU detected — CPU fallback available"}'
fi

# ─── Step 5: Build run command ───────────────────────────────────────────────

# The run command Aegis will use to launch the skill
# stdin/stdout pipe (-i), auto-remove (--rm), shared volume
RUN_CMD="$DOCKER_CMD run -i --rm $USB_FLAG"
RUN_CMD="$RUN_CMD -v /tmp/aegis_detection:/tmp/aegis_detection"
RUN_CMD="$RUN_CMD --env AEGIS_SKILL_ID --env AEGIS_SKILL_PARAMS --env PYTHONUNBUFFERED=1"
RUN_CMD="$RUN_CMD $IMAGE_NAME:$IMAGE_TAG"

log "Runtime command: $RUN_CMD"

# ─── Step 6: Complete ────────────────────────────────────────────────────────

if [ "$TPU_FOUND" = true ]; then
    emit "{\"event\": \"complete\", \"status\": \"success\", \"tpu_found\": true, \"run_command\": \"$RUN_CMD\", \"message\": \"Coral TPU skill installed — Edge TPU ready\"}"
    log "Done! Edge TPU ready."
    exit 0
else
    emit "{\"event\": \"complete\", \"status\": \"partial\", \"tpu_found\": false, \"run_command\": \"$RUN_CMD\", \"message\": \"Coral TPU skill installed — no TPU detected (CPU fallback)\"}"
    log "Done with warning: no TPU detected. Connect Coral USB and restart."
    exit 2
fi
