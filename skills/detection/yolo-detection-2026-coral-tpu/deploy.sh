#!/usr/bin/env bash
# deploy.sh — Native local bootstrapper for Coral TPU Detection Skill
#
# Builds a local Python virtual environment and verifies Edge TPU connectivity.
# Called by Aegis skill-runtime-manager during installation.
#
# Exit codes:
#   0 = success
#   1 = fatal error (Python/pip not found)
#   2 = partial success (no TPU detected, will use CPU fallback)

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_PREFIX="[coral-tpu-deploy]"

log()  { echo "$LOG_PREFIX $*" >&2; }
emit() { echo "$1"; }  # JSON to stdout for Aegis to parse

# ─── Step 1: Detect Platform ────────────────────────────────────────────────

PLATFORM="$(uname -s)"
ARCH="$(uname -m)"
log "Platform: $PLATFORM ($ARCH)"
emit "{\"event\": \"progress\", \"stage\": \"platform\", \"message\": \"Platform: $PLATFORM/$ARCH\"}"

if [ "$PLATFORM" = "Linux" ]; then
    log "Linux: ensuring system packages are installed..."
    emit '{"event": "progress", "stage": "platform", "message": "Ensuring Linux dependencies..."}'
    sudo apt-get update >/dev/null 2>&1 || true
    sudo apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv libusb-1.0-0 >/dev/null 2>&1 || true
fi

# ─── Step 2: Ensure Python 3 ────────────────────────────────────────────────

if ! command -v python3 &>/dev/null; then
    log "ERROR: Python 3 not found."
    emit '{"event": "error", "stage": "python", "message": "Python 3 not found"}'
    exit 1
fi

PYTHON_CMD="python3"
log "Using Python: $($PYTHON_CMD --version)"
emit '{"event": "progress", "stage": "python", "message": "Python verified"}'

# ─── Step 3: Create Virtual Environment ─────────────────────────────────────

VENV_DIR="$SKILL_DIR/venv"
log "Setting up virtual environment in $VENV_DIR..."
emit '{"event": "progress", "stage": "build", "message": "Creating Python virtual environment..."}'

"$PYTHON_CMD" -m venv "$VENV_DIR"

# Ensure the venv works
if [ ! -f "$VENV_DIR/bin/python" ]; then
    log "ERROR: Failed to create virtual environment."
    emit '{"event": "error", "stage": "build", "message": "Failed to create venv"}'
    exit 1
fi

# ─── Step 4: Install Dependencies ───────────────────────────────────────────

log "Installing Python dependencies (this may take a minute)..."
emit '{"event": "progress", "stage": "build", "message": "Installing ai-edge-litert and dependencies..."}'

# Upgrade pip securely
"$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null 2>&1 || true

# Install requirements
if ! "$VENV_DIR/bin/python" -m pip install -r "$SKILL_DIR/requirements.txt"; then
    log "ERROR: Failed to install Python dependencies."
    emit '{"event": "error", "stage": "build", "message": "pip install failed"}'
    exit 1
fi

log "Dependencies installed successfully."
emit '{"event": "progress", "stage": "build", "message": "Python environment ready"}'

# ─── Step 5: Probe for Edge TPU devices ──────────────────────────────────────

log "Probing for Edge TPU devices natively..."
emit '{"event": "progress", "stage": "probe", "message": "Checking for physical Edge TPU..."}'

TPU_FOUND=false
# Run probe inside the venv
PROBE_OUTPUT=$("$VENV_DIR/bin/python" "$SKILL_DIR/scripts/tpu_probe.py" 2>/dev/null) || true

if echo "$PROBE_OUTPUT" | grep -q '"available": true'; then
    TPU_COUNT=$(echo "$PROBE_OUTPUT" | "$VENV_DIR/bin/python" -c "import sys,json; print(json.load(sys.stdin)['count'])" 2>/dev/null || echo "?")
    TPU_FOUND=true
    log "Edge TPU detected: $TPU_COUNT device(s)"
    emit "{\"event\": \"progress\", \"stage\": \"probe\", \"message\": \"Found $TPU_COUNT Edge TPU device(s) natively\"}"
else
    log "WARNING: No Edge TPU detected — skill will run in CPU fallback mode"
    emit '{"event": "progress", "stage": "probe", "message": "No Edge TPU detected — CPU fallback available"}'
fi

# ─── Step 6: Complete ────────────────────────────────────────────────────────

if [ "$TPU_FOUND" = true ]; then
    emit "{\"event\": \"complete\", \"status\": \"success\", \"tpu_found\": true, \"message\": \"Native Coral TPU skill installed — Edge TPU ready\"}"
    log "Done! Edge TPU ready."
    exit 0
else
    emit "{\"event\": \"complete\", \"status\": \"partial\", \"tpu_found\": false, \"message\": \"Native Coral TPU skill installed — no TPU detected (CPU fallback)\"}"
    log "Done with warning: no TPU detected. Connect Coral USB and restart."
    exit 2
fi
