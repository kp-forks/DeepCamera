#!/usr/bin/env bash
# deploy.sh — Native bootstrapper for Coral TPU Detection Skill
#
# Installs ai-edge-litert in a Python venv and verifies Edge TPU connectivity.
# Called by Aegis skill-runtime-manager during installation.
#
# The Edge TPU hardware driver (libedgetpu) must be installed separately —
# this script detects the platform and provides instructions or auto-installs.
#
# Exit codes:
#   0 = success (Edge TPU detected)
#   1 = fatal error (Python not found, install failed)
#   2 = partial success (no TPU detected, will use CPU fallback)

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_PREFIX="[coral-tpu-deploy]"
VENV_DIR="$SKILL_DIR/.venv"

log()  { echo "$LOG_PREFIX $*" >&2; }
emit() { echo "$1"; }  # JSON to stdout for Aegis to parse

# ─── Step 1: Find Python ────────────────────────────────────────────────────

find_python() {
    # Prefer venv if already created
    if [ -f "$VENV_DIR/bin/python3" ]; then
        echo "$VENV_DIR/bin/python3"
        return 0
    fi
    # Search system
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            ver=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -eq 3 ] && [ "$minor" -ge 9 ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON=$(find_python) || {
    log "ERROR: Python 3.9+ not found. Install Python 3.9 or newer."
    emit '{"event": "error", "stage": "python", "message": "Python 3.9+ not found"}'
    exit 1
}

PY_VER=$("$PYTHON" --version 2>&1)
log "Found Python: $PYTHON ($PY_VER)"

# ─── Step 2: Create virtual environment ─────────────────────────────────────

PLATFORM="$(uname -s)"
ARCH="$(uname -m)"
emit "{\"event\": \"progress\", \"stage\": \"platform\", \"message\": \"Platform: $PLATFORM/$ARCH\"}"

if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment at $VENV_DIR ..."
    emit '{"event": "progress", "stage": "venv", "message": "Creating Python virtual environment..."}'
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Activate venv
VENV_PIP="$VENV_DIR/bin/pip"
VENV_PYTHON="$VENV_DIR/bin/python3"

if [ ! -f "$VENV_PIP" ]; then
    log "ERROR: venv creation failed — $VENV_PIP not found"
    emit '{"event": "error", "stage": "venv", "message": "Failed to create virtual environment"}'
    exit 1
fi

log "Virtual environment ready"

# ─── Step 3: Install Python dependencies ────────────────────────────────────

log "Installing dependencies..."
emit '{"event": "progress", "stage": "install", "message": "Installing ai-edge-litert and dependencies..."}'

"$VENV_PIP" install --upgrade pip -q 2>&1 | while read -r line; do log "$line"; done
"$VENV_PIP" install -r "$SKILL_DIR/requirements.txt" -q 2>&1 | while read -r line; do log "$line"; done

log "Python dependencies installed"
emit '{"event": "progress", "stage": "install", "message": "Dependencies installed"}'

# ─── Step 3b: Download Edge TPU model if not present ─────────────────────────

MODEL_DIR="$SKILL_DIR/models"
mkdir -p "$MODEL_DIR"

if ! ls "$MODEL_DIR"/*.tflite 1>/dev/null 2>&1; then
    log "No .tflite model found — downloading default EfficientDet-Lite0 Edge TPU model..."
    emit '{"event": "progress", "stage": "model", "message": "Downloading Edge TPU detection model..."}'

    # EfficientDet-Lite0 — 320x320 INT8, compiled for Edge TPU
    # Source: https://coral.ai/models/object-detection/
    MODEL_URL="https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    LABELS_URL="https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt"

    if curl -fSL "$MODEL_URL" -o "$MODEL_DIR/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite" 2>&1; then
        log "Model downloaded: ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    else
        log "ERROR: Failed to download Edge TPU model"
        emit '{"event": "error", "stage": "model", "message": "Failed to download model"}'
        exit 1
    fi

    # Also download the CPU-only variant for fallback
    CPU_MODEL_URL="https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
    curl -fSL "$CPU_MODEL_URL" -o "$MODEL_DIR/ssd_mobilenet_v2_coco_quant_postprocess.tflite" 2>&1 || true

    # Download labels
    curl -fSL "$LABELS_URL" -o "$MODEL_DIR/coco_labels.txt" 2>&1 || true

    emit '{"event": "progress", "stage": "model", "message": "Model downloaded ✓"}'
else
    log "Model already present in $MODEL_DIR"
fi

# ─── Step 4: Check libedgetpu (platform-specific) ───────────────────────────

log "Checking for libedgetpu hardware driver..."
emit '{"event": "progress", "stage": "driver", "message": "Checking for Edge TPU driver (libedgetpu)..."}'

LIBEDGETPU_FOUND=false

case "$PLATFORM" in
    Linux)
        # Check if libedgetpu is installed
        if ldconfig -p 2>/dev/null | grep -q "libedgetpu"; then
            LIBEDGETPU_FOUND=true
            log "libedgetpu found via ldconfig"
        elif [ -f "/usr/lib/libedgetpu.so.1" ] || [ -f "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1" ] || [ -f "/usr/lib/x86_64-linux-gnu/libedgetpu.so.1" ]; then
            LIBEDGETPU_FOUND=true
            log "libedgetpu found in /usr/lib"
        else
            log "libedgetpu not found. Installing from Coral apt repository..."
            # Try auto-install on Debian/Ubuntu
            if command -v apt-get &>/dev/null; then
                emit '{"event": "progress", "stage": "driver", "message": "Installing libedgetpu from Coral repository..."}'
                echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list > /dev/null 2>&1 || true
                curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - 2>/dev/null || true
                sudo apt-get update -qq 2>/dev/null || true
                if sudo apt-get install -y libedgetpu1-std 2>&1 | while read -r line; do log "$line"; done; then
                    LIBEDGETPU_FOUND=true
                    log "libedgetpu installed successfully"
                else
                    log "WARNING: Failed to install libedgetpu via apt"
                fi
            else
                log "WARNING: Non-Debian system — install libedgetpu manually"
                log "  See: https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime"
            fi
        fi
        ;;
    Darwin)
        # macOS: check for libedgetpu dylib
        if [ -f "/usr/local/lib/libedgetpu.1.dylib" ] || [ -f "/opt/homebrew/lib/libedgetpu.1.dylib" ]; then
            LIBEDGETPU_FOUND=true
            log "libedgetpu found on macOS"
        else
            log "libedgetpu not found on macOS."
            log "Install instructions:"
            log "  curl -LO https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip"
            log "  unzip edgetpu_runtime_20221024.zip && cd edgetpu_runtime && sudo bash install.sh"
            emit '{"event": "progress", "stage": "driver", "message": "libedgetpu not found — see stderr for install instructions"}'
        fi
        ;;
    MINGW*|MSYS*|CYGWIN*)
        log "Windows: libedgetpu must be installed manually."
        log "  Download: https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip"
        log "  Extract and run install.bat"
        emit '{"event": "progress", "stage": "driver", "message": "Windows: install libedgetpu manually (see stderr)"}'
        ;;
esac

# ─── Step 5: Probe for Edge TPU devices ──────────────────────────────────────

log "Probing for Edge TPU devices..."
emit '{"event": "progress", "stage": "probe", "message": "Checking for Edge TPU devices..."}'

TPU_FOUND=false
PROBE_OUTPUT=$("$VENV_PYTHON" "$SKILL_DIR/scripts/tpu_probe.py" 2>/dev/null) || true

if echo "$PROBE_OUTPUT" | grep -q '"available": true'; then
    TPU_FOUND=true
    log "Edge TPU detected and accessible"
    emit '{"event": "progress", "stage": "probe", "message": "Edge TPU detected ✓"}'
else
    log "No Edge TPU detected — skill will run in CPU fallback mode"
    if [ "$LIBEDGETPU_FOUND" = false ]; then
        emit '{"event": "progress", "stage": "probe", "message": "No Edge TPU — libedgetpu driver not installed"}'
    else
        emit '{"event": "progress", "stage": "probe", "message": "No Edge TPU connected — CPU fallback available"}'
    fi
fi

# ─── Step 6: Complete ────────────────────────────────────────────────────────

RUN_CMD="$VENV_PYTHON $SKILL_DIR/scripts/detect.py"

if [ "$TPU_FOUND" = true ]; then
    emit "{\"event\": \"complete\", \"status\": \"success\", \"accelerator_found\": true, \"run_command\": \"$RUN_CMD\", \"message\": \"Coral TPU skill installed — Edge TPU ready\"}"
    log "Done! Edge TPU ready."
    exit 0
else
    emit "{\"event\": \"complete\", \"status\": \"partial\", \"accelerator_found\": false, \"run_command\": \"$RUN_CMD\", \"message\": \"Coral TPU skill installed — no TPU detected (CPU fallback)\"}"
    log "Done with warning: no TPU detected. Connect Coral USB and restart."
    exit 2
fi
