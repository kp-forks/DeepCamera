#!/usr/bin/env bash
# deploy.sh — Zero-assumption bootstrapper for YOLO 2026 Detection Skill
#
# Probes the system for Python, GPU backends, and installs the minimum
# viable stack. Called by Aegis skill-runtime-manager during installation.
#
# Exit codes:
#   0  = success
#   1  = fatal error (no Python found and cannot install)
#   2  = partial success (CPU-only fallback)

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SKILL_DIR/.venv"
LOG_PREFIX="[YOLO-2026-deploy]"

log()  { echo "$LOG_PREFIX $*" >&2; }
emit() { echo "$1"; }  # JSON to stdout for Aegis to parse

# ─── Step 1: Find or install Python ─────────────────────────────────────────

find_python() {
    # Check common Python 3 locations
    for cmd in python3.12 python3.11 python3.10 python3.9 python3; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver="$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')"
            local major minor
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done

    # Check conda
    if command -v conda &>/dev/null; then
        log "No system Python >=3.9 found, but conda is available"
        log "Creating conda environment..."
        conda create -n aegis-yolo2026 python=3.11 -y >/dev/null 2>&1
        # shellcheck disable=SC1091
        eval "$(conda shell.bash hook 2>/dev/null)"
        conda activate aegis-yolo2026
        echo "python3"
        return 0
    fi

    # Check pyenv
    if command -v pyenv &>/dev/null; then
        log "No system Python >=3.9 found, using pyenv..."
        pyenv install -s 3.11.9
        pyenv local 3.11.9
        echo "$(pyenv which python3)"
        return 0
    fi

    return 1
}

PYTHON_CMD=$(find_python) || {
    log "ERROR: No Python >=3.9 found. Install Python 3.9+ and retry."
    emit '{"event": "error", "stage": "python", "message": "No Python >=3.9 found"}'
    exit 1
}

log "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"
emit "{\"event\": \"progress\", \"stage\": \"python\", \"message\": \"Found $($PYTHON_CMD --version 2>&1)\"}"

# ─── Step 2: Create virtual environment ─────────────────────────────────────

if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PIP="$VENV_DIR/bin/pip"

# Upgrade pip
"$PIP" install --upgrade pip -q 2>/dev/null || true

emit '{"event": "progress", "stage": "venv", "message": "Virtual environment ready"}'

# ─── Step 3: Detect compute backend ─────────────────────────────────────────

BACKEND="cpu"

detect_gpu() {
    # NVIDIA CUDA
    if command -v nvidia-smi &>/dev/null; then
        local cuda_ver
        cuda_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$cuda_ver" ]; then
            BACKEND="cuda"
            log "Detected NVIDIA GPU (driver: $cuda_ver)"
            return 0
        fi
    fi

    # AMD ROCm
    if command -v rocm-smi &>/dev/null || [ -d "/opt/rocm" ]; then
        BACKEND="rocm"
        log "Detected AMD ROCm"
        return 0
    fi

    # Apple Silicon MPS
    if [ "$(uname)" = "Darwin" ]; then
        local arch
        arch=$(uname -m)
        if [ "$arch" = "arm64" ]; then
            BACKEND="mps"
            log "Detected Apple Silicon (MPS)"
            return 0
        fi
    fi

    log "No GPU detected, using CPU backend"
    return 0
}

detect_gpu
emit "{\"event\": \"progress\", \"stage\": \"gpu\", \"backend\": \"$BACKEND\", \"message\": \"Compute backend: $BACKEND\"}"

# ─── Step 4: Install requirements ────────────────────────────────────────────

REQ_FILE="$SKILL_DIR/requirements_${BACKEND}.txt"

if [ ! -f "$REQ_FILE" ]; then
    log "WARNING: $REQ_FILE not found, falling back to CPU"
    REQ_FILE="$SKILL_DIR/requirements_cpu.txt"
    BACKEND="cpu"
fi

log "Installing dependencies from $REQ_FILE ..."
emit "{\"event\": \"progress\", \"stage\": \"install\", \"message\": \"Installing $BACKEND dependencies...\"}"

"$PIP" install -r "$REQ_FILE" -q 2>&1 | tail -5 >&2

# ─── Step 5: CoreML pre-conversion (MPS only) ───────────────────────────────

if [ "$BACKEND" = "mps" ]; then
    log "Pre-converting default model to CoreML for ANE acceleration..."
    emit '{"event": "progress", "stage": "coreml", "message": "Converting model to CoreML (~30s)..."}'

    "$VENV_DIR/bin/python" -c "
from ultralytics import YOLO
model = YOLO('yolo26n.pt')
exported = model.export(format='coreml', half=True, nms=False)
print(f'CoreML model exported: {exported}')
" 2>&1 | while read -r line; do log "$line"; done

    if [ $? -eq 0 ]; then
        emit '{"event": "progress", "stage": "coreml", "message": "CoreML conversion complete"}'
    else
        log "WARNING: CoreML conversion failed, will use PyTorch MPS at runtime"
        emit '{"event": "progress", "stage": "coreml", "message": "CoreML conversion failed — PyTorch MPS fallback"}'
    fi
fi

# ─── Step 6: Verify installation ────────────────────────────────────────────

log "Verifying installation..."
"$VENV_DIR/bin/python" -c "
from ultralytics import YOLO
import torch
device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): device = 'mps'
print(f'OK: ultralytics loaded, torch device={device}')
" 2>&1 | while read -r line; do log "$line"; done

emit "{\"event\": \"complete\", \"backend\": \"$BACKEND\", \"message\": \"YOLO 2026 skill installed ($BACKEND backend)\"}"
log "Done! Backend: $BACKEND"

