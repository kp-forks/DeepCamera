#!/bin/bash
# deploy.sh — Platform-aware dependency install for Depth Estimation
#
# macOS:  CoreML only (fast ~10s install, Neural Engine inference)
# Other:  Full PyTorch stack (torch + torchvision + depth-anything-v2)
#
# The Aegis deployment agent calls this instead of raw pip install.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MODELS_DIR="$HOME/.aegis-ai/models/feature-extraction"
COREML_VARIANT="DepthAnythingV2SmallF16"
COREML_HF_REPO="apple/coreml-depth-anything-v2-small"

echo "=== Depth Estimation (Privacy) — Setup ==="
echo "Platform: $(uname -s) / $(uname -m)"

# ── Create venv ──────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

# Upgrade pip
"$PIP" install --upgrade pip --quiet

# ── Platform detection ───────────────────────────────────────────────
if [ "$(uname -s)" = "Darwin" ]; then
    echo ""
    echo "=== macOS detected — CoreML backend (Neural Engine) ==="
    echo "Installing CoreML dependencies only (fast)..."
    "$PIP" install --quiet \
        "coremltools>=8.0" \
        "huggingface_hub>=0.20.0" \
        "numpy>=1.24.0" \
        "opencv-python-headless>=4.8.0" \
        "Pillow>=10.0.0" \
        "matplotlib>=3.7.0"

    echo "✅ CoreML dependencies installed"

    # ── Download CoreML model if not present ─────────────────────────
    MODEL_PATH="$MODELS_DIR/$COREML_VARIANT.mlpackage"
    if [ -d "$MODEL_PATH" ]; then
        echo "✅ CoreML model already present: $MODEL_PATH"
    else
        echo "Downloading CoreML model: $COREML_VARIANT from $COREML_HF_REPO..."
        mkdir -p "$MODELS_DIR"
        "$PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '$COREML_HF_REPO',
    local_dir='$MODELS_DIR',
    allow_patterns=['$COREML_VARIANT.mlpackage/**'],
)
print('✅ CoreML model downloaded')
"
    fi

    # Verify
    "$PYTHON" -c "
import coremltools, cv2, numpy, PIL
from pathlib import Path
model_path = Path('$MODEL_PATH')
assert model_path.exists(), f'Model not found: {model_path}'
print(f'✅ Verified: coremltools={coremltools.__version__}, model={model_path.name}')
"

else
    echo ""
    echo "=== Non-macOS — PyTorch backend ==="
    echo "Installing full PyTorch dependencies..."
    "$PIP" install --quiet -r "$SCRIPT_DIR/requirements.txt"

    echo "✅ PyTorch dependencies installed"

    # Verify
    "$PYTHON" -c "
import torch, cv2, numpy, PIL
from depth_anything_v2.dpt import DepthAnythingV2
print(f'✅ Verified: torch={torch.__version__}, CUDA={torch.cuda.is_available()}')
"
fi

echo ""
echo "=== Setup complete ==="
