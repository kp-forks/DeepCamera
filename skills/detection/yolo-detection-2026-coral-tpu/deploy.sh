#!/usr/bin/env bash
# deploy.sh — Native local bootstrapper for Coral TPU Detection Skill
#
# Builds a local Python virtual environment and installs the Google Coral Edge TPU
# natively on the host OS. Safely prompts for sudo inline to execute driver hooks.
#
# Exit codes:
#   0 = success
#   1 = fatal error (Python/pip not found or sudo denied)
#   2 = partial success (no TPU detected, will use CPU fallback)

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_PREFIX="[coral-tpu-deploy]"

log()  { echo "$LOG_PREFIX $*" >&2; }
emit() { echo "$1"; }

# ─── Inline Sudo Wrapper ────────────────────────────────────────────────────
ask_sudo() {
    local cmd="$1"
    local desc="$2"
    local manual="$3"
    
    emit "{\"event\": \"progress\", \"stage\": \"platform\", \"message\": \"Sudo required: $desc\"}"
    log "Attempting to run: $desc"
    
    # Fast path: If passwordless sudo works, execute immediately.
    if sudo -n true 2>/dev/null; then
        eval "sudo $cmd"
        return $?
    fi

    echo ""
    echo -e "\033[1;33m[SUDO REQUIRED]\033[0m Aegis needs administrator privileges to $desc."
    echo "A secure native OS authorization window should appear shortly."
    echo "If it does not appear or you cancel the prompt, manual fallback instructions will be provided."
    echo ""
    
    local success=0
    
    if [ "$PLATFORM" = "Darwin" ]; then
        # Use native macOS GUI authorization (TouchID/Password)
        if ! osascript -e "do shell script \"$cmd\" with administrator privileges" >/dev/null 2>&1; then
            success=1
        fi
    elif [ "$PLATFORM" = "Linux" ]; then
        # Check if pkexec is available for native Linux GUI authorization
        if command -v pkexec &>/dev/null; then
            if ! pkexec bash -c "$cmd" >/dev/null 2>&1; then
                success=1
            fi
        else
            # Fallback to sudo if run within an interactive TTY
            if ! sudo -v &>/dev/null || ! eval "sudo $cmd"; then
                success=1
            fi
        fi
    else
        success=1
    fi

    if [ "$success" -ne 0 ]; then
        echo ""
        echo -e "\033[1;31m[MANUAL SETUP REQUIRED]\033[0m Sudo prompt was skipped or user aborted."
        echo "Please execute the following fragile instructions manually in a global OS terminal:"
        echo ""
        echo -e "\033[1;36m$manual\033[0m"
        echo ""
        echo "Once completed, re-run this deployment."
        emit '{"event": "error", "stage": "platform", "message": "Manual OS setup required (sudo skipped)"}'
        exit 1
    fi
}

PLATFORM="$(uname -s)"
ARCH="$(uname -m)"
log "Platform: $PLATFORM ($ARCH)"

# ─── Step 1: Install Native OS TPU Drivers ──────────────────────────────────
if [ "$PLATFORM" = "Linux" ]; then
    log "Linux: ensuring Coral Edge TPU system packages..."
    
    MANUAL_LINUX="echo \"deb https://packages.cloud.google.com/apt coral-edgetpu-stable main\" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y libedgetpu1-max libusb-1.0-0"

    ask_sudo "apt-get update && apt-get install -y --no-install-recommends curl gnupg && \
              echo 'deb https://packages.cloud.google.com/apt coral-edgetpu-stable main' | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
              curl -sL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
              apt-get update && \
              apt-get install -y libedgetpu1-max libusb-1.0-0 python3 python3-pip python3-venv" \
             "Install native Google Coral APT repository and TPU runtime" \
             "$MANUAL_LINUX"

elif [ "$PLATFORM" = "Darwin" ]; then
    log "macOS: Installing libusb and Native Edge TPU Driver..."

    # Ensure libusb exists via brew without sudo
    if command -v brew &>/dev/null; then
        brew install libusb || true
    fi

    if [ "$ARCH" = "arm64" ]; then
        MANUAL_MAC="curl -sSLO https://github.com/feranick/libedgetpu/releases/download/16.0TF2.19.1-1/libedgetpu-16.0-tf2.19.1-1_MacOS_Silicon.zip
unzip -q -o libedgetpu-16.0-tf2.19.1-1_MacOS_Silicon.zip
sudo mkdir -p /usr/local/lib
sudo cp libedgetpu.1.dylib /usr/local/lib/
rm -rf libedgetpu*"

        log "Downloading Apple Silicon arm64 driver payload..."
        TMP_DIR=$(mktemp -d)
        cd "$TMP_DIR"
        curl -sSLO https://github.com/feranick/libedgetpu/releases/download/16.0TF2.19.1-1/libedgetpu-16.0-tf2.19.1-1_MacOS_Silicon.zip
        unzip -q -o libedgetpu-16.0-tf2.19.1-1_MacOS_Silicon.zip
        ask_sudo "mkdir -p /usr/local/lib && cp libedgetpu.1.dylib /usr/local/lib/" "Install libedgetpu.1.dylib to /usr/local/lib" "$MANUAL_MAC"
        cd "$SKILL_DIR"
        rm -rf "$TMP_DIR"
    else
        MANUAL_MAC="curl -LO https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip
unzip edgetpu_runtime_20221024.zip
cd edgetpu_runtime
sudo bash install.sh"

        log "Downloading Google official x86_64 installer..."
        TMP_DIR=$(mktemp -d)
        cd "$TMP_DIR"
        curl -sSLO https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip
        unzip -q -o edgetpu_runtime_20221024.zip
        cd edgetpu_runtime
        ask_sudo "bash install.sh </dev/null" "Execute official Google Coral install.sh" "$MANUAL_MAC"
        cd "$SKILL_DIR"
        rm -rf "$TMP_DIR"
    fi
fi

# ─── Step 2: Ensure Python 3 ────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    log "ERROR: Python 3 not found."
    emit '{"event": "error", "stage": "python", "message": "Python 3 not found"}'
    exit 1
fi

PYTHON_CMD="python3"
log "Using Python: $($PYTHON_CMD --version)"

# ─── Step 3: Create Virtual Environment ─────────────────────────────────────
VENV_DIR="$SKILL_DIR/venv"
log "Setting up virtual environment with system packages in $VENV_DIR..."
emit '{"event": "progress", "stage": "build", "message": "Creating Python virtual environment..."}'

# Use --system-site-packages so Linux can inherit python3-pycoral if installed globally
"$PYTHON_CMD" -m venv --system-site-packages "$VENV_DIR"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    log "ERROR: Failed to create virtual environment."
    emit '{"event": "error", "stage": "build", "message": "Failed to create venv"}'
    exit 1
fi

# ─── Step 4: Install PyCoral & Dependencies ─────────────────────────────────
log "Installing Python dependencies (including pycoral)..."
emit '{"event": "progress", "stage": "build", "message": "Fetching pycoral specific wheels..."}'

"$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null 2>&1 || true

# Explicitly fetch pycoral using the extra URL provided in the instructions
if ! "$VENV_DIR/bin/python" -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0; then
    log "WARNING: pip install pycoral failed. Will attempt to install standard requirements anyway."
fi

if ! "$VENV_DIR/bin/python" -m pip install -r "$SKILL_DIR/requirements.txt"; then
    log "ERROR: Failed to install Python dependencies."
    emit '{"event": "error", "stage": "build", "message": "pip install failed"}'
    exit 1
fi

log "Dependencies installed successfully."

# ─── Step 5: Probe for Edge TPU devices ──────────────────────────────────────
log "Probing for Edge TPU devices natively..."
emit '{"event": "progress", "stage": "probe", "message": "Checking for physical Edge TPU..."}'

TPU_FOUND=false
PROBE_OUTPUT=$("$VENV_DIR/bin/python" "$SKILL_DIR/scripts/tpu_probe.py" 2>/dev/null) || true

if echo "$PROBE_OUTPUT" | grep -q '"available": true'; then
    TPU_COUNT=$(echo "$PROBE_OUTPUT" | "$VENV_DIR/bin/python" -c "import sys,json; print(json.load(sys.stdin)['count'])" 2>/dev/null || echo "?")
    TPU_FOUND=true
    emit "{\"event\": \"progress\", \"stage\": \"probe\", \"message\": \"Found $TPU_COUNT Edge TPU device(s) natively\"}"
else
    emit '{"event": "progress", "stage": "probe", "message": "No Edge TPU detected — CPU fallback available"}'
fi

# ─── Step 6: Complete ────────────────────────────────────────────────────────
if [ "$TPU_FOUND" = true ]; then
    emit "{\"event\": \"complete\", \"status\": \"success\", \"tpu_found\": true, \"message\": \"Native Coral TPU skill installed — Edge TPU ready\"}"
    exit 0
else
    emit "{\"event\": \"complete\", \"status\": \"partial\", \"tpu_found\": false, \"message\": \"Native Coral TPU skill installed — no TPU detected (CPU fallback)\"}"
    exit 2
fi
