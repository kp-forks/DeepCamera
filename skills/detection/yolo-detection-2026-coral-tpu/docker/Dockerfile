# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Coral Edge TPU Model Compiler
#
# Converts YOLO 2026 nano (.pt) → ONNX → TFLite INT8 → Edge TPU (.tflite)
#
# IMPORTANT: Must run on linux/amd64. edgetpu_compiler only exists for
# x86_64 Linux. On Apple Silicon or Windows, Docker Desktop handles emulation
# via QEMU automatically — just pass --platform linux/amd64.
#
# Why tensorflow 2.13?
#   edgetpu_compiler v16.0 only understands TFLite flatbuffer schema ≤ v3.
#   tensorflow-cpu 2.13.1 is the last version that produces schema v3.
#   TF 2.14+ produces schema v6 → "Invalid model" error from the compiler.
#
# Why NOT onnx2tf?
#   onnx2tf ≥ 1.17 + TF 2.13 hits a TF-Keras symbolic-tensor TypeError
#   during INT8 calibration. We bypass it entirely by using the ONNX Runtime
#   directly to run calibration, then drive TFLiteConverter ourselves.
#   This is the approach used in compile_model.py's two-stage pipeline.
# ─────────────────────────────────────────────────────────────────────────────

FROM --platform=linux/amd64 python:3.9-slim

LABEL maintainer="Aegis-AI / DeepCamera"
LABEL description="Compiles YOLO 2026 nano to Google Coral Edge TPU .tflite"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        apt-transport-https \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Google Coral edgetpu_compiler (x86_64 Debian package) ────────────────────
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] \
        https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        > /etc/apt/sources.list.d/coral-edgetpu.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends edgetpu-compiler \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
# Pipeline: ultralytics exports ONNX, then we convert ONNX→TFLite INT8
# using onnx-tf (not onnx2tf) + tensorflow 2.13 + tflite-support calibrator.
#
# tensorflow-cpu==2.13.1  → TFLite schema v3, compatible with edgetpu_compiler v16
# onnx-tf==1.10.0         → stable ONNX→TF SavedModel, works with TF 2.13
# ultralytics             → YOLO .pt download + ONNX export only (not tflite export)
# onnxruntime             → ONNX inference for INT8 representative dataset generation
# numpy<2                 → required by tensorflow 2.13
RUN pip install --no-cache-dir \
        "tensorflow-cpu==2.13.1" \
        "tensorflow_probability==0.21.0" \
        "keras==2.13.1" \
        "onnx-tf==1.10.0" \
        "onnx>=1.14.0,<1.17.0" \
        "onnxruntime==1.17.3" \
        "ultralytics>=8.3.0" \
        "numpy>=1.24.0,<2.0" \
        "Pillow>=10.0.0"

# ── Copy compile script ───────────────────────────────────────────────────────
WORKDIR /compile
COPY scripts/compile_model.py /compile/compile_model.py

# ── Output volume ─────────────────────────────────────────────────────────────
VOLUME ["/compile/output"]

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "/compile/compile_model.py"]
CMD ["--model", "yolo26n", "--size", "320", "--output", "/compile/output"]
