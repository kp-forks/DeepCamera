#!/usr/bin/env python3
"""
Coral TPU Model Compiler — converts YOLO models to Edge TPU format.

Pipeline: YOLO (.pt) → TFLite INT8 → edgetpu_compiler → _edgetpu.tflite

Requirements:
  - Ultralytics (pip install ultralytics)
  - edgetpu_compiler (x86_64 Linux only, or Docker --platform linux/amd64)
  - Calibration images for INT8 quantization

Usage:
  python scripts/compile_model.py --model yolo26n --size 320 --output models/
  python scripts/compile_model.py --model yolo26s --size 640 --output models/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_edgetpu_compiler():
    """Check if edgetpu_compiler is available."""
    try:
        result = subprocess.run(
            ["edgetpu_compiler", "--version"],
            capture_output=True, text=True, timeout=10
        )
        print(f"[compile] edgetpu_compiler: {result.stdout.strip()}")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def export_tflite_int8(model_name, imgsz):
    """Export YOLO model to TFLite INT8 via Ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[compile] ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model_file = f"{model_name}.pt"
    if not os.path.exists(model_file):
        print(f"[compile] Downloading {model_file}...")

    print(f"[compile] Loading model: {model_file}")
    model = YOLO(model_file)

    print(f"[compile] Exporting to TFLite INT8 (imgsz={imgsz})...")
    # Export with full integer quantization for Edge TPU
    result = model.export(
        format="tflite",
        imgsz=imgsz,
        int8=True,       # Full INT8 quantization
        nms=False,        # Edge TPU handles raw output, NMS in post-process
    )

    tflite_path = result
    if not tflite_path or not os.path.exists(str(tflite_path)):
        # Ultralytics may save with different naming
        candidates = list(Path(".").glob(f"**/{model_name}*_int8.tflite"))
        if not candidates:
            candidates = list(Path(".").glob(f"**/{model_name}*.tflite"))
        if candidates:
            tflite_path = str(candidates[0])
        else:
            print("[compile] ERROR: TFLite export failed — no output file found")
            sys.exit(1)

    print(f"[compile] TFLite INT8 model: {tflite_path}")
    return str(tflite_path)


def compile_for_edgetpu(tflite_path, output_dir):
    """Run edgetpu_compiler on the INT8 TFLite model."""
    if not check_edgetpu_compiler():
        print("[compile] ERROR: edgetpu_compiler not found.")
        print("[compile] This tool only runs on x86_64 Linux.")
        print("[compile] Install: https://coral.ai/docs/edgetpu/compiler/")
        print("[compile] Or run inside Docker: --platform linux/amd64")
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compile] Running edgetpu_compiler on {tflite_path}...")
    result = subprocess.run(
        ["edgetpu_compiler", "-s", "-o", str(output_dir), tflite_path],
        capture_output=True, text=True, timeout=300
    )

    print(result.stdout)
    if result.returncode != 0:
        print(f"[compile] ERROR: edgetpu_compiler failed:\n{result.stderr}")
        sys.exit(1)

    # Find the output file
    base_name = Path(tflite_path).stem
    edgetpu_model = output_dir / f"{base_name}_edgetpu.tflite"
    if not edgetpu_model.exists():
        # Look for any _edgetpu.tflite in output dir
        matches = list(output_dir.glob("*_edgetpu.tflite"))
        if matches:
            edgetpu_model = matches[0]
        else:
            print("[compile] ERROR: No _edgetpu.tflite output found")
            sys.exit(1)

    size_mb = edgetpu_model.stat().st_size / (1024 * 1024)
    print(f"[compile] ✓ Edge TPU model: {edgetpu_model} ({size_mb:.1f} MB)")

    # Check compilation log for segment info
    log_file = output_dir / f"{base_name}_edgetpu.log"
    if log_file.exists():
        log_text = log_file.read_text()
        print(f"[compile] Compilation log:\n{log_text}")
        if "not mapped" in log_text.lower():
            print("[compile] WARNING: Some operations not mapped to Edge TPU — will fall back to CPU")

    return str(edgetpu_model)


def main():
    parser = argparse.ArgumentParser(description="Compile YOLO model for Coral Edge TPU")
    parser.add_argument("--model", default="yolo26n", help="YOLO model name (e.g., yolo26n, yolo26s)")
    parser.add_argument("--size", type=int, default=320, help="Input image size (320 or 640)")
    parser.add_argument("--output", default="models/", help="Output directory for compiled model")
    parser.add_argument("--skip-export", action="store_true", help="Skip TFLite export, use existing .tflite")
    parser.add_argument("--tflite", help="Path to existing TFLite INT8 model (with --skip-export)")
    args = parser.parse_args()

    print(f"[compile] Model: {args.model}, Size: {args.size}×{args.size}")

    if args.skip_export and args.tflite:
        tflite_path = args.tflite
    else:
        tflite_path = export_tflite_int8(args.model, args.size)

    edgetpu_path = compile_for_edgetpu(tflite_path, args.output)
    print(f"\n[compile] Done! Model ready at: {edgetpu_path}")
    print(f"[compile] Copy to skills/detection/yolo-detection-2026-coral-tpu/models/")


if __name__ == "__main__":
    main()
