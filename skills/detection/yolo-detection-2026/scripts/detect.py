#!/usr/bin/env python3
"""
YOLO 2026 Detection Skill — Real-time object detection for SharpAI Aegis.

Communicates via JSON lines over stdin/stdout:
  stdin:  {"event": "frame", "frame_id": N, "camera_id": "...", "frame_path": "...", ...}
  stdout: {"event": "detections", "frame_id": N, "camera_id": "...", "objects": [...]}

On Apple Silicon (MPS), auto-converts to CoreML for ~2x faster inference via ANE.

Usage:
  python detect.py --config config.json
  python detect.py --model-size nano --confidence 0.5 --device auto
"""

import sys
import json
import argparse
import signal
from pathlib import Path


# Model size → ultralytics model name mapping (YOLO26, released Jan 2026)
MODEL_SIZE_MAP = {
    "nano": "yolo26n",
    "small": "yolo26s",
    "medium": "yolo26m",
    "large": "yolo26l",
}


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 2026 Detection Skill")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--model-size", type=str, default="nano",
                        choices=["nano", "small", "medium", "large"])
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--classes", type=str, default="person,car,dog,cat")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps", "rocm"])
    parser.add_argument("--fps", type=float, default=5)
    return parser.parse_args()


def load_config(args):
    """Load config from JSON file, CLI args, or AEGIS_SKILL_PARAMS env var."""
    import os

    # Priority 1: AEGIS_SKILL_PARAMS env var (set by Aegis skill-runtime-manager)
    env_params = os.environ.get("AEGIS_SKILL_PARAMS")
    if env_params:
        try:
            return json.loads(env_params)
        except json.JSONDecodeError:
            pass

    # Priority 2: Config file
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)

    # Priority 3: CLI args
    return {
        "model_size": args.model_size,
        "confidence": args.confidence,
        "classes": args.classes.split(","),
        "device": args.device,
        "fps": args.fps,
    }


def select_device(preference: str) -> str:
    """Select the best available inference device."""
    if preference not in ("auto", ""):
        return preference
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        # ROCm exposes as CUDA in PyTorch with ROCm builds
    except ImportError:
        pass
    return "cpu"


def emit(event: dict):
    """Write a JSON line to stdout."""
    print(json.dumps(event), flush=True)


def log(msg: str):
    """Write a log message to stderr (visible in Aegis deploy console)."""
    print(f"[YOLO-2026] {msg}", file=sys.stderr, flush=True)


def try_coreml_export(model, model_name: str) -> "Path | None":
    """Export PyTorch model to CoreML. Returns path to .mlpackage or None on failure."""
    coreml_path = Path(f"{model_name}.mlpackage")

    # Already exported
    if coreml_path.exists():
        log(f"CoreML model found: {coreml_path}")
        return coreml_path

    try:
        log(f"Exporting {model_name}.pt → CoreML (one-time, ~30s)...")
        exported = model.export(format="coreml", half=True, nms=False)
        exported_path = Path(exported)
        if exported_path.exists():
            log(f"CoreML export complete: {exported_path}")
            return exported_path
        log(f"CoreML export returned path {exported} but file not found")
    except Exception as e:
        log(f"CoreML export failed: {e}")

    return None


def load_model(model_name: str, device: str, use_coreml: bool):
    """Load YOLO model — CoreML on MPS if available, PyTorch otherwise."""
    from ultralytics import YOLO

    model_format = "pytorch"

    # Try CoreML on Apple Silicon
    if device == "mps" and use_coreml:
        pt_model = YOLO(f"{model_name}.pt")
        coreml_path = try_coreml_export(pt_model, model_name)

        if coreml_path:
            try:
                model = YOLO(str(coreml_path))
                model_format = "coreml"
                log(f"Loaded CoreML model ({coreml_path})")
                return model, model_format
            except Exception as e:
                log(f"CoreML load failed, falling back to PyTorch MPS: {e}")

        # Fallback: use the already-loaded PyTorch model on MPS
        pt_model.to(device)
        return pt_model, model_format

    # Non-CoreML path: standard PyTorch
    model = YOLO(f"{model_name}.pt")
    model.to(device)
    return model, model_format


def main():
    args = parse_args()
    config = load_config(args)

    # Resolve config values
    model_size = config.get("model_size", "nano")
    device = select_device(config.get("device", "auto"))
    confidence = config.get("confidence", 0.5)
    fps = config.get("fps", 5)
    use_coreml = config.get("use_coreml", True)

    # Coerce use_coreml from string "true"/"false" if passed via env
    if isinstance(use_coreml, str):
        use_coreml = use_coreml.lower() in ("true", "1", "yes")

    # Map size to ultralytics model name
    model_name = MODEL_SIZE_MAP.get(model_size, "yolo26n")

    target_classes = config.get("classes", ["person", "car", "dog", "cat"])
    if isinstance(target_classes, str):
        target_classes = [c.strip() for c in target_classes.split(",")]

    # Load YOLO model (with CoreML auto-conversion on MPS)
    try:
        model, model_format = load_model(model_name, device, use_coreml)
        emit({
            "event": "ready",
            "model": f"yolo2026{model_size[0]}",
            "model_size": model_size,
            "device": device,
            "format": model_format,
            "classes": len(model.names),
            "fps": fps,
            "available_sizes": list(MODEL_SIZE_MAP.keys()),
        })
    except Exception as e:
        emit({"event": "error", "message": f"Failed to load model: {e}", "retriable": False})
        sys.exit(1)

    # Graceful shutdown
    running = True
    def handle_signal(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Main loop: read frames from stdin, output detections to stdout
    for line in sys.stdin:
        if not running:
            break

        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        if msg.get("command") == "stop":
            break

        if msg.get("event") == "frame":
            frame_path = msg.get("frame_path")
            frame_id = msg.get("frame_id")
            camera_id = msg.get("camera_id", "unknown")
            timestamp = msg.get("timestamp", "")

            if not frame_path or not Path(frame_path).exists():
                emit({
                    "event": "error",
                    "frame_id": frame_id,
                    "message": f"Frame not found: {frame_path}",
                    "retriable": True,
                })
                continue

            # Run inference
            try:
                results = model(frame_path, conf=confidence, verbose=False)
                objects = []
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        if cls_name in target_classes or not target_classes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            objects.append({
                                "class": cls_name,
                                "confidence": round(float(box.conf[0]), 3),
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            })

                emit({
                    "event": "detections",
                    "frame_id": frame_id,
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "objects": objects,
                })
            except Exception as e:
                emit({
                    "event": "error",
                    "frame_id": frame_id,
                    "message": f"Inference error: {e}",
                    "retriable": True,
                })


if __name__ == "__main__":
    main()
