#!/usr/bin/env python3
"""
Coral TPU Object Detection — JSONL stdin/stdout protocol
Runs inside Docker container with Edge TPU access.
Same protocol as yolo-detection-2026/scripts/detect.py.

Communication:
  stdin:  {"event": "frame", "frame_id": N, "frame_path": "...", ...}
  stdout: {"event": "detections", "frame_id": N, "objects": [...]}
  stderr: Debug logs (ignored by Aegis parser)
"""

import json
import os
import sys
import time
import signal
from pathlib import Path

import numpy as np
from PIL import Image

# ─── Edge TPU imports ─────────────────────────────────────────────────────────
try:
    from pycoral.adapters import common
    from pycoral.adapters import detect
    from pycoral.utils.edgetpu import list_edge_tpus, make_interpreter
    HAS_EDGETPU = True
except ImportError:
    HAS_EDGETPU = False
    sys.stderr.write("[coral-detect] WARNING: pycoral not available, running in CPU-fallback mode\n")

try:
    import tflite_runtime.interpreter as tflite
    HAS_TFLITE = True
except ImportError:
    HAS_TFLITE = False


# ─── COCO class names (80 classes) ───────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


class PerfTracker:
    """Tracks per-frame timing and emits aggregate stats."""

    def __init__(self, emit_interval=50):
        self.emit_interval = emit_interval
        self.timings = []
        self.total_frames = 0

    def record(self, timing_dict):
        self.timings.append(timing_dict)
        self.total_frames += 1

    def should_emit(self):
        return len(self.timings) >= self.emit_interval

    def emit_and_reset(self):
        if not self.timings:
            return None

        stats = {"event": "perf_stats", "total_frames": len(self.timings), "timings_ms": {}}
        for key in self.timings[0]:
            values = sorted([t[key] for t in self.timings])
            n = len(values)
            stats["timings_ms"][key] = {
                "avg": round(sum(values) / n, 2),
                "p50": round(values[n // 2], 2),
                "p95": round(values[int(n * 0.95)], 2),
                "p99": round(values[int(n * 0.99)], 2),
            }
        self.timings = []
        return stats


class CoralDetector:
    """Edge TPU object detector using pycoral."""

    def __init__(self, params):
        self.params = params
        self.confidence = float(params.get("confidence", 0.5))
        self.input_size = int(params.get("input_size", 320))
        self.tpu_device = params.get("tpu_device", "auto")
        self.clock_speed = params.get("clock_speed", "standard")
        self.interpreter = None
        self.tpu_count = 0

        # Parse target classes
        classes_str = params.get("classes", "person,car,dog,cat")
        self.target_classes = set(c.strip().lower() for c in classes_str.split(","))

        self._load_model()

    def _find_model_path(self):
        """Find the compiled Edge TPU model."""
        model_dir = Path("/app/models")
        script_dir = Path(__file__).parent.parent / "models"

        for d in [model_dir, script_dir]:
            for pattern in ["*_edgetpu.tflite", "*.tflite"]:
                matches = list(d.glob(pattern))
                if matches:
                    return str(matches[0])

        return None

    def _load_model(self):
        """Load model onto Edge TPU (or CPU fallback)."""
        model_path = self._find_model_path()
        if not model_path:
            log("ERROR: No .tflite model found in /app/models/")
            emit_json({"event": "error", "message": "No Edge TPU model found", "retriable": False})
            sys.exit(1)

        # Enumerate TPUs
        if HAS_EDGETPU:
            tpus = list_edge_tpus()
            self.tpu_count = len(tpus)
            log(f"Found {self.tpu_count} Edge TPU(s): {tpus}")

            if self.tpu_count == 0:
                log("WARNING: No Edge TPU detected — falling back to CPU TFLite")
                self._load_cpu_fallback(model_path)
                return

            # Select TPU device
            device_idx = None
            if self.tpu_device != "auto":
                device_idx = int(self.tpu_device)
                if device_idx >= self.tpu_count:
                    log(f"WARNING: TPU index {device_idx} not available, using auto")
                    device_idx = None

            try:
                if device_idx is not None:
                    device_str = f":{ device_idx}"
                    self.interpreter = make_interpreter(model_path, device=device_str)
                else:
                    self.interpreter = make_interpreter(model_path)
                self.interpreter.allocate_tensors()
                self.device_name = "coral"
                log(f"Loaded model on Edge TPU: {model_path}")
            except Exception as e:
                log(f"ERROR loading on Edge TPU: {e}, falling back to CPU")
                self._load_cpu_fallback(model_path)
        else:
            self._load_cpu_fallback(model_path)

    def _load_cpu_fallback(self, model_path):
        """Fallback to CPU-only TFLite interpreter."""
        if not HAS_TFLITE:
            log("FATAL: Neither pycoral nor tflite-runtime available")
            emit_json({"event": "error", "message": "No inference runtime available", "retriable": False})
            sys.exit(1)

        # Use a non-edgetpu model if available
        cpu_path = model_path.replace("_edgetpu.tflite", ".tflite")
        if not os.path.exists(cpu_path):
            cpu_path = model_path  # Try with edgetpu model (may fail)

        self.interpreter = tflite.Interpreter(model_path=cpu_path)
        self.interpreter.allocate_tensors()
        self.device_name = "cpu"
        log(f"Loaded model on CPU: {cpu_path}")

    def detect_frame(self, frame_path):
        """Run detection on a single frame. Returns list of detection dicts."""
        t0 = time.perf_counter()

        # Read and resize image
        try:
            img = Image.open(frame_path).convert("RGB")
        except Exception as e:
            log(f"ERROR reading frame: {e}")
            return [], {}

        t_read = time.perf_counter()

        # Resize to model input size
        input_details = self.interpreter.get_input_details()[0]
        input_shape = input_details["shape"]
        h, w = input_shape[1], input_shape[2]
        orig_w, orig_h = img.size
        img_resized = img.resize((w, h), Image.LANCZOS)

        # Set input tensor
        input_data = np.expand_dims(np.array(img_resized, dtype=np.uint8), axis=0)
        self.interpreter.set_tensor(input_details["index"], input_data)

        # Run inference
        t_pre = time.perf_counter()
        self.interpreter.invoke()
        t_infer = time.perf_counter()

        # Parse output — pycoral detect API if available
        objects = []
        if HAS_EDGETPU and self.device_name == "coral":
            try:
                raw_detections = detect.get_objects(
                    self.interpreter, score_threshold=self.confidence
                )
                for det in raw_detections:
                    class_id = det.id
                    if class_id < len(COCO_CLASSES):
                        class_name = COCO_CLASSES[class_id]
                    else:
                        class_name = f"class_{class_id}"

                    if self.target_classes and class_name not in self.target_classes:
                        continue

                    bbox = det.bbox
                    # Scale bbox from model input coords to original image coords
                    x_min = int(bbox.xmin * orig_w / w)
                    y_min = int(bbox.ymin * orig_h / h)
                    x_max = int(bbox.xmax * orig_w / w)
                    y_max = int(bbox.ymax * orig_h / h)

                    objects.append({
                        "class": class_name,
                        "confidence": round(float(det.score), 3),
                        "bbox": [x_min, y_min, x_max, y_max]
                    })
            except Exception as e:
                log(f"ERROR parsing detections: {e}")
        else:
            # CPU fallback: manual output parsing
            output_details = self.interpreter.get_output_details()
            if len(output_details) >= 4:
                boxes = self.interpreter.get_tensor(output_details[0]["index"])[0]
                classes = self.interpreter.get_tensor(output_details[1]["index"])[0]
                scores = self.interpreter.get_tensor(output_details[2]["index"])[0]
                count = int(self.interpreter.get_tensor(output_details[3]["index"])[0])

                for i in range(min(count, 25)):
                    score = float(scores[i])
                    if score < self.confidence:
                        continue
                    class_id = int(classes[i])
                    if class_id < len(COCO_CLASSES):
                        class_name = COCO_CLASSES[class_id]
                    else:
                        class_name = f"class_{class_id}"

                    if self.target_classes and class_name not in self.target_classes:
                        continue

                    y1, x1, y2, x2 = boxes[i]
                    objects.append({
                        "class": class_name,
                        "confidence": round(score, 3),
                        "bbox": [
                            int(x1 * orig_w), int(y1 * orig_h),
                            int(x2 * orig_w), int(y2 * orig_h)
                        ]
                    })

        t_post = time.perf_counter()

        timings = {
            "file_read": round((t_read - t0) * 1000, 2),
            "preprocess": round((t_pre - t_read) * 1000, 2),
            "inference": round((t_infer - t_pre) * 1000, 2),
            "postprocess": round((t_post - t_infer) * 1000, 2),
            "total": round((t_post - t0) * 1000, 2),
        }

        return objects, timings


# ─── Helpers ──────────────────────────────────────────────────────────────────

def log(msg):
    """Write to stderr (ignored by Aegis parser)."""
    sys.stderr.write(f"[coral-detect] {msg}\n")
    sys.stderr.flush()


def emit_json(obj):
    """Emit JSONL to stdout."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# ─── Main loop ───────────────────────────────────────────────────────────────

def main():
    # Parse params from environment
    params_str = os.environ.get("AEGIS_SKILL_PARAMS", "{}")
    try:
        params = json.loads(params_str)
    except json.JSONDecodeError:
        params = {}

    log(f"Starting with params: {json.dumps(params)}")

    # Initialize detector
    detector = CoralDetector(params)
    perf = PerfTracker(emit_interval=50)

    # Emit ready event
    emit_json({
        "event": "ready",
        "model": "yolo26n_edgetpu",
        "device": detector.device_name,
        "format": "edgetpu_tflite" if detector.device_name == "coral" else "tflite_cpu",
        "tpu_count": detector.tpu_count,
        "classes": len(COCO_CLASSES),
        "input_size": detector.input_size,
        "fps": params.get("fps", 5),
    })

    # Handle graceful shutdown
    running = True
    def on_signal(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    # JSONL request-response loop
    log("Ready — waiting for frame events on stdin")
    for line in sys.stdin:
        if not running:
            break

        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            log(f"Invalid JSON: {line[:100]}")
            continue

        # Handle stop command
        if msg.get("command") == "stop" or msg.get("event") == "stop":
            log("Received stop command")
            break

        # Handle frame event
        if msg.get("event") == "frame":
            frame_id = msg.get("frame_id", 0)
            frame_path = msg.get("frame_path", "")
            camera_id = msg.get("camera_id", "")
            timestamp = msg.get("timestamp", "")

            if not frame_path or not os.path.exists(frame_path):
                log(f"Frame not found: {frame_path}")
                emit_json({
                    "event": "detections",
                    "frame_id": frame_id,
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "objects": [],
                })
                continue

            objects, timings = detector.detect_frame(frame_path)

            # Emit detections
            emit_json({
                "event": "detections",
                "frame_id": frame_id,
                "camera_id": camera_id,
                "timestamp": timestamp,
                "objects": objects,
            })

            # Track performance
            if timings:
                perf.record(timings)
                if perf.should_emit():
                    stats = perf.emit_and_reset()
                    if stats:
                        emit_json(stats)

    log("Shutting down")


if __name__ == "__main__":
    main()
