---
name: yolo-detection-2026-coral-tpu
description: "Google Coral Edge TPU — real-time object detection via Docker"
version: 1.0.0
icon: assets/icon.png
entry: scripts/monitor.js
deploy: deploy.sh
runtime: node

requirements:
  platforms: ["linux", "macos", "windows"]

manual_setup:
  macos: |
    # 1. Fix Homebrew permissions and install libusb
    sudo chown -R $(whoami) /opt/homebrew
    brew install libusb

    # 2. Add Apple Silicon Edge TPU native driver
    curl -sSLO https://github.com/feranick/libedgetpu/releases/download/16.0TF2.19.1-1/libedgetpu-16.0-tf2.19.1-1_MacOS_Silicon.zip
    unzip -q -o libedgetpu-16.0-tf2.19.1-1_MacOS_Silicon.zip
    sudo mkdir -p /usr/local/lib
    sudo cp libedgetpu.1.dylib /usr/local/lib/
    rm -rf libedgetpu*

parameters:
  - name: auto_start
    label: "Auto Start"
    type: boolean
    default: false
    description: "Start this skill automatically when Aegis launches"
    group: Lifecycle

  - name: confidence
    label: "Confidence Threshold"
    type: number
    min: 0.1
    max: 1.0
    default: 0.5
    description: "Minimum detection confidence — lower than GPU models due to INT8 quantization"
    group: Model

  - name: classes
    label: "Detect Classes"
    type: string
    default: "person,car,dog,cat"
    description: "Comma-separated COCO class names (80 classes available)"
    group: Model

  - name: fps
    label: "Processing FPS"
    type: select
    options: [0.2, 0.5, 1, 3, 5, 15]
    default: 5
    description: "Frames per second — Edge TPU handles 15+ FPS easily"
    group: Performance

  - name: input_size
    label: "Input Resolution"
    type: select
    options: [320, 640]
    default: 320
    description: "320 fits fully on TPU (~4ms), 640 partially on CPU (~20ms)"
    group: Performance

  - name: tpu_device
    label: "TPU Device"
    type: select
    options: ["auto", "0", "1", "2", "3"]
    default: "auto"
    description: "Which Edge TPU to use — auto selects first available"
    group: Performance

  - name: clock_speed
    label: "TPU Clock Speed"
    type: select
    options: ["standard", "max"]
    default: "standard"
    description: "Max is faster but runs hotter — needs active cooling for sustained use"
    group: Performance

capabilities:
  live_detection:
    script: scripts/detect.py
    description: "Real-time object detection on live camera frames via Edge TPU"

category: detection
mutex: detection
---

# Coral TPU Object Detection

Real-time object detection using Google Coral Edge TPU accelerator. Runs inside Docker for cross-platform support. Detects 80 COCO classes (person, car, dog, cat, etc.) with ~4ms inference on 320×320 input.

## Requirements

- **Google Coral USB Accelerator** (USB 3.0 port recommended)
- **Docker Desktop 4.35+** (all platforms — Linux, macOS, Windows)
- USB 3.0 cable (the included cable is recommended)
- Adequate cooling for sustained inference

## How It Works

```
┌─────────────────────────────────────────────────────┐
│ Host (Aegis-AI)                                     │
│   frame.jpg → /tmp/aegis_detection/                 │
│   stdin  ──→ ┌──────────────────────────────┐       │
│              │ Docker Container              │       │
│              │   detect.py                   │       │
│              │   ├─ loads _edgetpu.tflite     │       │
│              │   ├─ reads frame from volume   │       │
│              │   └─ runs inference on TPU    │       │
│   stdout ←── │   → JSONL detections          │       │
│              └──────────────────────────────┘       │
│   USB ──→ /dev/bus/usb (Linux) or USB/IP (Mac/Win) │
└─────────────────────────────────────────────────────┘
```

1. Aegis writes camera frame JPEG to shared `/tmp/aegis_detection/` volume
2. Sends `frame` event via stdin JSONL to Docker container
3. `detect.py` reads frame, runs inference on Edge TPU
4. Returns `detections` event via stdout JSONL
5. Same protocol as `yolo-detection-2026` — Aegis sees no difference

## Platform Setup

### Linux
```bash
# USB Coral should be auto-detected
# Docker uses --device /dev/bus/usb for direct access
./deploy.sh
```

### macOS (Docker Desktop 4.35+)
```bash
# Docker Desktop USB/IP handles USB passthrough
# ARM64 Docker image builds natively on Apple Silicon
./deploy.sh
```

### Windows
```powershell
# Docker Desktop 4.35+ with USB/IP support
# Or WSL2 backend with usbipd-win
.\deploy.bat
```

## Model

Ships with pre-compiled `yolo26n_edgetpu.tflite` (YOLO 2026 nano, INT8 quantized, 320×320). To compile custom models:

```bash
# Requires x86_64 Linux or Docker --platform linux/amd64
python scripts/compile_model.py --model yolo26s --size 320
```

## Performance

| Input Size | Inference | On-chip | Notes |
|-----------|-----------|---------|-------|
| 320×320 | ~4ms | 100% | Fully on TPU, best for real-time |
| 640×640 | ~20ms | Partial | Some layers on CPU (model segmented) |

> **Cooling**: The USB Accelerator aluminum case acts as a heatsink. If too hot to touch during continuous inference, it will thermal-throttle. Consider active cooling or `clock_speed: standard`.

## Protocol

Same JSONL as `yolo-detection-2026`:

### Skill → Aegis (stdout)
```jsonl
{"event": "ready", "model": "yolo26n_edgetpu", "device": "coral", "format": "edgetpu_tflite", "tpu_count": 1, "classes": 80}
{"event": "detections", "frame_id": 42, "camera_id": "front_door", "objects": [{"class": "person", "confidence": 0.85, "bbox": [100, 50, 300, 400]}]}
{"event": "perf_stats", "total_frames": 50, "timings_ms": {"inference": {"avg": 4.1, "p50": 3.9, "p95": 5.2}}}
```

### Bounding Box Format
`[x_min, y_min, x_max, y_max]` — pixel coordinates (xyxy).

## Installation

```bash
./deploy.sh
```

The deployer builds the Docker image locally, probes for TPU devices, and sets the runtime command. No packages pulled from external registries beyond Docker base images and Coral apt repo.
