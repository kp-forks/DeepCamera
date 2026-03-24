---
name: yolo-detection-2026-openvino
description: "OpenVINO — real-time object detection via Docker (NCS2, Intel GPU, CPU)"
version: 1.0.0
icon: assets/icon.png
entry: scripts/detect.py
deploy: deploy.sh
runtime: docker

requirements:
  docker: ">=20.10"
  platforms: ["linux", "macos", "windows"]

parameters:
  - name: auto_start
    label: "Auto Start"
    type: boolean
    default: false
    group: Lifecycle

  - name: confidence
    label: "Confidence Threshold"
    type: number
    min: 0.1
    max: 1.0
    default: 0.5
    group: Model

  - name: classes
    label: "Detect Classes"
    type: string
    default: "person,car,dog,cat"
    group: Model

  - name: fps
    label: "Processing FPS"
    type: select
    options: [0.2, 0.5, 1, 3, 5, 15]
    default: 5
    group: Performance

  - name: input_size
    label: "Input Resolution"
    type: select
    options: [320, 640]
    default: 640
    group: Performance

  - name: device
    label: "Inference Device"
    type: select
    options: ["AUTO", "CPU", "GPU", "MYRIAD"]
    default: "AUTO"
    description: "AUTO lets OpenVINO pick the fastest available device"
    group: Performance

  - name: precision
    label: "Model Precision"
    type: select
    options: ["FP16", "INT8", "FP32"]
    default: "FP16"
    group: Performance

capabilities:
  live_detection:
    script: scripts/detect.py
    description: "Real-time object detection via OpenVINO"

category: detection
mutex: detection
---

# OpenVINO Object Detection

Real-time object detection using Intel OpenVINO runtime. Runs inside Docker for cross-platform support. Supports Intel NCS2 USB stick, Intel integrated GPU, Intel Arc discrete GPU, and any x86_64 CPU.

## Requirements

- **Docker Desktop 4.35+** (all platforms)
- **Optional hardware**: Intel NCS2 USB, Intel iGPU, Intel Arc GPU
- Falls back to CPU if no accelerator present

## How It Works

```
┌─────────────────────────────────────────────────────┐
│ Host (Aegis-AI)                                     │
│   frame.jpg → /tmp/aegis_detection/                 │
│   stdin  ──→ ┌──────────────────────────────┐       │
│              │ Docker Container              │       │
│              │   detect.py                   │       │
│              │   ├─ loads OpenVINO IR model   │       │
│              │   ├─ reads frame from volume   │       │
│              │   └─ runs inference on device  │       │
│   stdout ←── │   → JSONL detections          │       │
│              └──────────────────────────────┘       │
│   USB ──→ /dev/bus/usb (NCS2)                       │
│   DRI ──→ /dev/dri (Intel GPU)                      │
└─────────────────────────────────────────────────────┘
```

## Supported Devices

| Device | Flag | Precision | ~Speed |
|--------|------|-----------|--------|
| Intel NCS2 | `MYRIAD` | FP16 | ~15ms |
| Intel iGPU | `GPU` | FP16/INT8 | ~8ms |
| Intel Arc | `GPU` | FP16/INT8 | ~4ms |
| Any CPU | `CPU` | FP32/INT8 | ~25ms |
| Auto | `AUTO` | Best | Auto |

## Protocol

Same JSONL as `yolo-detection-2026`:

```jsonl
{"event": "ready", "model": "yolo26n_openvino", "device": "GPU", "format": "openvino_ir", "classes": 80}
{"event": "detections", "frame_id": 42, "camera_id": "front_door", "objects": [{"class": "person", "confidence": 0.85, "bbox": [100, 50, 300, 400]}]}
{"event": "perf_stats", "total_frames": 50, "timings_ms": {"inference": {"avg": 8.1, "p50": 7.9, "p95": 10.2}}}
```

## Installation

```bash
./deploy.sh
```
