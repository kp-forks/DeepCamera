---
name: yolo-detection-2026
description: "YOLO 2026 — state-of-the-art real-time object detection"
version: 1.0.0
icon: assets/icon.png
entry: scripts/detect.py

parameters:
  - name: model_size
    label: "Model Size"
    type: select
    options: ["nano", "small", "medium", "large"]
    default: "nano"
    description: "Larger models are more accurate but slower"
    group: Model

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
    description: "Comma-separated COCO class names (80 classes available)"
    group: Model

  - name: fps
    label: "Processing FPS"
    type: select
    options: [0.2, 0.5, 1, 3, 5, 15]
    default: 5
    description: "Frames per second — higher = more CPU/GPU usage"
    group: Performance

  - name: device
    label: "Inference Device"
    type: select
    options: ["auto", "cpu", "cuda", "mps", "rocm"]
    default: "auto"
    description: "auto = best available GPU, else CPU"
    group: Performance

capabilities:
  live_detection:
    script: scripts/detect.py
    description: "Real-time object detection on live camera frames"
---

# YOLO 2026 Object Detection

Real-time object detection using the latest YOLO 2026 models. Detects 80+ COCO object classes including people, vehicles, animals, and everyday objects. Outputs bounding boxes with labels and confidence scores.

## Model Sizes

| Size | Speed | Accuracy | Best For |
|------|-------|----------|----------|
| nano | Fastest | Good | Real-time on CPU, edge devices |
| small | Fast | Better | Balanced speed/accuracy |
| medium | Moderate | High | Accuracy-focused deployments |
| large | Slower | Highest | Maximum detection quality |

## Protocol

Communicates via **JSON lines** over stdin/stdout.

### Aegis → Skill (stdin)
```jsonl
{"event": "frame", "frame_id": 42, "camera_id": "front_door", "timestamp": "...", "frame_path": "/tmp/aegis_detection/frame_front_door.jpg", "width": 1920, "height": 1080}
```

### Skill → Aegis (stdout)
```jsonl
{"event": "ready", "model": "yolo2026n", "device": "mps", "classes": 80, "fps": 5}
{"event": "detections", "frame_id": 42, "camera_id": "front_door", "timestamp": "...", "objects": [
  {"class": "person", "confidence": 0.92, "bbox": [100, 50, 300, 400]}
]}
{"event": "error", "message": "...", "retriable": true}
```

### Bounding Box Format
`[x_min, y_min, x_max, y_max]` — pixel coordinates (xyxy).

### Stop Command
```jsonl
{"command": "stop"}
```

## Hardware Support

| Platform | Backend | Performance |
|----------|---------|-------------|
| Apple Silicon (M1+) | MPS | 20-30 FPS |
| NVIDIA GPU | CUDA | 25-60 FPS |
| AMD GPU | ROCm | 15-40 FPS |
| CPU (modern x86) | CPU | 5-15 FPS |
| Raspberry Pi 5 | CPU | 2-5 FPS |

## Installation

The `deploy.sh` bootstrapper handles everything — Python environment, GPU backend detection, and dependency installation. No manual setup required.

```bash
./deploy.sh
```
