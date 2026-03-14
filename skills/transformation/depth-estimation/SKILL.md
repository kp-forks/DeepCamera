---
name: depth-estimation
description: "Real-time depth map estimation for privacy transforms using Depth Anything v2"
version: 1.1.0
category: privacy

parameters:
  - name: model
    label: "Depth Model"
    type: select
    options: ["depth-anything-v2-small", "depth-anything-v2-base", "depth-anything-v2-large", "midas-small"]
    default: "depth-anything-v2-small"
    group: Model

  - name: blend_mode
    label: "Display Mode"
    type: select
    options: ["depth_only", "overlay", "side_by_side"]
    default: "depth_only"
    group: Display

  - name: opacity
    label: "Overlay Opacity"
    type: number
    min: 0.0
    max: 1.0
    default: 0.5
    group: Display

  - name: colormap
    label: "Depth Colormap"
    type: select
    options: ["inferno", "viridis", "plasma", "magma", "jet"]
    default: "inferno"
    group: Display

  - name: device
    label: "Device"
    type: select
    options: ["auto", "cpu", "cuda", "mps"]
    default: "auto"
    group: Performance

capabilities:
  live_transform:
    script: scripts/transform.py
    description: "Real-time depth estimation overlay on live feed"
---

# Depth Estimation (Privacy)

Real-time monocular depth estimation using Depth Anything v2. Transforms camera feeds with colorized depth maps — near objects appear warm, far objects appear cool.

When used for **privacy mode**, the `depth_only` blend mode fully anonymizes the scene while preserving spatial layout and activity, enabling security monitoring without revealing identities.

## What You Get

- **Privacy anonymization** — depth-only mode hides all visual identity
- **Depth overlays** on live camera feeds
- **Distance estimation** — approximate distance to detected objects
- **3D scene understanding** — spatial layout of the scene

## Interface: TransformSkillBase

This skill implements the `TransformSkillBase` interface. Any new privacy skill can be created by subclassing `TransformSkillBase` and implementing two methods:

```python
from transform_base import TransformSkillBase

class MyPrivacySkill(TransformSkillBase):
    def load_model(self, config):
        # Load your model, return {"model": "...", "device": "..."}
        ...

    def transform_frame(self, image, metadata):
        # Transform BGR image, return BGR image
        ...
```

## Protocol

### Aegis → Skill (stdin)
```jsonl
{"event": "frame", "frame_id": "cam1_1710001", "camera_id": "front_door", "frame_path": "/tmp/frame.jpg", "timestamp": "..."}
{"command": "config-update", "config": {"opacity": 0.8, "blend_mode": "overlay"}}
{"command": "stop"}
```

### Skill → Aegis (stdout)
```jsonl
{"event": "ready", "model": "depth-anything-v2-small", "device": "mps"}
{"event": "transform", "frame_id": "cam1_1710001", "camera_id": "front_door", "transform_data": "<base64 JPEG>"}
{"event": "perf_stats", "total_frames": 50, "timings_ms": {"transform": {"avg": 45.2, ...}}}
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
