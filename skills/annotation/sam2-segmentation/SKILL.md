---
name: sam2-segmentation
description: "Interactive click-to-segment using Segment Anything 2"
version: 1.0.0

parameters:
  - name: model
    label: "SAM2 Model"
    type: select
    options: ["sam2-tiny", "sam2-small", "sam2-base", "sam2-large"]
    default: "sam2-small"
    group: Model

  - name: device
    label: "Device"
    type: select
    options: ["auto", "cpu", "cuda", "mps"]
    default: "auto"
    group: Performance

capabilities:
  live_transform:
    script: scripts/segment.py
    description: "Interactive segmentation on frames"
---

# SAM2 Interactive Segmentation

Click anywhere on a video frame to segment objects using Meta's Segment Anything 2. Generates pixel-perfect masks for annotation, tracking, and video compositing.

## What You Get

- **Click-to-segment** — click on any object to get its mask
- **Video propagation** — segment in one frame, track through the video
- **Annotation** — export masks for dataset creation (COCO format)
- **Background removal** — isolate objects from scenes

## Protocol

### Aegis → Skill (stdin)
```jsonl
{"event": "frame", "camera_id": "front_door", "frame_path": "/tmp/frame.jpg", "timestamp": "..."}
{"event": "click", "x": 450, "y": 320, "label": 1}
{"event": "propagate", "direction": "forward", "num_frames": 30}
```

### Skill → Aegis (stdout)
```jsonl
{"event": "ready", "model": "sam2-small", "device": "mps"}
{"event": "segmentation", "frame_number": 0, "mask_path": "/tmp/mask_001.png", "score": 0.95, "bbox": [100, 50, 350, 420]}
{"event": "propagation_complete", "frames_processed": 30, "masks_dir": "/tmp/masks/"}
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py --model sam2-small
```
