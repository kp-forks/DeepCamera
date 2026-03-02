#!/usr/bin/env python3
"""
DINOv3 Visual Grounding Skill — Open-vocabulary object detection.

Detects objects based on natural language prompts.
Communicates via JSON lines over stdin/stdout.
"""

import sys
import json
import argparse
import signal
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv3 Grounding Skill")
    parser.add_argument("--config", type=str, help="Config JSON path")
    parser.add_argument("--model", type=str, default="dinov3-base", choices=["dinov3-base", "dinov3-large"])
    parser.add_argument("--prompt", type=str, default="person . car . dog . cat")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_config(args):
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            return json.load(f)
    return {
        "model": args.model,
        "prompt": args.prompt,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "device": args.device,
    }


def select_device(pref):
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    except ImportError:
        pass
    return "cpu"


def emit(event):
    print(json.dumps(event), flush=True)


def main():
    args = parse_args()
    config = load_config(args)
    device = select_device(config.get("device", "auto"))

    try:
        from groundingdino.util.inference import load_model, predict
        import cv2
        import numpy as np

        model = load_model(
            "groundingdino/config/GroundingDINO_SwinT_OGC.py",
            f"weights/{config['model']}.pth"
        )
        emit({"event": "ready", "model": config["model"], "device": device})
    except Exception as e:
        emit({"event": "error", "message": f"Failed to load model: {e}", "retriable": False})
        sys.exit(1)

    running = True
    def handle_signal(s, f):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    prompt = config.get("prompt", "person . car")
    box_thresh = config.get("box_threshold", 0.3)
    text_thresh = config.get("text_threshold", 0.25)

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
            if not frame_path or not Path(frame_path).exists():
                emit({"event": "error", "message": f"Frame not found: {frame_path}", "retriable": True})
                continue

            try:
                import cv2
                image = cv2.imread(frame_path)
                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption=prompt,
                    box_threshold=box_thresh,
                    text_threshold=text_thresh,
                )
                h, w = image.shape[:2]
                objects = []
                for box, logit, phrase in zip(boxes, logits, phrases):
                    cx, cy, bw, bh = box.tolist()
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    objects.append({
                        "class": phrase,
                        "confidence": round(float(logit), 3),
                        "bbox": [x1, y1, x2, y2],
                    })
                emit({
                    "event": "detections",
                    "camera_id": msg.get("camera_id", "unknown"),
                    "timestamp": msg.get("timestamp", ""),
                    "objects": objects,
                })
            except Exception as e:
                emit({"event": "error", "message": f"Inference error: {e}", "retriable": True})


if __name__ == "__main__":
    main()
