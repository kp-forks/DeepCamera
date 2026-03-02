#!/usr/bin/env python3
"""
Depth Estimation Skill — Real-time monocular depth maps.

Transforms camera frames with Depth Anything v2 colorized depth overlays.
"""

import sys
import json
import argparse
import signal
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Depth Estimation Skill")
    parser.add_argument("--config", type=str)
    parser.add_argument("--model", type=str, default="depth-anything-v2-small")
    parser.add_argument("--colormap", type=str, default="inferno")
    parser.add_argument("--blend-mode", type=str, default="overlay")
    parser.add_argument("--opacity", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_config(args):
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            return json.load(f)
    return {
        "model": args.model,
        "colormap": args.colormap,
        "blend_mode": args.blend_mode,
        "opacity": args.opacity,
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


COLORMAP_MAP = {
    "inferno": 1,   # cv2.COLORMAP_INFERNO
    "viridis": 16,  # cv2.COLORMAP_VIRIDIS
    "plasma": 13,   # cv2.COLORMAP_PLASMA
    "magma": 12,    # cv2.COLORMAP_MAGMA
    "jet": 2,       # cv2.COLORMAP_JET
}


def main():
    args = parse_args()
    config = load_config(args)
    device = select_device(config.get("device", "auto"))

    try:
        import torch
        import cv2
        import numpy as np

        model_name = config.get("model", "depth-anything-v2-small")
        model = torch.hub.load("LiheYoung/Depth-Anything-V2", model_name.replace("-", "_"), trust_repo=True)
        model.to(device)
        model.eval()

        emit({"event": "ready", "model": model_name, "device": device})
    except Exception as e:
        emit({"event": "error", "message": f"Failed to load model: {e}", "retriable": False})
        sys.exit(1)

    running = True
    def handle_signal(s, f):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    colormap_id = COLORMAP_MAP.get(config.get("colormap", "inferno"), 1)
    opacity = config.get("opacity", 0.5)
    blend_mode = config.get("blend_mode", "overlay")

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
                continue

            try:
                import torch
                import cv2
                import numpy as np

                image = cv2.imread(frame_path)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                with torch.no_grad():
                    depth = model.infer_image(rgb)

                # Normalize depth to 0-255
                depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_norm, colormap_id)

                if blend_mode == "overlay":
                    output = cv2.addWeighted(image, 1 - opacity, depth_colored, opacity, 0)
                elif blend_mode == "side_by_side":
                    output = np.hstack([image, depth_colored])
                else:  # depth_only
                    output = depth_colored

                out_path = tempfile.mktemp(suffix=".jpg", dir="/tmp")
                cv2.imwrite(out_path, output, [cv2.IMWRITE_JPEG_QUALITY, 90])

                emit({
                    "event": "transformed_frame",
                    "camera_id": msg.get("camera_id", "unknown"),
                    "timestamp": msg.get("timestamp", ""),
                    "frame_path": out_path,
                    "metadata": {
                        "min_depth": float(depth.min()),
                        "max_depth": float(depth.max()),
                    },
                })
            except Exception as e:
                emit({"event": "error", "message": f"Depth error: {e}", "retriable": True})


if __name__ == "__main__":
    main()
