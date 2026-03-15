#!/usr/bin/env python3
"""
SAM2 Segmentation Skill — Interactive click-to-segment.

Generates pixel-perfect masks from point/box prompts using Segment Anything 2.
"""

import sys
import json
import argparse
import signal
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 Segmentation Skill")
    parser.add_argument("--config", type=str)
    parser.add_argument("--model", type=str, default="sam2-small")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_config(args):
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            return json.load(f)
    return {"model": args.model, "device": args.device}


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
        import torch
        import numpy as np
        import cv2
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model_cfg = {
            "sam2-tiny": "sam2_hiera_t.yaml",
            "sam2-small": "sam2_hiera_s.yaml",
            "sam2-base": "sam2_hiera_b+.yaml",
            "sam2-large": "sam2_hiera_l.yaml",
        }

        model_name = config.get("model", "sam2-small")
        checkpoint = f"models/{model_name}.pt"

        sam2 = build_sam2(model_cfg.get(model_name, "sam2_hiera_s.yaml"), checkpoint)
        predictor = SAM2ImagePredictor(sam2)
        predictor.model.to(device)

        emit({"event": "ready", "model": model_name, "device": device})
    except Exception as e:
        emit({"event": "error", "message": f"Failed to load SAM2: {e}", "retriable": False})
        sys.exit(1)

    running = True
    current_image = None

    def handle_signal(s, f):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

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

        event = msg.get("event")

        if event == "frame":
            frame_path = msg.get("frame_path")
            if frame_path and Path(frame_path).exists():
                current_image = cv2.imread(frame_path)
                current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
                predictor.set_image(current_image)

        elif event == "click" and current_image is not None:
            x, y = msg.get("x", 0), msg.get("y", 0)
            label = msg.get("label", 1)  # 1=foreground, 0=background

            try:
                point = np.array([[x, y]])
                point_label = np.array([label])

                masks, scores, _ = predictor.predict(
                    point_coords=point,
                    point_labels=point_label,
                    multimask_output=True,
                )

                # Use highest-scoring mask
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                score = float(scores[best_idx])

                # Save mask
                mask_path = tempfile.mktemp(suffix=".png", dir="/tmp")
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

                # Compute bbox from mask
                ys, xs = np.where(mask)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

                emit({
                    "event": "segmentation",
                    "frame_number": msg.get("frame_number", 0),
                    "mask_path": mask_path,
                    "score": round(score, 3),
                    "bbox": bbox,
                })
            except Exception as e:
                emit({"event": "error", "message": f"Segmentation error: {e}", "retriable": True})


if __name__ == "__main__":
    main()
