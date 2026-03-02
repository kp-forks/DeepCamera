#!/usr/bin/env python3
"""
Person Recognition (ReID) Skill — Track individuals across cameras.

Extracts appearance embeddings from detected person crops and matches
against a gallery of known identities.
"""

import sys
import json
import argparse
import signal
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Person Recognition Skill")
    parser.add_argument("--config", type=str)
    parser.add_argument("--model", type=str, default="mgn-r50")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--gallery-size", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_config(args):
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            return json.load(f)
    return {
        "model": args.model,
        "similarity_threshold": args.threshold,
        "gallery_size": args.gallery_size,
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


class IdentityGallery:
    """Simple in-memory gallery of known person embeddings."""

    def __init__(self, max_size=100, threshold=0.7):
        self.embeddings = {}  # identity_id -> embedding
        self.labels = {}      # identity_id -> label
        self.max_size = max_size
        self.threshold = threshold
        self._next_id = 0

    def match(self, embedding):
        """Find the closest matching identity, or create new one."""
        import numpy as np

        best_id = None
        best_sim = 0.0

        for identity_id, stored_emb in self.embeddings.items():
            sim = float(np.dot(embedding, stored_emb) /
                       (np.linalg.norm(embedding) * np.linalg.norm(stored_emb) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_id = identity_id

        if best_sim >= self.threshold and best_id is not None:
            return best_id, self.labels.get(best_id, best_id), best_sim

        # New identity
        if len(self.embeddings) < self.max_size:
            new_id = f"person_{self._next_id:04d}"
            self._next_id += 1
            self.embeddings[new_id] = embedding
            self.labels[new_id] = new_id
            return new_id, new_id, 1.0

        return None, "unknown", 0.0


def main():
    args = parse_args()
    config = load_config(args)
    device = select_device(config.get("device", "auto"))

    try:
        import torchreid
        import torch
        import cv2
        import numpy as np

        extractor = torchreid.utils.FeatureExtractor(
            model_name="osnet_ain_x1_0",
            device=device,
        )
        gallery = IdentityGallery(
            max_size=config.get("gallery_size", 100),
            threshold=config.get("similarity_threshold", 0.7),
        )
        emit({"event": "ready", "model": config["model"], "device": device, "gallery_size": 0})
    except Exception as e:
        emit({"event": "error", "message": f"Failed to load model: {e}", "retriable": False})
        sys.exit(1)

    running = True
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

        if msg.get("event") == "frame":
            frame_path = msg.get("frame_path")
            detections = msg.get("detections", [])
            camera_id = msg.get("camera_id", "unknown")

            if not frame_path or not Path(frame_path).exists():
                continue

            try:
                image = cv2.imread(frame_path)
                results = []

                for det in detections:
                    if det.get("class") != "person":
                        results.append(det)
                        continue

                    x1, y1, x2, y2 = det["bbox"]
                    crop = image[max(0, y1):y2, max(0, x1):x2]
                    if crop.size == 0:
                        continue

                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    features = extractor([crop_rgb])
                    embedding = features[0].cpu().numpy()

                    identity_id, label, confidence = gallery.match(embedding)
                    results.append({
                        **det,
                        "identity": label,
                        "identity_id": identity_id,
                        "confidence": round(confidence, 3),
                        "track_id": identity_id,
                    })

                emit({
                    "event": "detections",
                    "camera_id": camera_id,
                    "timestamp": msg.get("timestamp", ""),
                    "objects": results,
                })
            except Exception as e:
                emit({"event": "error", "message": f"ReID error: {e}", "retriable": True})


if __name__ == "__main__":
    main()
