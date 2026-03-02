#!/usr/bin/env python3
"""
VLM Scene Analysis Skill — Offline clip understanding via vision language models.

Analyzes recorded video clips and generates natural language descriptions.
"""

import sys
import json
import argparse
import signal
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Scene Analysis Skill")
    parser.add_argument("--config", type=str)
    parser.add_argument("--model", type=str, default="smolvlm2-500m")
    parser.add_argument("--prompt", type=str,
                        default="Describe what is happening in this security camera footage. Focus on people, vehicles, and any unusual activity.")
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_config(args):
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            return json.load(f)
    return {
        "model": args.model,
        "prompt": args.prompt,
        "max_frames": args.max_frames,
        "device": args.device,
    }


def emit(event):
    print(json.dumps(event), flush=True)


def extract_frames(video_path, max_frames=4):
    """Extract evenly spaced frames from a video clip."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = [int(i * total / max_frames) for i in range(max_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def main():
    args = parse_args()
    config = load_config(args)

    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
        import cv2
        import base64

        model_path = Path(f"models/{config['model']}.gguf")
        if not model_path.exists():
            emit({"event": "error", "message": f"Model not found: {model_path}. Run: python scripts/download_model.py --model {config['model']}", "retriable": False})
            sys.exit(1)

        chat_handler = MiniCPMv26ChatHandler(clip_model_path=str(model_path.with_suffix(".mmproj")))
        llm = Llama(model_path=str(model_path), chat_handler=chat_handler, n_ctx=4096)

        emit({"event": "ready", "model": config["model"], "device": config.get("device", "cpu")})
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

        if msg.get("event") == "clip_ready":
            video_path = msg.get("video_path")
            clip_id = msg.get("clip_id", "unknown")
            camera_id = msg.get("camera_id", "unknown")

            if not video_path or not Path(video_path).exists():
                emit({"event": "error", "message": f"Video not found: {video_path}", "retriable": True})
                continue

            try:
                frames = extract_frames(video_path, config.get("max_frames", 4))
                if not frames:
                    emit({"event": "error", "message": "No frames extracted", "retriable": True})
                    continue

                # Encode frames as base64 for VLM
                images = []
                for frame in frames:
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    images.append(f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}")

                content = [{"type": "text", "text": config["prompt"]}]
                for img in images:
                    content.append({"type": "image_url", "image_url": {"url": img}})

                result = llm.create_chat_completion(messages=[
                    {"role": "user", "content": content}
                ])

                description = result["choices"][0]["message"]["content"]
                emit({
                    "event": "analysis_result",
                    "clip_id": clip_id,
                    "camera_id": camera_id,
                    "description": description,
                    "objects": [],  # Could be extracted from description
                    "confidence": 0.9,
                })
            except Exception as e:
                emit({"event": "error", "message": f"Analysis error: {e}", "retriable": True})


if __name__ == "__main__":
    main()
