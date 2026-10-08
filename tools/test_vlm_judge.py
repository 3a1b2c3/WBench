# -*- coding: utf-8 -*-
"""Smoke test for the WBench VLM judge endpoint (video mode and image-frame mode).

Reads VLM_API_URL, VLM_API_KEY, VLM_MODEL_NAME (and VLM_ALLOW_CUSTOM_MODEL=1 for a non-Doubao
model) from the environment. The key is never printed.

Usage (from the WBench root):
    python tools/test_vlm_judge.py path/to/clip.mp4
"""
import argparse
import os
import sys

import cv2
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.metrics.vlm.vlm_evaluator import VLMClient, encode_video_to_b64url

QUESTION = "Is there any visible content in this video?"
CONTROL = "Is this video completely black and empty?"


def read_frames(video_path, count):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Cannot read frames from {video_path}")
    frames = []
    for i in range(count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * (total - 1) // max(count - 1, 1))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def check(label, client, **inputs):
    try:
        raw = client.ask(QUESTION + "\nAnswer with 'yes' or 'no' only.", max_tokens=10, **inputs)
        answer = client.ask_binary(QUESTION, **inputs)
        control = client.ask_binary(CONTROL, **inputs)
    except Exception as e:
        print(f"[FAIL] {label}: request error: {e}")
        return False
    print(f"[{label}] raw reply (10 tokens): {raw!r}")
    print(f"[{label}] '{QUESTION}' -> {answer} (expect True)")
    print(f"[{label}] '{CONTROL}' -> {control} (expect False)")
    if not raw:
        print(f"[FAIL] {label}: empty reply. A reasoning model may have used all 10 tokens thinking.")
        return False
    ok = answer is True and control is False
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to a short, non-black mp4 clip")
    parser.add_argument("--frames", type=int, help="Frames for image mode (0 skips it)", default=4)
    parser.add_argument("--max-short-side", type=int, help="Downscale video before upload", default=480)
    args = parser.parse_args()

    client = VLMClient()
    print(f"URL:   {client.api_url}")
    print(f"Model: {client.model_name} (doubao={client._is_doubao})")

    results = [check("video", client, video_url=encode_video_to_b64url(args.video, args.max_short_side))]
    if args.frames > 0:
        results.append(check("frames", client, images=read_frames(args.video, args.frames)))

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
