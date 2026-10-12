"""
Physical plausibility — PAVRM reward model scoring.

Method: PAVRM (qwen3vl_a3b) assigns a 0~5 physical plausibility score per video.
- raw_score: PAVRM raw score (0~5)
- score = raw_score / 5.0 (normalized to 0~1, higher = better)

Usage:
    from src.metrics.physical.visual_plausibility import PhysicalPlausibilityEvaluator
    evaluator = PhysicalPlausibilityEvaluator(model_path="path/to/qwen3vl_a3b")
    results = evaluator.score_videos(["video1.mp4", "video2.mp4"])
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

METRIC_NAME = "visual_plausibility"

VIDEO_QUALITY_PROMPT = (
    "Suppose you are an expert in judging and evaluating the quality of AI-generated videos, "
    "please watch the above provided video and give scores for the video's truthfulness and "
    "rationality. i.e., whether the video's overall appreance and motion are consistent with "
    "our common-sense, physical principles.\n"
    "Your rating should be chosen from the following five catefories: Perfect, Good, Fair, Poor, "
    "and Bad. Now please rate this video:"
)

ANCHOR_TOKEN_IDS = [51041, 15216, 60795, 84103, 17082]
ANCHOR_WEIGHTS = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])


class PhysicalPlausibilityEvaluator:
    """Local PAVRM inference using Qwen3-VL reward model (transformers backend)."""

    def __init__(self, model_path: str = None, device: str = "cuda"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if model_path is None:
            model_path = os.environ.get("PAVRM_MODEL_PATH", "")
        if not model_path or not os.path.isdir(model_path):
            _default = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))), "weights", "qwen3vl-a3b-visual-plausibility")
            if os.path.isdir(_default):
                model_path = _default
            else:
                raise RuntimeError(
                    f"PAVRM model path not found: {model_path}. "
                    "Set PAVRM_MODEL_PATH env var or pass model_path argument."
                )

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        except (ImportError, ValueError):
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, dtype=torch.bfloat16,
                device_map="auto",
            )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device

    @torch.no_grad()
    def score_video(self, video_path: str, fps: float = 2.0) -> Dict[str, Any]:
        """Score a single video for physical plausibility.

        Returns:
            Dict with raw_score (0-5), score (0-1), error
        """
        if not os.path.exists(video_path):
            return {"raw_score": None, "score": None, "error": f"Video not found: {video_path}"}

        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path,
                     "fps": fps, "max_pixels": 602112},
                    {"type": "text", "text": VIDEO_QUALITY_PROMPT},
                ],
            }]

            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs, max_new_tokens=1, do_sample=False,
                output_logits=True, return_dict_in_generate=True)

            logits = outputs.logits[0]
            token_ids = torch.tensor(ANCHOR_TOKEN_IDS, device=logits.device)
            logits_gathered = torch.gather(logits[0], dim=-1, index=token_ids)
            probs = torch.softmax(logits_gathered, dim=-1)
            raw_score = float(torch.sum(probs * ANCHOR_WEIGHTS.to(logits.device)))

            return {
                "raw_score": round(raw_score, 4),
                "score": round(raw_score / 5.0, 4),
                "error": None,
            }
        except Exception as e:
            return {"raw_score": None, "score": None, "error": str(e)}

    def score_videos(self, video_paths: List[str], fps: float = 2.0) -> List[Dict[str, Any]]:
        """Score multiple videos sequentially.

        Args:
            video_paths: List of video file paths
            fps: Frames per second to sample from videos

        Returns:
            List of dicts with raw_score (0-5), score (0-1), error
        """
        results = []
        for vp in video_paths:
            results.append(self.score_video(vp, fps))
        return results


def score_video_api(video_path: str, endpoint: str = None) -> Dict[str, Any]:
    """Score a video via NVIDIA inference API endpoint."""
    import requests
    import base64
    import sys

    print(f"[DEBUG] score_video_api called with video_path={video_path}", file=sys.stderr)

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    print(f"[DEBUG] NVIDIA_API_KEY present: {bool(api_key)}", file=sys.stderr)

    if not api_key:
        err = "NVIDIA_API_KEY not set. Set env var for NVIDIA inference API."
        print(f"[DEBUG] {err}", file=sys.stderr)
        return {"raw_score": None, "score": None, "error": err}

    if not os.path.exists(video_path):
        return {"raw_score": None, "score": None, "error": f"Video not found: {video_path}"}

    endpoint = endpoint or "https://inference-api.nvidia.com/v1/chat/completions"

    try:
        # Read video and encode as base64
        with open(video_path, "rb") as f:
            video_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # Get video extension for media type
        ext = os.path.splitext(video_path)[1].lower().lstrip(".")
        media_type_map = {"mp4": "video/mp4", "webm": "video/webm", "mov": "video/quicktime"}
        media_type = media_type_map.get(ext, "video/mp4")

        # Call NVIDIA inference API with video
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "nvidia/qwen/qwen3-vl-a3b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{video_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": VIDEO_QUALITY_PROMPT,
                        },
                    ],
                }
            ],
            "temperature": 0.7,
            "max_tokens": 10,
        }

        resp = requests.post(endpoint, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()

        data = resp.json()
        if "error" in data:
            return {"raw_score": None, "score": None, "error": data["error"]["message"]}

        # Parse response for score
        response_text = data["choices"][0]["message"]["content"].lower()
        score_map = {"perfect": 5.0, "good": 4.0, "fair": 3.0, "poor": 2.0, "bad": 1.0}

        raw_score = None
        for word, score in score_map.items():
            if word in response_text:
                raw_score = score
                break

        if raw_score is None:
            return {"raw_score": None, "score": None,
                    "error": f"Could not parse score from response: {response_text}"}

        return {"raw_score": round(raw_score, 4), "score": round(raw_score / 5.0, 4), "error": None}
    except Exception as e:
        return {"raw_score": None, "score": None, "error": str(e)}


def compute_case(video_path: str, model_path: str = None,
                 endpoint: str = None, device: str = "cuda") -> Dict[str, Any]:
    """Evaluate physical plausibility. Tries local model first, falls back to API."""
    model_path = model_path or os.environ.get("PAVRM_MODEL_PATH", "")
    if not model_path:
        _default = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))), "weights", "qwen3vl-a3b-visual-plausibility")
        if os.path.isdir(_default):
            model_path = _default
    endpoint = endpoint or os.environ.get("PAVRM_API_URL", "")

    if model_path and os.path.isdir(model_path):
        evaluator = PhysicalPlausibilityEvaluator(model_path=model_path, device=device)
        result = evaluator.score_video(video_path)
    elif endpoint:
        result = score_video_api(video_path, endpoint)
    else:
        return {"score": None, "error": "Neither PAVRM_MODEL_PATH nor PAVRM_API_URL is set"}

    # Fail if scoring returned no result
    if result["score"] is None and result["error"] is None:
        return {"score": None, "error": "Scoring produced no result (score and error both None)"}

    return {
        "score": result["score"],
        "details": {"raw_score": result["raw_score"]},
        "params": {"method": "pavrm_qwen3vl_a3b", "scale": "raw/5"},
        "error": result["error"],
    }


def build_case_record(
    case_id: str,
    video_path: str,
    raw_score: Optional[float] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    score = round(raw_score / 5.0, 4) if raw_score is not None and error is None else None
    return {
        "case_id": str(case_id),
        "video_path": str(video_path),
        "score": score,
        "details": {"raw_score": round(raw_score, 4) if raw_score is not None else None},
        "params": {"method": "pavrm_qwen3vl_a3b", "scale": "raw/5"},
        "error": error,
    }


def summarize_model_results(case_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in case_records if r.get("score") is not None]
    if not valid:
        return {"score": 0.0, "n_cases": 0}
    scores = [r["score"] for r in valid]
    raws = [r["details"]["raw_score"] for r in valid if r["details"].get("raw_score")]
    return {
        "score": round(float(np.mean(scores)), 4),
        "raw_mean": round(float(np.mean(raws)), 4) if raws else 0.0,
        "n_cases": len(valid),
    }
