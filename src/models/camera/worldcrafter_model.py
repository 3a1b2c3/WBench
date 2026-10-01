"""WorldCrafter integration for WBench (camera-conditioned).

    from src.models import get_model
    model = get_model("worldcrafter")
    model.generate_multi_turn(case, "work_dirs/worldcrafter/videos/case_1_combined.mp4", "data")

Environment
===========
WorldCrafter (TencentARC) needs its own PyTorch 2.10 / CUDA 12.8 /
flash-attn-3 environment (``WorldCrafter/uvenv``), separate from WBench's
own ``wbench-main`` env -- neither exists yet on this machine (checked
directly). Following the same convention as ``../action/h3world.py`` for
the same reason: WorldCrafter's package isn't importable from WBench's own
environment, so this integration shells out to ``inference.py`` as a
subprocess rather than importing ``worldcrafter`` in-process.

Camera-convention mismatch (the actual integration problem)
=============================================================
WBench's ``poses`` dict (``../camera/poses.py``) is one pose per *latent*
frame, with a per-frame (in practice constant) 3x3 intrinsic ``K``.
WorldCrafter's ``camera.npy`` (``worldcrafter/inference.py``) is a plain
``[T, 3, 4]`` extrinsic-only array whose length ``T`` must be a multiple of
``CAMERA_CHUNK_FRAMES = 33``, and only rows at local offsets
``0, 4, 8, ..., 32`` within each 33-row chunk are ever read as real camera
state (confirmed via ``worldcrafter/ucpe/bridge.py``'s
``pose_chunk[:, ::vae_scale_factor_temporal]`` stride-4 slicing and the
identical indexing in ``worldcrafter/repencoder/trajectory_memory_provider.
py``'s ``_anchor_raw_frames``). The other 29/33 rows per chunk just need to
pass ``load_camera``'s orthonormality/finiteness checks -- they're filled
here by linear interpolation between the real poses, purely so the file
loads cleanly.

Intrinsics are a second mismatch: WorldCrafter takes a single *global*
``camera_x_fov`` + ``camera_xi`` scalar pair for the whole run (Unified
Camera Model, ``worldcrafter/ucpe/camera.py``), not a per-frame K matrix --
there is no finer-grained intrinsics path in WorldCrafter at all. WBench's
K is constant across frames in practice anyway (``poses.py``: "Intrinsic K
is fixed for 1920x1080"), so converting that single K to an equivalent
horizontal FOV is a lossless translation, not an approximation of
something WorldCrafter could otherwise do better.

Frame-count reconciliation
=============================
WorldCrafter always produces ``num_chunks * 33`` raw frames (chunk-
quantized); WBench wants exactly ``video_length``. After generation, the
output is trimmed or the last frame held to pad, to hit the exact count
``generate_multi_turn`` requires.

Prompt
======
``generate_with_poses`` receives no text prompt from WBench at all --
camera-conditioned models are purely navigation-driven
(``../conditioning.py``). This uses WorldCrafter's own auto-captioning
(``--prompt auto-first-person`` / ``auto-third-person``, backed by
Qwen3-VL) instead, selected from the ``perspective`` kwarg WBench passes
into ``generate_with_poses``.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .example_model import CameraConditionedModel

CAMERA_CHUNK_FRAMES = 33
LATENT_STRIDE = 4
LATENT_POSES_PER_CHUNK = CAMERA_CHUNK_FRAMES // LATENT_STRIDE + 1  # 9, offsets 0,4,...,32

# WBench's reference intrinsic (poses.py: "fixed for 1920x1080").
_WBENCH_REF_WIDTH = 1920
_WBENCH_REF_FX = 969.6969696969696


def _fov_from_fx(fx: float, width: int = _WBENCH_REF_WIDTH) -> float:
    """Horizontal FOV (degrees) equivalent to a pinhole fx at the given width."""
    return math.degrees(2.0 * math.atan(width / (2.0 * fx)))


def _poses_to_camera_npy(poses: Dict[str, Any], out_path: Path) -> int:
    """Build a WorldCrafter-compatible [T,3,4] camera.npy from WBench's
    per-latent pose dict. Returns T (= num_chunks * CAMERA_CHUNK_FRAMES).
    """
    items = sorted(poses.items(), key=lambda kv: int(kv[0]))
    extrinsics = [np.asarray(v["extrinsic"], dtype=np.float64)[:3, :] for _, v in items]
    if not extrinsics:
        raise RuntimeError("no poses for this case")

    num_chunks = max(1, -(-len(extrinsics) // LATENT_POSES_PER_CHUNK))  # ceil
    total_frames = num_chunks * CAMERA_CHUNK_FRAMES
    camera = np.empty((total_frames, 3, 4), dtype=np.float64)

    # Place real poses at local offsets 0,4,...,32 within each chunk, holding
    # the last available pose once the dict runs short for the final chunk.
    real_idx_to_frame: List[int] = []
    for chunk in range(num_chunks):
        for local in range(LATENT_POSES_PER_CHUNK):
            real_idx_to_frame.append(chunk * CAMERA_CHUNK_FRAMES + local * LATENT_STRIDE)
    for frame_idx, real_i in zip(real_idx_to_frame, range(len(real_idx_to_frame))):
        src = extrinsics[min(real_i, len(extrinsics) - 1)]
        camera[frame_idx] = src

    # Fill the remaining rows by linear interpolation between surrounding
    # real rows -- never read as real camera state (see module docstring),
    # only needs to pass load_camera's orthonormality/finiteness checks.
    real_frames = sorted(set(real_idx_to_frame))
    for lo, hi in zip(real_frames[:-1], real_frames[1:]):
        gap = hi - lo
        if gap <= 1:
            continue
        for f in range(lo + 1, hi):
            t = (f - lo) / gap
            camera[f] = (1.0 - t) * camera[lo] + t * camera[hi]
    # Tail past the last real pose (final chunk's rows 33+ never assigned
    # when the dict runs short): hold the last real pose.
    for f in range(real_frames[-1] + 1, total_frames):
        camera[f] = camera[real_frames[-1]]

    np.save(out_path, camera)
    return total_frames


def _read_video_frames(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open WorldCrafter output: {path}")
    frames: List[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"{path} produced zero readable frames")
    return frames


def _reconcile_frame_count(frames: List[np.ndarray], video_length: int) -> List[np.ndarray]:
    if len(frames) >= video_length:
        return frames[:video_length]
    frames = list(frames)
    while len(frames) < video_length:
        frames.append(frames[-1])
    return frames


def _resolve_wc_python(wc_root: Path, wc_python: Optional[Path]) -> Path:
    """Path to WorldCrafter's own venv interpreter.

    WorldCrafter/uvenv/.venv does not exist on this machine yet (checked
    directly) -- its README's `uv sync --project uvenv --frozen --extra demo`
    step must be run first. Pass wc_python explicitly to override.
    """
    if wc_python is not None:
        return wc_python
    for candidate in ("bin/python", "Scripts/python.exe"):
        p = wc_root / "uvenv" / ".venv" / candidate
        if p.exists():
            return p
    raise RuntimeError(
        f"WorldCrafter venv not found under {wc_root / 'uvenv' / '.venv'}. "
        "Run its README's `uv sync --project uvenv --frozen --extra demo` first, "
        "or pass wc_python= to override."
    )


class WorldCrafterModel(CameraConditionedModel):
    """WBench adapter for WorldCrafter (camera-conditioned, 6-DoF pose)."""

    def __init__(
        self,
        model_name: str = "worldcrafter",
        wc_root: Optional[str] = None,
        wc_python: Optional[str] = None,
        model_path: Optional[str] = None,
        model_type: str = "fast",
        seed: int = 0,
        num_inference_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        repo_root = Path(__file__).resolve().parents[3]  # .../WBench
        self.wc_root = Path(wc_root) if wc_root else repo_root.parent / "WorldCrafter"
        self._wc_python_override = Path(wc_python) if wc_python else None
        self.model_path = Path(model_path) if model_path else (
            self.wc_root / "weights" / (
                "WorldCrafter-Fast" if model_type == "fast" else "WorldCrafter-Base"
            )
        )
        self.model_type = model_type
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.infer_script = self.wc_root / "inference.py"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "worldcrafter",
            "class": "WorldCrafterModel",
            "wc_root": str(self.wc_root),
            "model_path": str(self.model_path),
            "model_type": self.model_type,
        }

    def generate_with_poses(self, image: str, poses: Dict[str, Any],
                            video_length: int, perspective: Optional[str] = None,
                            **kwargs) -> List[np.ndarray]:
        if not self.infer_script.exists():
            raise RuntimeError(f"inference.py not found: {self.infer_script}")
        if not self.model_path.exists():
            raise RuntimeError(
                f"WorldCrafter weights not found: {self.model_path}. "
                f"Download with: hf download TencentARC/WorldCrafter-"
                f"{'Fast' if self.model_type == 'fast' else 'Base'} "
                f"--local-dir {self.model_path}"
            )
        wc_python = _resolve_wc_python(self.wc_root, self._wc_python_override)

        auto_prompt = "auto-first-person" if perspective == "first_person" else "auto-third-person"

        with tempfile.TemporaryDirectory(prefix="worldcrafter_wbench_") as td:
            td_path = Path(td)
            camera_path = td_path / "camera.npy"
            _poses_to_camera_npy(poses, camera_path)
            fov = _fov_from_fx(_WBENCH_REF_FX)
            out_path = td_path / "output.mp4"

            cmd = [
                str(wc_python), str(self.infer_script),
                "--model-type", self.model_type,
                "--model-path", str(self.model_path),
                "--mode", "i2v",
                "--image-path", str(Path(image).resolve()),
                "--prompt", auto_prompt,
                "--camera-path", str(camera_path),
                "--output-path", str(out_path),
                "--camera-x-fov", str(fov),
                "--camera-xi", "0.0",
                "--seed", str(self.seed),
            ]
            if self.num_inference_steps is not None:
                cmd += ["--num-inference-steps", str(self.num_inference_steps)]

            result = subprocess.run(cmd, cwd=str(self.wc_root))
            if result.returncode != 0:
                raise RuntimeError(
                    f"WorldCrafter inference.py failed (exit {result.returncode}): "
                    f"{' '.join(cmd)}"
                )

            frames = _read_video_frames(out_path)
            return _reconcile_frame_count(frames, video_length)
