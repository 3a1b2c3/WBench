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

import atexit
import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

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

    # Fill the remaining rows by interpolating between surrounding real rows
    # -- never read as real camera state (see module docstring), but still
    # has to pass load_camera's orthonormality check, so the rotation block
    # is slerped (a linear blend of two rotation matrices is not itself a
    # rotation matrix -- determinant drifts away from +1) while translation
    # stays a plain lerp.
    real_frames = sorted(set(real_idx_to_frame))
    for lo, hi in zip(real_frames[:-1], real_frames[1:]):
        gap = hi - lo
        if gap <= 1:
            continue
        rotations = Rotation.concatenate(
            [Rotation.from_matrix(camera[lo, :, :3]), Rotation.from_matrix(camera[hi, :, :3])]
        )
        slerp = Slerp([0.0, 1.0], rotations)
        for f in range(lo + 1, hi):
            t = (f - lo) / gap
            camera[f, :, :3] = slerp([t]).as_matrix()[0]
            camera[f, :, 3] = (1.0 - t) * camera[lo, :, 3] + t * camera[hi, :, 3]
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
        self._server_script = self.wc_root / "serve_inference.py"
        self._server_process: subprocess.Popen | None = None
        atexit.register(self._close_server)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "worldcrafter",
            "class": "WorldCrafterModel",
            "wc_root": str(self.wc_root),
            "model_path": str(self.model_path),
            "model_type": self.model_type,
        }

    def _ensure_server(self) -> None:
        """Start the persistent inference server if it isn't already running.

        Loading WorldCrafter's diffusion model and the Qwen3-VL caption
        model (``inference.py``'s per-case cost -- the latter by design,
        see ``caption.py``'s ``generate_caption``) once and reusing them for
        every case in a batch is the whole point of ``serve_inference.py``;
        reverting to a fresh subprocess per case would bring the reload cost
        straight back.
        """
        if self._server_process is not None and self._server_process.poll() is None:
            return
        if not self._server_script.exists():
            raise RuntimeError(f"serve_inference.py not found: {self._server_script}")
        if not self.model_path.exists():
            raise RuntimeError(
                f"WorldCrafter weights not found: {self.model_path}. "
                f"Download with: hf download TencentARC/WorldCrafter-"
                f"{'Fast' if self.model_type == 'fast' else 'Base'} "
                f"--local-dir {self.model_path}"
            )
        wc_python = _resolve_wc_python(self.wc_root, self._wc_python_override)
        cmd = [
            str(wc_python), str(self._server_script),
            "--model-type", self.model_type,
            "--model-path", str(self.model_path),
            "--seed", str(self.seed),
        ]
        self._server_process = subprocess.Popen(
            cmd, cwd=str(self.wc_root),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        # Skip blank/non-JSON noise rather than trusting the first line --
        # stray output ahead of the real `{"ready": true}` print is a real
        # failure mode (observed: a lone blank line), not hypothetical. True
        # EOF (the process died before becoming ready) still fails loudly.
        while True:
            ready_line = self._server_process.stdout.readline()
            if not ready_line:
                self._close_server()
                raise RuntimeError(
                    "WorldCrafter inference server exited before becoming ready."
                )
            ready_line = ready_line.strip()
            if not ready_line:
                continue
            try:
                ready = json.loads(ready_line)
            except json.JSONDecodeError:
                continue
            if ready.get("ready"):
                break

    def _close_server(self) -> None:
        """Terminate the persistent inference server, releasing its GPU memory."""
        process = self._server_process
        self._server_process = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=30)
        except Exception:
            process.kill()

    def generate_with_poses(self, image: str, poses: Dict[str, Any],
                            video_length: int, perspective: Optional[str] = None,
                            **kwargs) -> List[np.ndarray]:
        self._ensure_server()
        auto_prompt = "auto-first-person" if perspective == "first_person" else "auto-third-person"

        with tempfile.TemporaryDirectory(prefix="worldcrafter_wbench_") as td:
            td_path = Path(td)
            camera_path = td_path / "camera.npy"
            _poses_to_camera_npy(poses, camera_path)
            fov = _fov_from_fx(_WBENCH_REF_FX)
            out_path = td_path / "output.mp4"

            request = {
                "mode": "i2v",
                "image_path": str(Path(image).resolve()),
                "prompt": auto_prompt,
                "negative_prompt": None,
                "camera_path": str(camera_path),
                "output_path": str(out_path),
                "camera_x_fov": fov,
                "camera_xi": 0.0,
                "seed": self.seed,
            }
            if self.num_inference_steps is not None:
                request["num_inference_steps"] = self.num_inference_steps

            assert self._server_process is not None
            assert self._server_process.stdin is not None
            assert self._server_process.stdout is not None
            self._server_process.stdin.write(json.dumps(request) + "\n")
            self._server_process.stdin.flush()
            # Same defensive read as _ensure_server(): skip blank/non-JSON
            # noise rather than trusting the next line to be the response.
            response = None
            while response is None:
                response_line = self._server_process.stdout.readline()
                if not response_line:
                    self._close_server()
                    raise RuntimeError("WorldCrafter inference server exited unexpectedly.")
                response_line = response_line.strip()
                if not response_line:
                    continue
                try:
                    response = json.loads(response_line)
                except json.JSONDecodeError:
                    continue
            if not response.get("ok"):
                raise RuntimeError(f"WorldCrafter inference server failed: {response.get('error')}")

            frames = _read_video_frames(out_path)
            return _reconcile_frame_count(frames, video_length)
