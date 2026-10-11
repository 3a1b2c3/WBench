"""InSpatio-World 1.5 integration for WBench (camera-conditioned, first-frame image + 6-DoF trajectory).

    from src.models import get_model
    model = get_model("inspatio")
    model.generate_multi_turn(case, "work_dirs/inspatio/videos/case_1_combined.mp4", "data")

Environment
===========
InSpatio-World needs its own venv (`.venv`, created by its `setup_env.sh`; checkpoints come from its
`download_models.sh`), so it is driven as a subprocess, like WorldCrafter and Matrix-Game-3.5. The
checkout is expected next to WBench (``../inspatio-world-v1.5``); pass ``inspatio_root=`` to override.

How a case becomes an InSpatio scene
====================================
InSpatio-World renders a depth-warped condition video from a source view plus per-frame target cameras and
then runs the causal video model on it. For a single image it needs depth, source intrinsics and the target
trajectory, none of which WBench supplies, so `inspatio_prepare_scene.py` (run inside InSpatio's venv)
builds a scene directory: the first frame is cropped to 832x480, Depth Anything 3 estimates its depth and
intrinsics, and the WBench trajectory is written as per-frame OpenCV world-to-camera matrices relative to
the first camera. `run_scene_inference.py --scene_dir` then renders and infers.

- WBench poses are camera-to-world, one per latent frame (6 per second); InSpatio wants one pose per output
  frame, so they are interpolated to 24 fps (linear translation, slerp rotation) and the last pose is held
  past the final latent.
- WBench's intrinsics K are not used: InSpatio fixes the target intrinsics to the estimated source
  intrinsics ("estimated_source_view_00_fixed").
- Translation units: WBench steps are 0.08 world units per latent, while depth from Depth Anything 3 is in
  metres. ``translation_scale`` rescales the trajectory if the motion looks too small or too large; the
  WBench navigation metric normalises absolute scale.
- Output is always 832x480. The result is trimmed or last-frame-padded to the requested ``video_length``.
- There is no per-case text, so ``prompt`` is a generic default; override it through the constructor.

Scenes are kept under ``<inspatio_root>/wbench_scenes/<id>``; the id is a hash of the image, trajectory,
prompt and scale, so a rerun of an unchanged case reuses InSpatio's finished result. InSpatio refuses to
reuse an output id whose scene directory was deleted, so do not delete scene directories without also
removing the matching entry from ``<inspatio_root>/output/manifest.json``.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .example_model import CameraConditionedModel
from .poses import FPS, LATENT_RATE

PREPARE_SCRIPT = Path(__file__).resolve().with_name("inspatio_prepare_scene.py")
VENV_VARIABLES = ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH")


def _interpolate_c2w(poses: Dict[str, Any], video_length: int) -> np.ndarray:
    """Per-latent camera-to-world poses -> one pose per output frame at FPS."""
    items = sorted(poses.items(), key=lambda kv: int(kv[0]))
    if not items:
        raise RuntimeError("no poses for this case")
    latent = np.stack([np.asarray(v["extrinsic"], dtype=np.float64) for _, v in items])
    latent_times = np.arange(len(latent)) / LATENT_RATE
    frame_times = np.clip(np.arange(video_length) / FPS, 0.0, latent_times[-1])

    if len(latent) == 1:
        return np.repeat(latent, video_length, axis=0)

    slerp = Slerp(latent_times, Rotation.from_matrix(latent[:, :3, :3]))
    translations = np.stack(
        [np.interp(frame_times, latent_times, latent[:, axis, 3]) for axis in range(3)], axis=1
    )
    frames = np.tile(np.eye(4), (video_length, 1, 1))
    frames[:, :3, :3] = slerp(frame_times).as_matrix()
    frames[:, :3, 3] = translations
    return frames


def _read_video_frames(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open InSpatio output: {path}")
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
    return frames + [frames[-1]] * (video_length - len(frames))


def _resolve_python(root: Path, override: Optional[Path]) -> Path:
    if override is not None:
        return override
    for candidate in ("bin/python", "Scripts/python.exe"):
        path = root / ".venv" / candidate
        if path.exists():
            return path
    raise RuntimeError(
        f"InSpatio-World venv not found under {root / '.venv'}. "
        "Run its `setup_env.sh` and `download_models.sh` first, or pass inspatio_python= to override."
    )


class InSpatioModel(CameraConditionedModel):
    """WBench adapter for InSpatio-World 1.5 (camera-conditioned, first-frame image)."""

    def __init__(
        self,
        model_name: str = "inspatio",
        inspatio_root: Optional[str] = None,
        inspatio_python: Optional[str] = None,
        prompt: str = "A real-world scene, camera moving naturally.",
        seed: int = 0,
        translation_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        repo_root = Path(__file__).resolve().parents[3]  # .../WBench
        self.inspatio_root = Path(inspatio_root) if inspatio_root else repo_root.parent / "inspatio-world-v1.5"
        self._python_override = Path(inspatio_python) if inspatio_python else None
        self.prompt = prompt
        self.seed = seed
        self.translation_scale = translation_scale

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "inspatio",
            "class": "InSpatioModel",
            "inspatio_root": str(self.inspatio_root),
            "translation_scale": self.translation_scale,
        }

    def _env(self) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in VENV_VARIABLES}
        env["PYTHONPATH"] = str(self.inspatio_root)
        return env

    def generate_with_poses(self, image: str, poses: Dict[str, Any],
                            video_length: int, **kwargs) -> List[np.ndarray]:
        root = self.inspatio_root
        if not (root / "run_scene_inference.py").exists():
            raise RuntimeError(f"run_scene_inference.py not found under {root}")
        python = _resolve_python(root, self._python_override)
        image_path = Path(image).resolve()
        target_c2w = _interpolate_c2w(poses, video_length)

        digest = hashlib.sha1()
        for part in (image_path.read_bytes(), target_c2w.tobytes(), self.prompt.encode(),
                     repr(self.translation_scale).encode()):
            digest.update(part)
        output_id = f"wbench_{digest.hexdigest()[:16]}"
        scene_dir = root / "wbench_scenes" / output_id
        env = self._env()

        if not (scene_dir / ".prepared").exists():
            with tempfile.TemporaryDirectory(prefix="inspatio_wbench_") as td:
                trajectory = Path(td) / "target_c2w.npz"
                np.savez(trajectory, target_c2w=target_c2w)
                prepare = [
                    str(python), str(PREPARE_SCRIPT),
                    "--image", str(image_path),
                    "--target-c2w", str(trajectory),
                    "--prompt", self.prompt,
                    "--fps", str(FPS),
                    "--scene-dir", str(scene_dir),
                    "--output-id", output_id,
                    "--translation-scale", str(self.translation_scale),
                    "--depth-model-path", str(root / "checkpoints" / "depth"),
                ]
                result = subprocess.run(prepare, cwd=str(root), env=env)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"InSpatio scene preparation failed (exit {result.returncode}): {' '.join(prepare)}"
                    )

        infer = [str(python), str(root / "run_scene_inference.py"),
                 "--scene_dir", str(scene_dir), "--seed", str(self.seed)]
        result = subprocess.run(infer, cwd=str(root), env=env)
        if result.returncode != 0:
            raise RuntimeError(
                f"InSpatio run_scene_inference.py failed (exit {result.returncode}): {' '.join(infer)}"
            )

        frames = _read_video_frames(root / "output" / output_id / "pred.mp4")
        return _reconcile_frame_count(frames, video_length)
