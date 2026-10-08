"""SolarWM MiniMax-H3 Stage0.5 integration for WBench (camera-conditioned).

    from src.models import get_model
    model = get_model("h3")
    model.generate_multi_turn(case, "work_dirs/h3/videos/case_1_combined.mp4", "data")

Environment
===========
SolarWM-H3 needs its own environment (setup_env_h3.sh), separate from WBench's
wbench-venv. This integration shells out to h3_camera_infer.py as a subprocess
since H3's packages aren't importable from WBench's environment.

Camera conventions
==================
H3 expects a c2w (camera-to-world) trajectory as a series of 4x4 poses.
The camera coordinate system uses COLMAP convention (Z forward, Y down).
This conversion builds that trajectory from WBench's extrinsic poses.

WARNING: h3_camera_infer.py is UNTESTED on real hardware. Camera direction
signs (left/right/straight) may be inverted and will need verification on
first run. See h3_camera_infer.py for detailed caveats.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.spatial.transform import Rotation

from .example_model import CameraConditionedModel

# SolarWM-H3 model paths (from download_h3_models.sh)
H3_MODEL_ROOT = Path.home() / ".cache" / "huggingface" / "hub"
H3_BASE = "SolarWM-h3-33B-base"
H3_ADAPTER = "SolarWM-h3-33B-bid-stage0p5-158f"

# H3 inference script location
H3_INFER = Path(__file__).parent.parent.parent.parent / "h3_camera_infer.py"


def _poses_to_c2w(poses: Dict[str, Any], video_length: int) -> np.ndarray:
    """Convert WBench poses to H3's c2w trajectory format.

    Args:
        poses: {"<latent_idx>": {"extrinsic": 4x4, "K": 3x3}, ...}
        video_length: total RGB frames needed

    Returns:
        c2w trajectory as (num_frames, 4, 4) array, filled via interpolation
        from latent-frame poses to match video_length.
    """
    # Extract latent-frame poses in order
    latent_indices = sorted([int(k) for k in poses.keys()])
    latent_poses = [poses[str(i)]["extrinsic"] for i in latent_indices]

    if not latent_poses:
        # Fallback: identity pose if none provided
        return np.tile(np.eye(4), (video_length, 1, 1))

    # Interpolate from latent frames to all video frames
    # Assume latent stride is constant (typically 4 frames per latent frame)
    stride = max(1, video_length // len(latent_poses))

    c2w_trajectory = []
    for frame_idx in range(video_length):
        latent_idx = frame_idx // stride
        latent_idx = min(latent_idx, len(latent_poses) - 1)

        if latent_idx + 1 < len(latent_poses) and frame_idx % stride != 0:
            # Interpolate between two latent poses
            alpha = (frame_idx % stride) / stride
            pose_a = latent_poses[latent_idx]
            pose_b = latent_poses[latent_idx + 1]

            # SLERP rotation, linear translation
            rot_a = Rotation.from_matrix(pose_a[:3, :3])
            rot_b = Rotation.from_matrix(pose_b[:3, :3])
            slerp = Rotation.from_quat(
                (rot_a * (1 - alpha)).as_quat() + (rot_b * alpha).as_quat()
            )
            trans = pose_a[:3, 3] * (1 - alpha) + pose_b[:3, 3] * alpha

            pose = np.eye(4)
            pose[:3, :3] = slerp.as_matrix()
            pose[:3, 3] = trans
        else:
            pose = latent_poses[min(latent_idx, len(latent_poses) - 1)]

        c2w_trajectory.append(pose)

    return np.array(c2w_trajectory, dtype=np.float32)


class H3Model(CameraConditionedModel):
    """SolarWM MiniMax-H3 Stage0.5 camera-conditioned video generation."""

    def __init__(self, model_name: str = "h3", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        # Verify h3_camera_infer.py exists
        if not H3_INFER.exists():
            raise FileNotFoundError(f"h3_camera_infer.py not found at {H3_INFER}")

    def generate_with_poses(
        self,
        image: str,
        poses: Dict[str, Any],
        video_length: int,
        **kwargs
    ) -> List[np.ndarray]:
        """Generate video using SolarWM-H3 Stage0.5.

        Args:
            image: path to first-frame image
            poses: {"<latent_idx>": {"extrinsic": 4x4, "K": 3x3}, ...}
            video_length: number of RGB frames to generate

        Returns:
            list of BGR uint8 frames (np.ndarray HxWx3)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Build camera trajectory
            c2w = _poses_to_c2w(poses, video_length)
            camera_path = tmpdir / "camera_c2w.npy"
            np.save(camera_path, c2w)

            # Generate output path
            output_video = tmpdir / "output.mp4"

            # Find H3 model weights
            base_model = H3_MODEL_ROOT / H3_BASE
            adapter = H3_MODEL_ROOT / H3_ADAPTER

            # Shell out to h3_camera_infer.py
            # Assuming H3 has its own venv at ~/SolarWM/venv-h3 or similar
            cmd = [
                "python", str(H3_INFER),
                "--base-model", str(base_model),
                "--adapter", str(adapter),
                "--image", image,
                "--camera-c2w", str(camera_path),
                "--out", str(output_video),
            ]

            # Try to use H3 venv if available; fall back to current Python
            h3_venv = Path.home() / "SolarWM" / "venv-h3" / "bin" / "python"
            if h3_venv.exists():
                cmd[0] = str(h3_venv)

            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"H3 inference failed:\nstdout: {e.stdout}\nstderr: {e.stderr}"
                ) from e

            # Load generated video frames
            if not output_video.exists():
                raise RuntimeError(f"H3 did not produce output video at {output_video}")

            frames = self._load_video_frames(str(output_video))

            # Trim or pad to exact video_length
            if len(frames) > video_length:
                frames = frames[:video_length]
            elif len(frames) < video_length:
                # Pad with last frame
                frames.extend([frames[-1]] * (video_length - len(frames)))

            return frames[:video_length]

    @staticmethod
    def _load_video_frames(video_path: str) -> List[np.ndarray]:
        """Load frames from an MP4 video file."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
