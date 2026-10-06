"""Matrix-Game-3.5 integration for WBench (camera-conditioned).

    from src.models import get_model
    model = get_model("matrix_game35")
    model.generate_multi_turn(case, "work_dirs/matrix_game35/videos/case_1_combined.mp4", "data")

Batched generation (avoiding per-case checkpoint reload)
=========================================================
generate_with_poses() spawns one infer.py subprocess per case, which
reloads Matrix-Game-3.5's full checkpoint stack (DiT + Wan2.2-TI2V-5B +
umt5-xxl + DA3NESTED) every time -- fine for a single case, expensive for
a sweep. generate_batch_with_poses() instead builds a --batch-manifest
(infer.py's own `build_batch_workspace()`/`--num_val_batches` support, not
a new server) and loads the model once per person group. WBench's own
generate.py loop calls generate_multi_turn/generate_with_poses once per
case and isn't wired up to call the batch method -- a caller that wants
the reload savings needs to collect its cases up front and call
generate_batch_with_poses() directly instead of going through generate.py.
UNTESTED end-to-end; see the method's own docstring for what to check
first if it doesn't come back clean on a real run.

Environment
===========
Matrix-Game-3.5 needs its own venv (`.venv`, created by `setup_env.sh`, torch
from the cu128 wheel index per its own requirements.txt header) -- not
present yet on this machine (checked directly), and no checkpoints are
downloaded (`checkpoints/` is empty, `download_models.sh` pulls
RiemannDynamics/Matrix-Game-3.5-{Base,Distilled}, Wan-AI/Wan2.2-TI2V-5B,
depth-anything/DA3NESTED-GIANT-LARGE-1.1). Same subprocess-via-separate-venv
pattern as WorldCrafter (`./worldcrafter_model.py`) and H3-World
(`../action/h3world.py`), for the same reason: not importable in-process
from WBench's own environment.

Camera-convention mismatch (the actual integration problem)
=============================================================
Confirmed by direct inspection of infer.py and a real example
(`samples/first_person/case_0/camera.npz`):

- `--camera` is an `.npz` with `extrinsics_c2w` `(N,4,4)` float32
  (camera-to-world) and `intrinsics` -- accepts `(N,4)`/`(4,)`/`(3,3)`/
  `(N,3,3)`, normalized internally to per-frame `[fx,fy,cx,cy]` in pixels
  of the anchor image resolution (`infer.py:110-129`).
- **Poses are per RAW output frame at 24fps**, not per latent frame
  (`infer.py`'s block indexing: block `k` consumes poses
  `[1+84*k, 84*(k+1)]`, `MIN_POSES=86`). This is a different mismatch than
  WorldCrafter's (which was chunk-quantized but still latent-adjacent) --
  here WBench's per-*latent* poses (`LATENT_RATE=6/s`) must be upsampled
  4x (repeat each pose 4 times) to approximate a 24fps-per-raw-frame
  trajectory, since WBench carries no finer-grained inter-latent motion
  to interpolate from.
- No hard frame-count-multiple requirement (unlike WorldCrafter's
  `% 33 == 0`) -- `infer.py` pads short trajectories by repeating the
  last pose (`pad_poses`, `infer.py:132-140`), so under-supplying frames
  degrades gracefully rather than erroring.
- Intrinsics ARE per-frame here (unlike WorldCrafter's global-FOV-only
  Unified Camera Model) -- WBench's per-frame `K` converts directly,
  losslessly, no FOV-scalar approximation needed.

Frame-count reconciliation
=============================
`infer.py` writes a single `result.mp4` per run (not per-block --
`collect_outputs` copies the pipeline's `*_history.mp4`,
`infer.py:361-381`), covering `num_blocks * 80` frames (each block = 80
output frames). As with WorldCrafter, the result is trimmed or
last-frame-padded to exactly WBench's requested `video_length` after
reading the output video back.

Prompt
======
Unlike WorldCrafter, there's no auto-captioning fallback here --
`--prompt`/`--prompt-file` is plain text, no "auto-*" special values. Since
`generate_with_poses` receives no text prompt from WBench
(`../conditioning.py`), this uses a generic placeholder prompt by default
(override via the `prompt` constructor arg if a real one is needed for
quality).
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .example_model import CameraConditionedModel

RAW_FPS = 24
LATENT_STRIDE = 4  # WBench poses are per-latent; Matrix-Game-3.5 wants per-raw-frame
FRAMES_PER_BLOCK = 80
MIN_POSES = 86  # infer.py's own minimum before padding kicks in


def _poses_to_camera_npz(poses: Dict[str, Any], out_path: Path) -> int:
    """Build a Matrix-Game-3.5-compatible camera.npz from WBench's per-latent
    pose dict, upsampling 4x (repeat) to approximate the 24fps-per-raw-frame
    rate infer.py expects. Returns the number of raw-frame poses written.
    """
    items = sorted(poses.items(), key=lambda kv: int(kv[0]))
    if not items:
        raise RuntimeError("no poses for this case")

    extrinsics: List[np.ndarray] = []
    intrinsics: List[np.ndarray] = []
    for _, v in items:
        ext = np.asarray(v["extrinsic"], dtype=np.float32)
        K = np.asarray(v["K"], dtype=np.float32)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        for _ in range(LATENT_STRIDE):
            extrinsics.append(ext)
            intrinsics.append(np.array([fx, fy, cx, cy], dtype=np.float32))

    extrinsics_arr = np.stack(extrinsics, axis=0)  # [N,4,4]
    intrinsics_arr = np.stack(intrinsics, axis=0)  # [N,4]

    if len(extrinsics_arr) < MIN_POSES:
        # infer.py pads short trajectories itself (pad_poses), but note it
        # in case the padding behavior ever surprises a case's motion.
        pass

    np.savez(out_path, extrinsics_c2w=extrinsics_arr, intrinsics=intrinsics_arr)
    return len(extrinsics_arr)


def _read_video_frames(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open Matrix-Game-3.5 output: {path}")
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


def _resolve_mg35_python(mg35_root: Path, mg35_python: Optional[Path]) -> Path:
    """Path to Matrix-Game-3.5's own venv interpreter.

    Matrix-Game-3.5/.venv does not exist on this machine yet (checked
    directly) -- its README's `setup_env.sh` must be run first. Pass
    mg35_python explicitly to override.
    """
    if mg35_python is not None:
        return mg35_python
    for candidate in ("bin/python", "Scripts/python.exe"):
        p = mg35_root / ".venv" / candidate
        if p.exists():
            return p
    raise RuntimeError(
        f"Matrix-Game-3.5 venv not found under {mg35_root / '.venv'}. "
        "Run its `setup_env.sh` first, or pass mg35_python= to override."
    )


class MatrixGame35Model(CameraConditionedModel):
    """WBench adapter for Matrix-Game-3.5 (camera-conditioned, 6-DoF pose)."""

    def __init__(
        self,
        model_name: str = "matrix_game35",
        mg35_root: Optional[str] = None,
        mg35_python: Optional[str] = None,
        person: str = "first",
        prompt: str = "A real-world scene, camera moving naturally.",
        checkpoint: Optional[str] = None,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        repo_root = Path(__file__).resolve().parents[3]  # .../WBench
        self.mg35_root = Path(mg35_root) if mg35_root else repo_root.parent / "Matrix-Game-3.5"
        self._mg35_python_override = Path(mg35_python) if mg35_python else None
        self.person = person
        self.prompt = prompt
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.seed = seed
        self.infer_script = self.mg35_root / "infer.py"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "matrix_game35",
            "class": "MatrixGame35Model",
            "mg35_root": str(self.mg35_root),
            "person": self.person,
        }

    def generate_with_poses(self, image: str, poses: Dict[str, Any],
                            video_length: int, perspective: Optional[str] = None,
                            **kwargs) -> List[np.ndarray]:
        if not self.infer_script.exists():
            raise RuntimeError(f"infer.py not found: {self.infer_script}")
        mg35_python = _resolve_mg35_python(self.mg35_root, self._mg35_python_override)

        person = "first" if perspective == "first_person" else (
            "third" if perspective == "third_person" else self.person
        )

        with tempfile.TemporaryDirectory(prefix="matrix_game35_wbench_") as td:
            td_path = Path(td)
            camera_path = td_path / "camera.npz"
            num_raw_frames = _poses_to_camera_npz(poses, camera_path)
            num_blocks = max(1, -(-num_raw_frames // FRAMES_PER_BLOCK))  # ceil

            out_dir = td_path / "out"
            run_name = "case"

            cmd = [
                str(mg35_python), str(self.infer_script),
                "--person", person,
                # Absolute: the subprocess runs with cwd=mg35_root (below), so a
                # relative path from WBench's own root resolves against the
                # Matrix-Game-3.5 checkout instead and infer.py reports
                # "cannot read image data/images/case_N.jpg".
                "--image", str(Path(image).resolve()),
                "--camera", str(camera_path),
                "--prompt", self.prompt,
                "--num-blocks", str(num_blocks),
                "--seed", str(self.seed),
                "--output", str(out_dir),
                "--name", run_name,
                "--camera-convention", "c2w",
            ]
            if self.checkpoint is not None:
                cmd += ["--ckpt", str(self.checkpoint)]

            result = subprocess.run(cmd, cwd=str(self.mg35_root))
            if result.returncode != 0:
                raise RuntimeError(
                    f"Matrix-Game-3.5 infer.py failed (exit {result.returncode}): "
                    f"{' '.join(cmd)}"
                )

            result_path = out_dir / (
                "first_person" if person == "first" else "third_person"
            ) / run_name / "result.mp4"
            frames = _read_video_frames(result_path)
            return _reconcile_frame_count(frames, video_length)

    def generate_batch_with_poses(
        self, items: List[Dict[str, Any]], gpu_id: int = 0,
    ) -> List[Optional[List[np.ndarray]]]:
        """Batch counterpart of generate_with_poses: groups items by
        resolved person (checkpoints differ by person, so a single infer.py
        process can't mix them) and, within each group, calls infer.py
        exactly once via --batch-manifest instead of once per item -- this
        is what actually avoids the per-case checkpoint reload documented
        at the top of this file (same subprocess-via-separate-venv pattern
        as WorldCrafter/H3-World, but now loading the model once per group
        instead of once per case).

        Each item: {"image": str, "poses": dict, "video_length": int,
        "perspective": Optional[str]}. Returns a list aligned with `items`;
        an entry is None if that case's result.mp4 was missing after the
        batch run (see infer.py's collect_outputs_batch "no result.mp4"
        warning for why, e.g. a filename-matching mismatch -- this is the
        first thing to check if this comes back empty on a real run).

        UNTESTED end-to-end, same caveat as infer.py's --batch-manifest
        path: verify against a real run before trusting the output.
        """
        if not self.infer_script.exists():
            raise RuntimeError(f"infer.py not found: {self.infer_script}")
        mg35_python = _resolve_mg35_python(self.mg35_root, self._mg35_python_override)

        groups: Dict[str, List[int]] = {}
        for i, item in enumerate(items):
            perspective = item.get("perspective")
            person = "first" if perspective == "first_person" else (
                "third" if perspective == "third_person" else self.person
            )
            groups.setdefault(person, []).append(i)

        results: List[Optional[List[np.ndarray]]] = [None] * len(items)

        for person, indices in groups.items():
            with tempfile.TemporaryDirectory(prefix="matrix_game35_wbench_batch_") as td:
                td_path = Path(td)
                manifest: List[Dict[str, Any]] = []
                case_name_by_index: Dict[int, str] = {}
                for j, idx in enumerate(indices):
                    item = items[idx]
                    case_name = f"case_{j:03d}"
                    case_name_by_index[idx] = case_name
                    camera_path = td_path / f"{case_name}_camera.npz"
                    _poses_to_camera_npz(item["poses"], camera_path)
                    manifest.append({
                        "name": case_name,
                        "person": person,
                        # Absolute: subprocess runs with cwd=mg35_root (below).
                        "image": str(Path(item["image"]).resolve()),
                        "camera": str(camera_path),
                        "prompt": self.prompt,
                        "camera_convention": "c2w",
                    })

                manifest_path = td_path / "manifest.json"
                manifest_path.write_text(
                    json.dumps(manifest), encoding="utf-8")

                # All items in a batch share one --num-blocks (infer.py's
                # run_generation passes it as a single global CLI value --
                # see build_batch_workspace's docstring). Use the max
                # across the group so no item is starved of blocks; shorter
                # items just get more blocks than strictly needed, trimmed
                # back down below by _reconcile_frame_count.
                num_blocks = 1
                for idx in indices:
                    n_raw = max(
                        1,
                        len(items[idx]["poses"]) * LATENT_STRIDE // FRAMES_PER_BLOCK
                        + 1,
                    )
                    num_blocks = max(num_blocks, n_raw)

                out_dir = td_path / "out"
                cmd = [
                    str(mg35_python), str(self.infer_script),
                    "--person", person,
                    "--batch-manifest", str(manifest_path),
                    "--num-blocks", str(num_blocks),
                    "--seed", str(self.seed),
                    "--output", str(out_dir),
                    "--name", "batch",
                ]
                if self.checkpoint is not None:
                    cmd += ["--ckpt", str(self.checkpoint)]

                result = subprocess.run(cmd, cwd=str(self.mg35_root))
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Matrix-Game-3.5 infer.py --batch-manifest failed "
                        f"(exit {result.returncode}): {' '.join(cmd)}")

                person_dir = "first_person" if person == "first" else "third_person"
                for idx in indices:
                    case_name = case_name_by_index[idx]
                    result_path = (
                        out_dir / person_dir / "batch" / case_name / "result.mp4"
                    )
                    if not result_path.exists():
                        continue
                    frames = _read_video_frames(result_path)
                    results[idx] = _reconcile_frame_count(
                        frames, items[idx]["video_length"])

        return results
