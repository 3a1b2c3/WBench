"""
H3-World integration for WBench (action-conditioned).

    from src.models import get_model
    model = get_model("h3world")
    model.generate_multi_turn(case, "work_dirs/h3world/videos/case_1_combined.mp4", "data")

LICENSING -- read before using any output from this integration
==================================================================
H3-World is a LoRA on MiniMax-H3, whose Community License grants rights only
within its "Applicable Territory" -- which excludes the USA, EU, UK and South
Korea -- and SS V.4 extends that restriction to the model's *Outputs*,
meaning generated clips and any WBench score derived from them. See the
LICENSE shipped with the H3-World weights before publishing anything produced
through this integration. (Mirrors the same caveat in
physics-IQ-benchmark/PHYSICS_IQ_DRIVERS.md's "Licensing" section, where
H3-World was first integrated for the Physics-IQ benchmark.)

Why this file exists / how it's structured
============================================
H3-World has no Python API of its own -- ``H3-World/code/abot/infer.py`` is a
single-purpose CLI: one process, one first-frame image in, one mp4 clip out
(see its own docstring). This integration therefore shells out to it, the
same way ``physics-IQ-benchmark/physiq/drive_h3world.py`` does for the
Physics-IQ benchmark (read-only reference, not modified or imported here --
that driver is a separate, already-complete sibling task).

Multi-turn continuation
------------------------
``ActionConditionedModel.generate_multi_turn`` (``../action/example_model.py``)
calls ``generate_with_actions()`` **once per case**, with the *entire*
per-turn action list and the *total* ``video_length`` -- not once per turn.
There is no "continue this rollout" hook the base class gives you between
turns; all turn-to-turn plumbing is left to the model subclass.

Since infer.py can't be told "continue from where you left off", this
integration does the chaining itself, inside ``generate_with_actions``: it
loops over ``actions`` (one entry per WBench turn) and, for each turn, shells
out to infer.py once, using the *last frame of the previous turn's clip* as
the next turn's ``--first-frame`` (option (b) from the task brief -- re-running
from the original first frame every turn was rejected: it would ignore
everything earlier turns did to the scene and each turn would re-diverge from
a stale starting point instead of continuing the actual rollout). Each
per-turn clip's frame 0 is dropped before concatenating, since
``keyframe_indices=[0]`` in infer.py makes the given first frame *become*
output frame 0 verbatim (not a separate frame in addition to it) -- keeping
it would duplicate the seed frame into the rollout, exactly as noted in
drive_h3world.py's own docstring for the Physics-IQ driver.

Frame-count math
------------------
infer.py requires ``(--num-frames - 5) % 17 == 0`` (checked directly in
infer.py's ``main()``; see also ``abot_action.latent_t_for``). Each WBench
turn wants a specific number of *new* frames (``round(DEFAULT_FPS *
turn_duration)``, apportioned across turns so they sum exactly to the
requested ``video_length``). Since that target frame count is essentially
never itself ``17k+5``, each turn requests the smallest valid
``raw_frames = 17k + 5`` that leaves at least ``target_frames`` after
dropping frame 0, generates that many, then keeps only the first
``target_frames`` frames after the drop (discarding the small overshoot
tail) -- the same "smallest 17k+5, drop-and-trim" strategy
``drive_h3world.py.raw_frames_for`` uses for Physics-IQ.

Actions
--------
WBench's action layer (``../action/actions.py``) already gives us MG3-style
``keyboard``/``mouse`` per turn, which map directly onto H3-World's key
vocabulary (``code/abot/abot_action.py``'s ``KEY_COLS`` /
``code/abot/action_script.py``'s pan-key convention, confirmed by
``H3-World/examples/racer/convert_actions.py``'s comment: J = pan left,
L = pan right):

    keyboard[0] W -> "W"        mouse[1] (yaw)   > 0 -> "L" (pan right)
    keyboard[1] S -> "S"        mouse[1] (yaw)   < 0 -> "J" (pan left)
    keyboard[2] A -> "A"        mouse[0] (pitch) > 0 -> "K" (tilt up)
    keyboard[3] D -> "D"        mouse[0] (pitch) < 0 -> "I" (tilt down)

Rather than mapping onto infer.py's fixed ``--action-preset`` table (which
only covers single keys / two fixed fast-pan combos), this integration always
builds a raw ``[num_frames, 17]`` action matrix and passes it via
``--action-file`` -- the same, more general mechanism
``examples/racer/convert_actions.py`` uses, which also handles compound
holds (e.g. W+D) that no preset name covers. The 6 continuous
rotation/translation columns are left at zero: WBench's navigation layer only
carries directional intent, not magnitude, so there's nothing real to put
there (same reasoning ``convert_actions.py`` gives for the racer clip).
Non-navigation turns (subject_action / event_edit interactions) naturally
fall back to an all-zero ("still") hold, since ``navigation_to_keyboard_mouse``
already zeroes ``keyboard``/``mouse`` for them.

Environment / setup status (as of writing)
=============================================
``H3-World/.venv`` does **not** exist on this machine. The Physics-IQ driver
for the same model (``drive_h3world.py``) runs H3-World via a named conda env
(``minimax_h3``, ``conda run -n minimax_h3``) rather than a venv next to the
checkout. This integration follows the same convention by default; pass
``h3_python=`` to bypass it with a direct interpreter path once one exists.
Nothing in this file has been run -- no inference, no downloads, no installs
(none of that is possible without a working H3-World environment, which per
the above still needs to be set up).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..conditioning import DEFAULT_FPS
from .example_model import ActionConditionedModel

# abot_action.py's ACTION_COLS layout, hardcoded here rather than imported:
# H3-World's package isn't importable from WBench's own environment (they're
# separate venvs/envs, see the module docstring), so this integration treats
# H3-World purely as a subprocess CLI and never imports its code.
_KEY_COLS = ["W", "A", "S", "D", "Q", "E", "I", "J", "K", "L", "Space"]
_NUM_KEYS = len(_KEY_COLS)
_ACTION_DIM = _NUM_KEYS + 6  # + d_pitch, d_yaw, d_roll, d_x_right, d_y_down, d_z_fwd = 17


def _turn_key_holds(action: Dict[str, Any]) -> List[str]:
    """MG3-style {keyboard, mouse} (see ../action/actions.py) -> H3-World key names."""
    keyboard = action.get("keyboard", [0, 0, 0, 0, 0, 0])
    mouse = action.get("mouse", [0.0, 0.0])
    keys = []
    if keyboard[0]:
        keys.append("W")
    if keyboard[1]:
        keys.append("S")
    if keyboard[2]:
        keys.append("A")
    if keyboard[3]:
        keys.append("D")
    if mouse[0] > 0:
        keys.append("K")   # tilt up
    elif mouse[0] < 0:
        keys.append("I")   # tilt down
    if mouse[1] > 0:
        keys.append("L")   # pan right
    elif mouse[1] < 0:
        keys.append("J")   # pan left
    return keys


def _raw_frames_for(target_frames: int) -> int:
    """Smallest 17k+5 that leaves >= target_frames once frame 0 (the seed) is dropped."""
    k = 0
    while 17 * k + 5 < target_frames + 1:
        k += 1
    return 17 * k + 5


def _allocate_turn_frames(actions: List[Dict[str, Any]], fps: int, video_length: int) -> List[int]:
    """Split video_length across turns proportional to each turn's duration.

    Every turn has the same duration in practice (case_to_actions assigns a
    constant `self.duration` per turn), but this stays generic. Rounding
    drift is absorbed by the last turn so the total is always exactly
    video_length.
    """
    durations = [max(0.0, float(a.get("duration", 0.0))) for a in actions]
    total = sum(durations) or float(len(actions)) or 1.0
    counts = [max(1, round(video_length * d / total)) for d in durations] if any(durations) else \
             [max(1, video_length // len(actions))] * len(actions)
    counts[-1] += video_length - sum(counts)
    counts[-1] = max(1, counts[-1])
    return counts


def _build_action_matrix(keys: List[str], num_frames: int) -> np.ndarray:
    """Constant per-frame hold of `keys` for the whole window, all 6 continuous
    columns zero -- see the module docstring's "Actions" section for why.
    """
    mat = np.zeros((num_frames, _ACTION_DIM), dtype=np.float32)
    for k in keys:
        if k in _KEY_COLS:
            mat[:, _KEY_COLS.index(k)] = 1.0
    return mat


def _read_frame_range(video_path: Path, start: int, count: int) -> List[np.ndarray]:
    """Read frames [start, start+count) (BGR uint8, cv2's native order --
    matches what ActionConditionedModel expects back).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open H3-World output: {video_path}")
    frames: List[np.ndarray] = []
    idx = 0
    try:
        while len(frames) < count:
            ret, frame = cap.read()
            if not ret:
                break
            if idx >= start:
                frames.append(frame)
            idx += 1
    finally:
        cap.release()
    if len(frames) < count:
        raise RuntimeError(
            f"{video_path} only had {idx - start} usable frame(s) after dropping the seed "
            f"frame, needed {count}. This means _raw_frames_for() under-requested frames "
            f"from infer.py -- a bug in this integration, not an expected shortfall."
        )
    return frames


def _scene_prompt(case: Dict[str, Any]) -> str:
    """environment/character/perspective description -- prefixed by infer.py to the
    per-latent action clauses it generates itself from the --action-file, so
    this is scene *content*, not motion (H3-World's action_script already
    supplies the motion text from the held keys).
    """
    parts = [p for p in (
        case.get("environment_prompt", ""),
        case.get("character_prompt", ""),
        case.get("perspective_prompt", ""),
    ) if p]
    return " ".join(parts) or "A real-world scene."


def _resolve_h3_python(h3_root: Path, conda_env: str, h3_python: Optional[Path]) -> List[str]:
    """Command prefix to run a script inside H3-World's environment.

    H3-World/.venv does not exist on this machine (checked directly). The
    Physics-IQ driver for the same model (physics-IQ-benchmark/physiq/
    drive_h3world.py) instead runs it via a named conda env, `minimax_h3`,
    and this integration follows the same default. Pass h3_python for a
    direct interpreter path once/if a dedicated venv is set up.
    """
    if h3_python is not None:
        return [str(h3_python)]
    return ["conda", "run", "-n", conda_env, "--no-capture-output", "python3"]


class H3WorldModel(ActionConditionedModel):
    """WBench adapter for H3-World (action-conditioned, WASD + IJKL keys).

    See this module's docstring for the licensing caveat, the multi-turn
    chaining strategy, and the frame-count math -- all load-bearing, not
    incidental detail.
    """

    def __init__(
        self,
        model_name: str = "h3world",
        h3_root: Optional[str] = None,
        conda_env: str = "minimax_h3",
        h3_python: Optional[str] = None,
        checkpoint: Optional[str] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: int = 0,
        subject: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        repo_root = Path(__file__).resolve().parents[3]  # .../WBench
        self.h3_root = Path(h3_root) if h3_root else repo_root.parent / "H3-World"
        self.conda_env = conda_env
        self.h3_python = Path(h3_python) if h3_python else None
        self.checkpoint = Path(checkpoint) if checkpoint else (
            self.h3_root / "checkpoints" / "H3-World" / "step-10000.safetensors"
        )
        self.infer_script = self.h3_root / "code" / "abot" / "infer.py"
        # Left as None unless explicitly overridden, so the command line only
        # ever carries flags that differ from infer.py's own defaults.
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.subject = subject
        self._case: Optional[Dict[str, Any]] = None  # stashed by generate_multi_turn, see below

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "h3world",
            "class": "H3WorldModel",
            "h3_root": str(self.h3_root),
            "checkpoint": str(self.checkpoint),
            "license": "MiniMax-H3 Community License -- outputs restricted outside "
                       "USA/EU/UK/South Korea, see module docstring",
        }

    def generate_multi_turn(self, case: Dict[str, Any], output_path: str,
                            data_root: str = "data") -> Dict[str, Any]:
        """Stash `case` so generate_with_actions() can read environment_prompt /
        character_prompt / perspective_prompt for --scene-prompt -- the base
        class (ActionConditionedModel.generate_multi_turn) only forwards
        image/actions/video_length/perspective/navigation/chunk_length to the
        model hook, not the raw case, and those fields aren't reconstructible
        from that reduced set.
        """
        self._case = case
        try:
            return super().generate_multi_turn(case, output_path, data_root=data_root)
        finally:
            self._case = None

    def generate_with_actions(self, image: str, actions: List[Dict[str, Any]],
                              video_length: int, **kwargs) -> List[np.ndarray]:
        if not actions:
            raise RuntimeError("no actions for this case")
        if not self.infer_script.exists():
            raise RuntimeError(f"infer.py not found: {self.infer_script}")
        if not self.checkpoint.exists():
            raise RuntimeError(f"H3-World checkpoint not found: {self.checkpoint}")

        scene_prompt = _scene_prompt(self._case or {})
        turn_frames = _allocate_turn_frames(actions, DEFAULT_FPS, video_length)
        py_cmd = _resolve_h3_python(self.h3_root, self.conda_env, self.h3_python)

        all_frames: List[np.ndarray] = []
        current_image = image
        with tempfile.TemporaryDirectory(prefix="h3world_wbench_") as td:
            td_path = Path(td)
            for i, (action, target_frames) in enumerate(zip(actions, turn_frames)):
                raw_frames = _raw_frames_for(target_frames)
                keys = _turn_key_holds(action)
                action_file = td_path / f"turn{i}_actions.npy"
                np.save(action_file, _build_action_matrix(keys, raw_frames))

                out_path = td_path / f"turn{i}.mp4"
                cmd = py_cmd + [
                    str(self.infer_script),
                    "--checkpoint", str(self.checkpoint),
                    "--first-frame", str(current_image),
                    "--scene-prompt", scene_prompt,
                    "--action-file", str(action_file),
                    "--num-frames", str(raw_frames),
                    "--seed", str(self.seed),
                    "--out", str(out_path),
                ]
                if self.steps is not None:
                    cmd += ["--steps", str(self.steps)]
                if self.cfg_scale is not None:
                    cmd += ["--cfg-scale", str(self.cfg_scale)]
                if self.subject is not None:
                    cmd += ["--subject", self.subject]

                result = subprocess.run(cmd, cwd=str(self.h3_root))
                if result.returncode != 0:
                    raise RuntimeError(
                        f"H3-World infer.py failed on turn {i + 1}/{len(actions)} "
                        f"(exit {result.returncode}): {' '.join(cmd)}"
                    )

                # Frame 0 of the clip *is* current_image (keyframe_indices=[0] in
                # infer.py), not a new frame -- drop it, keep exactly target_frames.
                turn_output = _read_frame_range(out_path, start=1, count=target_frames)
                all_frames.extend(turn_output)

                if i < len(actions) - 1:
                    next_image = td_path / f"turn{i}_last_frame.jpg"
                    cv2.imwrite(str(next_image), turn_output[-1])
                    current_image = str(next_image)

        # Safety net against off-by-one drift in the per-turn allocation.
        if len(all_frames) > video_length:
            all_frames = all_frames[:video_length]
        elif len(all_frames) < video_length:
            raise RuntimeError(
                f"assembled {len(all_frames)} frames, expected {video_length} -- "
                f"turn allocation bug in _allocate_turn_frames()"
            )
        return all_frames
