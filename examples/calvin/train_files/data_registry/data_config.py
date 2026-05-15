"""
CALVIN ABC→D benchmark — data configs, embodiment tags, and dataset mixtures.

This file is auto-discovered by the starVLA registry at
``starVLA/dataloader/gr00t_lerobot/registry.py`` and merged into the global
ROBOT_TYPE_CONFIG_MAP and DATASET_NAMED_MIXTURES.

Configs defined here:
  - calvin_franka           : Base CALVIN config (abs EEF, action_horizon=50)
  - calvin_franka_state     : + proprioception (include_state=True)
  - calvin_history3         : + k=3 history frames
  - calvin_history15        : + k=15 history frames (for MEM ViT)
  - calvin_aug              : + ColorJitter augmentation
  - calvin_cot              : + supervised bbox CoT annotations
  - calvin_delta            : delta_qpos action space variant

Mixtures defined here:
  - calvin_task_ABC_D             : standard CALVIN ABC train split
  - calvin_task_ABC_D_history3    : with k=3 history frames
  - calvin_task_ABC_D_history15   : with k=15 history frames (MEM)
  - calvin_task_ABC_D_aug         : with ColorJitter augmentation
  - calvin_task_ABC_D_cot         : with supervised bbox CoT
  - calvin_task_D_D               : D→D (for quick sanity check)
  - calvin_pretrain_libero        : LIBERO pre-training mix (same action space)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# Base CALVIN DataConfig (Franka, absolute EEF, 50-step action horizon)
# ---------------------------------------------------------------------------

class CalvinFrankaDataConfig:
    """
    Base data config for the CALVIN ABC→D benchmark.

    The CALVIN dataset (converted to LeRobot format) stores:
      - video.primary_image  : static camera (200×200 → resized to 224)
      - video.wrist_image    : wrist/gripper camera
      - state.*              : 8-DOF EEF state (xyz + rpy + pad + gripper)
      - action.*             : 7-DOF EEF action (xyz + rpy + gripper)
      - annotation.*         : task language instruction

    Defaults: abs_eef action space, no history (k=1), no proprioception.
    """

    embodiment_tag = EmbodimentTag.FRANKA

    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]

    # Single-frame observation (no history)
    observation_indices = [0]
    # 50-step action horizon (supports replan_steps ∈ [1, 50])
    action_indices = list(range(50))
    state_indices = [0]

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self) -> ComposedModalityTransform:
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={
                        "action.x": "min_max",
                        "action.y": "min_max",
                        "action.z": "min_max",
                        "action.roll": "min_max",
                        "action.pitch": "min_max",
                        "action.yaw": "min_max",
                        "action.gripper": "binary",
                    },
                ),
            ]
        )


# ---------------------------------------------------------------------------
# Variant: with proprioception (E-2a)
# ---------------------------------------------------------------------------

class CalvinFrankaStateDataConfig(CalvinFrankaDataConfig):
    """CALVIN with proprioceptive state as additional model input."""

    state_indices = [0]

    def transform(self) -> ComposedModalityTransform:
        base = super().transform()
        state_transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "min_max",
                    "state.y": "min_max",
                    "state.z": "min_max",
                    "state.roll": "min_max",
                    "state.pitch": "min_max",
                    "state.yaw": "min_max",
                    "state.pad": "min_max",
                    "state.gripper": "binary",
                },
            ),
        ]
        return ComposedModalityTransform(transforms=state_transforms + list(base.transforms))


# ---------------------------------------------------------------------------
# Variant: k=3 history frames (E-2b, naive multi-image)
# ---------------------------------------------------------------------------

class CalvinHistory3DataConfig(CalvinFrankaDataConfig):
    """CALVIN with k=3 history frames: indices = [-2, -1, 0] → 3 frames."""

    observation_indices = [-2, -1, 0]


# ---------------------------------------------------------------------------
# Variant: k=15 history frames (E-2c, MEM ViT)
# ---------------------------------------------------------------------------

class CalvinHistory15DataConfig(CalvinFrankaDataConfig):
    """CALVIN with k=15 history frames (for MEM CausalTemporalVitPatcher)."""

    observation_indices = list(range(-14, 1))   # [-14, ..., 0], 15 frames


# ---------------------------------------------------------------------------
# Variant: delta action space (E-2d comparison)
# ---------------------------------------------------------------------------

class CalvinDeltaDataConfig(CalvinFrankaDataConfig):
    """CALVIN with delta-qpos action space (for comparison with abs_eef)."""

    def transform(self) -> ComposedModalityTransform:
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={
                        "action.x": "q99",
                        "action.y": "q99",
                        "action.z": "q99",
                        "action.roll": "q99",
                        "action.pitch": "q99",
                        "action.yaw": "q99",
                        "action.gripper": "binary",
                    },
                ),
            ]
        )


# ---------------------------------------------------------------------------
# Variant: ColorJitter augmentation (E-4)
# ---------------------------------------------------------------------------

class CalvinAugDataConfig(CalvinFrankaDataConfig):
    """CALVIN with ColorJitter augmentation for cross-env visual robustness."""

    def transform(self) -> ComposedModalityTransform:
        base_transforms = list(super().transform().transforms)
        aug = VideoColorJitter(
            apply_to=self.video_keys,
            brightness=0.2,
            contrast=0.15,
            saturation=0.1,
            hue=0.02,   # extremely small — avoid confusing colour-conditioned tasks
        )
        return ComposedModalityTransform(transforms=[aug] + base_transforms)


# ---------------------------------------------------------------------------
# Variant: supervised Bounding Box CoT (E-2f)
# ---------------------------------------------------------------------------

class CalvinCoTDataConfig(CalvinFrankaDataConfig):
    """
    CALVIN with supervised bbox CoT annotations.

    Loads bbox annotations from a JSON file produced by
    ``tools/annotate_bbox_for_calvin.py`` (Grounding DINO).
    The annotations are consumed by the dataloader to produce a ``"solution"``
    field that is passed to QwenPI.forward() for language-modelling supervision.

    The annotation JSON has the structure:
        { "episode_XXXXXX_YYYYYY": {"bbox": [x1,y1,x2,y2], "label": str, "score": float} | null }
    """

    bbox_annotation_path: str = "tools/calvin_bbox_annotations.json"
    _annotations: Optional[dict] = None

    def load_annotations(self) -> dict:
        if self._annotations is None:
            path = Path(self.bbox_annotation_path)
            if path.is_file():
                with open(path) as f:
                    object.__setattr__(self, "_annotations", json.load(f))
            else:
                object.__setattr__(self, "_annotations", {})
        return self._annotations

    def get_solution(self, ep_id: str, frame_idx: int) -> Optional[str]:
        """Return a formatted bbox string for CoT supervision, or None if unavailable."""
        anns = self.load_annotations()
        key = f"{ep_id}_{frame_idx:06d}"
        ann = anns.get(key)
        if ann is None:
            return None
        x1, y1, x2, y2 = ann["bbox"]
        return f"The {ann['label']} is at [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}]."


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ROBOT_TYPE_CONFIG_MAP = {
    "calvin_franka":       CalvinFrankaDataConfig(),
    "calvin_franka_state": CalvinFrankaStateDataConfig(),
    "calvin_history3":     CalvinHistory3DataConfig(),
    "calvin_history15":    CalvinHistory15DataConfig(),
    "calvin_delta":        CalvinDeltaDataConfig(),
    "calvin_aug":          CalvinAugDataConfig(),
    "calvin_cot":          CalvinCoTDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    k: EmbodimentTag.FRANKA for k in ROBOT_TYPE_CONFIG_MAP
}

# ---------------------------------------------------------------------------
# Dataset Mixtures
# ---------------------------------------------------------------------------

DATASET_NAMED_MIXTURES = {
    # ----- Standard CALVIN ABC→D train split -----
    "calvin_task_ABC_D": [
        ("calvin_task_ABC_D", 1.0, "calvin_franka"),
    ],

    # ----- D→D (for fast sanity checks / overfitting test) -----
    "calvin_task_D_D": [
        ("calvin_task_D_D", 1.0, "calvin_franka"),
    ],

    # ----- With proprioception (E-2a) -----
    "calvin_task_ABC_D_state": [
        ("calvin_task_ABC_D", 1.0, "calvin_franka_state"),
    ],

    # ----- k=3 history frames (E-2b) -----
    "calvin_task_ABC_D_history3": [
        ("calvin_task_ABC_D", 1.0, "calvin_history3"),
    ],

    # ----- k=15 history frames + MEM ViT (E-2c) -----
    "calvin_task_ABC_D_history15": [
        ("calvin_task_ABC_D", 1.0, "calvin_history15"),
    ],

    # ----- Delta action space (E-2d) -----
    "calvin_task_ABC_D_delta": [
        ("calvin_task_ABC_D", 1.0, "calvin_delta"),
    ],

    # ----- ColorJitter augmentation (E-4) -----
    "calvin_task_ABC_D_aug": [
        ("calvin_task_ABC_D", 1.0, "calvin_aug"),
    ],

    # ----- Supervised bbox CoT (E-2f) -----
    "calvin_task_ABC_D_cot": [
        ("calvin_task_ABC_D", 1.0, "calvin_cot"),
    ],

    # ----- LIBERO pre-training mix (E-3, same action space as CALVIN) -----
    "calvin_pretrain_libero": [
        ("libero_10_no_noops_1.0.0_lerobot",      0.4, "libero_franka"),  # LIBERO-Long
        ("libero_goal_no_noops_1.0.0_lerobot",    0.3, "libero_franka"),  # LIBERO-Goal
        ("libero_spatial_no_noops_1.0.0_lerobot", 0.3, "libero_franka"),  # LIBERO-Spatial
    ],
}
