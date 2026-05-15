"""
QwenPI variants with multi-frame temporal ViT encoding.

Registers two additional framework names:
  - ``QwenPI_History3``  : naive k=3 history frames (no ViT modification)
  - ``QwenPI_HistoryMEM``: k=15 history frames with CausalTemporalVitPatcher

Both variants inherit from ``Qwen_PI`` and override only the VLM encoding path.
No existing files are modified.

Usage (YAML):
    framework:
      name: QwenPI_History3   # or QwenPI_HistoryMEM

Dataset pairing:
    QwenPI_History3  → data_mix: calvin_task_ABC_D_history3
    QwenPI_HistoryMEM → data_mix: calvin_task_ABC_D_history15
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
from starVLA.model.modules.vlm.temporal_vit import CausalTemporalVitPatcher
from starVLA.model.tools import FRAMEWORK_REGISTRY


# ---------------------------------------------------------------------------
# QwenPI_History3 — naive k=3 history (no ViT modification)
# ---------------------------------------------------------------------------

@FRAMEWORK_REGISTRY.register("QwenPI_History3")
class Qwen_PI_History3(Qwen_PI):
    """
    QwenPI with k=3 naive history frames.

    The three frames (t-2, t-1, t) are concatenated as a sequence of images
    and passed directly to Qwen3.5-VL's multi-image processor.  No ViT
    architecture change is needed; the VLM processes each frame independently
    via its native multi-image interleaved attention.

    This serves as the ablation baseline for the MEM ViT variant (E-2b vs E-2c).
    """

    K: int = 3

    def _encode_vl_hidden_states(
        self,
        batch_images: List,
        instructions: List[str],
        solutions: Optional[List[Optional[str]]] = None,
    ) -> Tuple:
        """
        Pass all K frames to the VLM processor.

        batch_images: List[List[PIL.Image]] where each inner list contains
                      K × num_views images (e.g., K=3, 2 views → 6 images).
        """
        return super()._encode_vl_hidden_states(batch_images, instructions, solutions=solutions)


# ---------------------------------------------------------------------------
# QwenPI_HistoryMEM — k=15 history with CausalTemporalVitPatcher (MEM method)
# ---------------------------------------------------------------------------

@FRAMEWORK_REGISTRY.register("QwenPI_HistoryMEM")
class Qwen_PI_HistoryMEM(Qwen_PI):
    """
    QwenPI with k=15 history frames using the MEM (Memory-Efficient Multi-frame)
    causal temporal attention mechanism.

    The CausalTemporalVitPatcher injects lightweight causal temporal attention
    after every 4th ViT block, enabling the model to fuse 15-frame temporal
    context without the O(K²n²) cost of naive full attention.

    Temporal modules are trained jointly with the rest of the model.
    At inference time with K=1, the patcher is bypassed (no overhead).

    Reference: temporal_vit.py CausalTemporalVitPatcher
    """

    K: int = 15

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        # Locate ViT blocks in the Qwen3.5-VL visual encoder
        vit_blocks = self._get_vit_blocks()
        self.temporal_patcher = CausalTemporalVitPatcher(
            vit_blocks=vit_blocks,
            inject_every_n_layers=4,
        )

    def _get_vit_blocks(self):
        """Retrieve the ViT block ModuleList from the Qwen3.5-VL visual encoder."""
        vlm_model = self.qwen_vl_interface.model
        # Qwen3.5-VL visual encoder path:
        # model.visual.blocks  (Qwen3_5VisionTransformer)
        # Fallback paths for different VLM versions:
        for attr_path in [
            "model.visual.blocks",
            "model.model.visual.blocks",
            "model.visual.transformer.blocks",
        ]:
            obj = vlm_model
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                if hasattr(obj, "__iter__"):
                    return obj
            except AttributeError:
                continue
        raise AttributeError(
            "QwenPI_HistoryMEM: Cannot locate ViT blocks in the VLM. "
            "Tried paths: model.visual.blocks, model.model.visual.blocks, "
            "model.visual.transformer.blocks. "
            "Please check the VLM architecture and update _get_vit_blocks()."
        )

    def _encode_vl_hidden_states(
        self,
        batch_images: List,
        instructions: List[str],
        solutions: Optional[List[Optional[str]]] = None,
    ) -> Tuple:
        """
        Encode K=15 frames through the patched ViT and return layer-wise
        hidden states for the Action DiT.
        """
        self.temporal_patcher.enable(K=self.K)
        try:
            result = super()._encode_vl_hidden_states(batch_images, instructions, solutions=solutions)
        finally:
            self.temporal_patcher.disable()
        return result

    def to(self, device, *args, **kwargs):
        """Ensure temporal modules are moved to the correct device."""
        super().to(device, *args, **kwargs)
        self.temporal_patcher.to(device)
        return self
