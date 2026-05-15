"""
Causal Temporal Attention for multi-frame ViT encoding (MEM method).

This module implements a lightweight causal temporal attention mechanism that
can be injected into ViT blocks at runtime via ``CausalTemporalVitPatcher``.
It enables the model to fuse information from multiple historical frames
without increasing inference latency (compared to naively stacking K-frame
inputs).

Key properties:
  - No additional projection parameters (uses the existing hidden dim)
  - Causal masking ensures information flows from past to present only
  - Fixed sinusoidal positional encoding with zero-shift at the current frame
  - Fully reversible: ``CausalTemporalVitPatcher.disable()`` restores originals

Reference design (MEM = Memory-Efficient Multi-frame):
  - Inject after every ``inject_every_n_layers``-th ViT block (default: 4)
  - For k=15 frames and n=256 patch tokens: memory = O(15×256²) vs O(15²×256²)
    for naive full attention — approximately 15× more efficient

Usage::

    from starVLA.model.modules.vlm.temporal_vit import CausalTemporalVitPatcher

    patcher = CausalTemporalVitPatcher(vit_blocks, inject_every_n_layers=4)
    patcher.enable(K=15)        # call before encoding K frames
    # ... run vit forward on K-frame input ...
    patcher.disable()           # restore original block.forward methods
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal temporal positional encoding
# ---------------------------------------------------------------------------

def make_temporal_pe(K: int, d: int, device: torch.device) -> torch.Tensor:
    """
    Construct a fixed sinusoidal positional encoding of shape (K, d).

    The encoding is shifted so that PE[-1] = 0, meaning the *current* frame
    receives zero modification and only historical frames are shifted.  This
    design is numerically stable at K=1 (single-frame mode).

    Args:
        K: Number of temporal frames.
        d: Hidden dimension of the ViT patch tokens.
        device: Target device.

    Returns:
        Tensor of shape (K, d).
    """
    pos = torch.arange(K, device=device).float()
    half_d = d // 2
    freq = torch.pow(
        10000.0,
        -torch.arange(half_d, device=device).float() / half_d,
    )
    sinusoid = pos.unsqueeze(1) * freq.unsqueeze(0)     # (K, d//2)
    pe = torch.zeros(K, d, device=device)
    pe[:, 0::2] = torch.sin(sinusoid)
    pe[:, 1::2] = torch.cos(sinusoid[:, : d - half_d])
    return pe - pe[-1:]   # shift: current frame PE = 0


# ---------------------------------------------------------------------------
# CausalTemporalAttention — injected after a ViT spatial block
# ---------------------------------------------------------------------------

class CausalTemporalAttention(nn.Module):
    """
    Causal temporal attention block injected after a ViT spatial attention layer.

    Operates on per-spatial-position temporal sequences: for each of the n
    patch tokens, attends over the K-frame history with a causal mask.

    Input:  (B*K, n, d)  — B batches × K frames, each with n patch tokens
    Output: (B*K, n, d)  — temporally updated patch tokens (residual connection)

    Parameters:
        d_model: Hidden dimension of ViT patch tokens.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.pre_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, K: int) -> torch.Tensor:
        """
        Args:
            x: (B*K, n, d) — batch of K-frame patch token sequences.
            K: Number of temporal frames.

        Returns:
            (B*K, n, d) — residual-updated tokens.
        """
        BK, n, d = x.shape
        B = BK // K

        # 1. Reshape to (B, K, n, d) and add temporal positional encoding
        x_4d = x.view(B, K, n, d)
        pe = make_temporal_pe(K, d, x.device)           # (K, d)
        x_pe = x_4d + pe.unsqueeze(0).unsqueeze(2)      # (B, K, n, d)

        # 2. Transpose to (B*n, K, d) for per-spatial temporal attention
        x_t = x_pe.permute(0, 2, 1, 3).contiguous().view(B * n, K, d)
        x_norm = self.pre_norm(x_t)

        # 3. Causal scaled dot-product attention (no new weight matrices)
        causal_mask = torch.tril(
            torch.ones(K, K, dtype=torch.bool, device=x.device)
        )
        attn_out = F.scaled_dot_product_attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=causal_mask,
            dropout_p=0.0,
        )   # (B*n, K, d)

        # 4. Residual + reshape back to (B*K, n, d)
        x_t = x_t + attn_out
        return x_t.view(B, n, K, d).permute(0, 2, 1, 3).contiguous().view(BK, n, d)


# ---------------------------------------------------------------------------
# CausalTemporalVitPatcher — injects CausalTemporalAttention at runtime
# ---------------------------------------------------------------------------

class CausalTemporalVitPatcher:
    """
    Injects ``CausalTemporalAttention`` modules after every
    ``inject_every_n_layers``-th ViT block via monkey-patching.

    Fully reversible: ``disable()`` restores all original ``forward`` methods.

    Usage::

        patcher = CausalTemporalVitPatcher(vit_blocks, inject_every_n_layers=4)
        patcher.enable(K=15)          # patch blocks for 15-frame input
        out = vit_encoder(K_frame_tokens)
        patcher.disable()             # restore original blocks

    Args:
        vit_blocks: ``nn.ModuleList`` of ViT transformer blocks from the VLM.
        inject_every_n_layers: Inject after layers 3, 7, 11, … (0-indexed).
    """

    def __init__(
        self,
        vit_blocks: nn.ModuleList,
        inject_every_n_layers: int = 4,
    ):
        self.vit_blocks = vit_blocks
        self.inject_every_n_layers = inject_every_n_layers
        self._original_forwards: dict[int, object] = {}
        self._current_K: int = 1

        d_model = self._detect_hidden_dim()
        self._target_layers = [
            i for i in range(len(vit_blocks))
            if (i + 1) % inject_every_n_layers == 0
        ]
        self.temporal_modules = nn.ModuleList(
            [CausalTemporalAttention(d_model) for _ in self._target_layers]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_hidden_dim(self) -> int:
        """Infer hidden dim from the first LayerNorm weight found in block[0]."""
        for module in self.vit_blocks[0].modules():
            if isinstance(module, nn.LayerNorm) and module.weight is not None:
                return module.weight.shape[0]
        raise RuntimeError(
            "CausalTemporalVitPatcher: cannot detect ViT hidden dim. "
            "No LayerNorm found in vit_blocks[0]."
        )

    @staticmethod
    def _make_patched_forward(orig_fwd, t_attn: CausalTemporalAttention, patcher_ref):
        """Create a patched forward function that appends temporal attention."""
        def patched(x, *args, **kwargs):
            x = orig_fwd(x, *args, **kwargs)
            if patcher_ref._current_K > 1:
                x = t_attn(x, patcher_ref._current_K)
            return x
        return patched

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enable(self, K: int) -> None:
        """Patch ViT blocks to perform causal temporal attention for K frames."""
        self._current_K = K
        for layer_idx, temporal_attn in zip(self._target_layers, self.temporal_modules):
            block = self.vit_blocks[layer_idx]
            if layer_idx not in self._original_forwards:
                self._original_forwards[layer_idx] = block.forward
            block.forward = self._make_patched_forward(
                self._original_forwards[layer_idx], temporal_attn, self
            )

    def disable(self) -> None:
        """Restore all original block.forward methods."""
        for layer_idx, orig_fwd in self._original_forwards.items():
            self.vit_blocks[layer_idx].forward = orig_fwd
        self._original_forwards.clear()
        self._current_K = 1

    def to(self, device) -> "CausalTemporalVitPatcher":
        """Move temporal modules to device (call after model.to(device))."""
        self.temporal_modules.to(device)
        return self

    def parameters(self):
        """Expose temporal module parameters for optimizer registration."""
        return self.temporal_modules.parameters()
