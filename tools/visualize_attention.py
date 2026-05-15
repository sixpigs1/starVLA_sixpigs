"""
visualize_attention.py — Visualise ViT attention maps from a trained QwenPI model.

For a given image (or set of history frames), this tool:
  1. Loads a checkpoint via the policy server or directly.
  2. Runs a forward pass and extracts the multi-head attention weights from
     specified ViT layers.
  3. Overlays the attention map as a heatmap on the original image.
  4. Optionally shows temporal attention from CausalTemporalVitPatcher (E-2c).

Usage:
    python tools/visualize_attention.py \\
        --checkpoint results/Checkpoints/starvla_calvin_stage2/step_25000 \\
        --image     playground/Datasets/calvin/sample_frame.png \\
        --instruction "push the blue block" \\
        --output    results/figures/attention_map.png \\
        --layer     -1

    # For temporal attention (E-2c MEM ViT):
    python tools/visualize_attention.py \\
        --checkpoint results/Checkpoints/starvla_calvin_e2c.../step_25000 \\
        --images    frame_t-14.png frame_t-7.png frame_t.png \\
        --instruction "grasp the red cup" \\
        --temporal \\
        --output    results/figures/temporal_attention.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ViT attention visualisation for QwenPI checkpoints")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to StarVLA checkpoint directory")
    p.add_argument("--image", type=str, default=None,
                   help="Single RGB image for visualisation")
    p.add_argument("--images", nargs="+", default=None,
                   help="List of images for multi-frame (history) visualisation")
    p.add_argument("--instruction", type=str, required=True,
                   help="Task instruction string")
    p.add_argument("--output", type=str, default="attention_map.png",
                   help="Output figure path")
    p.add_argument("--layer", type=int, default=-1,
                   help="ViT block index to visualise (default: last block = -1)")
    p.add_argument("--head", type=int, default=0,
                   help="Attention head to visualise (default: 0)")
    p.add_argument("--temporal", action="store_true",
                   help="Visualise temporal attention maps (requires QwenPI_HistoryMEM)")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint: str, device: str):
    """
    Load a StarVLA model from a checkpoint directory.
    Infers the framework name from the YAML config inside the checkpoint.
    """
    import glob
    import yaml
    import torch

    # Find config YAML in checkpoint dir
    yaml_files = glob.glob(os.path.join(checkpoint, "*.yaml")) + \
                 glob.glob(os.path.join(checkpoint, "config.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML config found in checkpoint: {checkpoint}")
    config_path = yaml_files[0]
    print(f"[visualize_attention] Loading config: {config_path}")

    # Bootstrap StarVLA path
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from omegaconf import OmegaConf
    from starVLA.model.framework.base_framework import get_framework

    cfg = OmegaConf.load(config_path)
    model = get_framework(cfg)

    # Load weights
    ckpt_files = glob.glob(os.path.join(checkpoint, "model*.pt")) + \
                 glob.glob(os.path.join(checkpoint, "pytorch_model*.bin"))
    if ckpt_files:
        state_dict = torch.load(ckpt_files[0], map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print(f"[visualize_attention] Loaded weights from {ckpt_files[0]}")
    else:
        print("[visualize_attention] WARNING: No weight files found; using random init.")

    model = model.to(device).eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Attention extraction hooks
# ──────────────────────────────────────────────────────────────────────────────

def register_attention_hook(model, layer_idx: int):
    """
    Register a forward hook on a specific ViT block to capture attention weights.
    Returns a list that will be populated with (attn_weights,) after the forward pass.
    """
    captured = []

    def _find_vit_blocks(m):
        for name in ["visual.blocks", "model.visual.blocks", "visual.transformer.blocks"]:
            parts = name.split(".")
            try:
                obj = m
                for p in parts:
                    obj = getattr(obj, p)
                return obj
            except AttributeError:
                continue
        return None

    vit_blocks = None
    for submodule in [model, getattr(model, "qwen_vl_interface", None)]:
        if submodule is None:
            continue
        blocks = _find_vit_blocks(submodule)
        if blocks is not None:
            vit_blocks = blocks
            break

    if vit_blocks is None:
        raise RuntimeError("Could not find ViT blocks in model.")

    target_block = vit_blocks[layer_idx]

    def hook(module, input, output):
        # Try to retrieve attention weights from the output
        # (Qwen2-VL ViT blocks return a tuple or use output_attentions)
        if isinstance(output, tuple) and len(output) > 1:
            attn = output[1]
        else:
            attn = None
        captured.append(attn)

    handle = target_block.register_forward_hook(hook)
    return captured, handle


# ──────────────────────────────────────────────────────────────────────────────
# Attention-to-heatmap conversion
# ──────────────────────────────────────────────────────────────────────────────

def attn_to_heatmap(
    attn_weights: "torch.Tensor",
    image_size: Tuple[int, int],
    head_idx: int = 0,
    cls_token: bool = True,
) -> np.ndarray:
    """
    Convert attention weights to a spatial heatmap.

    attn_weights: (B, H, N, N) where N = num_patches + 1 (cls token)
    Returns: (H, W) float array normalised to [0, 1]
    """
    import torch

    w = attn_weights[0, head_idx]  # (N, N)
    # Attention from CLS token to all patch tokens
    if cls_token and w.shape[-1] > 1:
        cls_attn = w[0, 1:].detach().float().cpu().numpy()  # (N-1,)
    else:
        cls_attn = w.mean(0).detach().float().cpu().numpy()

    num_patches = cls_attn.shape[0]
    grid_size = int(num_patches ** 0.5)
    if grid_size * grid_size != num_patches:
        # non-square: best-effort reshape
        grid_size = int(num_patches ** 0.5) + 1
        cls_attn = np.pad(cls_attn, (0, grid_size * grid_size - num_patches))

    heatmap = cls_attn.reshape(grid_size, grid_size)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Resize to image size
    from PIL import Image as PILImage
    heatmap_img = PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(
        image_size, PILImage.BILINEAR
    )
    return np.array(heatmap_img) / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# Overlay and save
# ──────────────────────────────────────────────────────────────────────────────

def overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay a float heatmap on an RGB PIL image with a colour map."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        raise ImportError("matplotlib is required for visualisation.")

    colormap = cm.get_cmap("jet")
    colored_heatmap = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    colored_pil = Image.fromarray(colored_heatmap).resize(image.size, Image.BILINEAR)

    blended = Image.blend(image.convert("RGB"), colored_pil, alpha=alpha)
    return blended


def save_figure(images: List[Image.Image], titles: List[str], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # Fallback: save first image only
        images[0].save(output_path)
        return

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.array(img))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[visualize_attention] Saved figure to: {output_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Collect images
    if args.images:
        image_paths = args.images
    elif args.image:
        image_paths = [args.image]
    else:
        raise ValueError("Provide --image or --images")

    images_pil = [Image.open(p).convert("RGB") for p in image_paths]
    print(f"[visualize_attention] Loaded {len(images_pil)} image(s).")

    # Load model
    model = load_model(args.checkpoint, device=args.device)

    # Register attention hook
    captured, handle = register_attention_hook(model, layer_idx=args.layer)

    # Run forward pass (inference mode, no grad)
    import torch
    with torch.no_grad():
        example = {
            "image": images_pil if len(images_pil) > 1 else images_pil[0],
            "lang": args.instruction,
        }
        try:
            _ = model.predict_action([example])
        except Exception as e:
            print(f"[visualize_attention] Forward pass error (ignored): {e}")

    handle.remove()

    # Build output figure
    out_images = []
    out_titles = []
    for i, (img_pil, img_path) in enumerate(zip(images_pil, image_paths)):
        out_images.append(img_pil)
        out_titles.append(f"Original\n{Path(img_path).name}")

        if captured and captured[0] is not None:
            heatmap = attn_to_heatmap(captured[0], image_size=img_pil.size, head_idx=args.head)
            overlay = overlay_heatmap_on_image(img_pil, heatmap)
            out_images.append(overlay)
            out_titles.append(f"Attn layer {args.layer} head {args.head}\n{args.instruction[:30]}")
        else:
            print(f"[visualize_attention] WARNING: No attention weights captured for image {i}.")

    save_figure(out_images, out_titles, args.output)


if __name__ == "__main__":
    main()
