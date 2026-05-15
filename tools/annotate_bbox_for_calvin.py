"""
annotate_bbox_for_calvin.py — Automatically annotate CALVIN episodes with
bounding boxes using Grounding DINO, for supervised CoT training (E-2f).

For each (episode, frame_idx) in the CALVIN dataset, this tool:
  1. Loads the static RGB image.
  2. Runs Grounding DINO to detect objects matching the task language.
  3. Saves the resulting bbox string "[x1,y1,x2,y2]" to a JSON file.

The JSON output is consumed by CalvinCoTDataConfig.get_solution() in
examples/calvin/train_files/data_registry/data_config.py.

Requirements:
    pip install groundingdino-py Pillow tqdm

Usage:
    python tools/annotate_bbox_for_calvin.py \\
        --dataset_root  playground/Datasets/calvin/task_ABC_D \\
        --output_dir    playground/Datasets/calvin/bbox_annotations \\
        --model_config  GroundingDINO_SwinT_OGC.py \\
        --model_weights groundingdino_swint_ogc.pth \\
        --box_threshold 0.30 \\
        --text_threshold 0.25 \\
        --max_workers 4

Output structure:
    bbox_annotations/
        ep_XXXXXX.json     # dict: {frame_idx: "[x1,y1,x2,y2]"}
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Grounding DINO helpers (lazy import to keep the rest of the file importable)
# ──────────────────────────────────────────────────────────────────────────────

def _load_gdino(model_config: str, model_weights: str, device: str = "cuda"):
    """Load a Grounding DINO model. Returns the model object."""
    try:
        from groundingdino.util.inference import load_model
    except ImportError:
        raise ImportError(
            "groundingdino-py is not installed. "
            "Run: pip install groundingdino-py"
        )
    return load_model(model_config, model_weights, device=device)


def _predict_bbox(
    model,
    image: Image.Image,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str = "cuda",
) -> Optional[str]:
    """
    Run Grounding DINO on a single image and return the highest-confidence
    bounding box as a normalised "[x1,y1,x2,y2]" string (coords in [0,1]).
    Returns None if no detection passes the thresholds.
    """
    try:
        from groundingdino.util.inference import predict
        import torch
        from groundingdino.datasets import transforms as T
    except ImportError:
        raise ImportError("groundingdino-py not available.")

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image.convert("RGB"), None)

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    if boxes is None or len(boxes) == 0:
        return None

    # Pick highest-confidence box
    best_idx = int(logits.argmax())
    cx, cy, w, h = boxes[best_idx].tolist()
    x1 = max(0.0, cx - w / 2)
    y1 = max(0.0, cy - h / 2)
    x2 = min(1.0, cx + w / 2)
    y2 = min(1.0, cy + h / 2)
    # Format with 4 decimal places
    return f"[{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}]"


# ──────────────────────────────────────────────────────────────────────────────
# CALVIN dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

def _iter_episodes(dataset_root: pathlib.Path):
    """
    Yields (ep_id, ep_dir) for all episodes in the CALVIN training split.
    Each episode directory contains ep_XXXXXX.npz files.
    """
    train_dir = dataset_root / "training"
    if not train_dir.exists():
        train_dir = dataset_root  # fallback: root is already the split dir
    for ep_dir in sorted(train_dir.iterdir()):
        if ep_dir.is_dir():
            yield ep_dir.name, ep_dir


def _load_episode_frames(ep_dir: pathlib.Path):
    """
    Returns a list of (frame_idx, PIL.Image, task_lang) for each frame in
    the episode that has an associated language annotation.

    CALVIN stores frames as ep_XXXXXX.npz and annotations in
    lang_annotations/auto_lang_ann.npy or similar.
    """
    import numpy as np  # noqa: F811

    # Load language annotation
    lang_ann_path = ep_dir / "lang_ann.npy"
    if not lang_ann_path.exists():
        return []

    lang_annotations = np.load(lang_ann_path, allow_pickle=True).item()
    # lang_annotations: dict {"language": {"ann": [...], "task": [...], "emb": ...},
    #                          "info": {"indx": [(start, end), ...]}}
    episodes = lang_annotations.get("info", {}).get("indx", [])
    tasks = lang_annotations.get("language", {}).get("ann", [])

    results = []
    for i, (start, end) in enumerate(episodes):
        if i >= len(tasks):
            break
        task_lang = tasks[i]
        # Use the first frame of the episode for annotation
        frame_idx = start
        frame_path = ep_dir / f"ep_{frame_idx:07d}.npz"
        if not frame_path.exists():
            continue
        try:
            data = np.load(str(frame_path))
            # CALVIN stores RGB as "rgb_static" (H, W, 3)
            rgb = data.get("rgb_static", None)
            if rgb is None:
                continue
            image = Image.fromarray(rgb.astype(np.uint8))
            results.append((frame_idx, image, task_lang))
        except Exception:
            continue
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate CALVIN episodes with Grounding DINO bboxes")
    p.add_argument("--dataset_root", type=str, required=True,
                   help="Root of the CALVIN split (e.g. playground/Datasets/calvin/task_ABC_D)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write per-episode JSON annotation files")
    p.add_argument("--model_config", type=str, default="GroundingDINO_SwinT_OGC.py",
                   help="Path to Grounding DINO model config file")
    p.add_argument("--model_weights", type=str, default="groundingdino_swint_ogc.pth",
                   help="Path to Grounding DINO checkpoint weights")
    p.add_argument("--box_threshold", type=float, default=0.30)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_workers", type=int, default=1,
                   help="Number of parallel annotation workers (CPU-bound)")
    p.add_argument("--resume", action="store_true",
                   help="Skip episodes that already have annotation files")
    return p.parse_args()


def annotate_episode(
    ep_id: str,
    ep_dir: pathlib.Path,
    output_dir: pathlib.Path,
    model,
    box_threshold: float,
    text_threshold: float,
    device: str,
    resume: bool,
) -> int:
    out_file = output_dir / f"{ep_id}.json"
    if resume and out_file.exists():
        return 0

    frames = _load_episode_frames(ep_dir)
    if not frames:
        return 0

    annotations: Dict[str, Optional[str]] = {}
    for frame_idx, image, task_lang in frames:
        bbox_str = _predict_bbox(model, image, task_lang, box_threshold, text_threshold, device)
        annotations[str(frame_idx)] = bbox_str

    with open(out_file, "w") as f:
        json.dump(annotations, f)

    return len(annotations)


def main() -> None:
    args = parse_args()

    dataset_root = pathlib.Path(args.dataset_root)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[annotate_bbox] Loading Grounding DINO model from {args.model_weights} ...")
    model = _load_gdino(args.model_config, args.model_weights, device=args.device)
    print("[annotate_bbox] Model loaded.")

    episodes = list(_iter_episodes(dataset_root))
    print(f"[annotate_bbox] Found {len(episodes)} episode directories.")

    total_annotations = 0
    for ep_id, ep_dir in tqdm(episodes, desc="Annotating episodes"):
        count = annotate_episode(
            ep_id=ep_id,
            ep_dir=ep_dir,
            output_dir=output_dir,
            model=model,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
            resume=args.resume,
        )
        total_annotations += count

    print(f"\n[annotate_bbox] Done. Total annotations: {total_annotations}")
    print(f"[annotate_bbox] Output written to: {output_dir}")


if __name__ == "__main__":
    main()
