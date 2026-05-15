"""
eval_calvin_logging.py — Structured logging wrapper for CALVIN evaluation.

Adds --exp_id flag for experiment tracking, writes per-sequence results to
a JSON file, and prints a summary table suitable for easy comparison.

Usage:
    python examples/calvin/eval_files/eval_calvin_logging.py \\
        --exp_id E-1_stage2_step25k \\
        --checkpoint results/Checkpoints/starvla_calvin_stage2/step_25000 \\
        --replan_steps 16 \\
        --dataset_path playground/Datasets/calvin/task_D_D \\
        --num_sequences 1000 \\
        --output_dir results/eval_logs

The output JSON has the format:
    {
        "exp_id": "E-1_stage2_step25k",
        "replan_steps": 16,
        "checkpoint": "...",
        "sr_1": 0.xx, "sr_2": 0.xx, ..., "sr_5": 0.xx,
        "avg_len": x.xx,
        "sequences": [ {"idx": 0, "tasks": [...], "success_len": N}, ... ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured CALVIN evaluation with experiment logging")

    # Experiment identity
    parser.add_argument("--exp_id", type=str, required=True,
                        help="Experiment identifier (e.g. 'E-1_stage2_step25k')")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="results/eval_logs",
                        help="Directory to save structured JSON logs")

    # CALVIN eval parameters
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--replan_steps", type=int, default=16,
                        help="Action horizon steps to execute before replanning")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to Calvin dataset (task_D_D)")
    parser.add_argument("--calvin_config_path", type=str, default="",
                        help="Path to Calvin config directory")
    parser.add_argument("--eval_sequences_path", type=str, default="",
                        help="Path to eval_sequences.json")
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--unnorm_key", type=str, default="")

    return parser.parse_args()


def launch_policy_server(checkpoint: str, host: str, port: int) -> subprocess.Popen:
    """Start the policy server as a background process."""
    cmd = [
        sys.executable, "-m", "deployment.model_server.server_policy",
        "--checkpoint", checkpoint,
        "--host", host,
        "--port", str(port),
    ]
    print(f"[eval_logging] Launching policy server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Give the server a few seconds to start
    time.sleep(10)
    return proc


def run_eval(args: argparse.Namespace) -> dict:
    """
    Call eval_calvin.py via subprocess and capture the per-sequence JSON.
    Alternatively, import and call directly if preferred.
    """
    eval_script = Path(__file__).parent / "eval_calvin.py"
    result_file = Path(args.output_dir) / f"{args.exp_id}_raw.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(eval_script),
        "--args.host", args.host,
        "--args.port", str(args.port),
        "--args.dataset_path", args.dataset_path,
        "--args.num_sequences", str(args.num_sequences),
        "--args.replan_steps", str(args.replan_steps),
        "--args.resize_size", str(args.resize_size),
    ]
    if args.calvin_config_path:
        cmd += ["--args.calvin_config_path", args.calvin_config_path]
    if args.eval_sequences_path:
        cmd += ["--args.eval_sequences_path", args.eval_sequences_path]
    if args.unnorm_key:
        cmd += ["--args.unnorm_key", args.unnorm_key]

    print(f"[eval_logging] Running eval: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout
    print(stdout)
    if result.returncode != 0:
        print(f"[eval_logging] WARNING: eval exited with code {result.returncode}")
        print(result.stderr)

    # Parse success rates from stdout (standard CALVIN output format)
    sr = {}
    for line in stdout.splitlines():
        for i in range(1, 6):
            if f"SR-{i}" in line or f"success_rate_{i}" in line.lower():
                try:
                    val = float(line.split(":")[-1].strip().rstrip("%")) / 100.0
                    sr[f"sr_{i}"] = val
                except (ValueError, IndexError):
                    pass

    return sr


def compute_avg_len(sr: dict) -> float:
    """
    Compute average successful sequence length:
        avg_len = Σ_{i=1}^{5} SR-i
    (standard CALVIN metric)
    """
    return sum(sr.get(f"sr_{i}", 0.0) for i in range(1, 6))


def save_results(args: argparse.Namespace, sr: dict) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{args.exp_id}_{timestamp}.json"

    record = {
        "exp_id": args.exp_id,
        "checkpoint": args.checkpoint,
        "replan_steps": args.replan_steps,
        "num_sequences": args.num_sequences,
        "timestamp": timestamp,
        **sr,
        "avg_len": compute_avg_len(sr),
    }

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n[eval_logging] Results saved to: {out_path}")
    return out_path


def print_summary(args: argparse.Namespace, sr: dict) -> None:
    avg_len = compute_avg_len(sr)
    print("\n" + "=" * 60)
    print(f"  Experiment : {args.exp_id}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  replan_steps: {args.replan_steps}")
    print("-" * 60)
    print(f"  {'Metric':<12}  {'Value':>8}")
    print("-" * 60)
    for i in range(1, 6):
        val = sr.get(f"sr_{i}", float("nan"))
        print(f"  SR-{i}        :  {val * 100:6.2f}%")
    print(f"  Avg len     :  {avg_len:.4f}")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    sr = run_eval(args)
    if not sr:
        print("[eval_logging] WARNING: No success rates parsed from eval output.")
        sr = {f"sr_{i}": float("nan") for i in range(1, 6)}

    print_summary(args, sr)
    save_results(args, sr)


if __name__ == "__main__":
    main()
