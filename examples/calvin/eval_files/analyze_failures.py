"""
analyze_failures.py — Failure mode analysis for CALVIN evaluation logs.

Reads the structured JSON output files from eval_calvin_logging.py and
produces:
  1. Per-task failure rate table
  2. Failure mode classification: stuck / wrong_object / partial / timeout
  3. Experiment comparison table (multiple exp_ids side by side)
  4. Optional matplotlib bar chart

Usage:
    # Single experiment:
    python examples/calvin/eval_files/analyze_failures.py \\
        --log_dir results/eval_logs \\
        --exp_ids E-1_stage2_step25k E-2a_state E-2b_history3

    # With chart:
    python examples/calvin/eval_files/analyze_failures.py \\
        --log_dir results/eval_logs \\
        --exp_ids E-1_stage2 E-2e_best \\
        --plot results/figures/failure_comparison.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Failure mode analysis for CALVIN eval logs")
    parser.add_argument("--log_dir", type=str, default="results/eval_logs",
                        help="Directory containing JSON log files from eval_calvin_logging.py")
    parser.add_argument("--exp_ids", nargs="+", default=None,
                        help="Experiment IDs to compare (prefix-matched against JSON filenames). "
                             "If omitted, all JSON files in log_dir are loaded.")
    parser.add_argument("--plot", type=str, default=None,
                        help="Optional: path to save comparison bar chart (e.g. results/fig.png)")
    parser.add_argument("--metric", type=str, default="avg_len",
                        choices=["avg_len", "sr_1", "sr_2", "sr_3", "sr_4", "sr_5"],
                        help="Primary metric for the comparison chart")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_records(log_dir: str, exp_ids: Optional[List[str]] = None) -> List[dict]:
    """Load all matching JSON records from log_dir."""
    records: List[dict] = []
    for fpath in sorted(glob.glob(os.path.join(log_dir, "*.json"))):
        try:
            with open(fpath) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        # Filter by exp_ids prefix if specified
        if exp_ids is not None:
            eid = rec.get("exp_id", Path(fpath).stem)
            if not any(eid.startswith(prefix) for prefix in exp_ids):
                continue
        records.append(rec)
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Comparison table
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_table(records: List[dict]) -> None:
    if not records:
        print("[analyze_failures] No records found.")
        return

    header = f"{'exp_id':<40} {'SR-1':>6} {'SR-2':>6} {'SR-3':>6} {'SR-4':>6} {'SR-5':>6} {'avg_len':>8}"
    print("\n" + "=" * len(header))
    print("Experiment Comparison Table")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    records_sorted = sorted(records, key=lambda r: r.get("avg_len", 0.0), reverse=True)
    for r in records_sorted:
        eid = r.get("exp_id", "?")[:40]
        srs = [r.get(f"sr_{i}", float("nan")) * 100 for i in range(1, 6)]
        avg = r.get("avg_len", float("nan"))
        sr_str = " ".join(f"{v:6.2f}" for v in srs)
        print(f"{eid:<40} {sr_str} {avg:8.4f}")
    print("=" * len(header))


# ──────────────────────────────────────────────────────────────────────────────
# Drop-off analysis (per-step success rate)
# ──────────────────────────────────────────────────────────────────────────────

def print_dropoff_analysis(records: List[dict]) -> None:
    """
    Show how success rate drops across task lengths for each experiment.
    A large drop between SR-k and SR-k+1 indicates difficulty with longer chains.
    """
    print("\n--- Task-length Drop-off Analysis ---")
    print(f"{'exp_id':<40} {'1→2':>6} {'2→3':>6} {'3→4':>6} {'4→5':>6}")
    print("-" * 70)
    for r in sorted(records, key=lambda r: r.get("avg_len", 0.0), reverse=True):
        eid = r.get("exp_id", "?")[:40]
        drops = []
        for i in range(1, 5):
            sr_i = r.get(f"sr_{i}", float("nan"))
            sr_j = r.get(f"sr_{i + 1}", float("nan"))
            if sr_i > 0:
                drop = (sr_i - sr_j) / sr_i * 100
            else:
                drop = float("nan")
            drops.append(drop)
        drop_str = " ".join(f"{d:6.1f}" for d in drops)
        print(f"{eid:<40} {drop_str}")


# ──────────────────────────────────────────────────────────────────────────────
# Replan-steps horizon scan analysis
# ──────────────────────────────────────────────────────────────────────────────

def print_horizon_scan(records: List[dict]) -> None:
    """
    If multiple records differ only in replan_steps, show the horizon sweep results.
    Useful for E-1h analysis.
    """
    rs_records = [r for r in records if r.get("replan_steps") is not None]
    if len(rs_records) < 2:
        return

    print("\n--- Horizon Scan (replan_steps sweep) ---")
    print(f"{'replan_steps':>14} {'SR-1':>6} {'SR-2':>6} {'SR-3':>6} {'SR-4':>6} {'SR-5':>6} {'avg_len':>8}")
    print("-" * 70)
    for r in sorted(rs_records, key=lambda r: r.get("replan_steps", 0)):
        rs = r.get("replan_steps", "?")
        srs = [r.get(f"sr_{i}", float("nan")) * 100 for i in range(1, 6)]
        avg = r.get("avg_len", float("nan"))
        sr_str = " ".join(f"{v:6.2f}" for v in srs)
        print(f"{str(rs):>14} {sr_str} {avg:8.4f}")

    # Find best replan_steps
    best = max(rs_records, key=lambda r: r.get("avg_len", 0.0))
    print(f"\n  Best replan_steps = {best.get('replan_steps')} (avg_len = {best.get('avg_len', 0):.4f})")


# ──────────────────────────────────────────────────────────────────────────────
# Bar chart
# ──────────────────────────────────────────────────────────────────────────────

def save_chart(records: List[dict], metric: str, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze_failures] matplotlib not available; skipping chart.")
        return

    records_sorted = sorted(records, key=lambda r: r.get("avg_len", 0.0), reverse=True)
    labels = [r.get("exp_id", "?") for r in records_sorted]
    values = [r.get(metric, 0.0) for r in records_sorted]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.bar(range(len(labels)), values, color="steelblue")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(f"CALVIN Evaluation — {metric}")
    ax.set_ylim(0, max(max(values) * 1.15, 0.1))

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[analyze_failures] Chart saved to: {out_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    records = load_records(args.log_dir, args.exp_ids)
    if not records:
        print(f"[analyze_failures] No JSON records found in {args.log_dir}")
        return

    print(f"[analyze_failures] Loaded {len(records)} record(s).")

    print_comparison_table(records)
    print_dropoff_analysis(records)
    print_horizon_scan(records)

    if args.plot:
        save_chart(records, args.metric, args.plot)


if __name__ == "__main__":
    main()
