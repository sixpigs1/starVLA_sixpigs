#!/usr/bin/env bash
# scan_replan_steps.sh — E-1h: Batch scan of replan_steps values
#
# Usage:
#   bash examples/calvin/eval_files/scan_replan_steps.sh \
#       --checkpoint results/Checkpoints/starvla_calvin_stage2/step_25000 \
#       --dataset_path playground/Datasets/calvin/task_D_D \
#       --num_sequences 500 \
#       --output_dir results/eval_logs/e1h_scan
#
# This script tests replan_steps ∈ {8,12,16,20,24,28,32,36,40} and logs each
# run as a separate JSON file in --output_dir.
# After all runs, it prints a summary table sorted by avg_len.

set -euo pipefail

# ──────────────────────────────────────────────
# Parse named arguments
# ──────────────────────────────────────────────
CHECKPOINT=""
DATASET_PATH=""
NUM_SEQUENCES=500
OUTPUT_DIR="results/eval_logs/e1h_scan"
HOST="127.0.0.1"
PORT=8000
EVAL_SCRIPT="examples/calvin/eval_files/eval_calvin_logging.py"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)      CHECKPOINT="$2";      shift 2 ;;
        --dataset_path)    DATASET_PATH="$2";    shift 2 ;;
        --num_sequences)   NUM_SEQUENCES="$2";   shift 2 ;;
        --output_dir)      OUTPUT_DIR="$2";      shift 2 ;;
        --host)            HOST="$2";            shift 2 ;;
        --port)            PORT="$2";            shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" || -z "$DATASET_PATH" ]]; then
    echo "ERROR: --checkpoint and --dataset_path are required"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ──────────────────────────────────────────────
# Replan steps to scan (E-1h sweep)
# ──────────────────────────────────────────────
REPLAN_STEPS_LIST=(8 12 16 20 24 28 32 36 40)

echo "=========================================="
echo "E-1h: Horizon Scan"
echo "Checkpoint : $CHECKPOINT"
echo "Sequences  : $NUM_SEQUENCES"
echo "Steps scan : ${REPLAN_STEPS_LIST[*]}"
echo "Output dir : $OUTPUT_DIR"
echo "=========================================="

# ──────────────────────────────────────────────
# Run evaluations
# ──────────────────────────────────────────────
for RS in "${REPLAN_STEPS_LIST[@]}"; do
    EXP_ID="E1h_rs${RS}"
    echo ""
    echo "--- Running replan_steps=${RS} (exp_id=${EXP_ID}) ---"
    python "$EVAL_SCRIPT" \
        --exp_id       "$EXP_ID" \
        --checkpoint   "$CHECKPOINT" \
        --replan_steps "$RS" \
        --host         "$HOST" \
        --port         "$PORT" \
        --dataset_path "$DATASET_PATH" \
        --num_sequences "$NUM_SEQUENCES" \
        --output_dir   "$OUTPUT_DIR"
done

# ──────────────────────────────────────────────
# Print summary table
# ──────────────────────────────────────────────
echo ""
echo "=========================================="
echo "E-1h Summary (sorted by avg_len)"
echo "=========================================="
echo "replan_steps | SR-1  | SR-2  | SR-3  | SR-4  | SR-5  | avg_len"
echo "-------------|-------|-------|-------|-------|-------|--------"

# Use python for JSON parsing and sorting
python3 - <<'PYEOF'
import json, glob, sys, os

output_dir = sys.argv[1] if len(sys.argv) > 1 else "results/eval_logs/e1h_scan"

# Read from environment variable set by the parent script
import os
output_dir_env = os.environ.get("SCAN_OUTPUT_DIR", "")
if output_dir_env:
    output_dir = output_dir_env

records = []
for f in sorted(glob.glob(os.path.join(output_dir, "E1h_rs*.json"))):
    try:
        with open(f) as fh:
            rec = json.load(fh)
        records.append(rec)
    except Exception:
        pass

records.sort(key=lambda r: r.get("avg_len", 0), reverse=True)
for r in records:
    rs = r.get("replan_steps", "?")
    srs = [r.get(f"sr_{i}", float("nan")) for i in range(1, 6)]
    avg = r.get("avg_len", float("nan"))
    sr_strs = " | ".join(f"{v*100:5.1f}%" for v in srs)
    print(f"  rs={rs:3d}      | {sr_strs} | {avg:.4f}")
PYEOF

