#!/usr/bin/env bash
# One-click launcher: 4-GPU parallel IJB-C evaluation over the EXPS list
# defined in run_ijb.py.
#
# Usage:
#   bash run_ijb_4gpu.sh                  # uses GPUs 0,1,2,3
#   bash run_ijb_4gpu.sh "0,1,2,3"        # explicit GPU list

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

GPUS="${1:-0,1,2,3}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}

LOG_DIR="./IJB_logs"
mkdir -p "$LOG_DIR" ./IJB_result

echo "============================================================"
echo "[1/4] Pre-warming IJB-C metadata caches (CPU-only, ~30s)..."
echo "============================================================"
CUDA_VISIBLE_DEVICES= python run_ijb.py --prewarm

echo ""
echo "============================================================"
echo "[2/4] LPT schedule preview ($N shards):"
echo "============================================================"
CUDA_VISIBLE_DEVICES= python run_ijb.py --num-shards "$N" --dry-run

echo ""
echo "============================================================"
echo "[3/4] Launching $N shards on GPUs: $GPUS"
echo "============================================================"
PIDS=()
for ((i=0; i<N; i++)); do
    GPU="${GPU_ARR[$i]}"
    LOG="$LOG_DIR/shard${i}_gpu${GPU}.log"
    echo "  shard $i  ->  GPU $GPU  ->  $LOG"
    CUDA_VISIBLE_DEVICES="$GPU" \
        python -u run_ijb.py --shard "$i" --num-shards "$N" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

# wait for all shards; if any fails, propagate failure
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=1
    fi
done
if [[ "$FAIL" -ne 0 ]]; then
    echo "[!] One or more shards failed — check $LOG_DIR/*.log"
    exit 1
fi

echo ""
echo "============================================================"
echo "[4/4] Merging results from all shards..."
echo "============================================================"
python merge_ijb_results.py | tee "$LOG_DIR/final_table.md"

echo ""
echo "Done. Per-shard logs in $LOG_DIR/, npz in IJB_result/, final table above."
