#!/usr/bin/env bash
# Unified evaluation (runs in the TRL env).
#   eval.sh <run_dir> [subset]          adapter = <run_dir>/adapter, base model from run_info.txt
#   eval.sh baseline <model_dir> [subset]   zero-shot base model -> runs/baseline/<model>/eval/<subset>
# subset: n100 (90 pos + 10 neg) | n50 (45 + 5) | full (1,100). The model is split over all GPUs
# (device_map=auto) when it does not fit on one GPU, otherwise data-parallel with torchrun.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../common/env.sh"
source "$HERE/../common/hparams.sh"
source "$HERE/../common/strategy.sh"
use_env trl

if [[ "$1" == baseline ]]; then
  MODEL="$(model_dir_for "$2")"         # 4b | 32b | model dir
  SUBSET="${3:-n100}"
  ADAPTER_ARGS=()
  OUT="$RUNS/baseline/$(basename "$MODEL")/eval/$SUBSET"
else
  RUN_DIR="$(realpath_m "$1")"
  SUBSET="${2:-n100}"
  MODEL="$(sed -n 's/^model=//p' "$RUN_DIR/run_info.txt")"
  PKG="$(sed -n 's/^package=//p' "$RUN_DIR/run_info.txt")"
  [[ -f "$RUN_DIR/adapter/adapter_model.safetensors" ]] || { echo "no adapter in $RUN_DIR/adapter" >&2; exit 1; }
  ADAPTER_ARGS=(--adapter "$RUN_DIR/adapter")
  OUT="$RUN_DIR/eval/$SUBSET"
fi
export MODEL
require_package; require_model
assert_not_dropbox "$OUT"
rm -f "$OUT"/preds.rank*.jsonl "$OUT/metrics.json" "$OUT/predict_info.json"    # never keep stale results
mkdir -p "$OUT"
echo "evaluating $(basename "$MODEL")${ADAPTER_ARGS:+ + adapter} on $SUBSET with $NGPU GPU(s) -> $OUT"

unset STRATEGY
mode="$(pick_strategy)"
check_fit "$mode"                         # e.g. 32B on one 48 GB GPU cannot be evaluated
if [[ "$mode" == shard ]]; then
  "$PY" "$HERE/predict.py" --model "$MODEL" "${ADAPTER_ARGS[@]}" --pkg "$PKG" --subset "$SUBSET" \
    --out "$OUT" --device-map auto 2>&1 | tee "$OUT/predict.log"
else
  torchrun --standalone --nproc_per_node "$NGPU" "$HERE/predict.py" --model "$MODEL" "${ADAPTER_ARGS[@]}" \
    --pkg "$PKG" --subset "$SUBSET" --out "$OUT" 2>&1 | tee "$OUT/predict.log"
fi
python3 "$HERE/score.py" --pred-dir "$OUT" --pkg "$PKG"
python3 "$HERE/leaderboard.py" "$RUNS"
