#!/usr/bin/env bash
# Cross-framework correctness gate:  audit.sh <fw> [model_dir]
#   1. reference encoding + base-model loss on audit32 (TRL env; cached in runs/reference/)
#   2. <fw>/train.sh with AUDIT=1 NGPU=1 TRAIN_SPLIT=audit32 -> one optimizer step over all 32
#      samples through the framework's own pipeline; the hooks dump the collated batches, the
#      trainable-parameter census, the vision-grad probe and the step-1 loss
#   3. compare_audit.py: identical input ids / trained tokens / image tokens, census, frozen ViT,
#      step-1 loss within 3% of the reference
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FW="$1"
if [[ -n "${2:-}" ]]; then export MODEL="$2"; fi     # 4b | 32b | model dir (resolved by env.sh)
# One GPU: AUDIT_GPU, else the first device the caller allowed, else GPU 0. Nothing of the caller's run
# configuration may leak into the audit (it is always 1 step, replicate, audit32).
_first="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${AUDIT_GPU:-${_first%%,*}}" NGPU=1
unset MAX_STEPS RUN_TAG RUN_DIR_FILE QUANT SAVE_STEPS WARMUP_STEPS_OVERRIDE RUN GLOBAL_BS_TARGET TRAIN_SPLIT RUN_TIMEOUT
source "$HERE/env.sh"
source "$HERE/strategy.sh"
require_package; require_model
if ! fits_one_gpu; then
  echo "audit needs the model on ONE GPU; $(basename "$MODEL") does not fit here -- audit with a smaller" \
       "model of the same family (bash pipelines/common/audit.sh $FW 4b)" >&2
  exit 2
fi
TAG="$(model_tag)"
REF="$RUNS/reference/audit32_$(basename "$MODEL")_$(pkg_id).json"

if [[ ! -f "$REF" ]]; then
  (
    use_env trl
    source "$HERE/hparams.sh"
    "$PY" "$HERE/reference_encode.py" --pkg "$PKG" --model "$MODEL" --split audit32 --out "$REF" --loss
  )
fi

export RUN="$RUNS/$FW/audit-$TAG-$(date +%y%m%d_%H%M%S)"
AUDIT=1 TRAIN_SPLIT=audit32 STRATEGY=replicate bash "$PIPELINES/$FW/train.sh" \
  || echo "[audit] $FW training step exited non-zero (comparing what exists)"
python3 "$HERE/compare_audit.py" --reference "$REF" --run "$RUN"
