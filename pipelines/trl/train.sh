#!/usr/bin/env bash
# TRL + PEFT LoRA SFT of Qwen3-VL on the data package.
# Env overrides: MODEL NGPU STRATEGY(shard|replicate) MAX_STEPS TRAIN_SPLIT(train|audit32) AUDIT=1 RUN_TAG
#   bash pipelines/trl/train.sh                                  # full epoch, auto strategy, all GPUs
#   MAX_STEPS=30 MODEL=models/Qwen3-VL-4B-Instruct bash pipelines/trl/train.sh
#   AUDIT=1 NGPU=1 TRAIN_SPLIT=audit32 MODEL=... bash pipelines/trl/train.sh   # label audit + step-1 loss
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../common/env.sh"
source "$HERE/../common/hparams.sh"
source "$HERE/../common/strategy.sh"
source "$HERE/../common/run_lib.sh"
use_env trl
require_package; require_model
STRATEGY="$(pick_strategy)"; export STRATEGY
preflight_run "$STRATEGY"
if [[ "$STRATEGY" == shard && "$(model_tag)" == 32b ]]; then
  echo "warning: TRL + DeepSpeed ZeRO-3 with the 32B model ran out of memory on 4x48 GB cards (unresolved);" \
       "prefer GPUs where the model fits (replicate)" >&2
fi
export TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
export TRL_DATA="$FORMATS/trl"
"$PY" "$HERE/prep_data.py" --pkg "$PKG" --out "$TRL_DATA"
TRAIN_FILE="$TRL_DATA/$TRAIN_SPLIT.jsonl" derive_hparams

tag="${RUN_TAG:-}"
if [[ "${AUDIT:-0}" == 1 ]]; then export VLM_AUDIT_DUMP=1; tag="${tag:-audit}"; fi
start_run trl "$tag"

if [[ "$STRATEGY" == shard ]]; then
  sed "s#__ZERO3_JSON__#$PIPELINES/common/zero3.json#" "$HERE/accelerate/zero3.yaml" > "$RUN/accelerate.yaml"
else
  cp "$HERE/accelerate/ddp.yaml" "$RUN/accelerate.yaml"
fi
if (( NGPU == 1 )) && [[ "$STRATEGY" == replicate ]]; then
  cmd=("$PY" "$HERE/train.py")
else
  cmd=(accelerate launch --config_file "$RUN/accelerate.yaml" --num_processes "$NGPU"
       --main_process_port "$(free_port)" "$HERE/train.py")
fi
echo "${cmd[*]}" > "$RUN/cmd.txt"
rc=0
run_logged "${cmd[@]}" || rc=$?
finish_run "$rc"
exit "$FINAL_RC"
