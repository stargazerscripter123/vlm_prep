#!/usr/bin/env bash
# ms-swift LoRA SFT of Qwen3-VL. Consumes the data package directly (ROOT_IMAGE_DIR=$PKG).
# Env overrides: MODEL NGPU STRATEGY(shard|replicate) MAX_STEPS TRAIN_SPLIT(train|audit32) AUDIT=1 RUN_TAG
#   shard     -> DeepSpeed ZeRO-3 (pipelines/common/zero3.json, same file as TRL / Axolotl)
#   replicate -> plain DDP (no --deepspeed)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../common/env.sh"
source "$HERE/../common/hparams.sh"
source "$HERE/../common/strategy.sh"
source "$HERE/../common/run_lib.sh"
use_env swift
require_package; require_model
STRATEGY="$(pick_strategy)"; export STRATEGY
preflight_run "$STRATEGY"
export TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
if [[ "$TRAIN_SPLIT" == train ]]; then
  DATASET="$PKG/train.jsonl"
else
  "$PY" "$HERE/prep_data.py" --pkg "$PKG" --out "$FORMATS/swift" --splits "$TRAIN_SPLIT"
  DATASET="$FORMATS/swift/$TRAIN_SPLIT.jsonl"
fi
TRAIN_FILE="$DATASET" derive_hparams

tag="${RUN_TAG:-}"
if [[ "${AUDIT:-0}" == 1 ]]; then export VLM_AUDIT_DUMP=1; tag="${tag:-audit}"; fi
start_run swift "$tag"

export ROOT_IMAGE_DIR="$PKG"            # package image paths are relative to the package root
export IMAGE_MIN_TOKEN_NUM IMAGE_MAX_TOKEN_NUM
export NPROC_PER_NODE="$NGPU"
export MASTER_PORT="$(free_port)"
SAVE_EVERY=$(( SAVE_STEPS > 0 ? SAVE_STEPS : TOTAL_STEPS ))

args=(sft
  --model "$MODEL" --dataset "$DATASET" --split_dataset_ratio 0 --load_from_cache_file false
  --tuner_type lora --lora_rank "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT"
  --target_modules all-linear --freeze_vit true --freeze_aligner true --freeze_llm false
  --vit_gradient_checkpointing false --external_plugins "$HERE/no_vit_backward.py"
  --torch_dtype bfloat16 --attn_impl "$ATTN_IMPL" --max_length "$MAX_LENGTH"
  --learning_rate "$LR" --lr_scheduler_type "$LR_SCHEDULER" --warmup_steps "$WARMUP_STEPS"
  --max_steps "$TOTAL_STEPS" --weight_decay "$WEIGHT_DECAY" --adam_beta1 "$ADAM_BETA1"
  --adam_beta2 "$ADAM_BETA2" --adam_epsilon "$ADAM_EPS" --max_grad_norm "$MAX_GRAD_NORM" --optim adamw_torch
  --per_device_train_batch_size "$MICRO_BS" --gradient_accumulation_steps "$GRAD_ACC"
  --gradient_checkpointing true
  --logging_steps "$LOGGING_STEPS" --eval_strategy no --report_to none
  --save_strategy steps --save_steps "$SAVE_EVERY" --save_total_limit 2 --save_only_model true
  --dataloader_num_workers "$DATALOADER_WORKERS" --dataset_num_proc 4
  --seed "$SEED" --data_seed "$SEED"
  --output_dir "$RUN/out" --add_version false
)
if [[ "$STRATEGY" == shard ]]; then args+=(--deepspeed "$PIPELINES/common/zero3.json"); fi
echo "swift ${args[*]}" > "$RUN/cmd.txt"

rc=0
run_logged swift "${args[@]}" || rc=$?
# the checkpoint of the LAST step only; an earlier one (--save-steps) is not a result
ckpt="$(ls -d "$RUN"/out/checkpoint-"$TOTAL_STEPS" "$RUN"/out/*/checkpoint-"$TOTAL_STEPS" 2>/dev/null | head -1 || true)"
if [[ -n "$ckpt" && -f "$ckpt/adapter_model.safetensors" ]]; then
  mkdir -p "$RUN/adapter"
  cp "$ckpt/adapter_config.json" "$ckpt/adapter_model.safetensors" "$RUN/adapter/"
fi
finish_run "$rc"
exit "$FINAL_RC"
