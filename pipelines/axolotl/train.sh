#!/usr/bin/env bash
# Axolotl LoRA SFT of Qwen3-VL.
# Env overrides: MODEL NGPU STRATEGY(shard|replicate) MAX_STEPS TRAIN_SPLIT(train|audit32) AUDIT=1 RUN_TAG
#   shard     -> DeepSpeed ZeRO-3 (pipelines/common/zero3.json; FSDP2+LoRA has open bug #4079)
#   replicate -> plain DDP
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../common/env.sh"
source "$HERE/../common/hparams.sh"
source "$HERE/../common/strategy.sh"
source "$HERE/../common/run_lib.sh"
use_env axolotl
require_package; require_model
STRATEGY="$(pick_strategy)"; export STRATEGY
preflight_run "$STRATEGY"
export TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
export AXOLOTL_DATA="$FORMATS/axolotl"
"$PY" "$HERE/prep_data.py" --pkg "$PKG" --out "$AXOLOTL_DATA"
TRAIN_FILE="$AXOLOTL_DATA/$TRAIN_SPLIT.jsonl" derive_hparams

tag="${RUN_TAG:-}"
if [[ "${AUDIT:-0}" == 1 ]]; then export VLM_AUDIT_DUMP=1; tag="${tag:-audit}"; fi
start_run axolotl "$tag"
export PYTHONPATH="$HERE:$PYTHONPATH"          # vlm_fixes plugin

if (( SAVE_STEPS > 0 )); then
  export SAVE_BLOCK="save_strategy: steps
save_steps: $SAVE_STEPS"
else
  export SAVE_BLOCK="save_strategy: \"no\""
fi
if [[ "$STRATEGY" == shard ]]; then
  export DEEPSPEED_LINE="deepspeed: $PIPELINES/common/zero3.json"
  # Axolotl auto-enables fused LoRA kernels; under ZeRO-3 they keep gathered base weights alive through
  # backward (32B OOM on 48 GB cards: 46 GiB vs 28 GiB for ms-swift with the same zero3.json). Off for shard.
  export LORA_KERNEL_BLOCK="lora_qkv_kernel: false
lora_o_kernel: false
lora_mlp_kernel: false
lora_embedding_kernel: false"
else
  export DEEPSPEED_LINE=""
  export LORA_KERNEL_BLOCK="# fused LoRA kernels: Axolotl default (auto-enabled; identical math, audited)"
fi
python3 "$PIPELINES/common/render_template.py" "$HERE/qwen3vl_lora.yaml.tmpl" "$RUN/axolotl.yaml"

cmd=(axolotl train "$RUN/axolotl.yaml" --launcher torchrun -- --nproc_per_node="$NGPU"
     --master_port="$(free_port)")
echo "${cmd[*]}" > "$RUN/cmd.txt"
rc=0
run_logged "${cmd[@]}" || rc=$?
if [[ -f "$RUN/out/adapter_model.safetensors" ]]; then
  mkdir -p "$RUN/adapter"
  cp "$RUN/out/adapter_config.json" "$RUN/out/adapter_model.safetensors" "$RUN/adapter/"
fi
finish_run "$rc"
exit "$FINAL_RC"
