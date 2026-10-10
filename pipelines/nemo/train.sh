#!/usr/bin/env bash
# NeMo AutoModel LoRA SFT of Qwen3-VL (recipe FinetuneRecipeForVLM).
# Env overrides: MODEL NGPU STRATEGY(shard|replicate) MAX_STEPS TRAIN_SPLIT(train|audit32) AUDIT=1 RUN_TAG
# Always FSDP2 full sharding (label "fsdp2"): NeMo's VLM recipe has no autocast, so DDP cannot run fp32
# LoRA weights (bf16 x fp32 matmul error), and FSDP2 refuses dp_replicate_size == dp_size (no DDP-like
# HSDP). FSDP2's mixed-precision policy computes in bf16 with fp32 LoRA master weights.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../common/env.sh"
source "$HERE/../common/hparams.sh"
source "$HERE/../common/strategy.sh"
source "$HERE/../common/run_lib.sh"
use_env nemo
require_package; require_model
requested_strategy="$(pick_strategy)"
if fits_one_gpu; then preflight_run none; else preflight_run shard; fi     # FSDP2 always shards over NGPU
export STRATEGY=fsdp2
export TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
export NEMO_DATA="$FORMATS/nemo"
"$PY" "$HERE/prep_data.py" --pkg "$PKG" --out "$NEMO_DATA"
TRAIN_FILE="$NEMO_DATA/$TRAIN_SPLIT.jsonl" derive_hparams

tag="${RUN_TAG:-}"
if [[ "${AUDIT:-0}" == 1 ]]; then tag="${tag:-audit}"; fi
start_run nemo "$tag"
export PYTHONPATH="$HERE:$PYTHONPATH"         # person_rock_data.make_dataset

export CKPT_EVERY=$(( SAVE_STEPS > 0 ? SAVE_STEPS : TOTAL_STEPS ))
export DISTRIBUTED_BLOCK="distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none
  dp_replicate_size: 1
  tp_size: 1
  cp_size: 1
  sequence_parallel: false
  activation_checkpointing: true"
echo "nemo_distributed=fsdp2-full-shard (requested $requested_strategy)" >> "$RUN/run_info.txt"
python3 "$PIPELINES/common/render_template.py" "$HERE/qwen3vl_lora.yaml.tmpl" "$RUN/nemo.yaml"

if [[ "${AUDIT:-0}" == 1 ]]; then
  "$PY" "$HERE/audit_dump.py" --data "$NEMO_DATA/$TRAIN_SPLIT.jsonl" --model "$MODEL" \
    --out "$RUN/audit_dump.json" --min-pixels "$MIN_PIXELS" --max-pixels "$MAX_PIXELS"
fi

export PET_MASTER_PORT="$(free_port)"       # the CLI builds its torchrun arguments from PET_* variables
cmd=(automodel "$RUN/nemo.yaml" --nproc-per-node "$NGPU")
echo "${cmd[*]}" > "$RUN/cmd.txt"
rc=0
run_logged "${cmd[@]}" || rc=$?
# the adapter of the LAST step only (step numbers are 0-based); an earlier checkpoint is not a result
final="$(ls -d "$RUN"/checkpoints/epoch_*_step_$(( TOTAL_STEPS - 1 ))/model 2>/dev/null | tail -1 || true)"
if [[ -n "$final" && -f "$final/adapter_model.safetensors" ]]; then
  mkdir -p "$RUN/adapter"
  cp "$final/adapter_config.json" "$final/adapter_model.safetensors" "$RUN/adapter/"
fi
finish_run "$rc"
exit "$FINAL_RC"
