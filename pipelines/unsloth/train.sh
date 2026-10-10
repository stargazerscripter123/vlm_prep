#!/usr/bin/env bash
# Unsloth LoRA SFT of Qwen3-VL (DDP only -- Unsloth cannot shard).
# Env overrides: MODEL NGPU QUANT(bf16|4bit, auto) MAX_STEPS TRAIN_SPLIT(train|audit32) AUDIT=1 RUN_TAG
# QUANT defaults to bf16 when the model fits one GPU, else 4bit (32B on 48 GB cards: reference only).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../common/env.sh"
source "$HERE/../common/hparams.sh"
source "$HERE/../common/strategy.sh"
source "$HERE/../common/run_lib.sh"
use_env unsloth
preflight_run none          # Unsloth falls back to 4-bit when the model does not fit one GPU
fits="$(STRATEGY= pick_strategy)"
export QUANT="${QUANT:-$([[ "$fits" == replicate ]] && echo bf16 || echo 4bit)}"
export STRATEGY="replicate-$QUANT"      # DDP; label carries the precision
export TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
export UNSLOTH_DATA="$FORMATS/unsloth"
"$PY" "$HERE/prep_data.py" --pkg "$PKG" --out "$UNSLOTH_DATA"
TRAIN_FILE="$UNSLOTH_DATA/$TRAIN_SPLIT.jsonl" derive_hparams

tag="${RUN_TAG:-}"
if [[ "${AUDIT:-0}" == 1 ]]; then export VLM_AUDIT_DUMP=1; tag="${tag:-audit}"; fi
start_run unsloth "$tag"

if (( NGPU == 1 )); then
  cmd=("$PY" "$HERE/train.py")
else
  cmd=(torchrun --standalone --nproc_per_node "$NGPU" "$HERE/train.py")
fi
echo "${cmd[*]}" > "$RUN/cmd.txt"
rc=0
run_logged "${cmd[@]}" || rc=$?
finish_run "$rc"
exit "$FINAL_RC"
