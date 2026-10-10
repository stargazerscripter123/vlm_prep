#!/usr/bin/env bash
# Single source of truth for the hyper-parameters of all 5 frameworks. Integers where possible.
# Each framework's train.sh translates these into its own flags/config; the values a run actually
# used are written to $RUN/hparams.env (plus the rendered config / command line in the run dir).

export SEED=42

# LoRA on the language model only: q/k/v/o/gate/up/down of every decoder layer.
# Vision tower, mergers (incl. deepstack), lm_head and embeddings stay frozen.
export LORA_R=32
export LORA_ALPHA=64
export LORA_DROPOUT=0            # 0 keeps Unsloth's fast LoRA path and NeMo's fused LoRA MLP on

# Optimizer / schedule (AdamW, cosine to 0, linear warmup from 0, grad clip 1.0)
export LR="${LR:-5e-5}"
export WEIGHT_DECAY=0.0
export ADAM_BETA1=0.9
export ADAM_BETA2=0.999
export ADAM_EPS=1e-8
export MAX_GRAD_NORM=1.0
export LR_SCHEDULER=cosine

# Batch: micro-batch 1 per GPU; the global batch targets 32 samples per optimizer step.
# grad_accum = round(32 / NGPU): exactly 32 for 1, 2, 4, 8 GPUs; 33 / 30 / 30 / 35 for 3 / 5 / 6 / 7 GPUs
# (derive_hparams exports the effective GLOBAL_BS, so step counts and samples/s stay correct).
export GLOBAL_BS_TARGET="${GLOBAL_BS_TARGET:-32}"
export MICRO_BS=1

# Images: same pixel bounds everywhere (32x32 px per visual token -> 64..1024 tokens)
export MIN_PIXELS=65536
export MAX_PIXELS=1048576
export IMAGE_MIN_TOKEN_NUM=64     # ms-swift spelling of MIN_PIXELS
export IMAGE_MAX_TOKEN_NUM=1024   # ms-swift spelling of MAX_PIXELS

export MAX_LENGTH=4096            # no sample is longer than ~850 tokens; nothing is truncated
export ATTN_IMPL=sdpa
# DataLoader workers hand tensors to the trainer through /dev/shm. Containers often get only 64 MB
# there (workers then die with "Bus error"), so loading falls back to in-process when it is small.
# Results do not depend on this value (same sampler order); override with DATALOADER_WORKERS=<n>.
if [[ -z "${DATALOADER_WORKERS:-}" ]]; then
  DATALOADER_WORKERS=4
  _shm_kb="$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2 {print $4}')"
  if [[ "$_shm_kb" =~ ^[0-9]+$ ]] && (( _shm_kb < 4 * 1024 * 1024 )); then
    echo "note: /dev/shm has only $(( _shm_kb / 1024 )) MiB free -- DataLoader workers disabled" \
         "(mount a larger /dev/shm for the pod to get them back)" >&2
    DATALOADER_WORKERS=0
  fi
fi
export DATALOADER_WORKERS
export LOGGING_STEPS=1
export SAVE_STEPS="${SAVE_STEPS:-0}"   # 0 = save only the final adapter (run.sh --save-steps N)
export RUN_TIMEOUT="${RUN_TIMEOUT:-0}" # wall-clock limit of one training run, e.g. 12h; 0 = none (run.sh --timeout)

derive_hparams() {
  # Needs NGPU and TRAIN_FILE (or PKG). Sets GRAD_ACC, the effective GLOBAL_BS, N_TRAIN, TOTAL_STEPS
  # (one epoch of full batches unless MAX_STEPS is set) and WARMUP_STEPS (5%, >=1 when TOTAL_STEPS >= 2).
  local per_step=$(( NGPU * MICRO_BS ))
  export GRAD_ACC=$(( (2 * GLOBAL_BS_TARGET + per_step) / (2 * per_step) ))     # round to nearest
  if (( GRAD_ACC < 1 )); then GRAD_ACC=1; fi
  export GLOBAL_BS=$(( GRAD_ACC * per_step ))
  if (( GLOBAL_BS != GLOBAL_BS_TARGET )); then
    echo "note: $NGPU GPU(s) x grad_accum $GRAD_ACC = global batch $GLOBAL_BS (target $GLOBAL_BS_TARGET);" \
         "not exactly comparable with 1/2/4/8-GPU runs" >&2
  fi
  export N_TRAIN
  N_TRAIN="$(wc -l < "${TRAIN_FILE:-$PKG/train.jsonl}" | tr -d ' ')"
  export TOTAL_STEPS="${MAX_STEPS:-$(( N_TRAIN / GLOBAL_BS ))}"
  if (( TOTAL_STEPS < 1 )); then TOTAL_STEPS=1; fi
  local warmup=$(( (TOTAL_STEPS * 5 + 50) / 100 ))
  if (( TOTAL_STEPS >= 2 && warmup < 1 )); then warmup=1; fi
  export WARMUP_STEPS="${WARMUP_STEPS_OVERRIDE:-$warmup}"
}
