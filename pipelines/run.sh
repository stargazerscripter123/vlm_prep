#!/usr/bin/env bash
# Single entry point: LoRA SFT of Qwen3-VL (person grounding) with ms-swift / TRL / NeMo / Axolotl / Unsloth.
#
#   bash pipelines/run.sh setup [--model 4b,32b] [--fw all]
#       once per host: conda (Miniforge is installed into ./.conda if there is none), the framework
#       envs, base model(s), the data package, then host checks. Everything missing is created,
#       downloaded or built; safe to re-run. The 4B model is ALWAYS downloaded from Hugging Face with
#       the envs (whatever --model says): it is the quick-test model and the audit fallback.
#
#   bash pipelines/run.sh train --fw swift --model 4b --gpus 4
#       end to end for each framework: correctness audit -> train -> post-train check -> evaluate ->
#       leaderboard. Anything `setup` would have prepared is prepared on demand.
#
#   bash pipelines/run.sh status          leaderboard + the last results
#
# Options
#   --fw       swift|trl|nemo|axolotl|unsloth, a comma list, or all   (train: required; setup: all)
#   --model    4b | 32b | an existing model directory     (train: 32b; setup: 4b,32b -- 4b is always included)
#   --gpus     1..8  number of GPUs: the first N of the visible devices            (default: all)
#   --gpu-ids  e.g. 4,5,6,7  use exactly these devices (instead of / together with --gpus)
#   --steps    optimizer steps for a short run; default = one full epoch (171 steps at batch 32)
#   --lr       learning rate                                                       (default: 5e-5)
#   --eval     full | n100 | n50 | none: test subset evaluated after training      (default: full)
#   --save-steps N   also save a checkpoint every N steps                 (default: final adapter only)
#   --timeout  wall-clock limit per training run, e.g. 12h; 0 = none               (default: none)
#   --strategy shard|replicate   force the parallel strategy for swift/trl/axolotl (default: by GPU memory;
#                                NeMo always shards with FSDP2, Unsloth always replicates)
#   --dry-run       print the plan (devices, batch, steps, strategy per framework) and stop
#   --no-audit      skip the correctness audit (it is cached per framework/model/data/code/env anyway)
#   --no-baseline   skip the zero-shot baseline of the base model
#
# Exit code: 0 all requested frameworks succeeded, 1 at least one failed, 2 usage / precondition error.
#
# Examples
#   bash pipelines/run.sh train --fw all --model 32b --gpus 8             # H200: 5 frameworks, full epoch
#   bash pipelines/run.sh train --fw trl --model 4b --gpus 2 --steps 20 --eval n100
#   bash pipelines/run.sh train --fw all --model 32b --gpus 4 --dry-run   # what would run, and how
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALL_FW="swift trl nemo axolotl unsloth"

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
die() { echo "error: $*" >&2; exit 2; }

command="${1:-help}"
if (( $# > 0 )); then shift; fi
FW=""; MODELS=""; GPUS=""; GPU_IDS=""; STEPS=""; LR_ARG=""; SUBSET=full; SAVE_ARG=""; TIMEOUT_ARG=""
DO_AUDIT=1; DO_BASELINE=1; STRATEGY_ARG=""; DRY_RUN=0
while (( $# > 0 )); do
  case "$1" in
    --fw|--model|--models|--gpus|--gpu-ids|--steps|--lr|--eval|--strategy|--save-steps|--timeout)
      (( $# >= 2 )) || die "option $1 needs a value"
      case "$1" in
        --fw) FW="$2" ;;
        --model|--models) MODELS="$2" ;;
        --gpus) GPUS="$2" ;;
        --gpu-ids) GPU_IDS="$2" ;;
        --steps) STEPS="$2" ;;
        --lr) LR_ARG="$2" ;;
        --eval) SUBSET="$2" ;;
        --strategy) STRATEGY_ARG="$2" ;;
        --save-steps) SAVE_ARG="$2" ;;
        --timeout) TIMEOUT_ARG="$2" ;;
      esac
      shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-audit) DO_AUDIT=0; shift ;;
    --no-baseline) DO_BASELINE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

# ---- validate options -------------------------------------------------------------------
case "$command" in
  help|-h|--help) usage; exit 0 ;;
  setup) FW="${FW:-all}"; MODEL_LIST="${MODELS:-4b,32b}"; MODEL_LIST="${MODEL_LIST//,/ }"
         [[ " $MODEL_LIST " == *" 4b "* ]] || MODEL_LIST="4b $MODEL_LIST" ;;   # 4B always comes with the envs
  train) [[ -n "$FW" ]] || die "train needs --fw <swift|trl|nemo|axolotl|unsloth|all>"
         MODEL_LIST="${MODELS:-32b}"
         [[ "$MODEL_LIST" != *,* ]] || die "train takes exactly one --model (got '$MODEL_LIST')" ;;
  status) FW="${FW:-all}"; MODEL_LIST="" ;;
  *) die "unknown command '$command' (setup | train | status)" ;;
esac
if [[ "$FW" == all ]]; then FW="$ALL_FW"; else FW="${FW//,/ }"; fi
for fw in $FW; do
  [[ " $ALL_FW " == *" $fw "* ]] || die "unknown framework '$fw' (choose from: $ALL_FW, all)"
done
[[ "$SUBSET" =~ ^(full|n100|n50|none)$ ]] || die "--eval must be full, n100, n50 or none"
[[ -z "$GPUS" || "$GPUS" =~ ^[1-8]$ ]] || die "--gpus must be 1..8"
[[ -z "$GPU_IDS" || "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--gpu-ids must look like 0,1,2,3"
[[ -z "$STEPS" || "$STEPS" =~ ^[1-9][0-9]*$ ]] || die "--steps must be a positive integer"
[[ -z "$SAVE_ARG" || "$SAVE_ARG" =~ ^[0-9]+$ ]] || die "--save-steps must be a non-negative integer"
[[ -z "$TIMEOUT_ARG" || "$TIMEOUT_ARG" =~ ^[0-9]+[smhd]?$ ]] || die "--timeout must look like 90m, 12h or 0"
[[ -z "$STRATEGY_ARG" || "$STRATEGY_ARG" =~ ^(shard|replicate)$ ]] || die "--strategy must be shard or replicate"

# Nothing from the calling shell may steer a run except what was passed on this command line.
unset STRATEGY MAX_STEPS TRAIN_SPLIT AUDIT QUANT SAVE_STEPS RUN RUN_DIR_FILE RUN_TAG GLOBAL_BS_TARGET \
      WARMUP_STEPS_OVERRIDE NGPU LR RUN_TIMEOUT MODEL
if [[ -n "$GPU_IDS" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_IDS"; fi
if [[ -n "$GPUS" ]]; then export NGPU="$GPUS"; fi
if [[ -n "$LR_ARG" ]]; then export LR="$LR_ARG"; fi
if [[ -n "$SAVE_ARG" ]]; then export SAVE_STEPS="$SAVE_ARG"; fi
if [[ -n "$TIMEOUT_ARG" ]]; then export RUN_TIMEOUT="$TIMEOUT_ARG"; fi
if [[ "$command" == train ]]; then export MODEL="$MODEL_LIST"; fi

for tool in python3 timeout curl; do
  command -v "$tool" > /dev/null 2>&1 || die "'$tool' is required but not on PATH"
done
(( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 4) )) || die "bash >= 4.4 is required"
if [[ "$command" != status ]] && ! nvidia-smi --query-gpu=index --format=csv,noheader > /dev/null 2>&1; then
  nvidia-smi 2>&1 | head -n 3 >&2
  die "nvidia-smi does not work on this host (no NVIDIA GPU, or driver/library mismatch -- reboot?)"
fi

source "$HERE/common/env.sh"
source "$HERE/common/strategy.sh"
source "$HERE/common/ensure.sh"
mkdir -p "$RUNS"
STATUS_FILE="$RUNS/run_status.tsv"
record() {   # record <fw> <stage> <result> <detail>
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%F_%T)" "$(basename "$MODEL")" "$1" "$2" "$3" "$4" >> "$STATUS_FILE"
  log "$(printf '%-8s %-10s %-5s %s' "$1" "$2" "$3" "$4")"
}
trap 'trap - INT TERM; echo "interrupted -- stopping" >&2; kill -TERM 0' INT TERM   # reaches every child

# ---- status -----------------------------------------------------------------------------
if [[ "$command" == status ]]; then
  python3 "$HERE/eval/leaderboard.py" "$RUNS" > /dev/null && cat "$RUNS/leaderboard.md"
  echo; echo "last results ($STATUS_FILE):"; tail -n 20 "$STATUS_FILE" 2>/dev/null
  exit 0
fi

ensure_conda || exit 2

# ---- setup ------------------------------------------------------------------------------
if [[ "$command" == setup ]]; then
  failed=0
  log "host $(hostname): $GPUS_PRESENT GPU(s), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)," \
      "$(free_gib "$ROOT") GiB free under $ROOT"
  for fw in trl ${FW//trl/}; do    # trl first: it also runs the audit reference, evaluation and data build
    ensure_env "$fw" check || failed=1
  done
  for model in $MODEL_LIST; do
    ensure_model "$model" && log "model ready: $(model_dir_for "$model")" || failed=1
  done
  ensure_package && log "data package ready: $PKG" || failed=1
  log "preflight (driver / wheel index, model checksums, NCCL and GPU-to-GPU copy correctness)"
  bash "$HERE/common/preflight.sh" || failed=1
  if (( failed )); then log "SETUP INCOMPLETE -- fix the errors above and re-run"; exit 1; fi
  log "SETUP OK -- next: bash pipelines/run.sh train --fw <name|all> --model <4b|32b> --gpus <N>"
  exit 0
fi

# ---- train: prerequisites ---------------------------------------------------------------
if (( DRY_RUN )); then      # a dry run never downloads or builds anything
  model_complete "$MODEL" 2> /dev/null \
    || die "dry run: $MODEL is not here yet (a real run downloads it; or: bash pipelines/run.sh setup --model $MODEL_LIST)"
  [[ -n "${PKG:-}" && -f "$PKG/train.jsonl" ]] || die "dry run: no data package yet (a real run builds it; or: bash pipelines/run.sh setup)"
else
  # A host whose envs are being created right now (no `setup` was run): fetch the 4B model as well.
  new_host=0; [[ -f "$CONDA_BASE/envs/$(env_name trl)/.vlm_setup_stamp" ]] || new_host=1
  ensure_env trl || exit 2
  ensure_model "$MODEL_LIST" || exit 2
  if (( new_host )) && [[ "$MODEL_LIST" != 4b ]]; then
    ensure_model 4b || log "warning: the 4B model could not be downloaded (re-run: bash pipelines/run.sh setup --model 4b)"
  fi
  ensure_package || exit 2
fi

# The audit runs on ONE GPU: with the selected model when it fits there, otherwise with the 4B model of
# the same family (same pipeline code, data, template and loss masking) -- downloaded if it is missing.
audit_model="$MODEL"
if [[ "$DO_AUDIT" == 1 ]] && ! fits_one_gpu; then
  (( DRY_RUN )) || ensure_model 4b || die "the audit of $(basename "$MODEL") on this host needs the 4B model (or pass --no-audit)"
  audit_model="$(model_dir_for 4b)"
fi

# Evaluation feasibility is checked now, not after hours of training.
if [[ "$SUBSET" != none ]]; then
  eval_mode="$(STRATEGY= pick_strategy)"
  check_fit "$eval_mode" || die "evaluation of $(basename "$MODEL") is impossible with $NGPU GPU(s) here --" \
                                "use more GPUs or --eval none"
fi

# ---- train: plan ------------------------------------------------------------------------
source "$HERE/common/hparams.sh"
( TRAIN_FILE="$PKG/train.jsonl"; [[ -n "$STEPS" ]] && MAX_STEPS="$STEPS"; derive_hparams 2>/dev/null
  log "model $(basename "$MODEL") | $NGPU GPU(s): devices $CUDA_VISIBLE_DEVICES | global batch $GLOBAL_BS" \
      "(grad_accum $GRAD_ACC) | steps $TOTAL_STEPS | lr $LR | eval $SUBSET | data $(basename "$PKG")"
  if (( GLOBAL_BS != GLOBAL_BS_TARGET )); then
    log "note: global batch $GLOBAL_BS instead of $GLOBAL_BS_TARGET with $NGPU GPUs -- not exactly comparable with 1/2/4/8-GPU runs"
  fi )
declare -A PLAN RESULT
for fw in $FW; do
  PLAN[$fw]=run
  mode="$(STRATEGY="$STRATEGY_ARG" pick_strategy)"
  case "$fw" in
    nemo)    label="fsdp2 (always shards)"
             if ! fits_one_gpu && ! check_fit shard 2> /dev/null; then PLAN[$fw]="SKIP: does not fit $NGPU GPU(s)"; fi ;;
    unsloth) if fits_one_gpu; then label="replicate, bf16"; else label="replicate, 4-bit QLoRA (model does not fit one GPU; not comparable with bf16 runs)"; fi ;;
    *)       label="$mode"
             if ! reason="$(check_fit "$mode" 2>&1)"; then PLAN[$fw]="SKIP: $reason"; fi
             if [[ "$fw" == trl && "$mode" == shard && "$(model_tag)" == 32b && "${ALLOW_UNSUPPORTED:-0}" != 1 ]]; then
               PLAN[$fw]="SKIP: TRL + ZeRO-3 with 32B runs out of memory on 48 GB cards (known, unresolved; ALLOW_UNSUPPORTED=1 to try)"
             fi ;;
  esac
  note=""; if [[ "${PLAN[$fw]}" != run ]]; then note="${PLAN[$fw]}"; fi
  log "$(printf '  plan  %-8s %s  %s' "$fw" "$label" "$note")"
done
model_gib=$(( $(model_bytes "$MODEL") / 1073741824 )); host_gib=$(( $(host_mem_bytes) / 1073741824 ))
if (( model_gib * 12 / 10 > host_gib )); then
  log "warning: $host_gib GiB of host memory usable (available RAM / container limit), loading $(basename "$MODEL")" \
      "needs about $(( model_gib * 12 / 10 )) GiB -- the loader may be killed"
fi
if (( DRY_RUN )); then
  if [[ "$DO_AUDIT" == 1 ]]; then log "  audit on one GPU with $(basename "$audit_model")"; fi
  log "dry run: nothing was started"
  exit 0
fi

# ---- train: baseline --------------------------------------------------------------------
if [[ "$SUBSET" != none && "$DO_BASELINE" == 1 ]]; then
  baseline="$RUNS/baseline/$(basename "$MODEL")/eval/$SUBSET"
  stamp="$baseline/.for_$(pkg_id)"
  if [[ ! -f "$baseline/metrics.json" || ! -f "$stamp" ]]; then
    mkdir -p "$baseline"
    log "zero-shot baseline of $(basename "$MODEL") on $SUBSET (log: $baseline/eval.log)"
    if bash "$HERE/eval/eval.sh" baseline "$MODEL" "$SUBSET" > "$baseline/eval.log" 2>&1; then
      rm -f "$baseline"/.for_*; touch "$stamp"
      record baseline eval PASS "$(grep '\[score\]' "$baseline/eval.log" | tail -1 | sed 's/.*: n=/n=/')"
    else
      record baseline eval FAIL "see $baseline/eval.log"
    fi
  fi
fi

# ---- train: per framework ---------------------------------------------------------------
failures=0
for fw in $FW; do
  if [[ "${PLAN[$fw]}" != run ]]; then
    record "$fw" train SKIP "${PLAN[$fw]#SKIP: }"
    RESULT[$fw]="${PLAN[$fw]}"
    continue
  fi
  if ! ensure_env "$fw"; then RESULT[$fw]="FAIL (env)"; failures=$((failures + 1)); continue; fi

  if [[ "$DO_AUDIT" == 1 ]]; then
    key="$RUNS/audit_ok/${fw}__$(basename "$audit_model")__$(pkg_id)__code$(code_hash "$fw")__env$(env_id "$fw")"
    if [[ -f "$key" ]]; then
      record "$fw" audit PASS "cached ($(basename "$audit_model"))"
    else
      mkdir -p "$RUNS/audit_ok"
      audit_log="$RUNS/audit_${fw}_$(basename "$audit_model").log"
      log "[$fw] correctness audit on one GPU with $(basename "$audit_model") (log: $audit_log)"
      if bash "$HERE/common/audit.sh" "$fw" "$audit_model" > "$audit_log" 2>&1; then
        cp "$audit_log" "$key"
        record "$fw" audit PASS "$(grep 'step-1 loss' "$audit_log" | tail -1 | sed 's/^ *- //') ($(basename "$audit_model"))"
      else
        record "$fw" audit FAIL "$(grep -E '^FAIL' "$audit_log" | head -2 | tr '\n' ' ') -- $audit_log"
        RESULT[$fw]="FAIL (audit) $audit_log"; failures=$((failures + 1)); continue
      fi
    fi
  fi

  run_file="$(mktemp "$CACHE/tmp/run_dir.XXXXXX")"
  train_env=(RUN_DIR_FILE="$run_file" RUN_TAG=run)
  if [[ -n "$STEPS" ]]; then train_env+=(MAX_STEPS="$STEPS"); fi
  if [[ -n "$STRATEGY_ARG" ]]; then train_env+=(STRATEGY="$STRATEGY_ARG"); fi
  log "[$fw] training (progress: tail -f the train.log of the run dir printed below)"
  env "${train_env[@]}" bash "$HERE/$fw/train.sh" 2>&1 | sed 's/^/    /'
  if [[ "${PIPESTATUS[0]}" == 0 ]]; then train_ok=1; else train_ok=0; fi
  run="$(cat "$run_file" 2>/dev/null || true)"; rm -f "$run_file"
  if (( ! train_ok )); then
    record "$fw" train FAIL "${run:-no run directory was created}"
    RESULT[$fw]="FAIL (train) ${run:-}"; failures=$((failures + 1)); continue
  fi
  record "$fw" train PASS "$run"
  RESULT[$fw]="OK   $run"

  if [[ "$SUBSET" != none ]]; then
    log "[$fw] evaluating on $SUBSET (log: $run/eval_$SUBSET.log)"
    if bash "$HERE/eval/eval.sh" "$run" "$SUBSET" > "$run/eval_$SUBSET.log" 2>&1; then
      score="$(grep '\[score\]' "$run/eval_$SUBSET.log" | tail -1 | sed 's/.*: n=/n=/')"
      record "$fw" eval PASS "$score"
      RESULT[$fw]="OK   $run | $score"
    else
      record "$fw" eval FAIL "see $run/eval_$SUBSET.log"
      RESULT[$fw]="FAIL (eval) $run"; failures=$((failures + 1))
    fi
  fi
done

python3 "$HERE/eval/leaderboard.py" "$RUNS" > /dev/null || true
echo
log "=== $(basename "$MODEL"), $NGPU GPU(s) ==="
for fw in $FW; do printf '  %-8s %s\n' "$fw" "${RESULT[$fw]:-not run}"; done
echo "  leaderboard: $RUNS/leaderboard.md"
exit $(( failures > 0 ))
