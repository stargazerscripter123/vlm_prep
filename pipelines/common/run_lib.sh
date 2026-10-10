#!/usr/bin/env bash
# Run bookkeeping shared by every pipelines/<fw>/train.sh (source after env.sh / hparams.sh / strategy.sh).
#   preflight_run <strategy>  fail early: no data package / no model / model cannot fit the GPUs
#   start_run <fw> [tag]   create $RUN (runs/<fw>/<model>-<strategy>-<N>g[-tag]-<ts>), cd into it,
#                          record metadata + resolved hparams, start the 1 Hz GPU/host-memory monitor
#   run_logged <cmd...>    run (optional RUN_TIMEOUT); output timestamped into train.log; signals stop the group
#   finish_run <rc>        require $RUN/adapter, stop the monitor, write timing.json + run_summary.json;
#                          sets FINAL_RC (exit with it)

preflight_run() {
  # preflight_run <shard|replicate|none>: data package + model present, model fits the chosen GPUs.
  require_package
  require_model
  if [[ "$1" != none ]]; then check_fit "$1" || exit 1; fi
}

start_run() {
  local fw="$1" tag="${2:-}"
  export RUN_FW="$fw"
  if [[ -z "${RUN:-}" ]]; then
    RUN="$RUNS/$fw/$(model_tag)-${STRATEGY}-${NGPU}g${tag:+-$tag}-$(date +%y%m%d_%H%M%S)"
  fi
  export RUN
  assert_not_dropbox "$RUN"
  mkdir -p "$RUN"
  cd "$RUN"
  if [[ -n "${RUN_DIR_FILE:-}" ]]; then echo "$RUN" > "$RUN_DIR_FILE"; fi   # lets a caller find this run
  export VLM_RUN="$RUN"
  # instrumentation (pipelines/common/hooks/sitecustomize.py): steps.jsonl, param census, vision-grad probe
  export VLM_HOOKS=1
  export PYTHONPATH="$PIPELINES/common/hooks${PYTHONPATH:+:$PYTHONPATH}"
  export PYTHONUNBUFFERED=1
  {
    echo "framework=$fw"
    echo "model=$MODEL"
    echo "package=$PKG"
    echo "strategy=$STRATEGY"
    echo "ngpu=$NGPU"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "host=$(hostname)"
    echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    nvidia-smi -L
  } > run_info.txt
  "$PY" -m pip freeze > pip_freeze.txt 2>/dev/null || true
  env | grep -E '^(SEED|LORA_[A-Z_]+|LR|LR_SCHEDULER|WEIGHT_DECAY|ADAM_[A-Z0-9_]+|MAX_GRAD_NORM|GLOBAL_BS|GLOBAL_BS_TARGET|MICRO_BS|GRAD_ACC|N_TRAIN|TOTAL_STEPS|WARMUP_STEPS|MAX_STEPS|MIN_PIXELS|MAX_PIXELS|IMAGE_M[A-Z_]+|MAX_LENGTH|ATTN_IMPL|DATALOADER_WORKERS|SAVE_STEPS|STRATEGY|NGPU|CUDA_VISIBLE_DEVICES|TRAIN_SPLIT|QUANT|RUN_TIMEOUT)=' \
    | sort > hparams.env
  grep -q '^LORA_R=' hparams.env || { echo "internal error: LORA_R missing from hparams.env" >&2; exit 1; }
  (
    set +e +o pipefail          # a transient nvidia-smi hiccup must not end the monitor
    while true; do
      t="$(date +%s.%N)"
      nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits \
        | awk -v t="$t" -F', *' '{print t "," $1 "," $2 "," $3}' >> gpu.csv
      awk -v t="$t" '/^MemTotal/{tot=$2} /^MemAvailable/{av=$2} END{print t "," tot "," av}' /proc/meminfo >> host_mem.csv
      sleep 1
    done
  ) &
  export MONITOR_PID=$!
  trap '_stop_training; kill "$MONITOR_PID" 2>/dev/null || true' EXIT   # never leave anything running
  RUN_START="$(date +%s.%N)"
  export RUN_START
  echo "run dir: $RUN"
}

free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])'; }

_stop_training() {
  # The launcher runs under `timeout`, which leads its own process group: signal the whole group.
  local pid
  pid="$(cat "$RUN/.train_pid" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  fi
}

run_logged() {
  # run_logged <cmd...>: wall-clock limit RUN_TIMEOUT (0 = none), every output line timestamped into
  # train.log. Ctrl-C / TERM / HUP stop the whole training process group (no orphan keeps the GPUs).
  local rcfile="$RUN/.train_rc" pidfile="$RUN/.train_pid" pipe_pid rc
  rm -f "$rcfile" "$pidfile"
  set +e
  {
    timeout --signal=TERM --kill-after=120 "${RUN_TIMEOUT:-0}" "$@" 2>&1 &
    echo $! > "$pidfile"
    wait $!
    echo $? > "$rcfile"
  } | python3 -u "$PIPELINES/common/ts_lines.py" > "$RUN/train.log" &
  pipe_pid=$!
  trap '_stop_training' INT TERM HUP
  while kill -0 "$pipe_pid" 2>/dev/null; do wait "$pipe_pid"; done
  trap - INT TERM HUP
  rc="$(cat "$rcfile" 2>/dev/null || echo 143)"
  rm -f "$pidfile"
  set -e
  if [[ "$rc" == 124 || "$rc" == 137 ]]; then
    echo "TIMED OUT after RUN_TIMEOUT=${RUN_TIMEOUT:-0} (raise it with --timeout, or 0 for no limit)" >&2
  fi
  return "$rc"
}

finish_run() {
  # finish_run <rc>: a run only counts as successful when the HF PEFT adapter exists in $RUN/adapter.
  local rc="$1"
  if (( rc == 0 )) && [[ ! -s "$RUN/adapter/adapter_model.safetensors" || ! -s "$RUN/adapter/adapter_config.json" ]]; then
    echo "training exited 0 but no adapter was written to $RUN/adapter" >&2
    rc=3
  fi
  if (( rc == 0 )) && ! python3 "$PIPELINES/common/compare_audit.py" --run "$RUN" --post-train \
      > "$RUN/post_train_check.log" 2>&1; then
    echo "post-train check FAILED (wrong trainable parameters or backward through the frozen ViT):" >&2
    cat "$RUN/post_train_check.log" >&2
    rc=4
  fi
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  printf '{"rc": %s, "start": %s, "end": %s}\n' "$rc" "$RUN_START" "$(date +%s.%N)" > "$RUN/timing.json"
  python3 "$PIPELINES/common/summarize_run.py" "$RUN" || echo "warning: could not summarize $RUN" >&2
  if (( rc != 0 )); then
    echo "FAILED (rc=$rc) -- last lines of $RUN/train.log:" >&2
    grep -v "it/s\]" "$RUN/train.log" 2>/dev/null | tail -8 | cut -c 16-300 >&2 || true
  fi
  echo "run dir: $RUN (rc=$rc)"
  FINAL_RC="$rc"
}
