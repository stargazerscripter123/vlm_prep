#!/usr/bin/env bash
# Train + evaluate every framework for every model and GPU count, one combination after another.
#
#   bash pipelines/sweep.sh                          # 4b and 32b  x  2, 4, 6, 8 GPUs  x  5 frameworks,
#                                                    # each: full epoch + evaluation on the full test set
#   bash pipelines/sweep.sh --steps 5 --eval n50     # the same loop as a quick smoke test
#   bash pipelines/sweep.sh --dry-run                # only print the plan of every combination
#   MODELS="32b" GPU_COUNTS="4 8" bash pipelines/sweep.sh --fw swift,trl
#
# MODELS (default "4b 32b") and GPU_COUNTS (default "2 4 6 8") select the loop. Every argument is passed
# on to `run.sh train` (--fw --steps --eval --lr --save-steps --gpu-ids ...; see `run.sh --help`).
# A combination that fails or cannot run here does not stop the sweep.
# Output of each combination: runs/sweep_<time>/<model>_<N>g.log; every run also lands in the leaderboard.
# Exit code: 0 when every combination succeeded, 1 otherwise, 2 when setup failed.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="${MODELS:-4b 32b}"
GPU_COUNTS="${GPU_COUNTS:-2 4 6 8}"
RUNS="${RUNS:-${ROOT:-$(cd "$HERE/.." && pwd)}/runs}"
OUT="$RUNS/sweep_$(date +%y%m%d_%H%M%S)"
case "$OUT" in *Dropbox*) echo "refusing to write inside Dropbox: $OUT" >&2; exit 2 ;; esac

fw=all; dry=0; args=("$@")
for (( i = 0; i < ${#args[@]}; i++ )); do
  case "${args[i]}" in
    --fw) fw="${args[i + 1]:-all}" ;;
    --dry-run) dry=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
  esac
done

echo "sweep pid $$ -- to stop everything (the running training included):" \
     "kill -TERM -- -$(ps -o pgid= -p $$ | tr -d ' ')"

child=""
trap 'echo "sweep interrupted -- stopping the current run" >&2
      if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null; fi
      wait; exit 130' INT TERM

# Once, before hours of training: envs, models, data package, host checks (everything missing is prepared).
if (( ! dry )); then
  bash "$HERE/run.sh" setup --fw "$fw" --model "${MODELS// /,}" \
    || { echo "setup failed -- sweep not started" >&2; exit 2; }
fi

mkdir -p "$OUT"
results=(); failed=0
for model in $MODELS; do
  for n in $GPU_COUNTS; do
    log="$OUT/$(basename "$model")_${n}g.log"
    echo
    echo "########## $(date '+%F %T')  model $model | $n GPU(s) | log: $log"
    bash "$HERE/run.sh" train --fw all --model "$model" --gpus "$n" "$@" > "$log" 2>&1 &
    child=$!
    tail -n +1 -f --pid="$child" "$log" 2> /dev/null &     # live output; ends with the run
    wait "$child"; rc=$?
    wait
    child=""
    case "$rc" in
      0) verdict="OK" ;;
      1) verdict="FAILED"; failed=1 ;;
      *) verdict="NOT RUN (rc=$rc)"; failed=1 ;;
    esac
    results+=("$(printf '%-24s %-5s %-16s %s' "$(basename "$model")" "$n" "$verdict" "$log")")
  done
done

echo
echo "########## sweep summary"
printf '%-24s %-5s %-16s %s\n' model gpus result log
printf '%s\n' "${results[@]}"
if (( ! dry )); then
  echo
  echo "per framework:"
  for log in "$OUT"/*.log; do sed -n '/^\[[0-9:]*\] === /,$p' "$log" | grep -v '^  leaderboard:'; done
  echo
  echo "speed / memory / accuracy of every run: bash pipelines/run.sh status   ($RUNS/leaderboard.md)"
fi
exit "$failed"
