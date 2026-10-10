#!/usr/bin/env bash
# Decide STRATEGY=shard|replicate for the current MODEL on the current GPUs (override with STRATEGY=...).
#   replicate: every GPU holds the full BF16 model (DDP)       -> H200 32B, 6000b 4B
#   shard    : parameters sharded over GPUs (ZeRO-3 / FSDP2)  -> 6000b 32B
# Rule: replicate when model_bytes * 1.3 + 8 GiB fits in one GPU.

model_bytes() {
  python3 - "$1" <<'EOF'
import json, os, sys
from pathlib import Path
model = Path(sys.argv[1])
index = model / "model.safetensors.index.json"
if index.exists():
    print(json.loads(index.read_text())["metadata"]["total_size"])
else:
    print(sum(p.stat().st_size for p in model.glob("*.safetensors")))
EOF
}

gpu_mem_bytes() {      # smallest memory among the GPUs this run will use
  local mib
  mib="$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=memory.total --format=csv,noheader,nounits \
         | tr -d ' ' | sort -n | head -1)"
  echo $(( mib * 1024 * 1024 ))
}

host_mem_bytes() {     # memory this job can use: min(MemAvailable, cgroup limit of the container)
  local avail limit
  avail=$(( $(awk '/^MemAvailable/{print $2}' /proc/meminfo) * 1024 ))
  limit="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)"
  if [[ "$limit" =~ ^[0-9]+$ ]] && (( limit < avail )); then echo "$limit"; else echo "$avail"; fi
}

fits_one_gpu() {       # true when the BF16 model plus working memory fits on a single GPU
  local need have
  need=$(( $(model_bytes "$MODEL") * 13 / 10 + 8 * 1024 * 1024 * 1024 ))
  have="$(gpu_mem_bytes)"
  (( need <= have ))
}

pick_strategy() {
  if [[ -n "${STRATEGY:-}" ]]; then
    case "$STRATEGY" in
      shard|replicate) echo "$STRATEGY" ;;
      *) echo "unknown STRATEGY '$STRATEGY' (expected shard or replicate)" >&2; return 1 ;;
    esac
    return
  fi
  if fits_one_gpu; then echo replicate; else echo shard; fi
}

check_fit() {
  # check_fit <strategy> : fail early (clear message) when the model cannot fit the chosen GPUs.
  #   replicate: full model on every GPU.   shard: model split over NGPU GPUs + ~12 GiB working memory each.
  local strategy="$1" size have gib=$(( 1024 * 1024 * 1024 ))
  size="$(model_bytes "$MODEL")"
  have="$(gpu_mem_bytes)"
  case "$strategy" in
    replicate)
      if ! fits_one_gpu; then
        echo "$(basename "$MODEL") ($(( size / gib )) GiB) does not fit on one $(( have / gib )) GiB GPU;" \
             "replicate (DDP) is impossible here -- do not force replicate, or use a smaller model" >&2
        return 1
      fi ;;
    shard)
      if (( NGPU < 2 )); then
        if fits_one_gpu; then
          echo "STRATEGY=shard needs at least 2 GPUs ($(basename "$MODEL") fits one GPU: use replicate)" >&2
        else
          echo "$(basename "$MODEL") ($(( size / gib )) GiB) does not fit on one $(( have / gib )) GiB GPU" \
               "and 1 GPU cannot shard it -- use --gpus >= 2 or --model 4b" >&2
        fi
        return 1
      fi
      if (( size / NGPU + 12 * gib > have )); then
        echo "$(basename "$MODEL") sharded over $NGPU x $(( have / gib )) GiB GPUs needs about" \
             "$(( (size / NGPU + 12 * gib) / gib )) GiB per GPU -- use more GPUs" >&2
        return 1
      fi
      if (( (size / NGPU + 12 * gib) * 10 > have * 9 )); then
        echo "warning: sharding $(basename "$MODEL") over $NGPU x $(( have / gib )) GiB GPUs leaves <10%" \
             "headroom (estimate measured with ms-swift on 4 GPUs) -- untested, may run out of memory" >&2
      fi ;;
    *)
      echo "unknown strategy '$strategy' (expected shard or replicate)" >&2
      return 1 ;;
  esac
}

model_tag() {
  # Qwen3-VL-32B-Instruct -> 32b
  basename "$MODEL" | sed -E 's/.*-([0-9.]+[Bb]).*/\1/' | tr 'A-Z' 'a-z'
}
