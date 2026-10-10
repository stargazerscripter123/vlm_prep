#!/usr/bin/env bash
# Host preflight (called by `run.sh setup`). Exit 1 on hard failures.
#   bash pipelines/common/preflight.sh          # WARM=1 also pre-reads model weights into page cache
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
source "$HERE/detect_cuda.sh"
source "$HERE/strategy.sh"
fail=0

echo "== host: $(hostname) | cpus $(nproc) | $(free -g | awk '/^Mem/{print "RAM " $2 " GiB, available " $7 " GiB"}')"
host_gib=$(( $(host_mem_bytes) / 1024 / 1024 / 1024 ))
echo "   memory usable by this job (min of available RAM and the container limit): $host_gib GiB"
shm_kb="$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2 {print $4}')"
shm_mib=$(( ${shm_kb:-0} / 1024 ))
echo "   /dev/shm free: $shm_mib MiB$( (( shm_mib < 4096 )) && echo ' (small: DataLoader workers will be disabled; mount a larger /dev/shm to use them)')"
echo "== GPUs"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || { echo "no nvidia-smi"; exit 1; }
echo "== torch wheel index for this driver"
if tag="$(torch_index_tag)"; then
  echo "   $tag (pinned torch: $(sed -n 's/^create_framework_env \([a-z]*\) \([0-9.]*\) .*/\1 \2/p' "$PIPELINES"/*/setup_env.sh | tr '\n' ',' | sed 's/,$//; s/,/, /g'))"
else
  fail=1
fi
echo "== disk: $(df -h "$ROOT" | awk 'NR==2{print $4 " free on " $6}')"
echo "== models"
for model in "$ROOT"/models/*/; do
  model="${model%/}"
  [[ -f "$model/config.json" ]] || continue
  name="$(basename "$model")"
  size_gib=$(( $(model_bytes "$model") / 1024 / 1024 / 1024 ))
  if [[ -f "$model/SHA256SUMS" ]]; then
    # full checksums take minutes for 32B: redone only when a weight file changed (size or mtime)
    stamp="$CACHE/sha256_ok/$name.$( (cd "$model" && stat -c '%n %s %Y' SHA256SUMS ./*.safetensors) | sha256sum | cut -c1-16)"
    if [[ -f "$stamp" ]]; then
      echo "   $name ($size_gib GiB): sha256 OK (verified $(date -r "$stamp" +%F))"
    elif (cd "$model" && sha256sum -c --quiet SHA256SUMS > /dev/null 2>&1); then
      mkdir -p "$CACHE/sha256_ok"; rm -f "$CACHE/sha256_ok/$name".*; touch "$stamp"
      echo "   $name ($size_gib GiB): sha256 OK"
    else
      echo "   $name ($size_gib GiB): sha256 FAILED"; fail=1
    fi
  else
    echo "   $name ($size_gib GiB): no SHA256SUMS (structure checked by run.sh only)"
  fi
  if (( size_gib * 12 / 10 > host_gib )); then
    echo "   WARNING: $name needs about $(( size_gib * 12 / 10 )) GiB of host memory to load, $host_gib GiB usable"
  fi
  if [[ "${WARM:-0}" == 1 ]]; then cat "$model"/*.safetensors > /dev/null; echo "   warmed page cache"; fi
done
echo "== NCCL collectives and direct GPU-to-GPU copies ($NGPU GPU(s))"
torchrun_bin=""
for fw in trl swift unsloth axolotl nemo; do
  if [[ -x "$CONDA_BASE/envs/$(env_name "$fw")/bin/torchrun" ]]; then
    torchrun_bin="$CONDA_BASE/envs/$(env_name "$fw")/bin/torchrun"
    break
  fi
done
if [[ -z "$torchrun_bin" ]]; then
  echo "   skipped: no framework env yet (run again after the envs exist)"
else
  nccl_out="$("$torchrun_bin" --standalone --nproc_per_node "$NGPU" "$HERE/nccl_test.py" 2>&1)" || {
    echo "   NCCL test FAILED"
    fail=1
  }
  grep -E "NCCL collectives|WRONG" <<< "$nccl_out" | sed 's/^/   /' || { echo "$nccl_out" | tail -n 5; fail=1; }
  if grep -q "CORRUPTED" <<< "$nccl_out"; then
    echo "   WARNING: direct GPU-to-GPU copies are corrupted on this host. NCCL-based training (DDP / ZeRO-3 /"
    echo "            FSDP) is unaffected; single-process model splitting is handled by common/p2p_safe.py."
  fi
fi
exit $fail
