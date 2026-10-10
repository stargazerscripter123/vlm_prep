#!/usr/bin/env bash
# Sourced by every pipeline script. Resolves paths, pins every cache inside $ROOT, guards Dropbox.
# Overridable env vars: ROOT PKG MODEL (4b|32b|path) RUNS NGPU (1..N) CUDA_VISIBLE_DEVICES CONDA_BASE

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PIPELINES="$(cd "$_COMMON_DIR/.." && pwd)"
export ROOT="${ROOT:-$(cd "$PIPELINES/.." && pwd)}"

realpath_m() { python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"; }

assert_not_dropbox() {
  local resolved
  resolved="$(realpath_m "$1")"
  if [[ "$resolved" == *Dropbox* ]]; then
    echo "refusing to write inside Dropbox: $resolved" >&2
    exit 1
  fi
}

# --- data / model / runs ---------------------------------------------------------------
# PKG: explicit, else the most recently BUILT package (meta.json mtime; names do not sort by date).
if [[ -z "${PKG:-}" ]]; then
  PKG="$(ls -t "$ROOT"/data/package/vlm_sft_*/meta.json 2>/dev/null | head -1 || true)"
  PKG="${PKG%/meta.json}"
fi
if [[ -n "$PKG" ]]; then PKG="$(realpath_m "$PKG")"; fi
export PKG
pkg_id() {   # name + content hash: cache key for references / audits / baselines of this exact data
  echo "$(basename "$PKG")-$(cat "$PKG/train.jsonl" "$PKG/person_test.jsonl" 2>/dev/null | sha256sum | cut -c1-10)"
}

# MODEL: a size alias (4b | 32b -> $ROOT/models/Qwen3-VL-<N>B-Instruct) or an existing model directory
# (relative paths are tried against the current directory, then $ROOT).
model_is_alias() { [[ "$1" =~ ^[0-9]+[bB]$ ]]; }
model_dir_for() {
  local spec="$1"
  if [[ "$spec" =~ ^([0-9]+)[bB]$ ]]; then
    echo "$ROOT/models/Qwen3-VL-${BASH_REMATCH[1]}B-Instruct"
  elif [[ "$spec" != /* && ! -e "$spec" && -e "$ROOT/$spec" ]]; then
    realpath_m "$ROOT/$spec"
  else
    realpath_m "$spec"
  fi
}
model_repo_for() { echo "Qwen/$(basename "$(model_dir_for "$1")")"; }   # HF repo id (aliases only)
MODEL="$(model_dir_for "${MODEL:-32b}")"
export MODEL
RUNS="$(realpath_m "${RUNS:-$ROOT/runs}")"
export RUNS
export FORMATS="$ROOT/data/formats/$(basename "${PKG:-none}")"

require_package() {
  if [[ -z "${PKG:-}" || ! -f "$PKG/train.jsonl" ]]; then
    echo "no data package under $ROOT/data/package -- run: bash pipelines/run.sh setup" >&2
    exit 1
  fi
}
require_model() {
  if [[ ! -f "$MODEL/config.json" ]]; then
    echo "model not found: $MODEL -- run: bash pipelines/run.sh setup --model <4b|32b>" >&2
    exit 1
  fi
}

# --- caches: everything inside $ROOT/.cache (never ~/.cache, never Dropbox) ---------------
export CACHE="$ROOT/.cache"
export HF_HOME="$CACHE/huggingface"
export HF_MODULES_CACHE="$CACHE/huggingface/modules"
export TORCH_HOME="$CACHE/torch"
export TRITON_CACHE_DIR="$CACHE/triton"
export TMPDIR="$CACHE/tmp"
export XDG_CACHE_HOME="$CACHE/xdg"
export TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export CUDA_CACHE_PATH="$CACHE/nv"
export PIP_CACHE_DIR="$CACHE/pip"
export MODELSCOPE_CACHE="$CACHE/modelscope"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"       # models are local; set 0 only to download
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE=disabled
# no telemetry from any framework (Axolotl reports errors by default; HF hub; generic opt-out)
export AXOLOTL_DO_NOT_TRACK=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DIR="$CACHE/wandb"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" \
  "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$PIP_CACHE_DIR" \
  "$MODELSCOPE_CACHE" "$WANDB_DIR"
assert_not_dropbox "$CACHE"

# --- GPUs -------------------------------------------------------------------------------
# NGPU = number of GPUs to use (1..all). CUDA_VISIBLE_DEVICES always ends up listing EXACTLY NGPU
# devices (ms-swift silently splits the model when it sees >= 2x devices per process):
#   NGPU only                 -> devices 0..NGPU-1
#   CUDA_VISIBLE_DEVICES only -> NGPU = number of listed devices
#   both                      -> the first NGPU of the listed devices
GPUS_PRESENT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c '^[0-9]' || true)"
GPUS_PRESENT="${GPUS_PRESENT:-0}"
export GPUS_PRESENT
if (( GPUS_PRESENT > 0 )); then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _visible <<< "$CUDA_VISIBLE_DEVICES"
  else
    _visible=($(seq 0 $((GPUS_PRESENT - 1))))
  fi
  for _dev in "${_visible[@]}"; do          # entries are indices (or GPU UUIDs, which nvidia-smi also accepts)
    if [[ "$_dev" =~ ^[0-9]+$ ]] && (( _dev >= GPUS_PRESENT )); then
      echo "error: GPU $_dev does not exist here ($GPUS_PRESENT GPU(s): 0..$((GPUS_PRESENT - 1)))" >&2
      exit 2
    fi
  done
  if [[ -n "$(printf '%s\n' "${_visible[@]}" | sort | uniq -d)" ]]; then
    echo "error: a device is listed twice: $CUDA_VISIBLE_DEVICES" >&2
    exit 2
  fi
  NGPU="${NGPU:-${#_visible[@]}}"
  if ! [[ "$NGPU" =~ ^[0-9]+$ ]] || (( NGPU < 1 || NGPU > ${#_visible[@]} )); then
    echo "error: $NGPU GPU(s) requested but ${#_visible[@]} available here (devices ${_visible[*]})" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${_visible[*]:0:NGPU}")"
  export CUDA_VISIBLE_DEVICES
else
  NGPU="${NGPU:-0}"          # no NVIDIA GPU (e.g. the dev Mac): only non-GPU helpers work
fi
export NGPU

# --- conda ------------------------------------------------------------------------------
if [[ -z "${CONDA_BASE:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
  else
    for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda "$ROOT/.conda"; do
      if [[ -x "$candidate/bin/conda" ]]; then CONDA_BASE="$candidate"; break; fi
    done
  fi
fi
export CONDA_BASE="${CONDA_BASE:-}"

require_conda() {
  if [[ -z "$CONDA_BASE" || ! -x "$CONDA_BASE/bin/conda" ]]; then
    echo "conda not found (PATH, ~/miniconda3, ~/anaconda3, ~/miniforge3, /opt/conda, $ROOT/.conda)." >&2
    echo "Run: bash pipelines/run.sh setup   (installs Miniforge into $ROOT/.conda), or set CONDA_BASE." >&2
    exit 1
  fi
}

env_name() { echo "vlm${1}_NV_py312"; }   # swift|trl|nemo|axolotl|unsloth

use_env() {
  # Put a framework env first on PATH (console scripts + python), like `conda activate` without a shell hook.
  local name prefix
  name="$(env_name "$1")"
  prefix="$CONDA_BASE/envs/$name"
  if [[ ! -x "$prefix/bin/python" ]]; then
    echo "conda env $name not found under $CONDA_BASE/envs -- run pipelines/$1/setup_env.sh" >&2
    exit 1
  fi
  export CONDA_PREFIX="$prefix"
  export PATH="$prefix/bin:$PATH"
  export PY="$prefix/bin/python"
  if [[ -x "$prefix/bin/nvcc" ]]; then
    export CUDA_HOME="$prefix"     # deepspeed looks for nvcc here (installed by setup_env.sh)
  fi
}
