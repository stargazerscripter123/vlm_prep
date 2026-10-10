#!/usr/bin/env bash
# Idempotent "make sure it exists" helpers used by pipelines/run.sh (source after env.sh).
#   ensure_conda              a conda installation (Miniforge is installed into $ROOT/.conda if none exists)
#   ensure_env <fw> [check]   conda env of a framework (check = also import-test an existing env)
#   ensure_model <4b|32b|dir> base model: aliases are downloaded + sha256-verified when missing or
#                             incomplete; an explicit directory must already exist and be complete
#   ensure_package            data package under data/package/ (built from COCO when missing)
# Each returns non-zero with a pointer to its log when it cannot be satisfied.

log() { echo "[$(date +%H:%M:%S)] $*"; }

_trl_python() { echo "$CONDA_BASE/envs/$(env_name trl)/bin/python"; }

free_gib() { df -Pk "$1" | awk 'NR==2 {print int($4 / 1048576)}'; }

ensure_conda() {
  if [[ -n "$CONDA_BASE" && -x "$CONDA_BASE/bin/conda" ]]; then
    return 0
  fi
  local target="$ROOT/.conda" installer="$CACHE/tmp/miniforge.sh"
  log "no conda found -- installing Miniforge into $target"
  if ! curl -fsSL -o "$installer" \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"; then
    echo "could not download the Miniforge installer (no network?) -- install conda and set CONDA_BASE" >&2
    return 1
  fi
  bash "$installer" -b -p "$target" > "$CACHE/miniforge_install.log" 2>&1 || {
    echo "Miniforge install FAILED -- see $CACHE/miniforge_install.log" >&2
    return 1
  }
  rm -f "$installer"
  export CONDA_BASE="$target"
}

ensure_env() {
  local fw="$1" mode="${2:-}" prefix want have logfile torch_version
  prefix="$CONDA_BASE/envs/$(env_name "$fw")"
  logfile="$CACHE/setup_$fw.log"
  torch_version="$(sed -n 's/^create_framework_env [a-z]* \([0-9.]*\) .*/\1/p' "$PIPELINES/$fw/setup_env.sh")"
  want="$(sha256sum < "$PIPELINES/$fw/requirements.txt" | cut -c1-16)-torch$torch_version"
  have="$(cat "$prefix/.vlm_setup_stamp" 2>/dev/null || true)"
  if [[ "$want" == "$have" && "$mode" != check ]]; then
    return 0
  fi
  log "[$fw] conda env $(env_name "$fw") (log: $logfile)"
  if ! bash "$PIPELINES/$fw/setup_env.sh" > "$logfile" 2>&1; then
    echo "[$fw] environment setup FAILED -- last lines of $logfile:" >&2
    tail -n 8 "$logfile" | cut -c1-300 >&2
    return 1
  fi
  awk '/^python /{out=""} /^python |^  [a-z_]+ [0-9?]/{out=out $0 "\n"} END{printf "%s", out}' "$logfile"
}

model_complete() {
  # config / tokenizer / processor files present and every weight shard has exactly the size its
  # safetensors header implies (catches truncated copies, e.g. an interrupted rsync)
  python3 - "$1" <<'EOF'
import json, struct, sys
from pathlib import Path

model = Path(sys.argv[1])
small = ["config.json", "tokenizer.json", "preprocessor_config.json"]
index = model / "model.safetensors.index.json"
if not all((model / f).is_file() and (model / f).stat().st_size > 0 for f in small):
    sys.exit(1)
shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
for name in shards:
    path = model / name
    if not path.is_file():
        sys.exit(1)
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
    if path.stat().st_size != 8 + length + end:
        print(f"{name}: size {path.stat().st_size} != expected {8 + length + end} (truncated?)", file=sys.stderr)
        sys.exit(1)
EOF
}

ensure_model() {
  local spec="$1" dir repo logfile need
  dir="$(model_dir_for "$spec")"
  if model_complete "$dir"; then
    return 0
  fi
  if ! model_is_alias "$spec"; then
    echo "model directory missing or incomplete: $dir" >&2
    echo "(only the aliases 4b / 32b are downloaded automatically; pass an existing model directory)" >&2
    return 1
  fi
  repo="$(model_repo_for "$spec")"
  need=$([[ "$spec" =~ ^32 ]] && echo 75 || echo 15)
  if (( $(free_gib "$ROOT") < need )); then
    echo "not enough disk for $repo: need ~${need} GiB free under $ROOT, have $(free_gib "$ROOT") GiB" >&2
    return 1
  fi
  logfile="$CACHE/download_$(basename "$dir").log"
  ensure_env trl || return 1
  log "downloading $repo -> $dir (log: $logfile)"
  if ! HF_HUB_OFFLINE=0 "$(_trl_python)" "$ROOT/scripts/05_download_model.py" --repo "$repo" --out "$dir" \
      > "$logfile" 2>&1; then
    echo "model download FAILED -- last lines of $logfile:" >&2
    tail -n 8 "$logfile" | cut -c1-300 >&2
    return 1
  fi
  tail -n 2 "$logfile"
  model_complete "$dir"
}

ensure_package() {
  local logfile="$CACHE/build_package.log" py processor_model="" candidate name building final
  if [[ -n "${PKG:-}" && -f "$PKG/train.jsonl" && -f "$PKG/records/token_lengths.json" ]]; then
    return 0
  fi
  ensure_env trl || return 1
  py="$(_trl_python)"
  for candidate in "$MODEL" "$ROOT"/models/*/; do          # 04 needs a local Qwen3-VL processor
    if [[ -f "$candidate/preprocessor_config.json" ]]; then processor_model="${candidate%/}"; break; fi
  done
  if [[ -z "$processor_model" ]]; then
    echo "building the data package needs one local model (for its processor) -- ensure_model first" >&2
    return 1
  fi
  if [[ -n "${PKG:-}" && -f "$PKG/train.jsonl" ]]; then
    # a package that was copied here but never verified on this host: verify it in place
    log "verifying the data package $PKG (log: $logfile)"
    if ! "$py" "$ROOT/scripts/04_verify_package.py" --package "$PKG" --model "$processor_model" > "$logfile" 2>&1; then
      echo "data package verification FAILED -- last lines of $logfile:" >&2
      grep -v "it/s\]" "$logfile" | tail -n 8 | cut -c1-300 >&2
      return 1
    fi
    return 0
  fi
  name="vlm_sft_person_$(date +%y%m%d)"
  building="$ROOT/data/package/.building/$name"
  final="$ROOT/data/package/$name"
  if [[ -e "$final" ]]; then
    echo "$final exists but is not a package (no train.jsonl) -- remove it and re-run" >&2
    return 1
  fi
  log "building the data package: COCO person subset download + conversion + verification (log: $logfile)"
  # Every step must succeed (&& chain: `set -e` is ignored inside an `if` condition); the package only
  # appears under its final name after verification passed and no image was dropped.
  if ! ( cd "$ROOT" \
         && "$py" scripts/01_download_coco.py \
         && "$py" scripts/03_build_package.py --out "$building" --force \
         && "$py" scripts/04_verify_package.py --package "$building" --model "$processor_model" \
         && grep -q '"person": {}' "$building/meta.json" \
         && mv "$building" "$final" ) > "$logfile" 2>&1; then
    echo "data package build FAILED -- last lines of $logfile:" >&2
    grep -v "it/s\]\|img/s\]" "$logfile" | tail -n 8 | cut -c1-300 >&2
    echo "(a package is only accepted when every image downloaded and none was dropped)" >&2
    return 1
  fi
  export PKG="$final"
  export FORMATS="$ROOT/data/formats/$(basename "$PKG")"
  grep -E "VERIFY OK|person_(train|val|test):" "$logfile" | tail -n 4 || true
  return 0
}

code_hash() {   # code_hash <fw>: changes whenever the framework's pipeline or the shared code changes
  find "$PIPELINES/common" "$PIPELINES/$1" -type f \
      \( -name '*.sh' -o -name '*.py' -o -name '*.yaml' -o -name '*.tmpl' -o -name '*.json' -o -name '*.txt' \) \
      -not -path '*/__pycache__/*' -print0 | sort -z | xargs -0 cat | sha256sum | cut -c1-12
}

env_id() {      # env_id <fw>: identity of the installed env (changes when any package version changes)
  sha256sum < "$CONDA_BASE/envs/$(env_name "$1")/pip-freeze.txt" 2>/dev/null | cut -c1-8
}
