#!/usr/bin/env bash
# create_framework_env <fw> <torch_version> <requirements.txt> <required_modules> [extra torch-index pkgs...]
# Creates/completes conda env vlm<fw>_NV_py312 (python 3.12): torch from the driver-matched PyTorch
# index (or TORCH_INDEX_URL; pip's default index as a fallback), nvcc for deepspeed when needed, then the framework requirements with torch pinned by a
# constraints file (no dependency can swap torch). An env is only considered done when a stamp
# (hash of requirements + torch version) exists and the required modules import; an incomplete env
# is completed in place, FORCE=1 rebuilds from scratch.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/detect_cuda.sh"

self_check() {
  local py="$1" required="$2"
  REQUIRED="$required" "$py" - <<'EOF'
import importlib, os, sys
from importlib.metadata import PackageNotFoundError, version
import torch


def installed_version(name, module):
    if getattr(module, "__version__", None):
        return module.__version__
    try:
        return version(name.replace("_", "-"))
    except PackageNotFoundError:
        return "?"


print(f"python {sys.version.split()[0]} | torch {torch.__version__} | cuda {torch.version.cuda} "
      f"| gpu {torch.cuda.is_available()} x{torch.cuda.device_count()}")
missing = []
for name in os.environ["REQUIRED"].split():
    try:
        module = importlib.import_module(name)
        print(f"  {name} {installed_version(name, module)}")
    except Exception as exc:
        missing.append(f"{name} ({type(exc).__name__}: {exc})")
assert torch.cuda.is_available(), "CUDA not available"
assert not missing, f"required modules not importable: {missing}"
EOF
}

create_framework_env() {
  local fw="$1" torch_version="$2" requirements="$3" required="$4"
  shift 4
  local extra_torch_pkgs=("$@")
  local name prefix py url constraints stamp_value
  name="$(env_name "$fw")"
  prefix="$CONDA_BASE/envs/$name"
  py="$prefix/bin/python"
  stamp_value="$(cat "$requirements" | sha256sum | cut -c1-16)-torch$torch_version"

  require_conda
  if [[ "${FORCE:-0}" == 1 && -d "$prefix" ]]; then
    "$CONDA_BASE/bin/conda" env remove -y -p "$prefix"
  fi
  if [[ -f "$prefix/.vlm_setup_stamp" && "$(cat "$prefix/.vlm_setup_stamp")" == "$stamp_value" ]]; then
    echo "[$name] up to date -- self-check"
    self_check "$py" "$required" && return 0
    echo "[$name] self-check failed -- completing the install"
  fi
  if [[ ! -x "$py" ]]; then
    echo "[$name] creating (python 3.12) at $prefix"
    # explicit prefix (independent of the user's envs_dirs) and conda-forge only (no channel ToS prompt)
    "$CONDA_BASE/bin/conda" create -y -q -p "$prefix" -c conda-forge --override-channels python=3.12 pip
  fi
  "$py" -m pip install -q -U pip setuptools wheel packaging ninja

  local local_wheels="$CACHE/prefetch_$torch_version"
  if [[ -d "$local_wheels" ]] && "$py" -m pip install --no-index --find-links "$local_wheels" \
      "torch==$torch_version" torchvision "${extra_torch_pkgs[@]}"; then
    echo "[$name] torch installed from local wheels ($local_wheels)"
  else
    url="$(torch_index_url "$torch_version" "$py")" || exit 1
    echo "[$name] torch $torch_version from $url ${extra_torch_pkgs[*]}"
    if ! "$py" -m pip install "torch==$torch_version" torchvision "${extra_torch_pkgs[@]}" --index-url "$url"; then
      torch_install_help "$py" "$url" "$torch_version"
      # Same torch version from pip's default index (PyPI or the mirror configured on this host). Its
      # CUDA build is the PyPI default, not necessarily the driver-matched one: the self-check below
      # still requires torch.cuda.is_available().
      echo "[$name] falling back to pip's default index for torch==$torch_version"
      if ! "$py" -m pip install "torch==$torch_version" torchvision "${extra_torch_pkgs[@]}"; then
        echo "[$name] torch==$torch_version could not be installed from $url nor from pip's default index" >&2
        exit 1
      fi
    fi
  fi
  echo "[$name] installed: torch $("$py" -c 'import torch; print(torch.__version__, "| CUDA build", torch.version.cuda)')"

  if grep -qi deepspeed "$requirements"; then
    # deepspeed's setup.py needs nvcc (CUDA_HOME) even when no op is prebuilt: use the CUDA
    # version the installed torch was built for (13.0, 12.6, ...), installed inside this env.
    local cuda_version
    cuda_version="$("$py" -c 'import torch; print(torch.version.cuda)')"
    if [[ ! -x "$prefix/bin/nvcc" ]]; then
      echo "[$name] cuda-nvcc $cuda_version (for deepspeed)"
      "$CONDA_BASE/bin/conda" install -y -q -p "$prefix" -c nvidia -c conda-forge --override-channels \
        "cuda-nvcc=$cuda_version"
    fi
    export CUDA_HOME="$prefix" DS_BUILD_OPS=0
  fi

  constraints="$CACHE/constraints_${name}.txt"
  "$py" - > "$constraints" <<'EOF'
from importlib.metadata import version
for pkg in ("torch", "torchvision"):
    print(f"{pkg}=={version(pkg).split('+')[0]}")
EOF
  echo "[$name] requirements: $requirements (constraints: $(tr '\n' ' ' < "$constraints"))"
  # --no-build-isolation: deepspeed (sdist) imports torch in setup.py
  "$py" -m pip install --no-build-isolation -r "$requirements" -c "$constraints"

  "$py" -m pip freeze > "$prefix/pip-freeze.txt"
  self_check "$py" "$required"
  echo "$stamp_value" > "$prefix/.vlm_setup_stamp"
  echo "[$name] OK"
}
