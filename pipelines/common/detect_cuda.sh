#!/usr/bin/env bash
# torch_index_tag                        -> cu130 | cu126 for the host driver (no network)
# torch_index_url <torch_version> [python]
# Maps the host NVIDIA driver to a PyTorch wheel index and verifies the requested torch version
# exists there (pip index versions) before anything is installed.
#   driver >= 580 -> cu130 (torch 2.9 .. 2.14.x)
#   driver >= 560 -> cu126 (torch .. 2.14.x)   (cu128 stops at torch 2.11, cu129 at 2.13: not used)
#   older         -> unsupported for these torch versions

torch_index_tag() {      # driver -> cu130 | cu126 (no network access)
  local driver major
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  if [[ -z "$driver" ]]; then
    echo "nvidia-smi not available -- cannot pick a CUDA wheel" >&2
    return 1
  fi
  major="${driver%%.*}"
  if (( major >= 580 )); then
    echo cu130
  elif (( major >= 560 )); then
    echo cu126
  else
    echo "driver $driver is too old for the pinned torch versions (need >= 560)" >&2
    return 1
  fi
}

torch_index_url() {
  local version="$1" python="${2:-python3}" tag url
  tag="$(torch_index_tag)" || return 1
  url="https://download.pytorch.org/whl/$tag"
  if ! "$python" -m pip index versions torch --index-url "$url" 2>/dev/null \
      | sed -n 's/^Available versions: //p' | tr ',' '\n' | tr -d ' ' | sed 's/+.*//' | grep -qx "$version"; then
    echo "torch==$version not found on $url" >&2
    return 1
  fi
  echo "$url"
}
