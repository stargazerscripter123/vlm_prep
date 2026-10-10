#!/usr/bin/env bash
# torch_index_tag                        -> cu130 | cu126 for the host driver (no network)
# torch_index_url <torch_version> [python]  (TORCH_INDEX_URL=<url> overrides the detection)
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
  # Prints the wheel index for torch <version>. TORCH_INDEX_URL=<url> skips the detection (mirror,
  # or a host that cannot reach download.pytorch.org). The version listing is only a pre-check: when
  # the listing itself cannot be obtained (network, proxy, pip.conf) this warns and carries on -- the
  # install that follows shows the real error; only "listed, but this version is not there" is fatal.
  local version="$1" python="${2:-python3}" tag url listing offered
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    echo "$TORCH_INDEX_URL"
    return 0
  fi
  tag="$(torch_index_tag)" || return 1
  url="https://download.pytorch.org/whl/$tag"
  listing="$("$python" -m pip index versions torch --index-url "$url" 2>&1)" || true
  offered="$(sed -n 's/^Available versions: //p' <<< "$listing")"
  if [[ -z "$offered" ]]; then
    echo "warning: could not list the torch versions on $url -- pip said:" >&2
    { grep -v '^$' <<< "$listing" || true; } | tail -n 3 | cut -c1-300 | sed 's/^/    /' >&2
    echo "         continuing with the install (set TORCH_INDEX_URL=<url> to use another index or a mirror)" >&2
  elif ! tr ',' '\n' <<< "$offered" | tr -d ' ' | sed 's/+.*//' | grep -qx "$version"; then
    echo "torch==$version is not offered on $url for this python / platform; offered: ${offered:0:160}" >&2
    return 1
  fi
  echo "$url"
}

torch_install_help() {
  # torch_install_help <python> <index url> <torch version>: why did the torch install fail?
  local python="$1" url="$2" version="$3" facts http
  facts="$("$python" -c 'import platform, sys; print(sys.version.split()[0], platform.machine(), *platform.libc_ver())' 2>/dev/null)"
  http="$(curl -s -m 20 -o /dev/null -w '%{http_code}' "$url/torch/" 2>/dev/null || true)"
  {
    echo "---- torch==$version could not be installed from $url"
    echo "     python / arch / libc here: ${facts:-unknown}   (the wheels need x86_64 or aarch64 and glibc >= 2.28)"
    echo "     HTTP status of $url/torch/ from this host: ${http:-no answer}   (200 = reachable)"
    echo "     not reachable -> set HTTPS_PROXY, or TORCH_INDEX_URL=<a PyTorch wheel index this host can reach>"
    echo "     reachable but 'No matching distribution' -> glibc / architecture / pip.conf (try: PIP_CONFIG_FILE=/dev/null)"
  } >&2
}
