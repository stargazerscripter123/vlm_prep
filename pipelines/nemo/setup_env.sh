#!/usr/bin/env bash
# Create/complete the conda env for nemo (same script on popos-6000b and the H200 pod).
# FORCE=1 rebuilds from scratch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../common/setup_lib.sh"
create_framework_env nemo 2.10.0 "$HERE/requirements.txt" "nemo_automodel transformers qwen_vl_utils"
