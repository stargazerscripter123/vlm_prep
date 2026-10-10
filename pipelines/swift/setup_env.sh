#!/usr/bin/env bash
# Create/complete the conda env for swift (same script on popos-6000b and the H200 pod).
# FORCE=1 rebuilds from scratch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../common/setup_lib.sh"
create_framework_env swift 2.13.0 "$HERE/requirements.txt" "swift deepspeed transformers peft qwen_vl_utils"
