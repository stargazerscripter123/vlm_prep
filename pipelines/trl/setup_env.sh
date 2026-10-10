#!/usr/bin/env bash
# Create/complete the conda env for trl (same script on popos-6000b and the H200 pod).
# FORCE=1 rebuilds from scratch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../common/setup_lib.sh"
create_framework_env trl 2.14.1 "$HERE/requirements.txt" "trl peft transformers accelerate deepspeed datasets qwen_vl_utils"
