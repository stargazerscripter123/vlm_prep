#!/usr/bin/env bash
# Create/complete the conda env for axolotl (same script on popos-6000b and the H200 pod).
# FORCE=1 rebuilds from scratch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../common/setup_lib.sh"
create_framework_env axolotl 2.14.0 "$HERE/requirements.txt" "axolotl transformers peft deepspeed"
