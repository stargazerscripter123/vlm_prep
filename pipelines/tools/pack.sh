#!/usr/bin/env bash
# pack.sh [out.tar] -- code + data package(s) for an offline pod (no raw data, caches, runs, models, secrets).
# Models are not packed: `bash pipelines/run.sh setup` downloads and verifies them on the pod (or copy
# models/<name>/ yourself -- setup then only verifies them).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/vlm_prep_code_data_$(date +%y%m%d).tar}"
export COPYFILE_DISABLE=1          # macOS tar: no AppleDouble ._ files
cd "$ROOT/.."
tar --exclude='.DS_Store' --exclude='__pycache__' --exclude='.env' --exclude='vlm_prep/.cache' \
    --exclude='vlm_prep/data/raw' --exclude='vlm_prep/data/preview' --exclude='vlm_prep/data/formats' \
    --exclude='vlm_prep/runs' --exclude='vlm_prep/models' --exclude='*.tar' \
    -cf "$OUT" "$(basename "$ROOT")"
echo "packed -> $OUT ($(du -h "$OUT" | cut -f1))"
echo "on the pod: tar xf $(basename "$OUT") && cd vlm_prep && bash pipelines/run.sh setup"
