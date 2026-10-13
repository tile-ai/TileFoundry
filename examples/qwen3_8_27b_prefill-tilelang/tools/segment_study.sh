#!/usr/bin/env bash
# Segmentation study: the whole prefill (tools/bench.py protocol, 2 + 7) at
# lengths between and around 1024 / 4096 / 8192, with the GEMM decomposition
# forced to each of the candidate configurations:
#   group 8,  split tail on   ("short" segment: what 1024 runs)
#   group 8,  split tail off
#   group 16, split tail on    (16 only where it divides the M-tiles)
#   group 16, split tail off   ("long" segment: what 4096 runs)
#   rule                       (the shipped per-length rule)
#   CUDA_VISIBLE_DEVICES=1 tools/segment_study.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
LENS=${LENS:-1024,2048,3072,4096,8192}
OUT=${OUT:-reports/segment}
for cfg in "8 on" "8 off" "16 on" "16 off" "0 rule"; do
  set -- $cfg
  out=$OUT/g$1_$2.json
  echo "== group $1 split $2 -> $out"
  $PY tools/bench.py --lens "$LENS" --gemm-group "$1" --gemm-split "$2" --label "group$1-split$2" --out "$out" 2>&1 | grep -E "^n=|Error"
done
