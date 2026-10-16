#!/usr/bin/env bash
# The step-2 evidence: `tilefoundry analyze` at every prompt length, per selection.
# The whole root is refused by analyze's inlining budget (20361 > 10000 nodes), so
# the prefill is analysed as its parts: one GDN layer (layer0; all 48 are the same
# program on different weights), one attention layer (layer3; 16 the same), and the
# root's own embed / final_norm / lm_head.
# usage: tools/analyze_all.sh VERSION [SOURCE]
set -euo pipefail
cd "$(dirname "$0")/.."
V=${1:?version tag}
SRC=${2:-model.py}
for T in 128 256 512 1024; do
  D=reports/analyze/$V/T$T
  mkdir -p $D
  for PART in layer0:Qwen3_8DeltaNetLayer.layer layer3:Qwen3_8AttentionLayer.layer embed:Qwen3_8_27B.embed final_norm:Qwen3_8_27B.final_norm lm_head:Qwen3_8_27B.lm_head; do
    NAME=${PART%%:*}
    SEL=${PART#*:}
    "${PYTHON:-python}" tools/tfdeep.py analyze $SRC:$SEL $D/$NAME.txt \
      --compute-cost --memory --roofline --performance --dim seq_len=$T &
    "${PYTHON:-python}" tools/tfdeep.py analyze $SRC:$SEL $D/$NAME.json \
      --compute-cost --memory --roofline --performance --dim seq_len=$T --json &
  done
done
wait
