#!/usr/bin/env bash
# analyze every prefill function of a model source at each prompt length.
#   tools/analyze_all.sh model.py v0 "1024 4096 8192"
set -uo pipefail
cd "$(dirname "$0")/.."
SRC=${1:-model.py}; TAG=${2:-v0}; LENS=${3:-"1024 4096 8192"}
OUT=reports/analyze/$TAG; mkdir -p "$OUT"
SELS=${SELS:-"Qwen3_8_27B.embed Qwen3_8_27B.final LinearAttentionLayer.input_norm LinearAttentionLayer.post_norm MLP.gate_up MLP.down GatedDeltaNet.in_proj GatedDeltaNet.prep GatedDeltaNet.delta_rule GatedDeltaNet.gated_norm GatedDeltaNet.out_proj Attention.qkv Attention.attention Attention.o_proj"}
for N in $LENS; do
  for s in $SELS; do
    f="$OUT/$s-$N.txt"
    start=$(date +%s)
    if timeout 400 tilefoundry analyze "$SRC:$s" "$f" --compute-cost --memory --roofline --performance --dim seq_len=$N 2>"$f.err"; then
      echo "$s n=$N ok $(( $(date +%s) - start ))s"; rm -f "$f.err"
    else
      echo "$s n=$N FAILED: $(tail -1 "$f.err")"
    fi
  done
done
