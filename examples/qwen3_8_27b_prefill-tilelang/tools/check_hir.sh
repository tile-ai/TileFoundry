#!/usr/bin/env bash
# `check` every function of the authored HIR (model.py) against the published
# model's own activations, as tools/hf_reference.py captured them.
#
#   CUDA_VISIBLE_DEVICES=1 tools/check_hir.sh 1024 "0 3"
#
# Predicates, and why each bound is what it is:
#   equal      the HIR states the published step's arithmetic in the same
#              order, so the bits must match.
#   allclose   rtol 2^-7: two implementations that land the same value in
#              bf16 by different f32 paths can disagree by one ulp, which is
#              up to 2^-7 of the value at the bottom of a binade.
#              atol: conv / delta_rule 1e-5, f32 reassociation near zero;
#              attention 2^-8 * max|v|, because both sides round the softmax
#              weights to bf16 (2^-9 each) and an output is a weighted sum of
#              v, so its error scales with |v| and not with the (cancelling)
#              output.
#   rel_l2     2^-9 * sqrt(landings): the help's rule for the aggregate --
#              prep's q, k land 3 times (conv, silu, l2norm), v twice;
#              attention twice (output, gate product); delta_rule once.
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-1024}
LAYERS=${2:-"0 3"}
R=${QWEN38_REFDATA:?set QWEN38_REFDATA (see paths.py)}/n$N
OUT=reports/check_hir/v1/n$N
mkdir -p "$OUT"
PY=${PYTHON:-python}

run() { # Module.function weight_scope dir predicates...
  local sel=$1 scope=$2 dir=$3; shift 3
  local label=${scope:+$scope.}$sel
  local ins
  ins=$(ls "$dir"/in*.pt | sort -V | paste -sd, -)
  echo "=== $sel  (seq_len=$N)"
  "$PY" ../check_scoped.py "model.py:$sel" --weight-scope "$scope" --inputs "files:$ins" --weights ckpt:$QWEN38_REFDATA/ckpt \
    --expected "$dir/expected.pt" --dim "seq_len=$N" "$@" | tee "$OUT/$label.txt" | grep -E "^  output|    |^PASS|^FAIL"
}
eq() { local a=(); for o in "$@"; do a+=(--out "$o" --fn equal); done; echo "${a[@]}"; }

run Qwen3_8_27B.embed "" "$R/embed" $(eq output)
run Qwen3_8_27B.final "" "$R/final" $(eq output)
for L in $LAYERS; do
  D=$R/v1/layer$L
  if [ -d "$D/in_proj" ]; then
    LAYER=LinearAttentionLayer
    MIXER=GatedDeltaNet
  else
    LAYER=FullAttentionLayer
    MIXER=Attention
  fi
  run "$LAYER.input_norm" "layers.$L" "$D/input_norm" $(eq output)
  run "$LAYER.post_norm" "layers.$L" "$D/post_norm" $(eq output)
  run MLP.gate_up "layers.$L.mlp" "$D/gate_up" $(eq output)
  run MLP.down "layers.$L.mlp" "$D/down" $(eq output)
  if [ -d "$D/in_proj" ]; then
    run "$MIXER.in_proj" "layers.$L.mixer" "$D/in_proj" $(eq 'output[0]' 'output[1]' 'output[2]' 'output[3]')
    run "$MIXER.prep" "layers.$L.mixer" "$D/prep" \
      --out 'output[0]' --fn allclose --rtol 7.8125e-3 --atol 1e-5 --fn rel_l2 --max 3.39e-3 \
      --out 'output[1]' --fn allclose --rtol 7.8125e-3 --atol 1e-5 --fn rel_l2 --max 3.39e-3 \
      --out 'output[2]' --fn allclose --rtol 7.8125e-3 --atol 1e-5 --fn rel_l2 --max 2.77e-3 \
      $(eq 'output[3]' 'output[4]')
    run "$MIXER.delta_rule" "layers.$L.mixer" "$D/delta_rule" --out output --fn allclose --rtol 7.8125e-3 --atol 1e-5 --fn rel_l2 --max 1.96e-3
    run "$MIXER.gated_norm" "layers.$L.mixer" "$D/gated_norm" $(eq output)
    run "$MIXER.out_proj" "layers.$L.mixer" "$D/out_proj" $(eq output)
  else
    run "$MIXER.qkv" "layers.$L.mixer" "$D/qkv" $(eq 'output[0]' 'output[1]' 'output[2]' 'output[3]')
    VMAX=$(python -c "import torch;print(float(torch.load('$D/attention/in2.pt').float().abs().max())*2**-8)")
    run "$MIXER.attention" "layers.$L.mixer" "$D/attention" --out output --fn allclose --rtol 7.8125e-3 --atol "$VMAX" --fn rel_l2 --max 2.77e-3
    run "$MIXER.o_proj" "layers.$L.mixer" "$D/o_proj" $(eq output)
  fi
done
