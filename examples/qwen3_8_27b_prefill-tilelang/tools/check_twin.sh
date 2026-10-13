#!/usr/bin/env bash
# `check` the runtime twin (runtime_model.py, tilelang kernels) against model.py,
# function by function, on the published model's own activations.
#
#   CUDA_VISIBLE_DEVICES=0 tools/check_twin.sh 1024 "0 3"
#
# No --expected: the reference is the evaluator running model.py's authored
# HIR on the same inputs and weights. The twin differs from it only where its
# kernels round or sum in another order, so each bound counts those places.
# U = 2^-7: two f32 values a hair apart can land one bf16 step apart, and one
# step is 2^-7 of the value at the bottom of a binade.
#
#   equal       embed: a row gather, no arithmetic.
#   GEMM        in_proj, qkv gate / v: the f32 sum runs in another order, the
#               result lands in bf16 once -> rtol U; atol 2^-16 max|out| for
#               the f32 reassociation under cancellation.
#   residual    down, out_proj, o_proj: bf16(h + bf16(acc)) lands twice, and
#               the first step is relative to |acc|, not to |out| ->
#               rtol U, atol U max|acc| (acc = expected - h from refdata).
#   gate_up     g, u, silu(g) and the product each land: rtol 4U, atol 2^-16 max|out|.
#   q / k       proj, norm, two rope products, their sum: the sum can cancel,
#               so a step of a term is not a step of the result -> rtol U,
#               atol U max|out|.
#   norms       rmsnorm, gated_norm: one landing after an f32 reduction in
#               another order -> rtol U, atol 2^-16 max|out|.
#   prep        q, k three landings (conv, silu, l2norm), v two; g f32 from
#               bf16 inputs; beta one: as tools/check_hir.sh.
#   attention   both sides land the softmax weights in bf16, the twin before
#               the 1/l division (flash), model.py after: up to a step of each
#               weight, so 2^-7 max|v| apart -> atol U max|v|, rtol U.
#   delta_rule  the twin keeps w, u, the masked q k^T and v_new in bf16 between
#               and inside its two kernels (tensor-core operands, as FLA's
#               production kernels do); model.py keeps f32. Four landings of
#               terms that sum into an output: atol 4 U/2 max|out| = 2^-6 max|out|,
#               rtol U; rel_l2 2^-9 sqrt(5).
#   final       the norm lands in bf16, the GEMV sums 5120 terms in another order
#               and lands once: rtol U, atol 2^-11 max|logit| (one step of a
#               normed input element moves a logit by |w| U |x|).
#   rel_l2      2^-9 sqrt(landings) on every output, the help's aggregate rule.
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-1024}
LAYERS=${2:-"0 3"}
R=${QWEN38_REFDATA:?set QWEN38_REFDATA (see paths.py)}/n$N
OUT=${OUTDIR:-reports/check_twin}/n$N
mkdir -p "$OUT"
TF=${TILEFOUNDRY:-tilefoundry}
PY=${PYTHON:-python}
B=0.0078125         # U = 2^-7
B4=0.03125          # 4 U
RL1=0.001953125     # 2^-9 sqrt(1)
RL2=0.00276         # 2^-9 sqrt(2)
RL3=0.00338         # 2^-9 sqrt(3)
RL5=0.00437         # 2^-9 sqrt(5)

stat() { # python expression over torch tensors -> one number
  $PY -c "import torch; t=lambda p: torch.load(p); f=lambda x: (x[0] if isinstance(x,(list,tuple)) else x).float(); print(float($1))"
}

run() { # selector dir predicates...
  local sel=$1 dir=$2; shift 2
  local ins
  ins=$(ls "$dir"/in*.pt | sort -V | paste -sd, -)
  echo "=== twin $sel  (seq_len=$N)"
  $TF check "runtime_model.py:Qwen3_8_27BTwin.$sel" --inputs "files:$ins" --weights ckpt:$QWEN38_REFDATA/ckpt \
    --dim "seq_len=$N" "$@" | tee "$OUT/$sel.txt" | grep -E "^  output|    |^PASS|^FAIL" || true
}

run embed "$R/embed" --out output --fn equal
FMAX=$(stat "f(t('$R/final/expected.pt')).abs().max()*2**-11")
run final "$R/final" --out output --fn allclose --rtol $B --atol "$FMAX" --fn rel_l2 --max $RL2
for L in $LAYERS; do
  D=$R/v1/layer$L
  for nrm in input_norm post_norm; do
    A=$(stat "f(t('$D/$nrm/expected.pt')).abs().max()*2**-16")
    run "layer$L.$nrm" "$D/$nrm" --out output --fn allclose --rtol $B --atol "$A" --fn rel_l2 --max $RL1
  done
  A=$(stat "f(t('$D/gate_up/expected.pt')).abs().max()*2**-16")
  run "layer$L.mlp.gate_up" "$D/gate_up" --out output --fn allclose --rtol $B4 --atol "$A" --fn rel_l2 --max $RL3
  A=$(stat "(f(t('$D/down/expected.pt'))-f(t('$D/down/in1.pt'))).abs().max()*2**-7")
  run "layer$L.mlp.down" "$D/down" --out output --fn allclose --rtol $B --atol "$A" --fn rel_l2 --max $RL2
  if [ -d "$D/in_proj" ]; then
    args=()
    for i in 0 1 2 3; do
      A=$(stat "t('$D/in_proj/expected.pt')[$i].float().abs().max()*2**-16")
      args+=(--out "output[$i]" --fn allclose --rtol $B --atol "$A" --fn rel_l2 --max $RL1)
    done
    run "layer$L.mixer.in_proj" "$D/in_proj" "${args[@]}"
    run "layer$L.mixer.prep" "$D/prep" \
      --out 'output[0]' --fn allclose --rtol 7.8125e-3 --atol 1e-5 --fn rel_l2 --max $RL3 \
      --out 'output[1]' --fn allclose --rtol 7.8125e-3 --atol 1e-5 --fn rel_l2 --max $RL3 \
      --out 'output[2]' --fn allclose --rtol 7.8125e-3 --atol 1e-5 --fn rel_l2 --max $RL2 \
      --out 'output[3]' --fn allclose --rtol 1e-5 --atol 1e-5 \
      --out 'output[4]' --fn allclose --rtol $B --atol 0
    A=$(stat "f(t('$D/delta_rule/expected.pt')).abs().max()*2**-6")
    run "layer$L.mixer.delta_rule" "$D/delta_rule" --out output --fn allclose --rtol $B --atol "$A" --fn rel_l2 --max $RL5
    A=$(stat "f(t('$D/gated_norm/expected.pt')).abs().max()*2**-16")
    run "layer$L.mixer.gated_norm" "$D/gated_norm" --out output --fn allclose --rtol $B --atol "$A" --fn rel_l2 --max $RL2
    A=$(stat "(f(t('$D/out_proj/expected.pt'))-f(t('$D/out_proj/in1.pt'))).abs().max()*2**-7")
    run "layer$L.mixer.out_proj" "$D/out_proj" --out output --fn allclose --rtol $B --atol "$A" --fn rel_l2 --max $RL2
  else
    QA=$(stat "t('$D/qkv/expected.pt')[0].float().abs().max()*2**-7")
    GA=$(stat "t('$D/qkv/expected.pt')[1].float().abs().max()*2**-16")
    KA=$(stat "t('$D/qkv/expected.pt')[2].float().abs().max()*2**-7")
    VA=$(stat "t('$D/qkv/expected.pt')[3].float().abs().max()*2**-16")
    run "layer$L.mixer.qkv" "$D/qkv" \
      --out 'output[0]' --fn allclose --rtol $B --atol "$QA" --fn rel_l2 --max $RL5 \
      --out 'output[1]' --fn allclose --rtol $B --atol "$GA" --fn rel_l2 --max $RL1 \
      --out 'output[2]' --fn allclose --rtol $B --atol "$KA" --fn rel_l2 --max $RL5 \
      --out 'output[3]' --fn allclose --rtol $B --atol "$VA" --fn rel_l2 --max $RL1
    VMAX=$(stat "f(t('$D/attention/in2.pt')).abs().max()*2**-7")
    run "layer$L.mixer.attention" "$D/attention" --out output --fn allclose --rtol $B --atol "$VMAX" --fn rel_l2 --max $RL3
    A=$(stat "(f(t('$D/o_proj/expected.pt'))-f(t('$D/o_proj/in1.pt'))).abs().max()*2**-7")
    run "layer$L.mixer.o_proj" "$D/o_proj" --out output --fn allclose --rtol $B --atol "$A" --fn rel_l2 --max $RL2
  fi
done
echo "summary n=$N:"; grep -l "^PASS" "$OUT"/*.txt | wc -l | xargs echo "  PASS:"; grep -l "^FAIL" "$OUT"/*.txt | xargs -r -n1 basename | sed 's/^/  FAIL: /'
