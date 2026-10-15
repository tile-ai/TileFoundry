#!/usr/bin/env bash
# Re-run all evidence after a kernel change, on GPUs 0 and 1, each part after its card is
# free of foreign compute processes (tools/wait_quiet.sh; other users share these cards).
#   GPU 0: per-layer twin vs HIR at 128/256, end-to-end twin vs transformers at 128/256
#   GPU 1: per-layer at 512/1024, end-to-end at 512/1024, first token; then (after GPU 0 is
#          done) the alternating bench and the per-kernel profiles.
# usage: tools/final_pipeline.sh DONE_DIR
set -u
cd "$(dirname "$0")/.."
D=${1:?done dir}; rm -f "$D"/fp0.done "$D"/fp1.done
mkdir -p reports/check/archive/pre_final && mv reports/check/twin_vs_hir_layers reports/check/archive/pre_final/twin_vs_hir_layers_$(date +%H%M%S) 2>/dev/null
(
  tools/wait_quiet.sh 0 86400
  CUDA_VISIBLE_DEVICES=0 "${PYTHON:-python}" tools/check_layers.py runtime/twin.py Qwen3_8_27BTwin 128 256 --vs-hir --tag twin_vs_hir --jobs 6 > reports/check/twin_vs_hir_layers_128_256.log 2>&1
  for T in 128 256; do CUDA_VISIBLE_DEVICES=0 tools/check_hir.sh $T runtime/twin.py Qwen3_8_27BTwin twin_vs_hf > reports/check/twin_vs_hf_T$T.log 2>&1; done
  touch "$D"/fp0.done
) &
(
  tools/wait_quiet.sh 1 86400
  CUDA_VISIBLE_DEVICES=1 "${PYTHON:-python}" tools/check_layers.py runtime/twin.py Qwen3_8_27BTwin 512 1024 --vs-hir --tag twin_vs_hir --jobs 6 > reports/check/twin_vs_hir_layers_512_1024.log 2>&1
  for T in 512 1024; do CUDA_VISIBLE_DEVICES=1 tools/check_hir.sh $T runtime/twin.py Qwen3_8_27BTwin twin_vs_hf > reports/check/twin_vs_hf_T$T.log 2>&1; done
  CUDA_VISIBLE_DEVICES=1 "${PYTHON:-python}" tools/first_token.py > reports/first_token.log 2>&1
  while [ ! -f "$D"/fp0.done ]; do sleep 30; done
  tools/wait_quiet.sh 1 86400
  "${PYTHON:-python}" bench/alternate.py --gpu 1 --rounds 6 --iters 10 --warmup 3 > bench/final_run.log 2>&1
  for T in 128 256 512 1024; do
    tools/wait_quiet.sh 1 86400
    CUDA_VISIBLE_DEVICES=1 "${PYTHON:-python}" runtime/profile_forward.py $T 2>&1 | grep -v -i 'warn\|tilelang\|USDT' > reports/profile/final_T$T.txt
  done
  touch "$D"/fp1.done
) &
wait
