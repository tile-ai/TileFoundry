#!/usr/bin/env bash
# End-to-end `check` of a whole prefill (selection SOURCE:ROOT, entry `prefill`)
# against the transformers reference for one prompt length.
#
# Bounds, check --help's rule: the longest serial f32->bf16 chain is ~1350 landings
# (about 21 per layer x 64 layers + final norm + head), rtol = 2^-9 * sqrt(1350) =
# 0.0717. Over 64 chained layers single-element differences are amplified (a 1-ulp
# change in one layer moves later elements by more than a random walk predicts), so
# end to end the predicates are aggregate: rel_l2 <= 0.0717 and cosine >= 1 - 0.0717^2/2
# = 0.9974, plus nan_inf. Layer by layer, teacher-forced, tools/check_layers.py applies
# the same rule with n = 21 landings per layer (rel_l2 <= 0.009, cosine >= 0.99996).
#
# usage: tools/check_hir.sh T [SOURCE ROOT TAG [hir]]
#   with a fifth argument "hir" the reference is the evaluator running model.py (no --expected)
set -euo pipefail
cd "$(dirname "$0")/.."
T=${1:?length}
SOURCE=${2:-model.py}
ROOT=${3:-Qwen3_8_27B}
TAG=${4:-hir_vs_hf}
R=reference/T$T
REF=(--expected $R/expected.pt)
[ "${5:-}" = "hir" ] && REF=()
mkdir -p reports/check
P=(--fn rel_l2 --max 0.0717 --fn cosine --min 0.9974 --fn nan_inf)
"${PYTHON:-python}" tools/tfdeep.py check "$SOURCE:$ROOT" \
  --inputs files:$R/token_ids.pt,$R/cos.pt,$R/sin.pt \
  --weights ckpt:ckpt "${REF[@]}" --dim seq_len=$T \
  --out 'output[0]' "${P[@]}" --out 'output[1]' "${P[@]}" --out 'output[2]' "${P[@]}" \
  --out 'output[3]' "${P[@]}" --out 'output[4]' "${P[@]}" | tee reports/check/${TAG}_T$T.txt
