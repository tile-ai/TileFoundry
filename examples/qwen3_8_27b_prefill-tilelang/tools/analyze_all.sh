#!/usr/bin/env bash
# analyze every prefill function of a model source at each prompt length.
#   tools/analyze_all.sh model.py v0 "1024 4096 8192"
set -uo pipefail
cd "$(dirname "$0")/.."
SRC=${1:-model.py}; TAG=${2:-v0}; LENS=${3:-"1024 4096 8192"}
OUT=reports/analyze/$TAG; mkdir -p "$OUT"
ROOT=$(python - "$SRC" <<'PY'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
for node in tree.body:
    if isinstance(node, ast.ClassDef):
        for d in node.decorator_list:
            if isinstance(d, ast.Call) and any(k.arg == "target" for k in d.keywords):
                print(node.name)
PY
)
SELS=${SELS:-"embed final layer0.input_norm layer0.residual_add layer0.post_norm layer0.mlp.mlp layer0.mixer.in_proj layer0.mixer.conv layer0.mixer.gates layer0.mixer.delta_rule layer0.mixer.out_proj layer3.mixer.qkv layer3.mixer.attention layer3.mixer.o_proj"}
for N in $LENS; do
  for s in $SELS; do
    f="$OUT/$s-$N.txt"
    start=$(date +%s)
    if timeout 400 tilefoundry analyze "$SRC:$ROOT.$s" "$f" --compute-cost --memory --roofline --performance --dim seq_len=$N 2>"$f.err"; then
      echo "$s n=$N ok $(( $(date +%s) - start ))s"; rm -f "$f.err"
    else
      echo "$s n=$N FAILED: $(tail -1 "$f.err")"
    fi
  done
done
