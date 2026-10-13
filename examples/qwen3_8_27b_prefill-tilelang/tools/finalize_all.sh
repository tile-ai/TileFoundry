#!/usr/bin/env bash
# Lower every scheduled HIR kernel function to TIR (tir/), one file per
# (function, length, kernel), in parallel. Weight-only pack funcs do not
# depend on the length: they are lowered once, from the 1024 module.
#
#   tools/finalize_all.sh                 # all
#   tools/finalize_all.sh Down QKV        # modules whose name starts with one of these
#   JOBS=16 tools/finalize_all.sh
#
# The list of (file, module, function) comes from tools/scheduled_index.py,
# which also BENCH.md section 7's correspondence table is built from.
set -euo pipefail
cd "$(dirname "$0")/.."
TF=${TILEFOUNDRY:-tilefoundry}
PY=${PYTHON:-python}
JOBS=${JOBS:-12}
mkdir -p tir reports/finalize
$PY tools/scheduled_index.py --jobs "$@" | \
  xargs -P "$JOBS" -L 1 bash -c 'src=$0; out=$1; log=reports/finalize/$(basename "$out" .py).log;
    if timeout 14400 '"$PY"' tools/finalize_deep.py "$src" "$out" > "$log" 2>&1; then echo "ok   $src -> $out"; else echo "FAIL $src ($(tail -1 "$log"))"; fi'
ls tir | wc -l
