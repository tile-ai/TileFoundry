#!/usr/bin/env bash
# Simulate every run under a directory: graph.json -> Chakra traces -> ASTRA-sim -> sim.chrome.json.
#
#   SIM_ROOT=<install dir> examples/qwen3_training/sim/run.sh runs
#
# A run whose sim.chrome.json is newer than its graph.json is skipped. A run that
# fails, deadlocks or takes over 600 s is reported with the last 20 lines of its
# log; the others go on, and the exit code is 1.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${SIM_ROOT:-/nonexistent}/src/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware
PY=${SIM_ROOT:-/nonexistent}/venv/bin/python
VISUALIZER=${SIM_ROOT:-/nonexistent}/venv/bin/chakra_timeline_visualizer
if [[ -z "${SIM_ROOT:-}" || ! -x "$BIN" || ! -x "$PY" || ! -x "$VISUALIZER" ]]; then
    echo "set SIM_ROOT to the directory README.md installs into" >&2
    exit 2
fi
if [[ $# -ne 1 ]]; then
    echo "usage: run.sh <runs directory>" >&2
    exit 2
fi

status=0
while IFS= read -r graph; do
    run=$(dirname "$graph")
    if [[ -f "$run/sim.chrome.json" && "$run/sim.chrome.json" -nt "$graph" ]]; then
        continue
    fi
    rm -rf "${run:?}/trace" "${run:?}/log" "${run:?}/sim.chrome.json"
    started=$(date +%s)
    if ! workload=$("$PY" "$HERE/to_chakra.py" "$run" 2>"$run/to_chakra.log"); then
        echo "FAILED $run: to_chakra"
        tail -n 20 "$run/to_chakra.log"
        status=1
        continue
    fi
    ranks=$(find "$run/trace" -name 'trace.*.et' | wc -l)
    timeout 600 "$BIN" \
        --workload-configuration="$workload" \
        --comm-group-configuration="$run/trace/comm_group.json" \
        --system-configuration="$HERE/system.json" \
        --remote-memory-configuration="$HERE/remote_memory.json" \
        --network-configuration="$HERE/net_switch8.yml" \
        --logging-folder="$run/log" >"$run/sim.log" 2>&1
    code=$?
    finished=$(grep -c "finished," "$run/sim.log")
    if [[ $code -ne 0 || $finished -ne $ranks ]]; then
        echo "FAILED $run: exit $code, $finished of $ranks ranks finished"
        tail -n 20 "$run/sim.log"
        status=1
        continue
    fi
    # spdlog rotates log.log into log.1.log .. log.9.log, oldest last: read the oldest first,
    # and keep the issue and callback lines without the logger's prefix.
    for index in 9 8 7 6 5 4 3 2 1 ""; do
        file="$run/log/log${index:+.$index}.log"
        if [[ -f "$file" ]]; then cat "$file"; fi
    done | sed -n 's/^.*\] \(\(issue\|callback\),.*\)$/\1/p' >"$run/events.log"
    if [[ -f "$run/log/log.9.log" ]]; then
        echo "WARNING $run: the simulator's log rotated 9 times; the oldest events may be lost"
    fi
    # The tick is a nanosecond and chrome's ts a microsecond: the tool divides by frequency * 1000.
    if ! "$VISUALIZER" --input_filename "$run/events.log" --output_filename "$run/sim.chrome.json" \
        --num_npus "$ranks" --npu_frequency 1 --log_filename "$run/visualizer.log" >/dev/null 2>&1; then
        echo "FAILED $run: chakra_timeline_visualizer"
        tail -n 20 "$run/visualizer.log"
        status=1
        continue
    fi
    # The events are in sim.chrome.json now; the raw log is several MB per run.
    rm -rf "${run:?}/events.log" "${run:?}/log"
    step=$(grep -o 'finished, [0-9]*' "$run/sim.log" | awk '{print $2}' | sort -n | tail -1)
    echo "$run  step ${step} ns  simulated in $(($(date +%s) - started)) s"
done < <(find "$1" -name graph.json | sort)
exit $status
