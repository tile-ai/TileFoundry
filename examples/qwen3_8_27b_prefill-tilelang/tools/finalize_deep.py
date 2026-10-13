"""`tilefoundry schedule finalize` for a deep scheduled HIR (long chains of
updates, e.g. the delta rule's triangular solve): the CLI's own `main`, run
with a larger Python recursion limit and thread stack. Exit code as the CLI's.

    python tools/finalize_deep.py scheduled/gdn.py:DeltaRuleIntra1024.intra_wup tir/delta_rule_intra_wup_1024.py
"""
import sys
import threading

from tilefoundry.cli import main

RESULT = {"rc": 1}


def run():
    sys.setrecursionlimit(200000)
    try:
        RESULT["rc"] = main(["schedule", "finalize", *sys.argv[1:]])
    except SystemExit as e:  # argparse / CLI exits
        RESULT["rc"] = e.code if isinstance(e.code, int) else 1


threading.stack_size(1 << 30)
t = threading.Thread(target=run)
t.start()
t.join()
sys.exit(RESULT["rc"] or 0)
