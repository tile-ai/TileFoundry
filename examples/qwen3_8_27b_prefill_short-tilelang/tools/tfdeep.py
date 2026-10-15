#!/usr/bin/env python3
"""Run the `tilefoundry` CLI unchanged, with a deep Python stack.

The whole-model selection (64 chained layer calls) walks deeper than CPython's
default recursion limit (RecursionError in analyze). This wrapper calls the same
entry point `tilefoundry.cli.main` with the same argv, on a thread with a 1 GiB
stack and a raised recursion limit. Nothing else changes.
"""
import sys
import threading

from tilefoundry.cli import main

sys.setrecursionlimit(1_000_000)
threading.stack_size(1 << 30)
status = [1]


def _run() -> None:
    sys.argv[0] = "tilefoundry"
    try:
        status[0] = main() or 0
    except SystemExit as exc:  # argparse / main() exit codes
        status[0] = exc.code if isinstance(exc.code, int) else 1


worker = threading.Thread(target=_run)
worker.start()
worker.join()
sys.exit(status[0])
