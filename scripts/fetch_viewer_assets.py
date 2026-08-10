#!/usr/bin/env python3
"""Populate the viewer's vendored JavaScript cache.

This is a thin shim over :func:`tilefoundry.inspection.viewer.assets.ensure_assets`
so the manifest + verification logic has a single home inside the package
(importable when tilefoundry is pip-installed; ``scripts/`` is not packaged).

Usage:
    python scripts/fetch_viewer_assets.py [--cache-dir DIR]

With no ``--cache-dir`` the default cache root is used
(``$TILEFOUNDRY_VIEWER_ASSET_DIR`` → ``$XDG_CACHE_HOME``/``~/.cache``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tilefoundry.inspection.viewer.assets import ensure_assets


def main() -> None:
    ap = argparse.ArgumentParser(description="Download + verify viewer JS assets.")
    ap.add_argument("--cache-dir", type=Path, default=None, help="target cache directory")
    args = ap.parse_args()
    paths = ensure_assets(cache_root=args.cache_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
