"""First-run vendoring of the viewer's browser-side JS assets.

The viewer renders graphs client-side via ``@hpcc-js/wasm`` (full
Graphviz WASM layout) + ``d3`` (pan/zoom only, no data-join); there is
**no** server-side ``dot``. Those JS files are large and must never live in
the repo. Instead they are downloaded once to a user cache directory, pinned
to exact versions and verified against a baked-in SHA256 manifest.

Supply-chain pinning: every hash below was computed
by downloading the exact pinned URL once and running ``sha256sum``. The
runtime only ever *verifies* — it never *learns* a hash. A present cache
file whose hash does not match the manifest is treated as tampering and
raises, rather than being silently re-fetched.

``scripts/fetch_viewer_assets.py`` is a thin CLI around :func:`ensure_assets`
for pre-populating an offline machine's cache.
"""
from __future__ import annotations

import hashlib
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetEntry:
    name: str       # local filename in the cache
    url: str        # exact pinned URL (no @<major> shorthand)
    sha256: str     # 64-char hex of the bytes the URL serves
    license: str    # SPDX id
    source: str     # human-readable upstream project / version / commit


# Exact-pinned manifest. Hashes computed locally on 2026-05-29.
# @hpcc-js/wasm `graphviz.umd.js` embeds the WASM binary (base64) — it is
# self-contained, so no separate `.wasm` entry is needed (verified: the
# M0b spike rendered with only these scripts).
_ASSET_MANIFEST: tuple[AssetEntry, ...] = (
    AssetEntry(
        name="graphviz.umd.js",
        url="https://unpkg.com/@hpcc-js/wasm@2.33.8/dist/graphviz.umd.js",
        sha256="7a583e531a3ec2aadbde835e785d4b98289f28fa6d489c8c15bb2ec08f27f254",
        license="Apache-2.0",
        source="@hpcc-js/wasm 2.33.8 (Graphviz WASM, embedded binary)",
    ),
    AssetEntry(
        name="d3.min.js",
        url="https://unpkg.com/d3@7.9.0/dist/d3.min.js",
        sha256="f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539",
        license="BSD-3-Clause",
        source="d3 7.9.0 (pan/zoom only)",
    ),
)

_DOWNLOAD_TIMEOUT = 30  # seconds per attempt
_ENV_OVERRIDE = "TILEFOUNDRY_VIEWER_ASSET_DIR"


def manifest_version(manifest: tuple[AssetEntry, ...] = _ASSET_MANIFEST) -> str:
    """Short stable id for a manifest — bumping any URL/hash migrates the
    cache to a fresh subdirectory and orphans the old one."""
    h = hashlib.sha1()  # noqa: S324 — cache-key only, not security
    for e in manifest:
        h.update(f"{e.name}\0{e.url}\0{e.sha256}\0".encode())
    return h.hexdigest()[:12]


def _resolve_cache_root(manifest: tuple[AssetEntry, ...] = _ASSET_MANIFEST) -> Path:
    """Cache directory holding the vendored JS.

    ``TILEFOUNDRY_VIEWER_ASSET_DIR`` (if set) is used verbatim — handy for
    pre-populated / offline machines and for tests. Otherwise the assets
    live under ``$XDG_CACHE_HOME``/``~/.cache`` in a manifest-versioned
    subdir so a version bump never collides with stale bytes.
    """
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return base / "tilefoundry" / "viewer-assets" / manifest_version(manifest)


def _selfcheck_pinned(manifest: tuple[AssetEntry, ...]) -> None:
    """Refuse to run if any unpkg URL slipped back to a bare ``@<major>``
    shorthand (a floating tag), or if a hash is not 64 hex chars. This is
    the static defence-in-depth that replaces a network redirect-follow
    (CI / sandboxes must never hit the network)."""
    for e in manifest:
        if len(e.sha256) != 64 or not re.fullmatch(r"[0-9a-f]{64}", e.sha256):
            raise RuntimeError(f"asset {e.name!r}: sha256 must be 64 hex chars, got {e.sha256!r}")
        if "unpkg.com/" in e.url and not re.search(r"@\d+\.\d+\.\d+", e.url):
            raise RuntimeError(
                f"asset {e.name!r}: unpkg URL must pin an exact x.y.z version, "
                f"got {e.url!r} (bare @<major> is forbidden)"
            )


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _download(entry: AssetEntry, target: Path) -> None:
    """Fetch ``entry.url`` to ``target`` atomically; verify SHA256. One
    retry on transient ``URLError``; otherwise raise with an offline hint."""
    last_err: Exception | None = None
    for _attempt in range(2):
        try:
            with urllib.request.urlopen(entry.url, timeout=_DOWNLOAD_TIMEOUT) as resp:  # noqa: S310 — pinned https URL
                data = resp.read()
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(data)
            actual = hashlib.sha256(data).hexdigest()
            if actual != entry.sha256:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"asset {entry.name!r}: SHA256 mismatch after download from "
                    f"{entry.url}\n  expected {entry.sha256}\n  actual   {actual}"
                )
            tmp.replace(target)
            return
        except urllib.error.URLError as exc:
            last_err = exc
    raise RuntimeError(
        f"asset {entry.name!r}: download failed from {entry.url}\n"
        f"  cache path: {target}\n"
        f"  cause: {last_err}\n"
        f"  offline? pre-populate the cache and set {_ENV_OVERRIDE} to that directory."
    )


def ensure_assets(
    *,
    manifest: tuple[AssetEntry, ...] | None = None,
    cache_root: Path | None = None,
) -> dict[str, Path]:
    """Ensure every manifest asset is present and hash-verified in the
    cache, downloading any that are missing. Returns ``{name: path}``.

    * present + hash matches → used as-is (no network).
    * present + hash mismatches → :class:`RuntimeError` (tampering signal;
      never silently re-fetched).
    * absent → downloaded, atomically written, hash-verified.
    """
    manifest = manifest if manifest is not None else _ASSET_MANIFEST
    _selfcheck_pinned(manifest)
    root = cache_root if cache_root is not None else _resolve_cache_root(manifest)
    root.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for entry in manifest:
        target = root / entry.name
        if target.exists():
            actual = _sha256_of(target)
            if actual != entry.sha256:
                raise RuntimeError(
                    f"asset {entry.name!r}: cached file SHA256 mismatch at {target}\n"
                    f"  expected {entry.sha256}\n  actual   {actual}\n"
                    f"  the cached file does not match the pinned manifest — refusing to use it."
                )
        else:
            _download(entry, target)
        paths[entry.name] = target
    return paths


__all__ = ["AssetEntry", "ensure_assets", "manifest_version"]
