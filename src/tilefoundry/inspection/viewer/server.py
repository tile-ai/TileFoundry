"""Minimal HTTP server for the HIR viewer.

No server-side ``dot`` — graphs are laid out and rendered in the browser
by ``@hpcc-js/wasm`` (``d3`` handles pan/zoom only, no data-join). This
server only:

* ``GET /``                      → the first-party ``static/index.html``
* ``GET /static/<name>``         → a first-party static file, else a
                                   vendored (cache-root) JS asset
* ``GET /api/dot?collapsed=<csv>`` → fresh ``ViewerBuilder(root, collapsed)
                                   .build().source`` as ``text/plain`` DOT
* ``GET /api/expr/<visual_id>``    → detail-panel JSON formatted on demand
                                   from the last build's ``DetailIndex``
* ``GET /api/palette``             → palette pools (so the panel re-colours
                                   DimVar / storage the same as the graph)

Everything else is 404. No server-side ``dot``.
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .builder import ViewerBuilder, format_detail
from .palette import palette_pools

_STATIC_DIR = Path(__file__).parent / "static"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


def _content_type(name: str) -> str:
    return _CONTENT_TYPES.get(Path(name).suffix, "application/octet-stream")


class ViewerHTTPServer(ThreadingHTTPServer):
    """Carries the HIR root + asset cache root for the handler, plus the
    most recent ``(dot_text, detail_index)`` so ``/api/expr`` can
    resolve ids from the same build the client is viewing."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, root, cache_root: Path) -> None:
        super().__init__(addr, ViewerHandler)
        self.viewer_root = root
        self.cache_root = cache_root
        self.last_build: tuple[str, object] | None = None


class ViewerHandler(BaseHTTPRequestHandler):
    server: ViewerHTTPServer  # narrow the type for callers

    def log_message(self, *args) -> None:  # noqa: D401 — silence default stderr spam
        pass

    # -- routing ---------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        parts = urlsplit(self.path)
        path = parts.path
        if path == "/":
            self._send_static("index.html")
        elif path.startswith("/static/"):
            self._send_static(path[len("/static/"):])
        elif path == "/api/dot":
            self._send_dot(parse_qs(parts.query))
        elif path == "/api/palette":
            self._send_json(200, palette_pools())
        elif path.startswith("/api/expr/"):
            self._send_expr(path[len("/api/expr/"):])
        else:
            self._send_json(404, {"error": "not found", "path": path})

    # -- handlers --------------------------------------------------------
    def _send_static(self, name: str) -> None:
        # Basename only — no path traversal. First-party repo files win;
        # vendored JS lives only in the cache root.
        name = Path(name).name
        candidates = (_STATIC_DIR / name, self.server.cache_root / name)
        for path in candidates:
            if path.is_file():
                self._send_bytes(200, _content_type(name), path.read_bytes())
                return
        self._send_json(404, {"error": "asset not found", "name": name})

    def _send_dot(self, query: dict[str, list[str]]) -> None:
        collapsed = self._parse_collapsed(query)
        builder = ViewerBuilder(self.server.viewer_root, collapsed=collapsed)
        graph = builder.build()
        self.server.last_build = (graph.source, builder.index)
        self._send_bytes(200, "text/plain; charset=utf-8", graph.source.encode())

    def _send_expr(self, visual_id: str) -> None:
        visual_id = unquote(visual_id)
        if self.server.last_build is None:
            # No /api/dot yet — populate the index from a default build so a
            # direct detail request still resolves.
            builder = ViewerBuilder(self.server.viewer_root)
            graph = builder.build()
            self.server.last_build = (graph.source, builder.index)
        _, index = self.server.last_build
        ref = index.get(visual_id)
        if ref is None:
            self._send_json(404, {"error": "unknown visual_id", "id": visual_id})
            return
        self._send_json(200, format_detail(visual_id, ref, index.mesh_name_map))

    @staticmethod
    def _parse_collapsed(query: dict[str, list[str]]) -> set[str]:
        raw = query.get("collapsed", [""])[0]
        return {tok for tok in raw.split(",") if tok}

    # -- low-level wire --------------------------------------------------
    def _send_bytes(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict) -> None:
        import json  # noqa: PLC0415 — only the error/4xx path needs it
        self._send_bytes(code, "application/json; charset=utf-8", json.dumps(payload).encode())


def start_server(root, *, port: int, cache_root: Path, host: str = "127.0.0.1") -> ViewerHTTPServer:
    """Bind a :class:`ViewerHTTPServer` on ``<host>:<port>`` and start
    serving in a background daemon thread. Returns the live server (its
    actual port is ``server.server_address[1]``). ``host`` defaults to
    loopback; pass ``0.0.0.0`` to expose it on the LAN."""
    server = ViewerHTTPServer((host, port), root, cache_root)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


__all__ = ["ViewerHTTPServer", "ViewerHandler", "start_server"]
