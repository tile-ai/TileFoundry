"""Serve the browser-rendered HIR viewer and its read-only APIs.

Routes provide the viewer shell and assets, fresh DOT for collapse state,
expression details from the last build, and palette data. Graphviz WASM performs
layout in the browser; the server never invokes ``dot``. All other routes return
404.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType
from urllib.parse import parse_qs, unquote, urlsplit

from .builder import Builder, format_detail
from .palette import palette_pools

STATIC_DIR = Path(__file__).parent / "static"

CONTENT_TYPES = MappingProxyType({
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
})


def content_type_of(name: str) -> str:
    return CONTENT_TYPES.get(Path(name).suffix, "application/octet-stream")


def parse_collapsed(query: dict[str, list[str]]) -> set[str]:
    raw = query.get("collapsed", [""])[0]
    return {tok for tok in raw.split(",") if tok}


class HTTPServer(ThreadingHTTPServer):
    """Represent HTTPServer.

    Carries the HIR root + asset cache root for the handler, plus the
    most recent ``(dot_text, detail_index)`` so ``/api/expr`` can
    resolve ids from the same build the client is viewing.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, root, cache_root: Path) -> None:
        super().__init__(addr, Handler)
        self.viewer_root = root
        self.cache_root = cache_root
        self.last_build: tuple[str, object] | None = None

    def record_build(self, source: str, index: object) -> None:
        """Remember the build the client is viewing, for ``/api/expr``."""
        self.last_build = (source, index)


class Handler(BaseHTTPRequestHandler):
    server: HTTPServer

    def log_message(self, *args) -> None:
        pass


    def do_GET(self) -> None:
        parts = urlsplit(self.path)
        path = parts.path
        if path == "/":
            self.send_static("index.html")
        elif path.startswith("/static/"):
            self.send_static(path[len("/static/"):])
        elif path == "/api/dot":
            self.send_dot(parse_qs(parts.query))
        elif path == "/api/palette":
            self.send_json(200, palette_pools())
        elif path.startswith("/api/expr/"):
            self.send_expr(path[len("/api/expr/"):])
        else:
            self.send_json(404, {"error": "not found", "path": path})


    def send_static(self, name: str) -> None:


        name = Path(name).name
        candidates = (STATIC_DIR / name, self.server.cache_root / name)
        for path in candidates:
            if path.is_file():
                self.send_bytes(200, content_type_of(name), path.read_bytes())
                return
        self.send_json(404, {"error": "asset not found", "name": name})

    def send_dot(self, query: dict[str, list[str]]) -> None:
        collapsed = parse_collapsed(query)
        builder = Builder(self.server.viewer_root, collapsed=collapsed)
        graph = builder.build()
        self.server.record_build(graph.source, builder.index)
        self.send_bytes(200, "text/plain; charset=utf-8", graph.source.encode())

    def send_expr(self, visual_id: str) -> None:
        visual_id = unquote(visual_id)
        if self.server.last_build is None:


            builder = Builder(self.server.viewer_root)
            graph = builder.build()
            self.server.record_build(graph.source, builder.index)
        _, index = self.server.last_build
        ref = index.get(visual_id)
        if ref is None:
            self.send_json(404, {"error": "unknown visual_id", "id": visual_id})
            return
        self.send_json(200, format_detail(visual_id, ref, index.context))

    def send_bytes(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, code: int, payload: dict) -> None:
        self.send_bytes(code, "application/json; charset=utf-8", json.dumps(payload).encode())


def start_server(root, *, port: int, cache_root: Path, host: str = "127.0.0.1") -> HTTPServer:
    """Start server.

    Bind a :class:`HTTPServer` on ``<host>:<port>`` and start
    serving in a background daemon thread. Returns the live server (its
    actual port is ``server.server_address[1]``). ``host`` defaults to
    loopback; pass ``0.0.0.0`` to expose it on the LAN.
    """
    server = HTTPServer((host, port), root, cache_root)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


__all__ = ["HTTPServer", "Handler", "start_server"]
