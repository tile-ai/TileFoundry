"""HIR graph viewer.

``Viewer(root).serve()`` ensures the vendored browser JS is cached
(downloading once on a cold cache), starts a small HTTP server that
serves the page + the per-request DOT, and returns the bound port.
Rendering happens entirely client-side (``@hpcc-js/wasm`` +
``d3-graphviz``); there is no server-side ``dot``.
"""
from __future__ import annotations

import time
import webbrowser

from .assets import ensure_assets
from .server import start_server


class Viewer:
    def __init__(self, root) -> None:
        self.root = root

    def serve(
        self,
        port: int = 0,
        open_browser: bool = False,
        *,
        host: str = "127.0.0.1",
        block: bool | None = None,
    ) -> int:
        """Serve the viewer and return the bound port."""
        assets = ensure_assets()
        cache_root = next(iter(assets.values())).parent
        server = start_server(self.root, port=port, cache_root=cache_root, host=host)
        actual_port = server.server_address[1]
        display_host = "127.0.0.1" if host in ("127.0.0.1", "0.0.0.0") else host
        url = f"http://{display_host}:{actual_port}/"
        print(f"tilefoundry viewer serving at {url}")
        if open_browser:
            webbrowser.open(url)

        if block is None:
            block = open_browser
        if block:
            try:
                while True:  # daemon thread serves; keep the process alive
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nviewer stopped.")
                server.shutdown()

        return actual_port


__all__ = ["Viewer"]
