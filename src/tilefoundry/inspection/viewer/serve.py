"""The ``Viewer`` entry: cache the vendored JS, serve the page, return the port."""
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
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nviewer stopped.")
                server.shutdown()

        return actual_port


def view(root, *, port: int = 0, open_browser: bool = True) -> int:
    """Start the interactive HIR viewer for *root* (Function or Module)."""
    return Viewer(root).serve(port=port, open_browser=open_browser)


__all__ = ["Viewer", "view"]
