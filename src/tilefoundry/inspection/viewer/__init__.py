"""HIR graph viewer.

``Viewer(root).serve()`` ensures the vendored browser JS is cached
(downloading once on a cold cache), starts a small HTTP server that
serves the page + the per-request DOT, and returns the bound port.
Rendering happens entirely client-side (``@hpcc-js/wasm`` layout +
``innerHTML``, with ``d3`` for pan/zoom only); there is no server-side
``dot``.
"""

from . import assets
from .builder import Builder, format_detail
from .serve import Viewer, view
