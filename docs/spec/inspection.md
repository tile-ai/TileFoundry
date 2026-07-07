# Inspection

## Scope

Developer-facing inspection facilities for TileFoundry IR: DOT graph,
Python DSL printer (round-trippable), interactive HTML viewer, and dump
integration.
Implementation: `src/tilefoundry/inspection/`.

## 1. HIR DOT

`hir_function_to_dot(fn: hir.Function) -> str` produces a Graphviz DOT
digraph from an SSA HIR function.  Each `Var`, `Call`, and `Constant`
node is rendered with its type, shape, dtype, and `ShardLayout`
distribution annotations (mesh axes, attrs).  Shared subexpressions are
deduplicated by object identity.

```python
from tilefoundry.inspection import hir_function_to_dot
print(hir_function_to_dot(fn))
```

`module_entry_to_dot(mod: Module) -> str` renders the entry function of
a module.

## 2. Python DSL Printer

Canonical Python DSL printer that outputs executable `@func` /
`@module` source.  The output is round-trippable: `as_script(fn)` can
be parsed back via `parse_script()` to produce a structurally equivalent
IR.

### 2.1 Function printer

`hir_function_to_python(fn: hir.Function) -> str` — standalone
`@func` DSL output.  Produces imports, `@func` signature with full
`Tensor[...]` type annotations, and SSA body with op calls.

### 2.2 Module printer

`module_to_python(fn: hir.Function, module_name: str = "M") -> str` —
wraps the function in `@module(entry="<fn>") class Name:`. Shared `Mesh` /
`Topology` definitions are emitted at module level (before the class) so the
class body stays a pure function container; sugar annotations are preserved.

### 2.3 DSL text forms

DSL text forms for tensor / layout / shard annotations are owned by
[parser](./parser.md). The printer reuses those forms only when they
round-trip without losing mesh / layout / storage information;
otherwise it falls back to the verbose `ShardLayout(...)`. Printer
output supports two modes derived from the same pretty-print core:

- `canonical` — round-trippable text used by `as_script()`, pass
  dumps, and viewer detail `code` blocks: the `Tensor[...]` form of
  [parser §1.5](./parser.md#15-layout-sugar) (storage as the string
  slot, `gmem` omitted).
- `compact` — abbreviated, **display-only / non-round-trip** text for
  summaries / labels: `dtype[shape] {value-state?} @storage`. It inlines
  what it can (a split into the shape, a `Partial` into the `{...}`
  suffix) and falls back to the canonical form when a layout cannot be
  rendered compactly.

Both modes MUST agree on semantics; only the level of detail differs.
The meaning of `Split` / `Partial` / `Broadcast` is owned by
[shard](./shard.md); these forms define only render syntax.

### 2.4 Pretty-print / debug display contract

Pretty print is the core presentation layer.  Sugar, debug dumps, and
viewer type/value text reuse the same DSL text forms in §2.3.  That keeps
round-trippable source, labels, and detail panes semantically aligned.

- op attributes that are `DType`, `TensorType`, `Layout`, or `ShardLayout` are
  rendered through the §2.3 printer, not through raw dataclass / enum `repr()`

`repr()` is a debug surface, not the source of truth.  It may delegate to
the §2.3 implementation for context-free values, but context-dependent
printing (for example, choosing stable mesh names across a whole function)
must use an explicit pretty-printer API rather than relying on no-argument
`repr()`.

### 2.5 Mesh name map

The printer collects unique `Mesh` objects from all `ShardLayout`
references in the function (params, return type, body `Reshard` ops)
and assigns variable names based on `mesh.topology.name`.  Mesh
definitions are emitted in the module prelude / standalone header.

### 2.6 Specialization printing

A dispatch prototype ([hir.md §5](./hir.md#5-dispatch-specializations))
prints as its base `@tilefoundry.func` with a `pass` body, followed by each
variant as an `@<name>.specialize(pattern)` block in declared order:

```python
@tilefoundry.func
def f(x: Tensor[(S,), "f32"]) -> Tensor[(S,), "f32"]:
    pass

@f.specialize(DimVarRangePat("S", 1, 4))
def _(x: Tensor[(S,), "f32"]) -> Tensor[(S,), "f32"]:
    ...
```

The pattern prints in its constructor form (`DimVarRangePat("S", 1, 4)`;
other `Pattern` subclasses fall back to `repr(pattern)`). The emitted form
mirrors the authoring surface
([parser.md §8](./parser.md#8-dispatch-specializations)). Because a
dispatch prototype has a `DimVar` parameter, its rendering is a
**display-only** surface (§2.7): human-readable, not a round-trip
validation artifact.

### 2.7 Round-trip contract

A rendering is one of two surfaces:

**Canonical** — the rendering of a function with no `DimVar` parameter. It
MUST round-trip: `print → parse → structural_equal` holds over

- Params: shape, dtype, storage, layout.attrs, layout.shape, layout.strides, mesh identity (topology name/size, layout shape, names)
- Body: op class, args, keyword attrs, types
- Partial layouts preserve mesh names through the canonical
  parser §1.5 value-state form, and preserve `Partial.reduction` plus the
  attrs-position mesh axis in the underlying IR

**Display-only** — the rendering of a function with a `DimVar` parameter (a
`DimVar` shape entry prints as its bare name, without its envelope bounds),
and therefore of any dispatch prototype and its `.specialize` variants
(§2.6). A display-only rendering is human-readable and MUST NOT be used as a
`parse_script` validation artifact.

## 3. Viewer

The viewer is the interactive HIR inspector. `Viewer(root).serve(port,
open_browser)` (root = `hir.Function` or `Module`) starts a local HTTP
server and opens a browser page that lays the graph out **client-side**
via the vendored WebAssembly Graphviz build (`@hpcc-js/wasm`) — there is
no server-side `dot` process. The page offers pan / zoom, a detail panel,
collapsible function regions, node search, and upstream/downstream
highlight on top of the rendered SVG.

### 3.1 Architecture

```
HIR Function / Module
        │  ViewerBuilder (visitor)
        ▼
graphviz.Digraph  ──►  /api/dot?collapsed=<csv>   (DOT text)
   + DetailIndex   ──►  /api/expr/<visual_id>       (detail JSON, on demand)
                   ──►  /api/palette                (colour palette)
        │
        ▼ (browser)
@hpcc-js/wasm layout(dot, "svg", "dot")  →  innerHTML  →  SVG in #graph
        │
        ▼
d3-zoom pan/zoom · click → detail + highlight · search · collapse toggle
```

- **Backend** is the `tilefoundry.inspection.viewer` package. `Viewer(root)
  .serve(...)` ensures the vendored JS is cached, then serves five GET
  routes and returns the bound port: `/` (page), `/static/<name>`
  (first-party files, then cached vendor JS), `/api/dot?collapsed=<csv>`
  (a fresh `ViewerBuilder(root, collapsed).build().source`),
  `/api/expr/<visual_id>` (detail JSON formatted on demand from the
  `DetailIndex`), and `/api/palette` (the colour palette). No `dot`
  subprocess exists anywhere in the package. The server runs on a
  background daemon thread; `serve` returns the bound port immediately
  unless `block` holds the process open. When `block` is unset it
  follows `open_browser` — an interactive call (`open_browser=True`)
  blocks until interrupted so the page stays reachable; a programmatic
  call returns.
- **Builder.** `ViewerBuilder` walks the HIR `Function` / `Module`
  directly into a `graphviz.Digraph`; there is no intermediate model. A
  `Call(target=hir.Function)` and a top-level `Function` share one
  unified emitter — a collapsed region renders as a stand-in node, an
  expanded one as a `subgraph cluster_<region_visual_id>`.
- **Frontend** is `static/index.html` + `static/viewer.js` (first-party,
  committed). On load it fetches `/api/palette`, then renders `/api/dot`;
  each collapse toggle re-fetches `/api/dot?collapsed=...` and re-renders.
  Rendering calls `@hpcc-js/wasm` `layout()` to produce an SVG string that
  is injected via `innerHTML` (no d3-graphviz data-join).
- **Vendored assets.** The browser JS is NOT committed. `ensure_assets()`
  downloads each exact-pinned URL once to a user cache
  (`$TILEFOUNDRY_VIEWER_ASSET_DIR` → `$XDG_CACHE_HOME` →
  `~/.cache/tilefoundry/viewer-assets/<manifest-version>`), verified against a
  baked-in SHA256 manifest; any mismatch raises. The repo's `static/`
  holds only the first-party page assets (an allowlisting `.gitignore`
  keeps vendor JS out).

### 3.2 Visual identity + detail index

`visual_id` is the stable id of every emitted artifact:
`"__".join(call_path) + "__" + local`. The call path namespaces inline
expansions, so two calls to the same callee produce disjoint node ids and
disjoint detail entries; a dispatch prototype's variants, which share the
base name, are likewise disambiguated by their canonical specialization
signature. `region_visual_id` follows the same scheme, and collapse state
is a set of region ids.

`DetailIndex` maps `visual_id → DetailRef(hir_expr, kind, call_path,
region_visual_id?, param_index?)`. It is a click-lookup index, NOT a graph
model: it holds live HIR references, never pre-formatted panel JSON. Each
`/api/dot` build rebuilds and atomically replaces the index, so
`/api/expr/<visual_id>` always resolves against the currently displayed
graph; an id that was collapsed away returns 404.

### 3.3 Node rendering

- **Function node.** Title row: a `▼/▶` toggle port, the clickable
  `fn <name>` title, then one cell per parameter showing the param name.
  When expanded the params span two rows — `:pin<i>` (top, where the
  caller connects) and `:pout<i>` (bottom, where the body reads) — so a
  single port is never both an external sink and an internal source. A
  collapsed stand-in instead carries `:out<i>` result ports. The cluster
  is tinted by nesting depth (an independent low-saturation channel).
- **Op call node.** Title row (op name + per-operand input ports labelled
  with the Op's declared field names), then one field row per non-input
  attribute (`axis: 2`, `new_shape: …`), then the result-type row.
- **Return producer.** In an expanded function the real body producer of
  each return slot carries a bottom `▼ out<i>` marker (no separate anchor
  node), and a region's direct return producers share one rank. Output
  ports live on the collapsed stand-in only — an expanded header has no
  output port (that would read as the function depending on itself).
- **Var / Constant / Tuple.** `Var` shows its name; `Constant` uses the
  compact pretty value (`const(0)` / `const([1.0f, …])`, truncated past 8
  elements); `Tuple` bundles its elements. Op attributes that are
  constants / types render through the §2.4 pretty-print, never raw
  `repr`.
- **Type text.** Graph labels use the §2.3 **compact** pretty mode
  (`bf16[4 @ trd.l, 64] {trd.t @ P("sum")} @smem`) with inline split /
  DimVar / storage colour; the detail panel uses the §2.3 **canonical**
  mode (`Tensor[(4, 64), "f32", ((4 @ trd.l, 64), {trd.t @ P("sum")}),
  "smem"]`). `Reshard` / layout attrs render through the same core (never
  raw `repr`). DimVar is a single token-class colour;
  storage classes draw from an ordered pool, and an unknown memory level
  hashes stably into the pool's spare slots rather than going colourless.

### 3.4 Interaction contract

- **Detail panel.** Clicking a node title fetches `/api/expr/<visual_id>`
  and renders `params` (name | type), `returns` (idx | type) and `attrs`
  (key | value), formatted on demand from the live HIR expr. Type text is
  canonical (§3.3); DimVar / `@storage` tokens are re-coloured client-side
  from `/api/palette` using the same rule as the graph. A stale id (after
  a collapse changed the index) yields 404 and the panel clears.
- **Upstream / downstream highlight.** Clicking a node also highlights its
  connectivity cone: unrelated nodes/edges dim, upstream edges take one
  colour and downstream edges another. A header selector chooses the
  direction — bidirectional / single / upstream / downstream — and
  re-applies to the current selection on change. Clicking empty canvas
  clears. Adjacency is derived from the rendered SVG (`g.edge` titles,
  ports stripped to bare node ids), transitively.
- **Search.** A header search box highlights nodes whose id or visible
  label contains the query and dims the rest; it re-applies after a
  re-render and is independent of the highlight-direction mode.
- **Collapse / expand.** The `▼/▶` toggle cell on a function node flips
  that region's collapse state, which re-fetches `/api/dot?collapsed=<csv>`
  and re-renders. Collapse state is mirrored in the URL hash so a refresh
  restores it. Highlight/search are pure client-side SVG class switching
  and never re-render the DOT.
- **Pan / zoom.** Mouse wheel zooms and dragging pans (d3-zoom), composed
  on top of Graphviz's own layout transform.

### 3.5 File layout

```
src/tilefoundry/inspection/viewer/
  __init__.py        # Viewer.serve()
  builder.py         # HIR/Module → graphviz.Digraph; DetailIndex; format_detail
  htmltable.py       # typed Table / Row / Cell / Span for DOT HTML labels
  palette.py         # colour palette (also served at /api/palette)
  server.py          # HTTP routes (no server-side dot)
  assets.py          # vendored-JS manifest + ensure_assets()
  static/
    index.html       # first-party page (committed)
    viewer.js        # first-party client (committed)
    .gitignore       # allowlist: only index.html / viewer.js / .gitignore
scripts/fetch_viewer_assets.py   # CLI to pre-populate the asset cache
```

Vendored browser JS lives only in the user cache, never in the repo.

## 4. Dump Integration

`tilefoundry.dump.DumpScope` + `FileDumper` / `MemoryDumper` / `NullDumper`
provide per-test, per-pass IR dumping (see [passes](./passes.md) §6).

```python
from tilefoundry.dump import DumpFlags, dump, current_scope
dump("ir.py", src, DumpFlags.PASS_IR)
```

Output rooted at `test_results/{worker_id}/{nodeid}/` (per-test
subdirectory).  `pytest.mark.no_dump` opt-out available.
