# Inspection

## Scope

Developer-facing inspection facilities for TileFoundry IR: DOT graph,
Python DSL printer (round-trippable), interactive HTML viewer, and dump
integration.

## 1. HIR DOT

```python
def hir_function_to_dot(fn: hir.Function) -> str:
    """Render an SSA HIR function as Graphviz DOT.

    Args:
        fn: Function to render.

    Returns:
        Graphviz DOT text.
    """
    ...


def module_entry_to_dot(module: Module) -> str:
    """Render a Module's entry function as Graphviz DOT.

    Args:
        module: Module whose entry is rendered.

    Returns:
        Graphviz DOT text.
    """
    ...
```

- constraints:
  - Each `Var`, `Call`, and `Constant` node MUST show its type, shape, dtype,
    and sharding annotations. Shared expressions MUST be deduplicated by
    object identity.

```python
# example
from tilefoundry.inspection import hir_function_to_dot
print(hir_function_to_dot(fn))
```

## 2. Python DSL Printer

Canonical Python DSL printer that outputs executable `@func` /
`@module` source. The output is round-trippable: write `as_script(fn)` to a
Python file and import it, and the real authoring decorators MUST produce a
structurally equivalent IR. Printing that rebuilt value MUST reproduce the
same canonical text.

### 2.1 Function printer

```python
class PythonPrintOptions:
    """Control optional non-canonical printer annotations.

    Attributes:
        show_types: attribute; Whether to append inferred types.
        comment_metadata_types: attribute; Metadata classes rendered as comments.
        comment_opt_in: attribute; Declared keys a reader asked a comment for.
    """

    show_types: bool = False
    comment_metadata_types: tuple[type[IRMetadata], ...] = ()
    comment_opt_in: frozenset[str] = frozenset()


def hir_function_to_python(
    fn: hir.Function, *, options: PythonPrintOptions | None = None
) -> str:
    """Render a HIR function as Python DSL source.

    Args:
        fn: Function to render.
        options: Optional inspection annotations.

    Returns:
        Python DSL source.
    """
    ...
```

- constraints:
  - The output MUST include imports, a bare `@func` signature with complete
    `Tensor[...]` annotations, and the SSA body.

### 2.2 Module printer

```python
def as_script(
    fn: hir.Function | Module,
    *,
    module: str | None = None,
    options: PythonPrintOptions | None = None,
) -> str:
    """Render a HIR function or Module as Python DSL source.

    Args:
        fn: Function or Module to render.
        module: Optional wrapper class name for a Function.
        options: Optional inspection annotations.

    Returns:
        Python DSL source.
    """
    ...


def module_to_python(fn: hir.Function, module_name: str = "M") -> str:
    """Render a function through the module-wrapper compatibility alias.

    Args:
        fn: Function to render.
        module_name: Wrapper class name.

    Returns:
        Python DSL source.
    """
    ...
```

- constraints:
  - Module input MUST emit every HIR Function and preserve shared `Mesh` and
    `Topology` definitions before the class. Mixed HIR/TIR Modules MUST be
    rejected.

The printer MUST emit a Module's whole tree: each nested Module prints as a
`@module` class inside its owner's body, so importing the file rebuilds the same
ownership. It MUST emit only what a Module *declares*, never what the Module
resolves to — an inherited Target or topology hierarchy prints nothing, so
the declaration-versus-inheritance split survives the round trip. A declared
Target prints as the `@module(target=...)` argument and a declared hierarchy
as the `@module(topologies=...)` argument
([parser §2.7](./parser.md#27-module-authoring-surface)).
Every dimension referenced only by a declared topology expression MUST still
be emitted in the dimension prelude. A topology `ShapeDim` MUST use the same
DSL expression text as tensor and Mesh geometry, including public constructors
such as `ceildiv`, so importing restores the same expression tree.

- constraints:
  - The decorator MUST print in its called form, `@module()` included. A bare
    decorator has not run while the class body is evaluated, so a body naming a
    child call could not resolve it ([parser §1.1](./parser.md#11-decorators)).
  - A nested Module MUST print before the owner's Functions, because a body
    calling one names the attribute it is bound to and a class body binds in the
    order it is written.
  - A call on the entry of a direct child MUST print as that child's attribute
    applied to the call's arguments, never as an attribute reach into the child
    and never by the callee's own name. Which child a call reaches MUST be read
    from the attached entry's identity and the recorded origin
    ([hir §1.1](./hir.md#11-function)), not from the name they share and not from
    the parser's consumed authoring record
    ([parser §4.2](./parser.md#42-closure-then-registry-callee-resolution)):
    anything may be called the same, and two attributes may hold copies of one
    Module.
  - Such a call MUST print exactly the arguments `Call.args` carries
    ([hir §1.1](./hir.md#11-function)) and no others; importing the source
    restores the callee's complete signature from the child's own class body.

The printer MUST import the concrete Target class from its provider module and
embed `repr(target)` as the constructor expression. That representation MUST
rebuild an equal value of the same concrete class when the emitted source is
executed. A Target subclass with a different constructor customizes ordinary
`__repr__`; there is no printer-specific Target hook or construction registry.

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

The same-line type annotation `show_types` appends is the `canonical` form on
one physical line, rendered through the same mesh name map
([§2.5](#25-mesh-name-map)) as the signature and the prelude: an annotated
layout MUST name the hoisted mesh rather than restate it, and a `Tuple[...]`
annotation MUST name it in every field. The verbose `ShardLayout(...)` fallback
is unchanged — a mesh with no named axes, or a layout the sugar cannot express,
still renders verbose, so no annotation loses information. The annotation is
**display-only** ([§2.7](#27-round-trip-contract)); what round-trips is the
emitted code, not its comments.

Canonical DType text is the descriptor's `name`. Tensor annotations and DType
op attributes MUST emit that name as a quoted DSL string. Compact labels MAY
omit the quotes, but MUST NOT use the descriptor's raw `repr()`.

### 2.4 Pretty-print / debug display contract

Pretty print is the core presentation layer.  Sugar, debug dumps, and
viewer type/value text reuse the same DSL text forms in [§2.3](#23-dsl-text-forms).  That keeps
round-trippable source, labels, and detail panes semantically aligned.

- op attributes that are `DType`, `TensorType`, `Layout`, or `ShardLayout` are
  rendered through the [§2.3](#23-dsl-text-forms) printer; `DType` uses its canonical name and these
  values do not use raw `repr()` output

`repr()` is a debug surface, not the source of truth.  It may delegate to
the [§2.3](#23-dsl-text-forms) implementation for context-free values, but context-dependent
printing (for example, choosing stable mesh names across a whole function)
must use an explicit pretty-printer API rather than relying on no-argument
`repr()`.

### 2.5 Mesh name map

The printer collects unique `Mesh` objects from all `ShardLayout`
references in the function (params, return type, body `Reshard` ops)
and assigns variable names from the first declared topology's name. Mesh
definitions are emitted in the module prelude / standalone header.

### 2.6 Specialization printing

A dispatch prototype ([hir.md §1.1](./hir.md#11-function))
prints as its base `@func` with a `pass` body, followed by each
variant as an `@<name>.specialize(pattern)` block in declared order:

```python
# example
@func
def f(x: Tensor[(S,), "f32"]) -> Tensor[(S,), "f32"]:
    pass

@f.specialize(DimVarRangePat("S", 1, 4))
def _(x: Tensor[(S,), "f32"]) -> Tensor[(S,), "f32"]:
    ...
```

The pattern prints in its constructor form (`DimVarRangePat("S", 1, 4)`;
other `Pattern` subclasses fall back to `repr(pattern)`). The emitted form
mirrors the authoring surface
([parser.md §1.1](./parser.md#11-decorators)). Because a
dispatch prototype has a `DimVar` parameter, its rendering is a
**display-only** surface ([§2.7](#27-round-trip-contract)): human-readable, not a round-trip
validation artifact.

### 2.7 Round-trip contract

A rendering is one of two surfaces:

**Canonical** — the rendering of a function with no `DimVar` parameter, or a
Module whose Functions have no `DimVar` parameter and whose symbolic dimensions
occur in round-trippable tensor, Mesh, or topology geometry. It
MUST round-trip: printing, importing, and the test-side structural comparison
MUST agree over

- Params: shape, dtype, storage, layout.attrs, layout.shape, layout.strides, mesh identity (topology name/size, layout shape, names)
- Body: op class, args, keyword attrs, types
- DType annotations and op attributes preserve the selected descriptor singleton
  through their canonical names
- Partial layouts preserve mesh names through the canonical
  [parser §1.5](./parser.md#15-layout-sugar) value-state form, and preserve `Partial.reduction` plus the
  attrs-position mesh axis in the underlying IR

**Display-only** — the rendering of a function with a `DimVar` parameter, and
therefore of any dispatch prototype and its `.specialize` variants ([§2.6](#26-specialization-printing)). A
display-only rendering is human-readable and MUST NOT be used as a
structural round-trip validation artifact: it is held to importing, not to the
same structural comparison.

A canonical grid loop MUST render each yielded expression under its own unique
binding. After the loop body has emitted every yielded expression, the printer
MUST emit one `carry = yield` assignment per carried value, in
`GridRegionExpr.carried_args` order. Those assignments are the final statements
in the loop body. This preserves references to the old carry until the update
point and lets the parser's final-RHS carry rule rebuild the same loop.

A `DimVar` shape entry prints as its bare name, and a shape-valued op attribute
holding one MUST print it the same way rather than as its repr. The rendering
MUST therefore also emit, once, a declaration binding each such name to the
`DimVar` it stands for, with the envelope bounds it was declared with. Without
those declarations the printed source names something nothing defines and cannot
be imported at all, which is a weaker artifact than display-only is meant to be
-- the whole point of printing a program is that somebody can be handed the file.
The bounds are not recoverable from the name, so they are restated rather than
inferred.

### 2.8 Record comment forms

A record attached to an expression is typed fields and nothing else
([core-ir §2](./core-ir.md#2-expr)). What one looks like on a printed line is
decided here, by walking its fields: a family that wrote its own form string
would be deciding presentation, and every family would then spell one value its
own way.

A record MUST declare that it renders as a comment, and MAY declare which keys
it emits: a field by its own name, anything else as a projection carrying its
key, its type, and where its value comes from. Metadata with no declaration
renders as nothing, which is how an annotation that is not a report -- a binding
name, a constraint -- stays off the line.

A record an analysis report projects ([analysis §2](./analysis.md#2-authored-hir-metrics))
MUST NOT emit a key that projection cannot state; it MAY emit fewer, because a
comment is read on a line and JSON is read by a program. A record no report
projects is comment-only -- `SourceSpanMetadata` is where an expression was
authored, which no family measures -- and there is no projection for it to be a
subset of.

One separator per layer, and a separator MUST NOT appear inside a value it
separates unless that value brackets itself:

| Layer | Separates | With |
|---|---|---|
| value | bytes read from bytes written | `/` |
| value | a whole quantity from one unit's share | `@` |
| value | a mapping's key from its value | `:` |
| value | a mapping's entries | `,` |
| field | a key from its value | `=` |
| record | one field from the next | space |
| line | one record from the next | `; ` |

- constraints:
  - A key MUST be its declared name with `_` written as `-`. It MUST NOT be
    derived from the value, and a unit MUST be part of the name -- `ideal-ns=13`,
    never `ideal=13ns` -- so the unit is stated once, where the field states it.
  - A value MUST be an `int`, a `str`, a mapping of them, or a type with a
    declared rendering. `int` and `str` MUST NOT be wrapped in a type that
    renders identically.
  - A `str` that is a token renders bare. Text that is a sentence MUST render as
    a quoted, escaped DSL string literal ([§2.3](#23-dsl-text-forms)), which is
    what lets it hold a separator the ladder uses; a reader splits a layer
    outside the quotes.
  - A key whose value equals what it declared it says nothing by MUST be left
    out, so a record that measured nothing collapses to its family name.
  - A record declaring exactly one key MUST use the family name as that key: for
    a record of one thing the family and the key state the same thing.
  - The family name MUST be the class name without a trailing `Metadata`, in
    kebab case. A record whose reported name differs MUST state that name where
    its keys are declared.
  - Part zero of a line is the value's own type, which carries no key: it is not
    a measurement of the value, it is the value, and it is DSL text
    ([§2.3](#23-dsl-text-forms)) that pastes back. Every later part is a record.

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
  constants / types render through the [§2.4](#24-pretty-print--debug-display-contract) pretty-print, never raw
  `repr`.
- **Type text.** Graph labels use the [§2.3](#23-dsl-text-forms) **compact** pretty mode
  (`bf16[4 @ trd.l, 64] {trd.t @ P("sum")} @smem`) with inline split /
  DimVar / storage colour; the detail panel uses the [§2.3](#23-dsl-text-forms) **canonical**
  mode (`Tensor[(4, 64), "f32", ((4 @ trd.l, 64), {trd.t @ P("sum")}),
  "smem"]`). `Reshard` / layout attrs render through the same core (never
  raw `repr`). DimVar is a single token-class colour;
  storage classes draw from an ordered pool, and an unknown memory level
  hashes stably into the pool's spare slots rather than going colourless.

### 3.4 Interaction contract

- **Detail panel.** Clicking a node title fetches `/api/expr/<visual_id>`
  and renders `params` (name | type), `returns` (idx | type) and `attrs`
  (key | value), formatted on demand from the live HIR expr. Type text is
  canonical ([§3.3](#33-node-rendering)); DimVar / `@storage` tokens are re-coloured client-side
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

Vendored browser JS lives only in the user cache, never in the repo.

## 4. Dump Integration

`tilefoundry.dump.DumpScope` + `FileDumper` / `MemoryDumper` / `NullDumper`
provide per-test, per-pass IR dumping (see [passes §6](./passes.md#6-top-level-api)).

```python
# example
from tilefoundry.dump import DumpFlags, dump, current_scope
dump("ir.py", src, DumpFlags.PASS_IR)
```

```python
class DumpFlags(IntFlag):
    """Enumerate dump categories."""

    NONE = 0
    PASS_IR = 1
    CODEGEN_SOURCE = 2
    BUILD_LOG = 4
    ALL = PASS_IR | CODEGEN_SOURCE | BUILD_LOG


class DumpScope:
    """Install or narrow a dynamically scoped dump destination."""

    def __init__(
        self,
        subdir: str | None = None,
        flags: DumpFlags | None = None,
        *,
        dumper: IDumpper | None = None,
    ) -> None: ...
```

- constraints:
  - `DumpScope(subdir, flags)` MUST nest beneath the active scope and intersect
    its flags with the parent's. Without a parent it MUST remain a no-op.
  - `DumpScope(dumper=..., flags=...)` MUST replace the active scope.
  - Plain child threads MUST start without the parent's scope; asyncio tasks
    MUST inherit a copy of the creating context.
  - Test output MUST be rooted at `test_results/<file-stem>/<test-name>/`.
    A non-master worker appends `__<worker-id>` to the test-name leaf rather
    than adding a worker directory. `pytest.mark.no_dump` disables it.

## 5. Compact TIR rendering

```python
def format_expr(expr) -> str:
    """Render one supported TIR expression compactly.

    Args:
        expr: Expression to render.

    Returns:
        Compact text.
    """
    ...


def format_pattern(pat: Pattern) -> str:
    """Render one dispatch pattern compactly.

    Args:
        pat: Pattern to render.

    Returns:
        Compact text.
    """
    ...


def format_symbol_call(call: Evaluate) -> str:
    """Render one symbol invocation compactly.

    Args:
        call: Function-call statement.

    Returns:
        Compact text.
    """
    ...


def format_abort(stmt: Abort) -> str:
    """Render one abort statement compactly.

    Args:
        stmt: Abort statement.

    Returns:
        Compact text.
    """
    ...


def format_dispatch_call(stmt: DispatchCall, indent: str = "") -> str:
    """Render a dispatch statement compactly.

    Args:
        stmt: Dispatch statement.
        indent: Prefix for each emitted line.

    Returns:
        Compact multiline text.
    """
    ...
```

- constraints:
  - These functions are display-only and MUST NOT be treated as parser input.
  - `format_dispatch_call` MUST preserve case order and render the fallback.

