# Loading and running a Module

This page is used twice. In step one it is how you check the reference against the
published implementation; in step two it is how you produce the reference the fast
implementation is compared to. That is why it is its own page rather than a
paragraph inside either step.

## Loading

A `Module` is IR: it holds no tensors. `load(resource)` reads the weights it
declares and hands back a `LoadedModule`, which does hold them.

```python
from tilefoundry.runtime import SafetensorsResource

loaded = Mod.load(SafetensorsResource(prepared_dir, alias=hf_alias(config)))
out = loaded.forward(*activations)          # weights come from the loading
```

`load` returns rather than mutating, so two loadings of one Module are independent
and neither sees the other's tensors.

Two other resources you will want:

```python
from tilefoundry.runtime import DictResource

# Tensors you already hold, canonical names, no alias table needed.
loaded = Mod.load(DictResource({"w_router": w}))

# One nested Module out of a checkpoint, without loading the tree above it:
# walk to the child, scope the resource by the same names, load only that.
child = next(m for m in Mod.modules if m.name == "router")
loaded_child = child.load(resource.subtree("router"))
```

Scoping the resource is how a leaf reads its own tensors and not the model's. The
`check` command does exactly this from a selector — `Mod.router.routing` walks the
Module and the resource through the same segment.

## The five access faces

One Module, five things you can ask it for. Mixing them up is the most common
first-day confusion, so they are listed together:

| expression | what you get |
|---|---|
| `Mod.some_func(*all parameters)` | runs it; nothing is bound, so `ConstTensor` parameters are passed explicitly |
| `Mod.some_child` | the child `Module` node |
| `Mod.lookup("some_func")` | the `ModuleFunction` IR node |
| `loaded.some_func(*activations only)` | runs it; `ConstTensor` parameters come from this loading's bindings |
| `loaded.some_child` | the child `LoadedModule` |

What `entry` means, how `forward` relates to `__call__`, and what a twin may and
may not declare are stated once, normatively, in `tilefoundry spec runtime 1.1`.
Read it there rather than here: a second copy is the one that goes stale.

## Cache is heterogeneous, and the layers do not know it

In a hybrid model the per-layer state is not one shape. A linear-attention layer
carries `(conv_state, recurrent_state)` — a fixed-size convolution window and a
fixed-size recurrent matrix. A full-attention layer carries `(k_cache, v_cache)`,
which grows by one position per step. Both are "the state of one layer", and any
harness that assumes one shape is wrong for one of them.

So the container is built per layer type, from the published cycle:

{{fixture: qwen3_5_35b_a3b/model.py:Qwen3_5Decoder.init_caches}}

and advanced the same way:

{{fixture: qwen3_5_35b_a3b/model.py:Qwen3_5Decoder.append_cache}}

The layer itself never learns which kind it got. State arrives as parameters, comes
back as results, and passes through the boundary untouched — the **caller owns it**.
That is what lets one stack hold two layer kinds without either kind knowing about
the other.

## Comparing

`tilefoundry check` is the one command that reports agreement, output by output.
Its predicates and their bounds — and the arithmetic for choosing a tolerance — are
in `tilefoundry check --help`, which is generated from the predicates themselves and
is therefore the copy that cannot go stale.
