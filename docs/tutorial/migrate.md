# Step one: describe the published model

The output of this step is a reference: authored HIR that computes what the
published model computes. Everything after it is measured against this, so it is
worth getting right before anything is made fast.

The code on this page is not an example written for the page. It is the shipped
source of `qwen3_5_35b_a3b`, quoted by name — a hybrid model, because a
dense one cannot show you the thing that catches people out: layers of two
different kinds in a published cycle.

`tilefoundry models qwen3_5_35b_a3b --source` names the directory that holds it
and lists the shipped files.

## The four things to get right

Each of these is a decision you make in the first ten minutes and pay for later.
The comment on the biting line in the shipped source says what it costs.

### 1. A weight is a `ConstTensor` parameter

Weights are declared, not passed. A `ConstTensor` parameter is what puts a tensor
in `Module.weights`, which is what `load(resource)` binds and what the runtime twin
reads by name. Declare it as a plain `Tensor` and none of that happens: the Module
owns no weights, `load` has nothing to bind, and every caller has to hand the
tensor over positionally, forever.

{{fixture: qwen3_5_35b_a3b/model.py:Qwen3_5Router}}

### 2. The decode contract: one token, a prior cache, and this step's own K/V

A decode step is handed the cache before it and the token it is adding. Two
declarations state that, and they are the whole contract:

{{fixture: qwen3_5_35b_a3b/model.py:S}}

{{fixture: qwen3_5_35b_a3b/model.py:C}}

`ctx_len` is the length of the **prior** cache and starts at 0 — a first step has
nothing cached and attends the one position it brings itself. The exclusive upper
bound is one past the longest prior cache the model admits, which for this model is
`config.max_ctx`; read the field's own meaning before copying the form, because it
differs between published models.

The step takes `k_cache` / `v_cache` and hands back the entry its caller appends.
Two forms fit different callers; which one fits is a property of the caller, not
a preference.

| Form | Fits when | Cost |
|---|---|---|
| Caller-managed concat | The cache grows each step, execution is eager, or `ctx_len` participates in types as a `DimVar`. | Each step replaces the cache buffer, so one fixed-address graph cannot replay it. |
| `cache_update` | The cache has fixed capacity, its write window advances each step, or it must be captured and replayed in a CUDA graph. | Shape stays static, the write window is runtime data, and traffic falls back to the whole tensor when its bounds cannot be derived. |

See [spec hir § CacheUpdate](../spec/hir.md#cacheupdate) for the operation's
contract.

{{fixture: qwen3_5_35b_a3b/model.py:Qwen3_5FullAttention.full_attention}}

### 3. Dimensions come from the published config, not from arithmetic

`head_dim` is a published field. It is **not** `hidden ÷ num_heads` — for this
model those differ, and a fixture that computed it would be wrong in a way every
shape check would accept. The same holds for `rotary_dim`, which is
`partial_rotary_factor` of the head, and for `vocab`.

{{fixture: qwen3_5_35b_a3b/model.py:_D}}

{{fixture: qwen3_5_35b_a3b/model.py:_ROT}}

### 4. The layer-type cycle is stated, not inferred

A hybrid model publishes which kind each layer is. The root reads that list and
builds the stack from it, so the cycle is a fact of the model rather than something
a reader of the fixture has to work out.

{{fixture: qwen3_5_35b_a3b/model.py:Qwen3_5Decoder}}

## When step one is finished

When the authored Module agrees with the published implementation on real weights,
at production dimensions. That is a comparison, which is what step two's tool does
— so the next page is the one that runs it, and you will use it here first.
