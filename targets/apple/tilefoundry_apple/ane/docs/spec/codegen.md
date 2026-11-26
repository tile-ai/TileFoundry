# ANE Codegen Spec

Scope: the **ISA** layer of the ANE backend — register fields → task descriptors → HWX
container. (TIR, scheduling and the `@emitter` dispatch live above this layer.)

## ISA

Two views — the class nesting (what contains what) and the build flow (what runs when,
and where each validation loop sits):

```text
ISA classes  (outer box contains inner; BitField ◁ Block ◁ TD ◁ Image)

┌───────────────────────────────────────────────────────────────────────┐
│ Image · HWX   1..N TD · validate() = layout                           │
│   ┌─────────────────────────────────────────────────────────────────┐ │
│   │ TD      header + register map · validate() = check rules        │ │
│   │   ┌───────────────────────────────────────────────────────────┐ │ │
│   │   │ Block   Common / L2 / NE / PE / TileDMA · + pipeline_rank │ │ │
│   │   │   ┌───────────────────────────────────┐                   │ │ │
│   │   │   │ BitField                          │                   │ │ │
│   │   │   │ addr · bits · put() = range check │                   │ │ │
│   │   │   └───────────────────────────────────┘                   │ │ │
│   │   └───────────────────────────────────────────────────────────┘ │ │
│   └─────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘


build flow   (──▶ validate · ◀┄ ✗ reject = fail-closed)

      ┌─────────────────┐
      │      TIR op     │
      └─────────────────┘
               │
               ▼
      ┌─────────────────┐
      │  emit @emitter  │
      └─────────────────┘
               │
               ▼
      ┌─────────────────┐                   ┌────────────────┐
      │   intrinsic →   │ ── per field ───▶ │  BitField.put  │
      │   register map  │ ◀┄┄┄ ✗ reject ┄┄┄ │     range      │
      └─────────────────┘                   └────────────────┘
               │
               ▼
      ┌─────────────────┐
      │   serialize →   │
      │     packets     │
      └─────────────────┘
               │
               ▼
      ┌─────────────────┐                   ┌────────────────┐
      │     TD built    │ ── per TD ──────▶ │  rules.check   │
      │ header + packets│ ◀┄┄┄ ✗ reject ┄┄┄ │  cross-field   │
      └─────────────────┘                   └────────────────┘
               │
               ▼
      ┌─────────────────┐                   ┌────────────────┐
      │  function_image │ ── across TDs ──▶ │ Image.validate │
      │  → Image · HWX  │ ◀┄┄┄ ✗ reject ┄┄┄ │     layout     │
      └─────────────────┘                   └────────────────┘
```

- **Validation is layered — one loop per nesting level, each fails closed (✗).**
  `BitField.put` range-checks every field as the intrinsic writes it; `rules.check`
  asserts the cross-field relations in `TD.validate()` once the register map is
  assembled; `Image.validate` checks layout across the TDs.
- `register map` = a TD's logical `{addr: word}` (what an intrinsic returns); serialized
  it becomes `Packet`s — the RUN/MASK wire units.  `TensorDesc` = a feature map's memory
  layout (C/H/W + its L2 strides), what an intrinsic takes as `source` / `destination`.
- `rules.py` also holds the reverse-engineered field **formulas** (guard / row_stride /
  shift / …), tagged by confidence; an unmeasured NE-staging shape fails closed.
