# `diff_means.py` — implementation & rationale

Companion to `lib/extraction/diff_means.py`. This documents every function I
implemented, the logic behind each line, and the edge cases I guarded against —
including several the original prompt did not mention.

The module is **pure numpy**: no torch, no model, no I/O. It turns two piles of
cached activations (affirm sentences, deny sentences) into a single direction
vector, projects new activations onto it, and reports how well that direction
separates the classes.

---

## The core idea (why difference-of-means works)

Each sentence's activation is a superposition: English grammar + entity name +
template syntax + topic + **mind-attribution**. Averaging ~200 affirm sentences
lets everything that *varies* across them (entities, templates) wash toward a
common mean, while everything they *share* stays put. The deny mean contains the
same shared grammar/style. Subtracting the two means cancels the shared part and
leaves only what the piles **systematically differ on**.

The danger baked into that math: the subtraction cannot distinguish
"mind-attribution" from *any* confound that differed between the piles (length,
sentiment, refusal tone). That is why the stimulus spec is strict, and why the
`cosine` control battery exists.

---

## Module-level decisions

| Decision | Why |
|---|---|
| **Cast inputs to `float64`** via `np.asarray(..., dtype=np.float64)` | Activations are usually cached as `float32`. Summing 200 rows and computing norms in `float32` accumulates rounding error that can matter for the degeneracy comparison. Casting is cheap at this size (200×2300) and makes results reproducible. |
| **`_DEGENERACY_RTOL = 1e-8`** as a named constant | The threshold appears in one place, documented, instead of a magic number buried in a branch. |
| **`_as_2d_float` helper** | Shape/emptiness/finiteness checks are identical for both piles; one helper keeps `extract_direction` readable and the error messages consistent. |

---

## `extract_direction(acts_affirm, acts_deny, normalize=True)`

**What it does:** `mean(affirm, axis=0) - mean(deny, axis=0)`, pointing
affirm-ward, optionally normalized to unit length.

### Logic, step by step
1. **Shape checks first** (`_as_2d_float`). Both must be 2D `(n_items, d_model)`,
   non-empty, and finite. `d_model` must match between the piles. A shape bug
   that slips through becomes a silent nonsense result three experiments later,
   so this fails loudly and early.
2. **`axis=0`** averages *down the rows* (across items), leaving one
   `d_model`-length vector. The wrong axis would collapse to a per-item scalar
   and break everything downstream in confusing ways.
3. **`direction = mu_affirm - mu_deny`.** That is the entire method.
4. **Degeneracy guard *before* normalizing.** If the two means nearly coincide,
   the difference is a tiny noise vector — and normalizing would scale that noise
   up to length 1, producing something that *looks* like a confident direction.
   We compare the raw norm to a reference activation norm and raise if it is
   below `1e-8 ×` that reference.
5. **Return** `direction / ‖direction‖` if `normalize` else the raw difference.

### Decisions
- **`normalize=True` by default** — makes projections comparable across layers
  and models, and forces steering strength to be an explicit coefficient rather
  than something smuggled in by how long the vector happens to be.
- **Unequal pile sizes are allowed** (the mean handles them) but shouldn't
  happen — balance is the stimulus file's job (`validate_balance`). This function
  does **not** silently rebalance; that would paper over a data bug.
- **No centering/standardizing inside** — any preprocessing here would have to be
  replicated identically inside `project()`, and the day someone forgets is the
  day every number goes quietly wrong.

### Edge cases (beyond the prompt)
- **Non-finite input (nan/inf):** a single nan in the activations would silently
  poison the mean and produce a nan direction that flows downstream undetected.
  `_as_2d_float` rejects it up front.
- **Degeneracy reference norm — I used `max(‖mu_affirm‖, ‖mu_deny‖)`** instead of
  the prompt's literal `‖mu_affirm‖`. The prompt says "e.g. ‖mu_affirm‖"; using
  the larger of the two means is strictly more robust, because if `mu_affirm`
  itself happened to be near zero, the original threshold would collapse to ~0
  and let noise through. Same guarantee, no blind spot.
- **Both means are the zero vector:** the reference norm is 0, so the relative
  test is meaningless (0 < 0 is false, which would *pass* a genuinely degenerate
  case). Caught explicitly with its own error.
- **Empty pile / zero-width `d_model`:** rejected by the non-empty check rather
  than producing a nan-mean (numpy's `mean` of an empty axis warns and returns
  nan).
- **Integer-typed activations:** the `float64` cast means `int` inputs still
  produce a correct float direction rather than integer-truncated means.

---

## `project(acts, direction, *, acts_layer=None, direction_layer=None)`

**What it does:** one scalar per activation — how far along `direction` it sits.
High = processed as mind-affirming, low = mind-denying.

### Logic
1. Check `d_model` agreement between `acts` (2D) and `direction` (1D).
2. **Re-normalize `direction` defensively**, even if the caller "should have."
   One line, and a direction loaded from an old results file can't quietly change
   the scale of your numbers.
3. **Return `acts @ direction`** — `(n, d) @ (d,)` gives `(n,)`, one scalar per
   item, in the same row order as the input.

### The layer trap, and how I actually guarded it
Nothing in the math stops you projecting layer-14 activations onto a layer-9
direction — you get plausible numbers that mean nothing. Because this is a pure
`(acts, direction)` numpy function, the vector itself carries no layer tag. So I
added **optional keyword-only params `acts_layer` / `direction_layer`**: when
both are supplied, a mismatch raises. This is the prompt's "make the caller pass
layer explicitly" option, implemented without breaking the positional signature
(callers that don't pass them behave exactly as before).

> **Note / limitation:** if a caller omits both tags, the guard cannot fire — a
> pure-numpy function has no other way to know layer identity. The durable fix
> lives one level up (carry a layer tag with every stored direction and assert on
> load). The hook is here; wiring it in is the orchestration layer's job.

### Why it must be one function
Third-person validation items and first-person H1 items go through this **same
code path with the same frozen direction** — "same ruler, swapped referent."
That identity *is* the experiment. If first-person items ever got special-cased
projection logic, the transfer claim would dissolve.

### Edge cases (beyond the prompt)
- **Zero-norm direction:** defensive normalization would divide by zero. Caught
  explicitly — you cannot project onto "no direction."
- **Non-finite direction:** rejected, for the same reason as in extraction.
- **1D `acts` (a single activation):** rejected in favor of requiring 2D, so the
  return is *always* `(n,)` with a predictable row correspondence. Callers with a
  single item pass `acts[None, :]`.

---

## `cosine(a, b)`

**What it does:** one number for whether two arrows point the same way.
`1` = identical heading, `0` = perpendicular, `-1` = opposite.

### Why it's the credibility core
This is the control battery. Four comparisons, each killing a different
objection:
- **vs the placebo direction** (robot durable/fragile, identical pipeline): must
  be **low**. High ⇒ you built a generic "statements about entities" detector and
  mind content was never the active ingredient.
- **vs the safety/refusal direction:** must be **low**, or the boring story ("the
  model just got cagey") explains everything.
- **vs a sentiment direction:** must be **low**. Deny sentences skew lexically
  negative (*lacks, nothing, fails*), so sentiment is a live impostor.
- **across adjacent layers:** should be **moderate-to-high**. A direction that
  flips heading between neighbors is fitting noise. (Diagnostic, not a control.)

### Logic
1. **Zero-norm guard → return `nan`** (documented), never divide silently.
2. `float(a @ b / (‖a‖ · ‖b‖))`.

### Decisions & edge cases
- **No `abs()`.** `extract_direction` always points affirm-ward by construction,
  so sign is real information — a strong negative cosine is as meaningful as a
  positive one. Taking absolute value would erase a finding.
- **`nan` over raise for zero-norm:** chosen so a single degenerate direction in
  a batch of control comparisons yields one `nan` cell rather than aborting the
  whole battery. Documented in the docstring.
- **Clamp to `[-1, 1]`:** floating-point error can push the ratio a hair outside
  the valid cosine range; `np.clip` keeps the output a legal cosine without
  hiding anything meaningful.
- **`.ravel()` on both inputs** tolerates `(d,)` or `(1, d)` shapes; a genuine
  length mismatch still raises.

---

## `extract_all_layers(acts_by_layer, labels)`

**What it does:** the same extraction once per layer, so you can ask *which depth*
represents mind-attribution most cleanly.

### Why sweep instead of pick
There's no principled prior for where the feature lives (early layers ≈ surface
tokens, late ≈ output commitments, interesting abstractions usually mid-stack —
but "usually" isn't a reason to guess). Sweeping is cheap once activations are
cached (an average and a subtraction per layer), so sweep and let held-out
performance choose.

### Logic
1. From `labels`, compute affirm/deny row indices **once, outside the loop** —
   the split is identical at every layer because it's the same items.
2. Loop layers (in **sorted order**, for determinism), slice each layer's matrix
   with those two index sets, hand the pieces to `extract_direction`.
3. Collect `{layer: direction}`. If a layer raises on degeneracy, **warn and
   continue** rather than crashing the sweep — early layers sometimes carry
   almost no signal, and that absence is itself a result worth seeing in the plot.

### The alignment bug I designed against
`labels[i]` must describe row `i` of **every** activation matrix. If the ordering
desynchronizes (different sort, a filtered item, a dict that lost order) you
extract a direction from an essentially random split — which doesn't crash and
doesn't look obviously wrong, just weak-but-plausible enough to survive into the
writeup. Guards I added:
- **Per-layer row-count assertion:** every matrix must be 2D with exactly
  `len(labels)` rows, or it raises naming the offending layer. Ragged matrices
  (a filtered/extra item) are caught instead of silently mis-sliced.
- **Deterministic `sorted()` iteration**, so output order never depends on dict
  insertion order.

> **Note:** true id-based alignment (carry `item_ids` alongside and assert they
> match) can only happen where the ids exist. This function receives just
> `labels` + `acts_by_layer`, so the strongest check available here is row-count
> consistency; the id assertion belongs in the caller that assembles these dicts.

### Edge cases (beyond the prompt)
- **Labels not in `{0, 1}`:** any third value would be silently dropped from
  *both* index sets, corrupting the split. Rejected up front.
- **A class entirely absent:** raises (a global data problem affecting every
  layer identically) rather than returning an empty/garbage sweep.
- **Degenerate single layer:** warned + skipped, so the returned dict simply
  omits it — the plot shows the gap.

---

## `summarize_projections(proj, labels)`

**What it does:** descriptive stats on one set of ruler readings, split by class:
`affirm_mean`, `deny_mean`, `separation`, `pooled_sd`, `cohens_d`.

### Why accuracy isn't enough
Two probes can both score 0.78 while telling completely different stories: one
with two tight, cleanly-offset clusters; another with heavy overlap and a few
outliers dragging the threshold. Accuracy is what the kill criteria check; effect
size is what goes in the paper. Both come from the same function so nobody
recomputes one by hand.

### The reason this function really exists
It's how you read the **mundane controls**. The preregistered prediction is that
"I was trained on text" lands **near zero** on the direction — *not* at the deny
pole. That distinction is the whole interpretation: deny items *actively deny*
experience; mundane items simply *don't mention it*. A binary accuracy number
cannot express "near zero" — only the distribution can. If mundane items sit at
the affirm pole, the direction is a self-reference detector and H1's reading
collapses. (The smoke test confirms this: mundane items projected to ~0.25 while
the deny mean sat at ~−3.5.)

### Logic
1. Split `proj` by label.
2. Means; `separation = affirm_mean - deny_mean`; pooled sd =
   `sqrt((var_a + var_d) / 2)` for equal-ish `n`.
3. `cohens_d = separation / pooled_sd`.
4. Return a dict with **stable key names** — figure code reads them by name, so
   renaming a key later would silently break plots.

### Decisions & edge cases
- **`ddof=1` (sample variance):** standard for effect-size estimates. Requires
  n ≥ 2 per class.
- **Class with < 2 items → `pooled_sd = nan`** (sample variance is undefined),
  which propagates to `cohens_d = nan` rather than raising mid-report.
- **`pooled_sd == 0` (both clusters perfectly tight) → `cohens_d = nan`** instead
  of `±inf`, so a divide-by-zero can't masquerade as an infinitely strong effect.
- **`separation` kept as its own key** (not just `cohens_d`) so a reviewer can see
  the raw gap in the direction's native units alongside the standardized one.
- **Label/length validation** mirrors the other functions: 1D labels, matching
  length, values ⊆ `{0, 1}`, both classes present.

---

## Cross-cutting edge-case summary

| Case | Where handled | Behavior |
|---|---|---|
| Non-2D / empty activations | `_as_2d_float`, `project`, `extract_all_layers` | raise |
| `d_model` mismatch | every function that takes two arrays | raise |
| nan/inf in activations or direction | `_as_2d_float`, `project` | raise |
| Coincident class means (noise direction) | `extract_direction` | raise before normalizing |
| Both means at origin | `extract_direction` | raise (reference norm 0) |
| Zero-norm direction | `project` (raise), `cosine` (nan) | documented per function |
| Cosine float overshoot | `cosine` | clip to [-1, 1] |
| Wrong-layer projection | `project` via optional layer tags | raise on mismatch |
| Labels ∉ {0,1} | `extract_all_layers`, `summarize_projections` | raise |
| Class missing from split | `extract_all_layers`, `summarize_projections` | raise |
| Degenerate single layer in sweep | `extract_all_layers` | warn + skip |
| < 2 items or zero variance in a class | `summarize_projections` | pooled_sd / cohens_d = nan |
| float32 accumulation error | module-wide `float64` cast | avoided |

## Preregistration constraints (caller's responsibility, noted here)
- Feed `extract_all_layers` **train-template items only (t1–t5)**. Held-out
  templates (t6, t7) test generalization past sentence shape and are contaminated
  the instant they touch extraction.
- Pick the layer by **held-out accuracy under the prereg rule**, not by eyeballing
  which layer gave the nicest number after seeing them all.
