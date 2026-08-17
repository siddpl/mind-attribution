# Preregistration — mind-attribution direction probing

**This file contains commitments only.** Rationale lives in
[`docs/method.md`](docs/method.md), definitions in
[`docs/glossary.md`](docs/glossary.md), defects in
[`docs/bug_log.md`](docs/bug_log.md), pinned versions in
[`docs/environment.md`](docs/environment.md), and unresolved decisions in
[`docs/open_items.md`](docs/open_items.md). Those documents are revisable. This
one is not, after it is committed.

**Status: DRAFT.** One item in `docs/open_items.md` (§ "Blocking a real
capture") remains — access to the analysis model's weights. It does not change
any quantity declared here. The commit that finalizes this file is the
timestamp of record.

| | |
|---|---|
| Repository | `github.com/siddpl/mind-attribution` |
| Analysis model | `google/gemma-2-2b` |
| Data captured on the analysis model so far | **none** |

---

## 1. Hypotheses

**H1 (transfer).** A direction extracted from *third-person* mind-attribution
contrasts separates *first-person* experiential from mundane self-reference.

The result is **the gap between two probes**, reported together, always:

| pattern | reading |
|---|---|
| direction ≈ ceiling | shared machinery — our axis captures the available signal |
| direction ≪ ceiling | signal exists but **not on our axis** — self-specific machinery. A positive finding, not a null. |
| both at chance | no linearly decodable signal |

**Kill-1.** Extraction from templates 1–5 must generalize to held-out templates
6–7:

```
PASS  iff  heldout_accuracy > chance_band(n_heldout, alpha_sd)
      AND  heldout_accuracy > placebo_direction_accuracy_on_mind_items
```

Both conditions, on the same items. Failing either kills the leg.

---

## 2. Capture

| commitment | value |
|---|---|
| Hook point | `resid_post` only — never `resid_pre`, `mlp_out`, `attn_out` |
| Layers | all, in a single forward pass |
| Token positions captured | `final` **and** `claim_end` |
| **Primary position** | **`claim_end`** (§2.1) |
| Model load precision | **`bfloat16`** (§2.2) |
| Weight processing | **none** — `from_pretrained_no_processing` (§2.3) |
| Stored activation precision | `float32` regardless of model dtype |
| Generation | none, anywhere. Forward passes only. |
| Gradients | `torch.no_grad()` around all passes |
| Cache key | `dataset_hash` — sha256 over sorted `item_id::text`, first 12 hex |
| Row alignment | `item_ids` stored inside each `.npz`; asserted against the stimulus file before analysis |
| Fallback recording | `used_fallback` stored per item; a `claim_end` that became a final-token capture is never silent |

### 2.2 Model load precision — `bfloat16` (BINDING)

Weights are loaded at `bfloat16`; activations are still **written** as
`float32`, so nothing downstream changes shape or dtype.

This is a hardware constraint made explicit, not an optimization.
`google/gemma-2-2b` ships fp32 weights (10.46 GB); the capture machine has
16 GB of RAM, and transformer_lens needs transient headroom during weight
conversion, so an fp32 load risks swapping or OOM mid-capture. bf16 halves the
resident footprint to ~5.2 GB.

It is declared because **it changes the numbers.** Activations captured at
bf16 and fp32 are not identical, and `dataset_hash` cannot detect the
difference — it hashes stimulus text, not the model stack. The dtype is
therefore recorded in every `manifest.json`, and `capture_activations` refuses
a cache hit whose recorded dtype differs from the requested one.

### 2.3 Weight processing — none (BINDING)

Weights are loaded with `from_pretrained_no_processing`, not
`from_pretrained`.

TransformerLens's default loader applies weight processing: folding LayerNorm
into subsequent weights, centering writing weights, and centering the unembed.
`center_writing_weights` alters `resid_post` directly — the exact tensor this
project caches — so the two loaders yield different activations for the same
sentence, and every direction here is computed from those values.

The library itself warns that at reduced precision the folding arithmetic
degrades and advises the no-processing loader. bf16 (§2.2) therefore forces the
choice: processing-plus-bf16 is the unsound pairing.

Consequence to state plainly: `resid_post` here is **unprocessed**, so it is
not directly comparable to TransformerLens results that use the default loader.
Recorded in every `manifest.json` as `weight_processing`.

### 2.1 Primary token position — `claim_end` (BINDING)

`claim_end` is the primary. `final` is a robustness check. Declared before
extraction so neither can be chosen after seeing which flatters the result.

Basis, measured on the generated stimuli with the Gemma tokenizer — distance
from the end of the claim to the final token:

| template | gap (tokens) |
|---|---|
| t1 reported_belief | **16.75** (15–19) |
| t2 observational | **1.00** |
| t3 interrogative_embedded | **6.00** |
| t4 narrative | **1.00** |
| t5 plain_declarative | **5.50** (5–6) |

A 1–19 token spread determined by template identity. At `final`, a
template-generalization failure could not be distinguished from a
readout-position artifact. Rationale in `docs/method.md` §3.

Stated limitations: `claim_end` equalizes semantic position, not absolute depth;
and in some items the resolved token is a subword fragment.

---

## 3. Extraction and evaluation

| commitment | value |
|---|---|
| Direction | difference of class means, affirm-ward, unit-normalized |
| Threshold | midpoint of class-mean projections; **not** optimized |
| Threshold fitting | TRAIN items only, applied frozen to eval |
| Train templates | 1–5 |
| Held-out templates | 6–7 — never touch extraction or threshold fitting |
| Chance band | `0.5 + alpha_sd × √(0.25/n)`, **`alpha_sd = 2.0`** |
| Layer selection | **highest margin over the chance band** (margin = accuracy − band). **Ties broken by the layer nearest the middle of the eligible range**; remaining ties resolve to the lower layer. |
| Analysis classes | `affirm` / `deny` only. No third class is scored. |

**On the selection rule.** Margin and raw accuracy rank identically for a
fixed held-out set, since the band is then a constant; margin is the declared
quantity because the two diverge as soon as held-out sets of different sizes are
compared (t6 alone vs t7 alone vs pooled — all three are reported, per §5).

**On the tie-break.** `np.argmax` returns the first maximum, i.e. the lowest
layer. A saturated sweep would therefore select layer 0, which reads token
identity rather than accumulated meaning — the worst available choice, arrived
at silently. "Nearest the middle of the eligible range" is deterministic, is
fixed before any data is seen, and cannot be steered by the result. The eligible
range is the set of layers that were successfully swept, not `0..n_layers-1`.

**On `affirm`/`deny` in the first-person sets:** the column names are
structural. In `first_person.csv` and `referent_ladder.csv`, `affirm_text` is
the *experiential* member and `deny_text` the *mundane* member. The mundane
class is scored as `deny`; there is no separate neutral class in the declared
analysis.

---

## 4. Control battery

Run at the selected layer. All reported regardless of outcome.

| control | criterion |
|---|---|
| `cosine(mind, placebo)` | near zero — placebo shares entities, templates, denial devices, frames, and length distribution; only the property is non-mental |
| `cosine(mind, negation_control)` | near zero — high means a negation detector was built, and first-person results would be uninterpretable |
| Placebo direction on **mind** items | at chance. (Its accuracy on its own data is expected to be high and is not reported as a control.) |
| Adjacent-layer cosine | stability diagnostic, reported without a threshold |

No safety-direction control is claimed. It was considered and **struck** because
the dataset was never built; it must not appear in any writeup as an unrun
control.

---

## 5. Confound mitigations

**Length asymmetry — measured, no action.** Deny is longer in 66.7% of pairs,
but a length-only classifier reaches **0.512** (chance). `deny_longer_frac` is
reported for every set.

**Denial-device leak across the train/held-out boundary — action binding.**
`'nothing more'` appears in training template 5 (20%) and held-out template 7
(50%). t6 uses none of the tracked devices.

Both parts binding:

1. **t6 and t7 are reported separately**, always, in addition to pooled. If
   t7 ≫ t6, pooled held-out accuracy is **not** reported as the generalization
   result.
2. **A sensitivity extraction excluding t5** (train on t1–t4) is run and
   reported alongside the primary.

Regeneration of t7's deny tail occurs only if t7 clearly outperforms t6, and
would invalidate the held-out `dataset_hash`.

---

## 6. Data-integrity gates

Enforced before any forward pass. `--force` does not override them: they are
statements about the stimuli being wrong, not the cache being stale.

| gate | threshold |
|---|---|
| duplicate texts | none permitted |
| polarity balance, overall and per template | affirm fraction within 0.45–0.55 |
| top denial device share | ≤ `1/n_templates + 0.10` (§6.1) |
| `claim_end` resolution failures | ≤ 20% of items |

### 6.1 AMENDMENT — denial-device gate, 2026-08-16

**Original, as committed:** top denial device share ≤ **0.30**, flat, for every
stimulus file.

**Amended to:** ≤ **`1/n_templates + 0.10`**, with a floor of 0.30 for files
having no template structure (see below).

**Why the original was unreachable.** A denial device confined to a single
template is structurally forced to 1/k of that file's deny items, where k is
the number of templates in the file. The held-out file contains exactly two
templates (t6, t7), and t7's frame ends `"...and nothing more,"` — so
`'nothing more'` is pinned at 0.50 no matter how the stimuli are written. A
flat 0.30 is unreachable for any k < 4. The gate was firing on arithmetic, not
on a defect in the data.

**What the amendment preserves.** The +0.10 is margin above what template
structure forces, so genuine concentration still fires: at k=2 the ceiling is
0.60, and a device at 0.90 in a two-template file would still stop the run. At
k=5 the ceiling is 0.30 — **identical to the original number** — so no file
that passed before is affected.

**Exception at k=1.** The formula yields 1.10 for files with no `template_id`
(`first_person.csv`, `referent_ladder.csv`), which no share can exceed and
which would disable the gate entirely on the H1 stimuli. Those files have no
template structure to appeal to, so nothing forces a device to dominate and
concentration is a real defect. The original flat 0.30 stands there. Both files
currently sit at 0.000.

**Provenance, stated so it can be checked rather than trusted.**

- The mis-calibration was recorded in `docs/open_items.md` §9 **before any
  capture was attempted**, as a known structural problem with the gate — not
  discovered while trying to get a specific file to pass.
- This amendment **predates any capture of the held-out set.** No held-out
  activations existed when it was written, so no result influenced it. The
  contrast-pairs capture running at the time of writing is unaffected: at k=5
  its ceiling is unchanged at 0.30.
- The original threshold is preserved above rather than overwritten.

The honest summary: the original number was wrong for two-template files, and
recording that is worth more than pretending it was right.

---

## 7. Stimulus construction

- Entity balanced by construction — every entity appears in both halves of every
  pair it generates.
- Matched clauses between affirm and deny halves.
- Minimal-edit justifications.
- Held-out templates fenced in code (`build_rows` raises without
  `allow_heldout=True`).
- **`the assistant` is excluded from extraction entities.** A self-referential
  entity in the extraction set pre-loads self-relevance into the direction
  before transfer is tested.
- Claim spans are literal slices of the template pattern, filled from the same
  slot values, and asserted verbatim at build time.

---

## 8. Analysis order (BINDING)

Running these in a different order than declared is a researcher degree of
freedom, so the order is fixed here.

1. `run_cache.py` on contrast pairs — gates must pass
2. `run_cache.py` on placebo and negation control
3. `run_e1.py` — layer sweep, argmax selection, control battery, Kill-1
4. Held-out reported **pooled, t6 alone, and t7 alone** (§5)
5. Sensitivity extraction excluding t5 (§5)
6. Only then: H1 first-person transfer

No step is skipped on the grounds that an earlier one looked good. If Kill-1
fails, step 6 is not run.

---

## 9. Reporting

- Both probes always reported together (§1).
- Every control reported regardless of outcome (§4).
- Absent controls recorded as absent, never as passed.
- The chance band accompanies every accuracy.
- Effect size (Cohen's d) accompanies every accuracy in the writeup.
- Results are timestamp-versioned; nothing in `results/` is overwritten.
