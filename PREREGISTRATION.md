# Preregistration — mind-attribution direction probing

**This file contains commitments only.** Rationale lives in
[`docs/method.md`](docs/method.md), definitions in
[`docs/glossary.md`](docs/glossary.md), defects in
[`docs/bug_log.md`](docs/bug_log.md), pinned versions in
[`docs/environment.md`](docs/environment.md), and unresolved decisions in
[`docs/open_items.md`](docs/open_items.md). Those documents are revisable. This
one is not, after it is committed.

**Status: DRAFT.** Three items in `docs/open_items.md` (§ "Blocking a real
capture") must be resolved and folded in here before any capture on the
analysis model. The commit that finalizes this file is the timestamp of record.

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
| Precision | `float32` regardless of model dtype |
| Generation | none, anywhere. Forward passes only. |
| Gradients | `torch.no_grad()` around all passes |
| Cache key | `dataset_hash` — sha256 over sorted `item_id::text`, first 12 hex |
| Row alignment | `item_ids` stored inside each `.npz`; asserted against the stimulus file before analysis |
| Fallback recording | `used_fallback` stored per item; a `claim_end` that became a final-token capture is never silent |

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
| Layer selection | **argmax of held-out accuracy.** Tie-break: unresolved (`open_items` §2) |
| Chance band | `0.5 + alpha_sd × √(0.25/n)`; `alpha_sd` unresolved (`open_items` §1) |
| Analysis classes | `affirm` / `deny` only. No third class is scored. |

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
| top denial device share | ≤ 0.30 |
| `claim_end` resolution failures | ≤ 20% of items |

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
