# Open items

Decisions and known gaps. **This is a tracker, not a commitment** — items here
get resolved and edited. Anything binding lives in
[`../PREREGISTRATION.md`](../PREREGISTRATION.md) and does not change after it
is committed.

Status: `OPEN` · `RESOLVED` (with what it was resolved to) · `STRUCK`

---

## Blocking a real capture

| # | item | status |
|---|---|---|
| 3 | **Model access.** `google/gemma-2-2b` is license-gated and 403s. No transformer_lens-compatible ungated mirror exists. | **OPEN** |

## Resolved

| # | item | resolution |
|---|---|---|
| 4 | Primary token position | **RESOLVED — `claim_end`.** Binding; see PREREGISTRATION §2.2. Measured 1–19 token template-dependent gap at `final`. |
| 5 | Safety-direction control | **STRUCK.** Dataset was never built. Removed from the control battery rather than carried as an unrun control. Placebo and negation control remain. |
| 6 | NEUTRAL polarity / `neutral_mean` | **STRUCK** from the preregistered analysis. `stimuli.py` emits only `affirm`/`deny`; the mundane class is carried as `deny_text` in the first-person and referent-ladder sets, where `deny` means *mundane*, not *denies*. `NEUTRAL = 2` remains supported in `linear_probe.py` for future use but is not part of the declared analysis. |
| 1 | `alpha_sd` | **RESOLVED — 2.0.** Declared as-is in PREREGISTRATION §3. |
| 2 | Layer selection + tie-break | **RESOLVED — highest margin over the chance band, ties broken by the layer nearest the middle of the eligible range**, remaining ties to the lower layer. Implemented in `run_e1.best_layer`; a 12-way tie now selects L5 rather than L0. |
| 7 | Claim columns on first-person / referent-ladder generators | **RESOLVED.** Both now emit `affirm_claim`/`deny_claim`; 100% verbatim, 0% claim_end failure. |

## Known gaps, not blocking

| # | item | status |
|---|---|---|
| 8 | **`'nothing more'` crosses the train/held-out boundary** (t5 20%, t7 50%). Mitigation is preregistered (§2.5b: report t6/t7 separately + a t5-excluded sensitivity run). Regeneration of t7 happens only if t7 clearly outperforms t6. | **OPEN — mitigated** |
| 9 | ~~**Device gate mis-calibrated for 2-template files.**~~ The 30% threshold assumes a multi-template file; the held-out file has two templates, so any device used by one is automatically ≥50%. The gate currently stops the held-out dry run on a structural fact. Either exempt held-out files or scale by template count. | **RESOLVED 2026-08-16** — amended to `1/n_templates + 0.10`, floor 0.30 at k=1. See PREREGISTRATION §6.1. |
| 10 | **Multiple comparisons.** Layers × 2 positions × sets are all scored. Either state a correction or state that selection is by the preregistered rule alone and none is applied. | **OPEN** |
| 11 | **`audience_frames` and `task_battery` have no claim columns** — and structurally cannot: they are question/prompt batteries with no affirm/deny pairs and therefore no claim span. If they are ever projected onto a direction, the position question has to be answered separately for them. | **OPEN** |
| 12 | **`data/scripts/data_net.py`** arrived with the data merge and has not been reviewed. It may duplicate the gate thresholds in `run_cache.py`; two implementations of the same thresholds will drift. | **OPEN** |
| 13 | **`data/depricated/`** contains an older `schema.json`, `models.py`, and `validate.py`. Unclear whether anything still reads them. | **OPEN** |
| 14 | Mid-word token-path divergence (28/244 on real Gemma) | **ACCEPTED.** Unreachable from `find_claim_end`, documented in bug_log §3, pinned by a test that fails if it disappears. |
