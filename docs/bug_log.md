# Bug log

Every defect found in this codebase, with **what it would have done to the
results**. Append-only; entries are not deleted when fixed.

The common thread: nearly every one produced **plausible, non-crashing, wrong
output**. None of them raised an exception in normal use. That is the failure
mode this project is structurally exposed to — almost every quantity here is a
number between 0 and 1, and none of them look broken.

Several were caught by a check that existed for a different reason, or by
running in an environment that differed from the usual one. That is recorded
where it happened, because it is the argument for keeping those checks.

| # | defect | would have caused | status |
|---|---|---|---|
| 1 | claim lookup collapsed 35 phrases into 1 | 97.1% silent fallback to final-token | fixed |
| 2 | `char_end == 0` captured `<bos>` | activations from outside the sentence, unflagged | fixed |
| 3 | untested token-resolution branch | the path that actually runs was never exercised | fixed |
| 4 | duplicate `summarize_projections` | mundane controls crash or plot, depending on import | fixed |
| 5 | schema typed CSV fields as integer | every row rejected | fixed |
| 6 | `run_e1` template ids mismatched | empty split, or silent training on a subset | fixed |
| 7 | tests faked `transformer_lens` but not `torch` | tests silently required the multi-GB stack | fixed |
| 8 | bare `pytest` could not collect | green locally, uncollectable in CI | fixed |
| 9 | two acceptance tests were unachievable / no-ops | false confidence from tests that could not fail | fixed |
| 10 | figure plotted the uninformative placebo series | a control that cannot fail, presented as a control | fixed |

---

## 1. Claim-phrase lookup collapsed 35 phrases into one — 97.1% silent failure

**Found:** running `annotate_claim_ends` on the first real generated stimuli.

`annotate_claim_ends` looked phrases up by `(claim_id, polarity)`. But one
`claim_id` renders **35 distinct phrases** — 7 entities × 5 templates — because
the rendered phrase embeds both the subject and the template's wording:

```
claim_id 'consciousness' ->
    'Maya genuinely has consciousness'
    'Maya plainly has consciousness'
    'Maya truly has consciousness, possessing consciousness'
    ... 32 more
```

A dict keyed on `claim_id` keeps whichever was written last and discards the
other 34.

**Measured:** 408/420 sentences failed (97.1%). Templates 1–4 at exactly 100%,
t5 partially surviving — the uniform-100%-except-one shape is what identified it
as structural rather than data noise.

**Would have caused:** every failed item silently falls back to final-token
capture. Not an error — a matrix containing a *mixture* of two token positions,
with the mixture determined by template. Exactly the confound `claim_end` exists
to remove, reintroduced invisibly.

**Why it was not caught earlier:** the key was reasonable when written. It
predates claims being rendered per row; at that time one `claim_id` really did
mean one phrase.

**Fix:** the row's own `affirm_claim` / `deny_claim` column wins; the lookup
remains as a fallback for files without claim columns. Now 0 failures across all
six pair files.

**Coverage:** 3 regression tests, each verified to **fail** against the pre-fix
code. A regression test that passes under the buggy code is theater.

---

## 2. `char_end == 0` silently captured the BOS token

**Found:** by an exhaustive sweep of every character offset through both
token-resolution paths, looking for disagreements.

The prefix path computes `1 + len(prefix_tokens) - 1`. With `char_end == 0` that
is `1 + 0 - 1 = 0`, and index 0 is `<bos>` — a marker that is not part of the
sentence at all. The offset-mapping path handled it correctly (no token starts
before character 0, so it reported unresolvable). The two paths **disagreed**.

**Would have caused:** a real activation vector, from a token outside the
sentence, with `used_fallback` **not** set. Plausible numbers from the wrong
place, with nothing marking them as suspect.

**Fix:** `cache.py` rejects `char_end` that is 0, negative, or past end-of-text,
and records a fallback.

**The general principle it produced:** `char_end` is computed in a different
module. `find_claim_end` does guard against emitting 0 — but that guard is one
function away in another file, and nothing enforces the connection. A
hand-edited CSV could reintroduce it tomorrow. **`cache.py` does not trust an
offset it did not compute.**

---

## 3. The token-resolution branch that actually runs had zero coverage

**Found:** while asking whether the prefix path was ever exercised against a
real tokenizer.

The test fixture used a stub tokenizer that split on whitespace. With that stub,
token boundaries and word boundaries always coincide, so the arithmetic *cannot*
fail — the tests verified a scenario in which bugs were impossible. Worse, the
stub had no offset map, so every test took the **prefix** path. Gemma's
tokenizer is fast and provides offsets, so real runs take the **offset-mapping**
path.

**The tested path was the one that would never run; the untested path was the
one that would.**

**Fix:** a real byte-level BPE tokenizer trained in-process (offline, genuine
offsets), plus the real Gemma tokenizer when available, plus a **disagreement
test**: assert the two paths return identical indices. That is stronger than
asserting a hand-computed answer, because a hand-computed expectation can share
the same `<bos>` misunderstanding as the code. Agreement between independent
mechanisms is evidence; agreement with your own assumption is not.

**Known residual, documented not fixed:** on the real Gemma tokenizer the paths
diverge at **28/244 mid-word offsets** (0/46 at word boundaries). Truncating
mid-word makes BPE re-segment the fragment, so the prefix path over-counts.
`find_claim_end` returns "just past a phrase," which always lands on a word
boundary, so real data cannot reach it. A test *searches* for mid-word
divergences and asserts some exist, so that if a future tokenizer change removes
them, the stale documentation fails loudly.

---

## 4. Duplicate `summarize_projections` with incompatible contracts

**Found:** while reviewing the two modules side by side.

Two copies existed: a 0/1-only version in `diff_means.py` that **raised** on
`label == 2`, and a neutral-aware version in `linear_probe.py`. Whichever copy
the figure code happened to import decided whether the mundane controls plotted
or crashed.

**Would have caused:** an import-path-dependent crash on exactly the data the
neutral handling exists to serve.

**Fix:** collapsed into one canonical implementation in `linear_probe.py`,
keeping the stricter validation the deleted copy had (float64 coercion, label
vocabulary check) and deliberately keeping the more permissive missing-class
behaviour (nan plus counts, rather than raising mid-report). A regression test
asserts the twin does not come back, since the failure was import-dependent and
"restoring" the function to where the docs said it lived is a plausible future
mistake.

---

## 5. Schema typed CSV fields as `integer`

**Found:** first attempt to run `run_cache.py` against real stimulus files.

```
row 2: schema violation: '1' is not of type 'integer'
```

`csv.DictReader` has no type system — every field arrives as `str`. A JSON
Schema `{"type": "integer"}` can therefore **never** match a CSV field. Both
`template_id` and `mindedness` were typed this way, so every row of every file
was rejected.

A second layer sat behind it: `additionalProperties: false` with
`affirm_claim` / `deny_claim` undeclared, which would have rejected all rows
once the first issue was fixed. Present-but-undeclared is worse than absent —
the generators emit them on all 998 sentence-halves.

**Fix:** `mindedness` → string; `template_id` → `{"type": "string", "pattern":
"^\\d+$"}`, preserving the intent while matching reality; claim columns
declared.

---

## 6. `run_e1.py` template ids would have matched zero rows

**Found:** while checking whether a proposed mitigation (reporting t6 and t7
separately) was even runnable.

Defaults were `--train-templates t1,t2,t3,t4,t5`; the generators write
`template_id` as bare digits `1,2,3,4,5`.

**Would have caused:** `np.isin` matches nothing, `split_indices` raises "empty
split" at the first real E1 run. The louder failure is the lucky case — had
*some* ids matched (a mixed-format file), it would have trained on a silent
subset.

**Fix:** `t5` and `5` normalized to the same template on both sides. The
train/held-out overlap guard still fires.

---

## 7. Tests faked `transformer_lens` but not `torch`

**Found:** by running the suite in a freshly created venv with only the analysis
dependencies installed. Green in the usual environment, three failures here.

`capture_activations` imports `torch` and `transformer_lens` in the same `try`
block. The fixtures faked only the latter. In the normal environment real torch
was installed and silently satisfied the import.

**Would have caused:** nothing to the science — but the tests' own docstrings
claimed they ran without the model half, and they did not. A CI job installing
only the analysis extra would have failed with an error implying a code bug.

**Fix:** fake `torch.no_grad` as well (a null context is a complete stand-in),
guarded so it never shadows a real torch.

**Note:** only an environment change could surface this. It is the argument for
running the suite in more than one environment.

---

## 8. Bare `pytest` could not collect the suite

**Found:** SP ran `pytest` directly; every module failed with
`ModuleNotFoundError: No module named 'lib'`.

`pytest` inserts a test module's first parent *without* an `__init__.py` —
`tests/`, not the repo root. `python -m pytest` prepends the CWD, which is why
every prior run worked. In the venv it was masked a second way, because
`pip install -e .` puts `lib` on the path outright.

**Would have caused:** a suite that is green locally under one specific
invocation and uncollectable in CI — failing at path resolution rather than at
anything real.

**Fix:** a root `conftest.py`, whose directory pytest inserts by the same rule.
Verified under bare `pytest` and `python -m pytest`, in both environments.

---

## 9. Two acceptance tests were unachievable or no-ops

**Found:** implementing the specified acceptance tests and watching them fail /
trivially pass.

**(a) The NULL test as specified** — "20 seeds, all inside `chance_band(n)`" —
is **unachievable by construction**. The band *is* 2 sd, so ~2–3% of pure-noise
draws exceed it (measured 7/200). Demanding 20/20 is demanding the band be
miscalibrated. Rewritten as a rate test: mean at chance, at most 2 excursions.

**(b) The NO-LEAK test** at the obvious parameters (n=20/class, weak signal)
showed **0.000** optimism from fitting the threshold on eval — the midpoint
threshold is estimated well enough at that n that there is nothing to steal. It
would have passed while testing nothing. Re-tuned to n=8/class where the effect
is ~0.015 over 400 seeds.

**Would have caused:** false confidence. (a) fails randomly ~40% of the time,
training everyone to ignore it; (b) passes forever without exercising the leak
it names.

---

## 10. The figure plotted the uninformative placebo series

**Found:** by rendering the chart and looking at it.

The layer sweep plotted the placebo direction's accuracy **on placebo's own
data** — which is expected to be high at every layer and carries no information
about the mind result. The informative series is the placebo direction scored on
**mind** items, which must sit at chance.

**Would have caused:** a figure presenting a control that cannot fail as though
it were a control.

**Fix:** replaced with placebo-on-mind, computed per layer. Also found in the
same pass: coincident series painted over each other (fixed with draw order and
dash patterns) and colliding direct labels (fixed with vertical de-collision).

---

## Test-fixture defects (mine, caught before they mattered)

Recorded because they are the same class of error as the above:

- A smoke fixture placed the claim at the **end** of the sentence, so
  `claim_end` trivially equalled `final`. The test would have passed while
  proving nothing about position resolution.
- A hardcoded mid-word offset happened to **agree** across both token paths,
  so the divergence test asserted the opposite of what it claimed. Replaced
  with a search over all mid-word offsets.
- A length-gap fixture was off by one word (13 vs 14), producing a −7.0 gap
  where the test expected −6.0.
