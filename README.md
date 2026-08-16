# mind-attribution

Does a language model represent "this entity has genuine subjective
experience" as a linear direction in its activation space — and if so, does
that same direction fire when the model talks about *itself*?

This repo is the analysis harness for that question, run against
`google/gemma-2-2b` (via TransformerLens). It captures activations for
matched affirm/deny sentences about minds, extracts a candidate direction with
difference-of-means, and scores it with two probes designed to fail loudly
and separately, so a positive result can't be explained away as a generic
sentiment or self-reference detector.

## Status

Early stage. The extraction, caching, and probing code paths are implemented
and unit-tested; no stimulus data, activation cache, or results exist in the
repo yet. `PREREGISTRATION.md` is a placeholder — the hypotheses, held-out
kill criteria, and primary-token-position decision need to be written and
committed *before* any real data is captured, per the project's own rule
(see `lib/harness/cache.py` and `lib/harness/tokenization_notes.md`).

## How the pieces fit together

```
stimuli.py  →  cache.py  →  diff_means.py  →  linear_probe.py
(sentences)    (activations)  (the direction)   (does it hold up?)
```

- **`lib/harness/stimuli.py`** — the only entry point for stimulus data.
  Loads affirm/deny sentence pairs from a schema-validated CSV, expands each
  pair into two rows, locates where the "mind claim" ends in each sentence,
  and reports balance diagnostics (length gaps, denial-phrase overuse,
  polarity skew per template/entity) that a hidden confound could exploit.
  Schema violations raise; distributional judgment calls are reported, not
  enforced, so a human decides whether the stimulus set is clean enough.

- **`lib/harness/cache.py`** — the only file that touches the model. One
  forward pass per sentence captures `resid_post` at every layer, at two
  token positions (`final`, and `claim_end` — the token where the mind-claim
  itself ends, which means the same thing across differently-shaped
  templates; `final` doesn't). Results are cached to `.npz` by a hash of the
  stimulus text, so editing a sentence invalidates the cache automatically.
  See `tokenization_notes.md` for why `claim_end` exists and the token-offset
  bugs that motivated its validation.

- **`lib/extraction/diff_means.py`** — pure numpy, no model or I/O. Turns two
  piles of cached activations into a single direction
  (`mean(affirm) − mean(deny)`), projects new activations onto it, and
  computes cosine similarity against control directions (placebo, refusal,
  sentiment) to check the direction isn't just detecting something boring.
  Full rationale and edge-case reasoning in
  `lib/extraction/diff_means_explanation.md`.

- **`lib/probes/linear_probe.py`** — two probes that must never be merged:
  a **direction probe** (thresholds the frozen extracted direction, no
  fitting on eval data — the actual hypothesis test) and a **ceiling probe**
  (cross-validated logistic regression, free to use any direction — the
  upper bound on available signal). The gap between them on first-person
  data is the headline result: small gap implies shared machinery between
  self- and other-attribution; a large gap implies self-specific machinery.

## Running tests

```
pytest tests/
```

`tests/test_cache.py` builds a real byte-level BPE tokenizer in-process (and
optionally checks against the real Gemma tokenizer, skipped if offline) so
the token-offset logic is tested against genuine subword splitting rather
than a space-splitting stand-in.

## Dependencies

`numpy`, `scikit-learn`, `jsonschema`, `pytest`. `torch` and
`transformer_lens` are only required to run `capture_activations` (the one
function that touches the model) — everything downstream is plain numpy over
cached `.npz` files, so analysis and tests run without either installed.
