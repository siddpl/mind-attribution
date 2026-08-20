# Analysis Pipeline (`lib/`)

Once stimulus data exists (in `data/`), the analysis harness in `lib/` runs against it. It is built to analyze models like `google/gemma-2-2b` via TransformerLens.

The pipeline flows as follows:
```
stimuli.py  →  cache.py  →  diff_means.py  →  linear_probe.py
(sentences)    (activations)  (the direction)   (does it hold up?)
```

## Components

- **`harness/stimuli.py`** — the only entry point for stimulus data.
  Loads affirm/deny sentence pairs from a schema-validated CSV, expands each pair into two rows, locates where the "mind claim" ends in each sentence, and reports balance diagnostics (length gaps, denial-phrase overuse, polarity skew per template/entity) that a hidden confound could exploit. Schema violations raise; distributional judgment calls are reported, not enforced, so a human decides whether the stimulus set is clean enough.

- **`harness/cache.py`** — the only file that touches the model. One forward pass per sentence captures `resid_post` at every layer, at two token positions (`final`, and `claim_end` — the token where the mind-claim itself ends, which means the same thing across differently-shaped templates; `final` doesn't). Results are cached to `.npz` by a hash of the stimulus text, so editing a sentence invalidates the cache automatically.

- **`extraction/diff_means.py`** — pure numpy, no model or I/O. Turns two piles of cached activations into a single direction (`mean(affirm) − mean(deny)`), projects new activations onto it, and computes cosine similarity against control directions (placebo, refusal, sentiment) to check the direction isn't just detecting something boring.

- **`probes/linear_probe.py`** — two probes that must never be merged: a **direction probe** (thresholds the frozen extracted direction, no fitting on eval data — the actual hypothesis test) and a **ceiling probe** (cross-validated logistic regression, free to use any direction — the upper bound on available signal). The gap between them on first-person data is the headline result: small gap implies shared machinery between self- and other-attribution; a large gap implies self-specific machinery.

## Sub-Documentation
- [`harness/tokenization_notes.md`](harness/tokenization_notes.md): Explains why `claim_end` exists and the token-offset bugs that motivated its validation.
- [`extraction/diff_means_explanation.md`](extraction/diff_means_explanation.md): Full rationale and edge-case reasoning for extraction.
- [`logit/logit_README.md`](logit/logit_README.md): Logit analysis details.
