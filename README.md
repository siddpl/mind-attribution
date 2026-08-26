# Mind Attribution

A project for probing LLMs on mind attribution claims across different
entities, templates, perspectives, and polarities.

Concretely: does a language model represent "this entity has genuine
subjective experience" as a linear direction in its activation space — and if
so, does that same direction fire when the model talks about *itself*? The
repo has two halves: `data/` builds and validates the affirm/deny stimulus
sentences, and `lib/` turns cached activations for those sentences into a
candidate direction and tests whether it holds up.

## Repository Structure

The project is structured into three main pillars, each with its own detailed documentation:

1. **[`data/`](data/README.md)**: Builds, validates, and manages the affirm/deny stimulus sentences.
2. **[`lib/`](lib/README.md)**: The analysis harness. Turns cached activations for sentences into a candidate direction and tests whether it holds up.
3. **[`scripts/`](scripts/README.md)**: The executable entry points for running the end-to-end experiments.

## Additional Documentation

- **[`PREREGISTRATION.md`](PREREGISTRATION.md)**: Hypotheses and held-out kill criteria.
- **`docs/`**:
  - [`E4_jlens_README.md`](docs/E4_jlens_README.md)
  - [`bug_log.md`](docs/bug_log.md)
  - [`environment.md`](docs/environment.md)
  - [`glossary.md`](docs/glossary.md)
  - [`method.md`](docs/method.md)
  - [`open_items.md`](docs/open_items.md)

## Status

Stimulus generation and validation (`data/`) and the extraction/caching/
probing code paths (`lib/`) are implemented and unit-tested. No activation
cache or results exist in the repo yet. `PREREGISTRATION.md` is a
placeholder — the hypotheses, held-out kill criteria, and primary
token-position decision need to be written and committed *before* any real
data is captured, per the project's own rule (see `lib/harness/cache.py` and
`lib/harness/tokenization_notes.md`).

## Running tests

```bash
pytest tests/
```

`tests/test_cache.py` builds a real byte-level BPE tokenizer in-process (and
optionally checks against the real Gemma tokenizer, skipped if offline) so
the token-offset logic is tested against genuine subword splitting rather
than a space-splitting stand-in.

## Dependencies

`numpy`, `scikit-learn`, `jsonschema`, `pytest` (see
`data/scripts/requirements.txt` for the data-scripts environment). `torch`
and `transformer_lens` are only required to run `capture_activations` (the
one function in `lib/` that touches the model) — everything downstream is
plain numpy over cached `.npz` files, so analysis and tests run without
either installed.
