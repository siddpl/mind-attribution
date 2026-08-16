# Environment

Pinned versions, why they are pinned, and what changes if you unpin them.
Update this file whenever a pin moves.

Definitions: [`glossary.md`](glossary.md) · Rationale: [`method.md`](method.md)

---

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"        # analysis half — no model stack
.venv/bin/pip install -e ".[capture]"    # adds torch + transformer_lens (~1.2 GB)
```

Use the explicit `.venv/bin/` prefix. `source .venv/bin/activate` does not
persist across separate shell invocations, and a bare `pip` will silently
install into the base interpreter instead.

The split is deliberate: **55 of the 60 tests need only the analysis half** and
run in ~7 seconds. Only `lib/harness/cache.py` touches the model stack.

| extra | contents | suite result |
|---|---|---|
| `[dev]` | numpy, scipy, scikit-learn, jsonschema, matplotlib, pytest | 43 passed, 9 skipped, 5 deselected |
| `[capture]` | + torch, transformers, tokenizers, transformer-lens | 55 passed, 5 deselected |
| `-m slow` | (needs `[capture]` + weights) | 5 passed on `gpt2` |

---

## The pins that are forced, not chosen

### `transformer_lens==3.7.2` requires `transformers>=5.9.0`

This is an upstream requirement, not a preference. Consequences:

**`EleutherAI/pythia-70m` cannot load.** transformer_lens's NeoX weight
conversion reads `neox.embed_out`, which transformers 5.x restructured:

```
AttributeError: 'GPTNeoXForCausalLM' object has no attribute 'embed_out'
```

Use `gpt2` for small-model smoke tests. It is what all pipeline verification
ran on. Getting pythia back means transformer_lens 2.x + transformers 4.x — a
different pin set, not a tweak.

**Installing this into a shared environment breaks other projects.** The
transformers 4.x → 5.x jump broke `control-arena`, `vllm`, `inspect-ai`, and
`streamlit` in the base anaconda install. This is the reason for the venv.

### `HookedTransformer` is deprecated — and unpinning could move the numbers

transformer_lens 3.7.2 warns that `HookedTransformer` will be removed in 4.0 in
favour of `TransformerBridge.boot_transformers(...)`, with *"HookedTransformer-
equivalent numerics"* available only under an explicit compatibility mode.

**This is a scientific hazard, not a maintenance note.** If activations shift
under the new API:

- `dataset_hash` **will not detect it.** The hash covers stimulus text, not the
  model stack. A cache captured before the change and one captured after would
  carry the same hash and different numbers.
- Captures taken across a version change are not comparable, and nothing in the
  pipeline would say so.

`lib/harness/cache.py` has exactly one call site (`HookedTransformer.from_pretrained`).
If the pin ever moves, **re-capture everything** and record the change here.

---

## Model access

| model | status |
|---|---|
| `google/gemma-2-2b` | **license-gated.** Weights 403 without accepting Google's terms on huggingface.co. The analysis model. |
| `unsloth/gemma-2-2b` | ungated mirror — **tokenizer only.** Not in transformer_lens's model registry, so it cannot be `SMOKE_MODEL` or `--model`. Used by the tokenizer tests. |
| `gpt2` | ungated, works, 12 layers / d_model 768. All pipeline verification. |
| `EleutherAI/pythia-70m` | **broken** with this pin set (see above). |

`get_pretrained_model_config("google/gemma-2-2b")` resolves without
authentication (26 layers, d_model 2304) — only the weights are gated. So a
capture will get past the config check and fail at the download.

---

## Node

`node` (Homebrew) is required only to run the data-viz palette validator. It is
not a project dependency. `pip install node.js` does not work — node is not a
Python package.

---

## Reproducing a run

Recorded per capture in `results/<model>/<hash>/manifest.json` and appended to
`results/run_log.jsonl`: model, dataset hash, n_items, n_layers, d_model,
positions, per-position fallback counts, UTC timestamp. Direction files carry a
sidecar with the **git commit**.

What is *not* captured automatically, and must be recorded here by hand when it
changes: the pinned versions above.
