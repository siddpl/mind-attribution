"""
tests/test_smoke_model.py — does the scanner half of the harness actually work?

Sentiment is the most linearly-decodable axis in any language model. If this
pipeline cannot find it, the bug is in hooks, token positions, layer indexing,
or caching — NOT in the hypothesis. That is the entire point: a failure here
is a plumbing failure, and it should be fixed before real data is ever scored.

ALL SLOW: these need model weights. `pytest` deselects them by default
(see pytest.ini); run them with `pytest -m slow`.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from lib.extraction.diff_means import extract_direction
from lib.harness import cache
from lib.harness.cache import capture_activations, load_activations
from lib.harness.stimuli import find_claim_end
from lib.probes.linear_probe import AFFIRM, DENY, direction_probe_accuracy, fit_threshold

pytestmark = pytest.mark.slow

# google/gemma-2-2b is LICENSE-GATED: weights 403 unless you have accepted
# Google's terms on huggingface.co and are logged in. NOTE: the ungated
# `unsloth/gemma-2-2b` mirror works for the TOKENIZER (see test_cache.py) but
# is NOT in transformer_lens's model registry, so it cannot be SMOKE_MODEL.
# Valid small alternatives for a pure plumbing check: "gpt2" (12 layers) or
# "EleutherAI/pythia-70m" (6 layers) — both ungated and quick to download.
MODEL = os.environ.get("SMOKE_MODEL", "google/gemma-2-2b")

# 5 matched pairs. Within each pair only the sentiment word changes, so word
# count and structure are held constant and only valence varies. Registers
# differ across pairs (plain, conversational, review, exclamatory, formal) so a
# result cannot ride on one sentence shape.
PAIRS = [
    ("This is wonderful.", "This is terrible.", "wonderful", "terrible"),
    ("Honestly, I found the experience delightful.",
     "Honestly, I found the experience miserable.", "delightful", "miserable"),
    ("The service was excellent, according to every review.",
     "The service was awful, according to every review.", "excellent", "awful"),
    ("What a fantastic result, everyone agreed.",
     "What a dreadful result, everyone agreed.", "fantastic", "dreadful"),
    ("I would describe the outcome as superb overall.",
     "I would describe the outcome as dismal overall.", "superb", "dismal"),
]


def _items() -> tuple[list[dict], np.ndarray]:
    """Build sentence rows with real claim-end offsets, plus their labels."""
    items, labels = [], []
    for i, (pos_text, neg_text, pos_word, neg_word) in enumerate(PAIRS):
        for text, word, pol, label in (
            (pos_text, pos_word, "aff", AFFIRM), (neg_text, neg_word, "den", DENY)
        ):
            items.append({
                "item_id": f"s{i}__{pol}",
                "text": text,
                "claim_end_char": find_claim_end(text, word),
            })
            labels.append(label)
    return items, np.array(labels)


ITEMS, LABELS = _items()


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    """Capture once, share across tests — a forward pass per test would be waste."""
    pytest.importorskip("transformer_lens", reason="capture needs transformer_lens")
    original_root = cache.CACHE_ROOT
    cache.CACHE_ROOT = tmp_path_factory.mktemp("smoke_cache")
    try:
        ds_hash = capture_activations(ITEMS, model_name=MODEL)
    except Exception as e:  # gated repo, offline, or out of memory
        cache.CACHE_ROOT = original_root
        pytest.skip(
            f"model weights unavailable for {MODEL} ({type(e).__name__}: {str(e)[:120]}); "
            f"try SMOKE_MODEL=gpt2 — note EleutherAI/pythia-70m is BROKEN with "
            f"transformers 5.x (NeoX conversion wants `embed_out`), and "
            f"unsloth/gemma-2-2b is a tokenizer-only mirror, not in transformer_lens"
        )
    import json
    manifest = json.loads((cache.cache_dir(MODEL, ds_hash) / "manifest.json").read_text())
    yield ds_hash, manifest
    cache.CACHE_ROOT = original_root


def test_capture_produces_expected_shapes(captured):
    """Breaking this means the cache is writing malformed or misaligned files,
    and every downstream number would be computed on the wrong rows."""
    ds_hash, manifest = captured
    d_model, n_layers = manifest["d_model"], manifest["n_layers"]

    for layer in range(n_layers):
        for position in cache.POSITIONS:
            got = load_activations(MODEL, ds_hash, layer, position)
            assert got is not None, f"missing file for layer {layer}, {position}"
            item_ids, acts, _ = got
            assert acts.shape == (10, d_model)
            assert acts.dtype == np.float32
            assert item_ids == [it["item_id"] for it in ITEMS]  # input order, exactly


def test_sentiment_direction_separates_perfectly(captured):
    """Breaking this means the harness cannot recover the easiest signal that
    exists in a language model — investigate hooks and layer indexing before
    pointing any of this at real stimuli."""
    ds_hash, manifest = captured
    layer = manifest["n_layers"] // 2
    _, acts, _ = load_activations(MODEL, ds_hash, layer, "final")

    # Deliberately in-sample: this is a plumbing check, not a generalization
    # claim, so there is nothing to leak into.
    direction = extract_direction(acts[LABELS == AFFIRM], acts[LABELS == DENY])
    threshold = fit_threshold(acts, LABELS, direction)

    assert direction_probe_accuracy(acts, LABELS, direction, threshold) == 1.0


def test_cache_is_reused_not_recomputed(captured, monkeypatch):
    """Breaking this means every rerun pays for a full CPU capture again —
    the weekend's scarcest resource spent on work already done."""
    import transformer_lens

    def explode(*a, **k):
        raise AssertionError("model was loaded despite a complete cache")

    monkeypatch.setattr(transformer_lens.HookedTransformer, "from_pretrained", explode)
    ds_hash, _ = captured
    assert capture_activations(ITEMS, model_name=MODEL) == ds_hash


def test_all_layers_present_and_distinct(captured):
    """Breaking this means a layer-indexing bug wrote one layer's activations
    into every file — which would pass every other test in this file."""
    ds_hash, manifest = captured
    for position in cache.POSITIONS:
        seen = []
        for layer in range(manifest["n_layers"]):
            got = load_activations(MODEL, ds_hash, layer, position)
            assert got is not None
            seen.append(got[1])
        for layer in range(1, len(seen)):
            assert not np.array_equal(seen[layer], seen[layer - 1]), (
                f"{position}: layer {layer} is identical to layer {layer - 1}"
            )


def test_positions_differ_where_expected(captured):
    """Breaking this means claim_end resolution is silently falling back to the
    final token, so the two positions would be one position wearing two names."""
    ds_hash, manifest = captured
    layer = manifest["n_layers"] // 2
    _, final_acts, _ = load_activations(MODEL, ds_hash, layer, "final")
    _, claim_acts, fallback = load_activations(MODEL, ds_hash, layer, "claim_end")

    assert not fallback.any(), "every fixture sentence has a resolvable claim phrase"
    assert manifest["fallback_counts"]["claim_end"] == 0
    # Every sentence ends with punctuation after the sentiment word, so claim_end
    # is a strictly earlier token than final for all ten.
    differing = [i for i in range(len(ITEMS))
                 if not np.array_equal(final_acts[i], claim_acts[i])]
    assert len(differing) == len(ITEMS), (
        f"only {len(differing)}/{len(ITEMS)} sentences differ between positions"
    )
