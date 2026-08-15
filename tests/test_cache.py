"""
tests/test_cache.py — the acceptance tests from the cache.py spec.

Tests 1-4 are pure numpy-on-disk and must run WITHOUT transformer_lens.
Test 5 checks the ImportError message is actionable when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from lib.harness import cache
from lib.harness.cache import (
    capture_activations,
    dataset_hash,
    is_cached,
    load_activations,
    save_activations,
)

ITEMS = [
    {"item_id": "a1", "text": "I feel a flicker of curiosity."},
    {"item_id": "d1", "text": "I do not feel anything at all."},
    {"item_id": "m1", "text": "I was trained on text."},
]


@pytest.fixture(autouse=True)
def _isolated_cache_root(tmp_path, monkeypatch):
    """Point CACHE_ROOT at a temp dir so tests never touch real results/."""
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "activation_cache")


# 1. save -> load round-trips acts and item_ids exactly, order preserved
def test_save_load_round_trip():
    rng = np.random.default_rng(0)
    ids = ["a1", "d1", "m1"]
    acts = rng.normal(size=(3, 8)).astype(np.float32)

    save_activations("google/gemma-2-2b", "abc123def456", 4, ids, acts)
    loaded_ids, loaded_acts = load_activations("google/gemma-2-2b", "abc123def456", 4)

    assert loaded_ids == ids  # exact order, not just same set
    np.testing.assert_array_equal(loaded_acts, acts)
    assert loaded_acts.dtype == np.float32


def test_save_rejects_row_mismatch():
    with pytest.raises(ValueError, match="item_ids"):
        save_activations("m", "h", 0, ["a1", "d1"], np.zeros((3, 4)))


# 2. changing ONE character in ONE sentence changes dataset_hash
def test_hash_changes_with_one_character():
    edited = [dict(it) for it in ITEMS]
    edited[1]["text"] = edited[1]["text"].replace("anything", "anythinG")

    assert dataset_hash(ITEMS) != dataset_hash(edited)
    assert dataset_hash(ITEMS) == dataset_hash(list(reversed(ITEMS)))  # order-free
    assert len(dataset_hash(ITEMS)) == 12


# 3. load returns None for an uncached layer, does not raise
def test_load_uncached_returns_none():
    assert load_activations("google/gemma-2-2b", "deadbeef0000", 7) is None


# 4. is_cached is False when one layer file is missing
def test_is_cached_requires_every_layer():
    acts = np.zeros((3, 8), dtype=np.float32)
    ids = ["a1", "d1", "m1"]
    for layer in [0, 1, 3]:  # layer 2 deliberately missing
        save_activations("m", "h", layer, ids, acts)

    assert not is_cached("m", "h", 4)
    save_activations("m", "h", 2, ids, acts)
    assert is_cached("m", "h", 4)


# 5. capture_activations raises a clear, actionable ImportError
def test_capture_without_transformer_lens_raises_actionable_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("transformer_lens"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="runs LOCALLY.*not in a sandbox"):
        capture_activations(ITEMS)
