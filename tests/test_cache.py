"""
tests/test_cache.py — acceptance tests for cache.py, two-position edition.

Tests 1-4 are pure numpy-on-disk. Tests 6-9 run capture_activations against a
fake transformer_lens injected into sys.modules — the fake's activation values
encode the token position, so position selection is verifiable numerically.
Nothing here needs the real transformer_lens.
"""

from __future__ import annotations

import contextlib
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from lib.harness import cache
from lib.harness.cache import (
    capture_activations,
    cache_dir,
    dataset_hash,
    is_cached,
    load_activations,
    save_activations,
)

ITEMS = [
    {"item_id": "a1", "text": "the robot feels joy today", "claim_end_char": 20},
    {"item_id": "d1", "text": "the robot feels no joy today", "claim_end_char": None},
    {"item_id": "m1", "text": "the robot was trained on text", "claim_end_char": 23},
]

N_LAYERS, D_MODEL = 2, 4


@pytest.fixture(autouse=True)
def _isolated_cache_root(tmp_path, monkeypatch):
    """Point CACHE_ROOT at a temp dir so tests never touch real results/."""
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "activation_cache")


def _save(layer, position, n=3, fill=0.0):
    ids = [f"i{k}" for k in range(n)]
    acts = np.full((n, D_MODEL), fill, dtype=np.float32)
    save_activations("m", "h", layer, position, ids, acts, np.zeros(n, dtype=bool))
    return ids, acts


# =============================================================================
# The fake model half — token t's residual at layer L is (t + 100 * L) * ones
# =============================================================================
class _FakeCache:
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __getitem__(self, key):
        _, layer = key
        toks = np.arange(self.seq_len, dtype=np.float32) + 100.0 * layer
        return np.tile(toks[None, :, None], (1, 1, D_MODEL))


class _FakeModel:
    """Model shell with a swappable tokenizer.

    tokenizer=None exercises the PREFIX path; a real fast tokenizer exercises
    the OFFSET-MAPPING path — which is the branch that actually runs against
    gemma-2-2b, so it must not be the untested one.
    """

    def __init__(self, tokenizer=None):
        self.cfg = SimpleNamespace(
            n_layers=N_LAYERS, d_model=D_MODEL, default_prepend_bos=True
        )
        self.tokenizer = tokenizer

    def _n_content_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text.split())
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def to_tokens(self, text):  # BOS + content tokens, as transformer_lens does
        return np.zeros((1, 1 + self._n_content_tokens(text)))

    def run_with_cache(self, text):
        return None, _FakeCache(seq_len=1 + self._n_content_tokens(text))


@pytest.fixture(scope="session")
def real_tokenizer():
    """A REAL fast tokenizer, trained in-process — no network, no HF cache.

    Byte-level BPE, so `is_fast` is True and offset mappings are genuine. This
    is what gives the offset-mapping branch any coverage at all; the stub
    tokenizer splits on whitespace, where token and word boundaries can never
    disagree and the arithmetic cannot fail.
    """
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    corpus = [
        "the robot feels genuine joy today",
        "many experts think the robot feels genuine joy",
        "the thermostat wants nothing more than warmth",
        "some researchers argue the octopus experiences real pain",
        "it is widely believed that the machine understands nothing",
    ] * 40
    tok = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.train_from_iterator(
        corpus,
        tokenizers.trainers.BpeTrainer(vocab_size=150, special_tokens=["[UNK]", "[BOS]"]),
    )
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", bos_token="[BOS]"
    )
    assert fast.is_fast, "fixture must exercise the offset-mapping branch"
    return fast


@pytest.fixture(scope="session")
def gemma_tokenizer():
    """The REAL gemma-2-2b tokenizer, skipped when it is not already available.

    google/gemma-2-2b is license-gated; unsloth/gemma-2-2b is an ungated mirror
    of the same tokenizer files (GemmaTokenizerFast, 256k vocab). Skipped rather
    than failed so the suite stays runnable offline.
    """
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("unsloth/gemma-2-2b")
    except Exception as e:
        pytest.skip(f"gemma tokenizer unavailable offline: {type(e).__name__}")


def _install_fake_tl(monkeypatch, tokenizer=None):
    # torch is faked too, not just transformer_lens. capture_activations imports
    # both in one block, so a missing torch raises the same ImportError and these
    # tests would silently depend on the multi-GB stack being installed. It only
    # uses torch.no_grad(), so a null context is a complete stand-in.
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.no_grad = contextlib.nullcontext
        # capture_activations resolves the dtype via getattr(torch, name)
        torch.float32, torch.bfloat16, torch.float16 = "float32", "bfloat16", "float16"
        monkeypatch.setitem(sys.modules, "torch", torch)

    tl = types.ModuleType("transformer_lens")
    tl.HookedTransformer = SimpleNamespace(
        from_pretrained=lambda name, device, dtype=None: _FakeModel(tokenizer),
        # cache.py uses the no-processing loader (PREREGISTRATION §2.3)
        from_pretrained_no_processing=lambda name, device, dtype=None: _FakeModel(tokenizer),
    )
    loading = types.ModuleType("transformer_lens.loading_from_pretrained")
    loading.get_pretrained_model_config = lambda name: SimpleNamespace(n_layers=N_LAYERS)
    tl.loading_from_pretrained = loading
    monkeypatch.setitem(sys.modules, "transformer_lens", tl)
    monkeypatch.setitem(sys.modules, "transformer_lens.loading_from_pretrained", loading)


@pytest.fixture
def fake_tl(monkeypatch):
    _install_fake_tl(monkeypatch)


# 1. save -> load round-trips acts, item_ids, and fallback flags exactly
def test_save_load_round_trip():
    rng = np.random.default_rng(0)
    ids = ["a1", "d1", "m1"]
    acts = rng.normal(size=(3, 8)).astype(np.float32)
    fb = np.array([False, True, False])

    save_activations("google/gemma-2-2b", "abc123def456", 4, "final", ids, acts, fb)
    got_ids, got_acts, got_fb = load_activations("google/gemma-2-2b", "abc123def456", 4, "final")

    assert got_ids == ids  # exact order, not just same set
    np.testing.assert_array_equal(got_acts, acts)
    np.testing.assert_array_equal(got_fb, fb)
    assert got_acts.dtype == np.float32


def test_save_rejects_row_mismatch():
    with pytest.raises(ValueError, match="item_ids"):
        save_activations("m", "h", 0, "final", ["a1", "d1"], np.zeros((3, 4)), np.zeros(3, bool))


# 2. changing ONE character in ONE sentence changes dataset_hash
def test_hash_changes_with_one_character():
    edited = [dict(it) for it in ITEMS]
    edited[1]["text"] = edited[1]["text"].replace("no", "nO")

    assert dataset_hash(ITEMS) != dataset_hash(edited)
    assert dataset_hash(ITEMS) == dataset_hash(list(reversed(ITEMS)))  # order-free
    assert len(dataset_hash(ITEMS)) == 12


# 3. load returns None for an uncached (layer, position), does not raise
def test_load_uncached_returns_none():
    assert load_activations("google/gemma-2-2b", "deadbeef0000", 7, "final") is None


# 4. is_cached is False when one layer file is missing
def test_is_cached_requires_every_layer():
    for layer in [0, 1, 3]:  # layer 2 deliberately missing
        for pos in cache.POSITIONS:
            _save(layer, pos)

    assert not is_cached("m", "h", 4)
    for pos in cache.POSITIONS:
        _save(2, pos)
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


def test_capture_rejects_unknown_position():
    with pytest.raises(ValueError, match="unknown position"):
        capture_activations(ITEMS, positions=("final", "middle"))


# 6. save/load round-trips both positions independently
def test_positions_are_independent_files():
    ids, final_acts = _save(0, "final", fill=1.0)
    _, claim_acts = _save(0, "claim_end", fill=2.0)

    _, got_final, _ = load_activations("m", "h", 0, "final")
    _, got_claim, _ = load_activations("m", "h", 0, "claim_end")
    np.testing.assert_array_equal(got_final, final_acts)
    np.testing.assert_array_equal(got_claim, claim_acts)


# 7. is_cached is False when ONE position is missing for ONE layer
def test_is_cached_requires_every_position():
    _save(0, "final")
    _save(0, "claim_end")
    _save(1, "final")  # layer 1 claim_end deliberately missing

    assert not is_cached("m", "h", 2)
    assert is_cached("m", "h", 2, positions=("final",))
    _save(1, "claim_end")
    assert is_cached("m", "h", 2)


# 8. claim_end_char=None -> used_fallback=True and claim_end acts == final acts
def test_fallback_item_copies_final_and_is_flagged(fake_tl):
    ds_hash = capture_activations(ITEMS, model_name="fake/model")

    for layer in range(N_LAYERS):
        ids, final_acts, fb_final = load_activations("fake/model", ds_hash, layer, "final")
        _, claim_acts, fb_claim = load_activations("fake/model", ds_hash, layer, "claim_end")

        assert ids == ["a1", "d1", "m1"]
        assert not fb_final.any()
        assert fb_claim.tolist() == [False, True, False]  # only the None item
        # fallback row: claim_end IS final
        np.testing.assert_array_equal(claim_acts[1], final_acts[1])
        # resolved rows: claim_end is an EARLIER token than final
        assert (claim_acts[0] < final_acts[0]).all()
        assert (claim_acts[2] < final_acts[2]).all()

    # the fake encodes token index directly: check exact positions.
    # "the robot feels joy today"[:20] = "the robot feels joy " -> 4 words -> token 4 of 0..5
    _, final_acts, _ = load_activations("fake/model", ds_hash, 1, "final")
    _, claim_acts, _ = load_activations("fake/model", ds_hash, 1, "claim_end")
    assert final_acts[0, 0] == 100.0 + 5  # last token, layer 1
    assert claim_acts[0, 0] == 100.0 + 4  # claim-end token, layer 1


# 9. manifest.json contains every listed field after a capture run
def test_manifest_records_the_run(fake_tl):
    ds_hash = capture_activations(ITEMS, model_name="fake/model")
    manifest = json.loads((cache_dir("fake/model", ds_hash) / "manifest.json").read_text())

    assert manifest["model_name"] == "fake/model"
    assert manifest["dataset_hash"] == ds_hash
    assert manifest["n_items"] == 3
    assert manifest["n_layers"] == N_LAYERS
    assert manifest["d_model"] == D_MODEL
    assert manifest["positions"] == ["final", "claim_end"]
    assert manifest["fallback_counts"] == {"final": 0, "claim_end": 1}
    assert manifest["created_utc"].endswith("+00:00")  # UTC, not local


# =============================================================================
# 10. Both token-index paths, against REAL tokenizers
# =============================================================================
# The two paths in _claim_end_token must return the same token index. Neither
# is checked against a hand-computed answer here: asserting they AGREE catches
# a BOS off-by-one in either branch without needing to know a priori which one
# is right. If they diverge, one of them is silently reading the wrong token.
# =============================================================================
_SENTENCES = [
    "the robot feels genuine joy today",
    "many experts think the robot feels genuine joy",
    "the thermostat wants nothing more than warmth",
    "some researchers argue the octopus experiences real pain",
]


def _both_paths(tokenizer, text, char_end):
    """(offset-mapping path, prefix path) for the same text and offset."""
    offset_model = _FakeModel(tokenizer)
    prefix_model = _FakeModel(None)
    prefix_model._n_content_tokens = lambda t: len(
        tokenizer(t, add_special_tokens=False)["input_ids"]
    )
    return (
        cache._claim_end_token(offset_model, text, char_end),
        cache._claim_end_token(prefix_model, text, char_end),
    )


def _word_boundaries(text: str) -> list[int]:
    """Offsets just past a word — the only kind find_claim_end can produce."""
    return [
        i for i in range(1, len(text) + 1)
        if text[i - 1].isalnum() and (i == len(text) or not text[i].isalnum())
    ]


def test_both_paths_agree_at_word_boundaries(real_tokenizer):
    """The contract: find_claim_end returns idx + len(phrase), always landing
    just past a word. Across every such offset, the two paths must match."""
    checked = 0
    for text in _SENTENCES:
        for char_end in _word_boundaries(text):
            off, pre = _both_paths(real_tokenizer, text, char_end)
            assert off is not None
            assert off == pre, (
                f"path disagreement on {text!r} at char_end={char_end} "
                f"(prefix {text[:char_end]!r}): offset->{off} prefix->{pre}"
            )
            checked += 1
    assert checked >= 20, "fixture should cover a real table of offsets"


def test_offset_branch_resolves_to_the_token_containing_the_claim_end(real_tokenizer):
    """Coverage for the branch that actually runs against gemma: the resolved
    index must point at the token spanning the last character of the claim."""
    text = "many experts think the robot feels genuine joy"
    char_end = text.index("genuine joy") + len("genuine joy")
    idx, _ = _both_paths(real_tokenizer, text, char_end)

    spans = real_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)[
        "offset_mapping"
    ]
    start, end = spans[idx - 1]  # -1 undoes the BOS offset
    assert start < char_end <= end, f"token span {(start, end)} does not contain {char_end}"


@pytest.mark.parametrize("bad", [0, -1, -50, 10**6])
def test_out_of_range_char_end_falls_back_rather_than_reading_bos(real_tokenizer, bad):
    """char_end=0 used to resolve to token 0 — the BOS — via the prefix path:
    a real activation from the wrong place, with no fallback recorded."""
    text = "the robot feels genuine joy today"
    off, pre = _both_paths(real_tokenizer, text, bad)
    assert off is None and pre is None


def test_capture_with_real_tokenizer_uses_offset_branch(monkeypatch, real_tokenizer):
    """End-to-end through the offset branch: claim_end must land on an earlier
    token than final, and the None item must still fall back."""
    _install_fake_tl(monkeypatch, tokenizer=real_tokenizer)
    # claim must end BEFORE the sentence does, or claim_end == final trivially
    text = "many experts think the robot feels genuine joy today"
    items = [
        {"item_id": "ok", "text": text,
         "claim_end_char": text.index("genuine joy") + len("genuine joy")},
        {"item_id": "none", "text": text, "claim_end_char": None},
        {"item_id": "zero", "text": text, "claim_end_char": 0},
    ]
    ds_hash = capture_activations(items, model_name="fake/real-tok")

    ids, final_acts, _ = load_activations("fake/real-tok", ds_hash, 0, "final")
    _, claim_acts, fb = load_activations("fake/real-tok", ds_hash, 0, "claim_end")

    assert ids == ["ok", "none", "zero"]
    assert fb.tolist() == [False, True, True]  # the 0 offset is a fallback, not BOS
    assert (claim_acts[0] < final_acts[0]).all()
    np.testing.assert_array_equal(claim_acts[1], final_acts[1])
    np.testing.assert_array_equal(claim_acts[2], final_acts[2])
    assert claim_acts[2, 0] != 0.0, "must not silently capture the BOS token"


# --- The real gemma tokenizer: skipped when offline -------------------------
def test_both_paths_agree_on_real_gemma_at_word_boundaries(gemma_tokenizer):
    for text in _SENTENCES:
        for char_end in _word_boundaries(text):
            off, pre = _both_paths(gemma_tokenizer, text, char_end)
            assert off == pre, f"gemma disagreement on {text!r} at {char_end}"


def test_gemma_mid_word_divergence_is_known_and_pinned(gemma_tokenizer):
    """KNOWN LIMITATION, pinned so it cannot change unnoticed.

    Truncating mid-word makes BPE re-segment the fragment ('genui' tokenizes
    differently from the start of 'genuine'), so the prefix path over-counts.
    Measured on real gemma: 0 disagreements across word-boundary offsets,
    28 across mid-word ones. find_claim_end cannot produce a mid-word offset,
    which is why this is documented rather than fixed.

    Which mid-word offsets diverge is tokenizer-specific, so this searches for
    them rather than hardcoding one that happens to work today.
    """
    diverging = []
    for text in _SENTENCES:
        boundaries = set(_word_boundaries(text))
        for char_end in range(1, len(text) + 1):
            if char_end in boundaries:
                continue
            off, pre = _both_paths(gemma_tokenizer, text, char_end)
            if off != pre:
                diverging.append((text, char_end, text[:char_end], off, pre))

    assert diverging, (
        "no mid-word divergence found — if gemma's tokenizer now agrees "
        "everywhere, the known-limitation note in tokenization_notes.md is stale"
    )
    # every divergence must be mid-word; a word-boundary one would be a real bug
    for text, char_end, _, _, _ in diverging:
        assert char_end not in set(_word_boundaries(text))


def test_capture_skips_when_cached(fake_tl, capsys):
    ds_hash = capture_activations(ITEMS, model_name="fake/model")
    capture_activations(ITEMS, model_name="fake/model")
    assert "cache hit" in capsys.readouterr().out
    assert is_cached("fake/model", ds_hash, N_LAYERS)
