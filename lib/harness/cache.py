"""
lib/harness/cache.py — sentences in, cached activation matrices out.

The ONLY file that touches the model. Everything downstream is numpy on .npz
files, so analysis can run anywhere; capture runs once, locally, on CPU.

TWO token positions per sentence, one file per (layer, position):
  "final"      token -1, always available.
  "claim_end"  the token where the mind-claim ends. Templates put the final
               token a DIFFERENT distance past the claim, so final-token
               position interacts with template identity; claim_end means the
               same thing in every template. Which one is the headline result
               is declared in PREREGISTRATION.md before extraction — the other
               is the robustness check, never a post-hoc choice.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CACHE_ROOT = Path("results/activation_cache")
POSITIONS = ("final", "claim_end")


def dataset_hash(items: list[dict]) -> str:
    """Turn the full set of sentences into a short fingerprint that changes if any sentence changes.

    LOGIC: hashed from sentence TEXT so editing any stimulus automatically
    invalidates the cache — analyzing a stale dataset produces plausible
    numbers and never crashes, so it must be impossible.
    """
    lines = sorted(f"{it['item_id']}::{it['text']}" for it in items)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:12]


def cache_dir(model_name: str, ds_hash: str) -> Path:
    """Give the folder where this model + dataset combination's activations live.

    Position lives in the FILENAME (layer_<LLL>_<position>.npz), not the path,
    so one manifest.json can describe the whole capture.
    """
    return CACHE_ROOT / model_name.replace("/", "_") / ds_hash


def _layer_path(model_name: str, ds_hash: str, layer: int, position: str) -> Path:
    return cache_dir(model_name, ds_hash) / f"layer_{layer:03d}_{position}.npz"


def save_activations(
    model_name: str, ds_hash: str, layer: int, position: str,
    item_ids: list[str], acts: np.ndarray, used_fallback: np.ndarray,
) -> Path:
    """Write one (layer, position) activation matrix to disk with its item ids and fallback flags.

    LOGIC: item_ids and used_fallback live INSIDE the npz, not a sidecar, so
    row alignment and "this row is secretly final-token" both travel with the
    data instead of being trusted from memory.
    """
    acts = np.asarray(acts, dtype=np.float32)
    used_fallback = np.asarray(used_fallback, dtype=bool)
    if not (len(item_ids) == acts.shape[0] == len(used_fallback)):
        raise ValueError(
            f"layer {layer}/{position}: {len(item_ids)} item_ids vs "
            f"{acts.shape[0]} rows vs {len(used_fallback)} fallback flags"
        )
    path = _layer_path(model_name, ds_hash, layer, position)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, acts=acts, item_ids=np.array(item_ids, dtype=str),
             used_fallback=used_fallback)
    return path


def load_activations(
    model_name: str, ds_hash: str, layer: int, position: str
) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    """Read one (layer, position) file back as (item_ids, acts, used_fallback), or None if absent."""
    path = _layer_path(model_name, ds_hash, layer, position)
    if not path.exists():
        return None
    with np.load(path) as z:
        return z["item_ids"].tolist(), z["acts"], z["used_fallback"]


def is_cached(
    model_name: str, ds_hash: str, n_layers: int, positions: tuple[str, ...] = POSITIONS
) -> bool:
    """Say whether every (layer, position) file is already on disk."""
    return all(
        _layer_path(model_name, ds_hash, L, p).exists()
        for L in range(n_layers) for p in positions
    )


def _to_f32(x) -> np.ndarray:
    """Make a float32 numpy vector out of a torch tensor or array, whatever the model's dtype."""
    if hasattr(x, "detach"):  # torch tensor (incl. bfloat16, which numpy can't hold)
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _claim_end_token(model, text: str, char_end: int | None) -> int | None:
    """Turn a character offset into the index of the token that finishes the claim, or None.

    Offset mapping when the tokenizer provides one; otherwise tokenize the
    prefix up to char_end and take its last token's index. Never raises —
    the caller treats None as "fall back to final and record it".

    DEFENDS LOCALLY: char_end arrives from another module (stimuli.py) or from
    a hand-edited CSV, so it is validated here rather than trusted. char_end=0
    is the dangerous one: the prefix path computes token index 0, which is the
    BOS token — a real activation from the wrong place, with nothing marking
    it. An offset this module did not compute does not get believed.
    """
    if char_end is None or char_end <= 0 or char_end > len(text):
        return None
    try:
        tok = getattr(model, "tokenizer", None)
        if tok is not None and getattr(tok, "is_fast", False):
            spans = tok(text, return_offsets_mapping=True,
                        add_special_tokens=False)["offset_mapping"]
            covering = [j for j, (s, _) in enumerate(spans) if s < char_end]
            if not covering:
                return None
            bos = 1 if getattr(model.cfg, "default_prepend_bos", True) else 0
            return covering[-1] + bos
        # prefix and full tokenization share the BOS, so len(prefix)-1 indexes
        # the prefix's last token within the full sequence
        return int(model.to_tokens(text[:char_end]).shape[-1]) - 1
    except Exception:
        return None


def capture_activations(
    items: list[dict],
    model_name: str = "google/gemma-2-2b",
    device: str = "cpu",
    force: bool = False,
    positions: tuple[str, ...] = POSITIONS,
    dtype: str = "float32",
) -> str:
    """Run every sentence through the model once and save what it was 'thinking' at each layer, at both token positions.

    One forward pass per item, no generation; resid_post at every block. All
    layers AND both positions come from the same pass — they are free once it
    is running, and a second pass would waste the weekend's scarcest resource.
    """
    for p in positions:
        if p not in POSITIONS:
            raise ValueError(f"unknown position {p!r}; expected subset of {POSITIONS}")
    try:
        import torch
        from transformer_lens import HookedTransformer
        from transformer_lens.loading_from_pretrained import get_pretrained_model_config
    except ImportError as e:
        raise ImportError(
            "transformer_lens (and torch) are required for capture_activations. "
            "This function runs LOCALLY on your machine, not in a sandbox — "
            "install with: pip install transformer_lens torch"
        ) from e

    ds_hash = dataset_hash(items)
    n_items = len(items)

    # Config is cheap; weights are not. Check the cache before paying for them.
    n_layers = get_pretrained_model_config(model_name).n_layers
    if is_cached(model_name, ds_hash, n_layers, positions) and not force:
        # dataset_hash hashes stimulus TEXT, not the model stack — so a cache
        # captured at a different dtype is indistinguishable by path. Refuse it
        # loudly rather than silently mixing precisions in one analysis.
        prior = cache_dir(model_name, ds_hash) / "manifest.json"
        if prior.exists():
            prior_dtype = json.loads(prior.read_text()).get("dtype", "float32")
            if prior_dtype != dtype:
                raise ValueError(
                    f"cache at {prior.parent} was captured with dtype={prior_dtype!r} "
                    f"but this run asks for {dtype!r}. The dataset_hash cannot see "
                    f"precision. Use --force to recapture, or the prior dtype."
                )
        print(f"cache hit for {model_name} @ {ds_hash} "
              f"({n_layers} layers x {positions}); skipping")
        return ds_hash

    torch_dtype = getattr(torch, dtype)
    # from_pretrained_no_processing, NOT from_pretrained. Weight processing
    # (fold_ln, center_writing_weights, center_unembed) alters resid_post — the
    # exact tensor cached here — and transformer_lens warns that the folding
    # arithmetic degrades at reduced precision. Combining bf16 with processing
    # is the unsound pairing; skipping processing is what the library advises.
    # Declared in PREREGISTRATION.md §2.3 because it changes the numbers.
    model = HookedTransformer.from_pretrained_no_processing(
        model_name, device=device, dtype=torch_dtype
    )
    d_model = model.cfg.d_model

    acts = {(L, p): np.zeros((n_items, d_model), dtype=np.float32)
            for L in range(n_layers) for p in positions}
    fallback = {p: np.zeros(n_items, dtype=bool) for p in positions}
    item_ids = [it["item_id"] for it in items]

    t0 = time.time()
    with torch.no_grad():
        for i, it in enumerate(items):  # row index == input list index, always
            _, cache = model.run_with_cache(it["text"])
            seq_len = cache["resid_post", 0].shape[1]
            tok_idx: dict[str, int] = {}
            for p in positions:
                idx = -1  # "final", and the fallback for everything else
                if p == "claim_end":
                    resolved = _claim_end_token(model, it["text"], it.get("claim_end_char"))
                    if resolved is not None and 0 <= resolved < seq_len:
                        idx = resolved
                    else:
                        fallback[p][i] = True  # recorded, never silent
                tok_idx[p] = idx
            for L in range(n_layers):
                resid = cache["resid_post", L]
                for p in positions:
                    acts[(L, p)][i] = _to_f32(resid[0, tok_idx[p]])
            if (i + 1) % 25 == 0 or i + 1 == n_items:
                print(f"item {i + 1} of {n_items}, {time.time() - t0:.0f}s elapsed")

    del model  # weights are the memory hog; drop them before the writes

    for (L, p), matrix in acts.items():
        save_activations(model_name, ds_hash, L, p, item_ids, matrix, fallback[p])

    # LOGIC: on Sunday, when a number looks strange, this is what lets you
    # reconstruct exactly what produced it instead of guessing.
    manifest = {
        "model_name": model_name,
        "dataset_hash": ds_hash,
        "n_items": n_items,
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "dtype": dtype,
        "weight_processing": "none (from_pretrained_no_processing)",
        "positions": list(positions),
        "fallback_counts": {p: int(fb.sum()) for p, fb in fallback.items()},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (cache_dir(model_name, ds_hash) / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return ds_hash
