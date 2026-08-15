"""
lib/harness/cache.py — sentences in, cached activation matrices out.

The ONLY file that touches the model. Everything downstream is numpy on .npz
files, so analysis can run anywhere; capture runs once, locally, on CPU.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

CACHE_ROOT = Path("results/activation_cache")


def dataset_hash(items: list[dict]) -> str:
    """Turn the full set of sentences into a short fingerprint that changes if any sentence changes.

    LOGIC: hashed from sentence TEXT so editing any stimulus automatically
    invalidates the cache — analyzing a stale dataset produces plausible
    numbers and never crashes, so it must be impossible.
    """
    lines = sorted(f"{it['item_id']}::{it['text']}" for it in items)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:12]


def cache_dir(model_name: str, ds_hash: str) -> Path:
    """Give the folder where this model + dataset combination's activations live."""
    return CACHE_ROOT / model_name.replace("/", "_") / ds_hash


def _layer_path(model_name: str, ds_hash: str, layer: int) -> Path:
    return cache_dir(model_name, ds_hash) / f"layer_{layer:03d}.npz"


def save_activations(
    model_name: str, ds_hash: str, layer: int, item_ids: list[str], acts: np.ndarray
) -> Path:
    """Write one layer's activation matrix to disk, with the item ids stored alongside it.

    LOGIC: item_ids live INSIDE the npz, not a sidecar, so row alignment
    travels with the data and the analysis side can assert on it rather than
    trusting row order.
    """
    acts = np.asarray(acts, dtype=np.float32)
    if len(item_ids) != acts.shape[0]:
        raise ValueError(
            f"layer {layer}: {len(item_ids)} item_ids vs {acts.shape[0]} activation rows"
        )
    path = _layer_path(model_name, ds_hash, layer)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, acts=acts, item_ids=np.array(item_ids, dtype=str))
    return path


def load_activations(
    model_name: str, ds_hash: str, layer: int
) -> tuple[list[str], np.ndarray] | None:
    """Read one layer's cached activations back, or report None if it was never saved."""
    path = _layer_path(model_name, ds_hash, layer)
    if not path.exists():
        return None
    with np.load(path) as z:
        return z["item_ids"].tolist(), z["acts"]


def is_cached(model_name: str, ds_hash: str, n_layers: int) -> bool:
    """Say whether every single layer's file is already on disk."""
    return all(
        _layer_path(model_name, ds_hash, layer).exists() for layer in range(n_layers)
    )


def capture_activations(
    items: list[dict],
    model_name: str = "google/gemma-2-2b",
    device: str = "cpu",
    force: bool = False,
) -> str:
    """Run every sentence through the model once and save what the model was 'thinking' at each layer.

    One forward pass per item, no generation. resid_post at every block, final
    token position only ([0, -1]) — both preregistered choices. All layers come
    from the same pass: they are free once it is running.
    """
    try:
        import torch
        from transformer_lens import HookedTransformer
    except ImportError as e:
        raise ImportError(
            "transformer_lens (and torch) are required for capture_activations. "
            "This function runs LOCALLY on your machine, not in a sandbox — "
            "install with: pip install transformer_lens torch"
        ) from e

    ds_hash = dataset_hash(items)
    n_items = len(items)

    # Config is cheap; weights are not. Check the cache before paying for them.
    from transformer_lens.loading_from_pretrained import get_pretrained_model_config

    n_layers = get_pretrained_model_config(model_name).n_layers
    if is_cached(model_name, ds_hash, n_layers) and not force:
        print(f"cache hit for {model_name} @ {ds_hash} ({n_layers} layers); skipping")
        return ds_hash

    model = HookedTransformer.from_pretrained(model_name, device=device)
    d_model = model.cfg.d_model

    per_layer = {L: np.zeros((n_items, d_model), dtype=np.float32) for L in range(n_layers)}
    item_ids = [it["item_id"] for it in items]

    t0 = time.time()
    with torch.no_grad():
        for i, it in enumerate(items):  # row index == input list index, always
            _, cache = model.run_with_cache(it["text"])
            for L in range(n_layers):
                per_layer[L][i] = (
                    cache["resid_post", L][0, -1].float().cpu().numpy()
                )
            if (i + 1) % 25 == 0 or i + 1 == n_items:
                print(f"item {i + 1} of {n_items}, {time.time() - t0:.0f}s elapsed")

    del model  # weights are the memory hog; drop them before the writes

    for L in range(n_layers):
        save_activations(model_name, ds_hash, L, item_ids, per_layer[L])
    return ds_hash
