"""
lib/steering/steer.py — the E3 steering pieces. Pure array maths, no model.

Works on torch tensors (on Kaggle) and numpy arrays (in the offline tests),
because both support the same slicing and += operations.
"""

from __future__ import annotations

import numpy as np


def make_steering_hook(vector, multiplier: float, prompt_len: int):
    """Return a TransformerLens hook that adds multiplier * vector to the residual stream,
    on the model's REPLY only (decision 2026-11-11, following CAA).

    resid has shape (batch, positions, d_model). The question's tokens are left alone; we
    start at prompt_len - 1, the last prompt token, because that position is the one that
    predicts the first reply token. During generation with a KV cache the model sees one new
    token per pass (positions == 1), and every such token is part of the reply.
    """
    def hook(resid, hook=None):
        if resid.shape[1] == 1:
            resid += multiplier * vector
        else:
            resid[:, prompt_len - 1:, :] += multiplier * vector
        return resid
    return hook


def orthogonalize(mind_raw: np.ndarray, placebo_unit: np.ndarray) -> np.ndarray:
    """The mind direction with its placebo (general affirm/deny) part removed, rescaled to
    the mind vector's original length so ×1 means the same size of push in both conditions."""
    p = placebo_unit / np.linalg.norm(placebo_unit)
    rest = mind_raw - (mind_raw @ p) * p
    return rest / np.linalg.norm(rest) * np.linalg.norm(mind_raw)


def random_direction(like: np.ndarray, seed: int = 0) -> np.ndarray:
    """A random direction with the same length as `like` (the random-vector control)."""
    r = np.random.default_rng(seed).normal(size=like.shape)
    return r / np.linalg.norm(r) * np.linalg.norm(like)
