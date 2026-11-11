"""Offline tests for lib/steering/steer.py: numpy stands in for torch, no model needed."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.steering.steer import make_steering_hook, orthogonalize, random_direction  # noqa: E402


def test_prompt_pass_steers_reply_only():
    resid = np.zeros((1, 6, 4))           # 6 positions, of which the prompt is the first 4
    vec = np.array([1.0, 0, 0, 0])
    make_steering_hook(vec, 2.0, prompt_len=4)(resid)
    assert np.all(resid[0, :3] == 0)      # question tokens untouched
    assert np.all(resid[0, 3:, 0] == 2.0)  # last prompt token + reply pushed by 2 × vector


def test_cached_generation_step_is_steered():
    resid = np.zeros((1, 1, 4))           # one new token per pass with a KV cache
    make_steering_hook(np.ones(4), -1.0, prompt_len=4)(resid)
    assert np.all(resid == -1.0)


def test_zero_multiplier_changes_nothing():
    resid = np.random.default_rng(0).normal(size=(1, 5, 4))
    before = resid.copy()
    make_steering_hook(np.ones(4), 0.0, prompt_len=2)(resid)
    assert np.array_equal(resid, before)


def test_orthogonalize_removes_placebo_part_and_keeps_length():
    rng = np.random.default_rng(1)
    mind, plc = rng.normal(size=8), rng.normal(size=8)
    out = orthogonalize(mind, plc / np.linalg.norm(plc))
    assert abs(out @ plc) < 1e-9
    assert np.isclose(np.linalg.norm(out), np.linalg.norm(mind))


def test_random_direction_matches_length_and_is_reproducible():
    v = np.arange(1.0, 9.0)
    a, b = random_direction(v, seed=3), random_direction(v, seed=3)
    assert np.array_equal(a, b) and np.isclose(np.linalg.norm(a), np.linalg.norm(v))
