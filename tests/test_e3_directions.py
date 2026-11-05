"""
Offline test for scripts/run_e3_directions.py — no model, no torch.

We write FAKE activation caches for the real stimulus files (so the dataset
hashes and the item-order assert are exercised for real), plant a known
"mind" signal along one axis, and check the script finds it, saves raw AND
unit vectors for every layer, and that raw/|raw| == unit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import lib.harness.cache as cache  # noqa: E402
import run_e3_directions as e3  # noqa: E402
from run_e1 import load_rows  # noqa: E402

N_LAYERS, D = 3, 8
SIGNAL_AXIS = {"core": 0, "placebo": 1}  # mind signal on axis 0, placebo on axis 1


def _write_fake_cache(model, ds_hash, stimuli, axis, rng):
    rows = load_rows(ROOT / stimuli, ROOT / "data/schema.json")
    ids = [r["item_id"] for r in rows]
    sign = np.array([1.0 if r["polarity"] == "affirm" else -1.0 for r in rows])
    for layer in range(N_LAYERS):
        acts = rng.normal(size=(len(rows), D)) + 10.0  # offset = "typical activation"
        acts[:, axis] += 3.0 * sign
        for pos in ("claim_end", "final"):
            cache.save_activations(model, ds_hash, layer, pos, ids, acts,
                                   np.zeros(len(rows), bool))
    (cache.cache_dir(model, ds_hash) / "manifest.json").write_text(
        json.dumps({"n_layers": N_LAYERS}))


@pytest.fixture
def fake_caches(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)  # script reads data/ relative to the repo root
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "cache")
    rng = np.random.default_rng(0)
    d = e3.DEFAULTS
    for model in ("fake/it", "fake/base"):
        _write_fake_cache(model, d["core_hash"], d["stimuli"], 0, rng)
        _write_fake_cache(model, d["heldout_hash"], d["heldout_stimuli"], 0, rng)
        _write_fake_cache(model, d["placebo_hash"], d["placebo_stimuli"], 1, rng)
        _write_fake_cache(model, d["placebo_heldout_hash"],
                          d["placebo_heldout_stimuli"], 1, rng)
    return tmp_path


def test_saves_raw_and_unit_for_every_layer(fake_caches):
    out = fake_caches / "out"
    assert e3.main(["--model", "fake/it", "--compare-model", "fake/base",
                    "--out-dir", str(out)]) == 0

    z = np.load(out / "fake_it_claim_end_directions.npz")
    assert z["mind_raw"].shape == (N_LAYERS, D) == z["mind_unit"].shape
    assert z["placebo_raw"].shape == (N_LAYERS, D)
    # unit is exactly raw rescaled to length 1
    norms = np.linalg.norm(z["mind_raw"], axis=1, keepdims=True)
    np.testing.assert_allclose(z["mind_raw"] / norms, z["mind_unit"], atol=1e-9)
    # raw keeps its size: planted gap is 2 * 3.0 = 6 along axis 0
    assert np.all(np.abs(norms.ravel() - 6.0) < 1.0)
    # it found the planted axes
    assert np.all(np.argmax(np.abs(z["mind_unit"]), axis=1) == 0)
    assert np.all(np.argmax(np.abs(z["placebo_unit"]), axis=1) == 1)

    s = json.loads((out / "fake_it_claim_end_directions.json").read_text())
    for row in s["per_layer"]:
        assert row["mind_heldout_acc"] > 0.95
        assert abs(row["cos_mind_placebo"]) < 0.3
        assert row["cos_with_base"] > 0.95  # same planted signal in both fakes


def test_missing_compare_cache_is_skipped_not_fatal(fake_caches):
    out = fake_caches / "out2"
    assert e3.main(["--model", "fake/it", "--compare-model", "fake/nothing",
                    "--out-dir", str(out)]) == 0
    s = json.loads((out / "fake_it_claim_end_directions.json").read_text())
    assert s["per_layer"][0]["cos_with_base"] is None
    assert s["cos_with_base_note"].startswith("skipped")
